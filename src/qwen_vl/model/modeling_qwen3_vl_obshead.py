# FutureNav eval-compatible wrapper.
# Auxiliary observation heads are retained for checkpoint compatibility.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext

from transformers.utils import logging
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)

try:
    from deepspeed.zero import GatheredParameters
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

from .modeling_qwen3_vl import Qwen3VLForConditionalGenerationForFutureNav

logger = logging.get_logger(__name__)

ACTION_MAP = {
    'STOP': 0,
    'MOVE_FORWARD': 1,
    'TURN_LEFT': 2,
    'TURN_RIGHT': 3,
}


class Qwen3VLForFutureNav(Qwen3VLForConditionalGenerationForFutureNav):
    """
    FutureNav model wrapper used by evaluation.

    The auxiliary observation heads are kept so checkpoints trained with those
    modules still load cleanly, but auxiliary losses are not computed here.
    """

    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.text_config.hidden_size
        vggt_dim = 2048

        self.forward_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vggt_dim),
        )

        self.inverse_next_proj = nn.Sequential(
            nn.LayerNorm(vggt_dim),
            nn.Linear(vggt_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.inverse_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )

        self.gen_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vggt_dim),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, mode=None, **kwargs):
        vggt_model_path = kwargs.pop("vggt_model_path", None)
        model = super(Qwen3VLForConditionalGenerationForFutureNav, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.mode = mode
        if vggt_model_path:
            from .vggt.models.vggt import VGGT
            print(f"Loading VGGT from {vggt_model_path}")
            vggt = VGGT.from_pretrained(vggt_model_path)
            vggt.camera_head = None
            vggt.track_head = None
            model.vggt = vggt
            for param in model.vggt.parameters():
                param.requires_grad = False
        return model

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer
        self._action_token_map = {}
        for action_name, action_id in ACTION_MAP.items():
            tokens = tokenizer.encode(action_name, add_special_tokens=False)
            self._action_token_map[tuple(tokens)] = action_id

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        images_vggt: Optional[List[torch.Tensor]] = None,
        boxes: Optional[List[torch.Tensor]] = None,
        action_labels: Optional[torch.Tensor] = None,
        prev_action_labels: Optional[torch.Tensor] = None,
        next_vggt_targets: Optional[List] = None,
        next_frame_vggt: Optional[List] = None,
        tag: str = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            self.vggt.eval()
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                batch_size = inputs_embeds.shape[0]
                image_embeds_3d = []

                vggt_params = list(self.vggt.parameters())
                vggt_context = GatheredParameters(vggt_params, modifier_rank=None) if HAS_DEEPSPEED and any(hasattr(p, 'ds_id') for p in vggt_params) else nullcontext()

                with vggt_context:
                    for i in range(batch_size):
                        if images_vggt[i].shape[0] > 0:
                            n_image = 1
                            height, width = images_vggt[i].shape[-2:]
                            vggt_patch_size = 14
                            merge_size = self.config.vision_config.spatial_merge_size
                            h_grid, w_grid = height // vggt_patch_size, width // vggt_patch_size
                            h_grid_after_merge = h_grid // merge_size
                            w_grid_after_merge = w_grid // merge_size

                            all_frames = images_vggt[i]
                            n_original_frames = all_frames.shape[0]

                            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                            with torch.no_grad():
                                with torch.amp.autocast('cuda', dtype=dtype):
                                    if not self.config.reference_frame == "first":
                                        all_frames = torch.flip(all_frames, dims=(0,))

                                    if self.mode == "evaluation":
                                        if self.past_key_values_vggt is None:
                                            self.past_key_values_vggt = [None] * self.vggt.aggregator.depth
                                    else:
                                        self.past_key_values_vggt = [None] * self.vggt.aggregator.depth

                                    current_features = None
                                    for k, frame in enumerate(all_frames):
                                        images = frame.unsqueeze(0).unsqueeze(0)
                                        aggregator_output = self.vggt.aggregator(
                                            images,
                                            past_key_values=self.past_key_values_vggt,
                                            use_cache=True,
                                            past_frame_idx=k,
                                        )

                                        if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                                            aggregated_tokens, patch_start_idx, past_key_values_vggt = aggregator_output
                                        else:
                                            aggregated_tokens, patch_start_idx = aggregator_output

                                        self.past_key_values_vggt = past_key_values_vggt
                                        if self.mode == "evaluation" and self.past_key_values_vggt is not None:
                                            self.past_key_values_vggt = self.kv_cache_vggt(self.past_key_values_vggt)

                                        feat = aggregated_tokens[-2][0, :, patch_start_idx:]

                                        if k == n_original_frames - 1:
                                            current_features = feat.clone()

                            features = current_features
                            if not self.config.reference_frame == "first":
                                features = torch.flip(features, dims=(0,))

                            features = features.view(n_image, h_grid, w_grid, -1)
                            features = features[:, :h_grid_after_merge * merge_size, :w_grid_after_merge * merge_size, :].contiguous()
                            features = features.view(n_image, h_grid_after_merge, merge_size, w_grid_after_merge, merge_size, -1)
                            features = features.permute(0, 1, 3, 2, 4, 5).contiguous().to(self.visual.dtype)
                            image_embeds_3d.append(self.merger_vggt(features))

                image_embeds_3d = torch.cat(image_embeds_3d, dim=0).to(self.visual.dtype)

                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

                n_qwen = image_embeds.shape[0]
                n_vggt = image_embeds_3d.shape[0]
                if n_qwen != n_vggt:
                    image_embeds_3d = F.interpolate(
                        image_embeds_3d.unsqueeze(0).permute(0, 2, 1),
                        size=n_qwen, mode='linear', align_corners=False,
                    ).permute(0, 2, 1).squeeze(0)

                image_embeds = image_embeds + self.lam * image_embeds_3d

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_mask = image_mask.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                visual_pos_masks = (input_ids == self.config.image_token_id)
            else:
                deepstack_image_embeds = None
                visual_pos_masks = None

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds, deepstack_video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}")
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_mask = video_mask.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if visual_pos_masks is not None:
                    video_pos_mask = (input_ids == self.config.video_token_id)
                    combined_mask = visual_pos_masks | video_pos_mask
                    deepstack_visual_embeds = []
                    image_mask_joint = visual_pos_masks[combined_mask]
                    video_mask_joint = video_pos_mask[combined_mask]
                    for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                        embed_joint = img_embed.new_zeros(combined_mask.sum(), img_embed.shape[-1]).to(img_embed.device)
                        embed_joint[image_mask_joint, :] = img_embed
                        embed_joint[video_mask_joint, :] = vid_embed
                        deepstack_visual_embeds.append(embed_joint)
                    visual_pos_masks = combined_mask
                else:
                    visual_pos_masks = (input_ids == self.config.video_token_id)
                    deepstack_visual_embeds = deepstack_video_embeds
            else:
                if deepstack_image_embeds is not None:
                    deepstack_visual_embeds = deepstack_image_embeds
                else:
                    deepstack_visual_embeds = None

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        else:
            visual_pos_masks = None
            deepstack_visual_embeds = None

        if position_ids is None and (attention_mask is None or (attention_mask is not None and attention_mask.ndim == 2)):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0))
            ):
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        _use_cache = use_cache if use_cache is not None else (False if self.training else True)
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=_use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits_float = logits.float()
            shift_logits = logits_float[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
        cache_position=None, position_ids=None, use_cache=True,
        pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
        images_vggt=None, **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, position_ids=position_ids,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, use_cache=use_cache,
            images_vggt=images_vggt, **kwargs,
        )
        return model_inputs
