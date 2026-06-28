# FutureNav adapter for Qwen3-VL (single-node version)
# Keeps the same VGGT integration logic as the Qwen2.5-VL version,
# but uses Qwen3VLForConditionalGeneration as backbone.

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLConfig,
)

try:
    import deepspeed
    from deepspeed.zero import GatheredParameters
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

from .vggt.models.vggt import VGGT

logger = logging.get_logger(__name__)


# ============ Shared utilities (same as Qwen2.5-VL version) ============

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VGGTMerger(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.input_dim = context_dim * (spatial_merge_size**2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ln_q = Qwen3RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        return x


def slice1d(x, start, end):
    return x[:, start:end, ...]

def slice2d(x, start, end):
    return x[:, :, start:end, ...]

def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

DIM_TO_SLICE = {1: slice1d, 2: slice2d, 3: slice3d}


class StartRecentKVCache:
    def __init__(self, start_size=4, recent_size=48, k_seq_dim=2, v_seq_dim=2):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat([self.k_slice(k, 0, self.start_size), self.k_slice(k, seq_len - self.recent_size, seq_len)], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, self.start_size), self.v_slice(v, seq_len - self.recent_size, seq_len)], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]


# ============ FutureNav Model for Qwen3-VL (single-node) ============

class Qwen3VLForConditionalGenerationForFutureNav(Qwen3VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # --- VGGT components (same as Qwen2.5 version) ---
        vggt = VGGT()
        vggt.camera_head = None
        vggt.track_head = None
        self.vggt = vggt
        self.past_key_values_vggt = None
        for param in self.vggt.parameters():
            param.requires_grad = False

        self.merger_vggt = VGGTMerger(
            output_dim=config.text_config.hidden_size,
            hidden_dim=getattr(config, "vggt_merger_hidden_dim", 4096),
            context_dim=2048,
            spatial_merge_size=config.vision_config.spatial_merge_size,
        )
        self.config.reference_frame = getattr(config, "reference_frame", "first")
        self.lam = getattr(config, "lam", 0.2)

        self.kv_cache_vggt = StartRecentKVCache(start_size=8, recent_size=48, k_seq_dim=2, v_seq_dim=2)

        if getattr(config, "add_ground_classifier", False):
            self.classifier = nn.Sequential(
                Qwen3RMSNorm(config.text_config.hidden_size, eps=1e-6),
                nn.Linear(config.text_config.hidden_size, config.classifier_hidden_dim),
                nn.GELU(),
                nn.Linear(config.classifier_hidden_dim, config.classifier_out_dim),
            )

        self.vocab_size = config.text_config.vocab_size
        self.rope_deltas = None
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, mode=None, **kwargs):
        vggt_model_path = kwargs.pop("vggt_model_path", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.mode = mode
        if vggt_model_path:
            print(f"Loading VGGT from {vggt_model_path}")
            vggt = VGGT.from_pretrained(vggt_model_path)
            vggt.camera_head = None
            vggt.track_head = None
            model.vggt = vggt
            for param in model.vggt.parameters():
                param.requires_grad = False
        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    @property
    def visual(self):
        return self.model.visual

    @property
    def language_model(self):
        return self.model.language_model

    def get_rope_index(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        return self.model.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

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

                # Gather VGGT parameters once to avoid repeated all-gather in ZeRO-3
                vggt_params = list(self.vggt.parameters())
                vggt_context = GatheredParameters(vggt_params, modifier_rank=None) if HAS_DEEPSPEED and any(hasattr(p, 'ds_id') for p in vggt_params) else nullcontext()

                with vggt_context:
                    for i in range(batch_size):
                        if images_vggt[i].shape[0] > 0:
                            n_image = 1
                            height, width = images_vggt[i].shape[-2:]
                            # VGGT uses DINOv2 with patch_size=14, not Qwen3's patch_size=16
                            vggt_patch_size = 14
                            merge_size = self.config.vision_config.spatial_merge_size
                            h_grid, w_grid = height // vggt_patch_size, width // vggt_patch_size
                            h_grid_after_merge = h_grid // merge_size
                            w_grid_after_merge = w_grid // merge_size

                            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                            with torch.no_grad():
                                with torch.amp.autocast('cuda', dtype=dtype):
                                    if not self.config.reference_frame == "first":
                                        images_vggt[i] = torch.flip(images_vggt[i], dims=(0,))

                                    if self.mode == "evaluation":
                                        if self.past_key_values_vggt is None:
                                            self.past_key_values_vggt = [None] * self.vggt.aggregator.depth
                                    else:
                                        self.past_key_values_vggt = [None] * self.vggt.aggregator.depth

                                    for k, frame in enumerate(images_vggt[i]):
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

                                        features = aggregated_tokens[-2][0, :, patch_start_idx:]

                            if not self.config.reference_frame == "first":
                                features = torch.flip(features, dims=(0,))

                            # Spatial merge
                            features = features.view(n_image, h_grid, w_grid, -1)
                            features = features[:, :h_grid_after_merge * merge_size, :w_grid_after_merge * merge_size, :].contiguous()
                            features = features.view(n_image, h_grid_after_merge, merge_size, w_grid_after_merge, merge_size, -1)
                            features = features.permute(0, 1, 3, 2, 4, 5).contiguous().to(self.visual.dtype)
                            image_embeds_3d.append(self.merger_vggt(features))

                image_embeds_3d = torch.cat(image_embeds_3d, dim=0).to(self.visual.dtype)

                # Get image embeddings from Qwen3-VL vision encoder
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

                # Align VGGT token count to Qwen3 vision encoder token count
                n_qwen = image_embeds.shape[0]
                n_vggt = image_embeds_3d.shape[0]
                if n_qwen != n_vggt:
                    image_embeds_3d = F.interpolate(
                        image_embeds_3d.unsqueeze(0).permute(0, 2, 1),
                        size=n_qwen,
                        mode='linear',
                        align_corners=False,
                    ).permute(0, 2, 1).squeeze(0)

                image_embeds = image_embeds + self.lam * image_embeds_3d

                # Scatter into inputs_embeds
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_mask = image_mask.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                # DeepStack
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

        # Position IDs
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

        # Call language model with DeepStack
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
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            if hasattr(self.config, "add_ground_classifier") and self.config.add_ground_classifier:
                shift_labels = shift_labels.masked_fill(shift_labels == self.config.box_3d_token_ids[1], -100)
            loss = loss_fct(shift_logits, shift_labels)

            if boxes is not None:
                select_hidden_states = hidden_states[input_ids == self.config.box_3d_token_ids[0]].view(-1, hidden_states.shape[-1])
                select_output = self.classifier(select_hidden_states)
                loss_grd = F.mse_loss(select_output, boxes)
                loss += loss_grd

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
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, use_cache=use_cache, **kwargs,
        )
        model_inputs["position_ids"] = None
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["images_vggt"] = None
        else:
            model_inputs["images_vggt"] = images_vggt
        return model_inputs
