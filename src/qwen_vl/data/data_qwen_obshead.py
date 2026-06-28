# Dataset wrapper for ObsHead training.
# Adds next-frame VGGT targets and current-action labels.

import os
import re
import hashlib
import torch
from typing import Dict, Sequence
from dataclasses import dataclass
import transformers

from .data_qwen import (
    LazySupervisedDataset,
    DataCollatorForSupervisedDataset,
)


ACTION_MAP = {'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2, 'TURN_RIGHT': 3}


class LazySupervisedDatasetObsHead(LazySupervisedDataset):
    """
    FutureNav wrapper:
    - next_vggt_feature / next_frame_vggt_img: next observation target for future gen.
    - action_label: action taken at the current step, for inverse dynamics.
    """

    def __init__(self, *args, vggt_features_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_episode_index()

        self._vggt_features = None
        self._online_mode = True
        if vggt_features_path and os.path.exists(vggt_features_path):
            print(f"Loading precomputed VGGT features from {vggt_features_path}...")
            self._vggt_features = torch.load(vggt_features_path, map_location='cpu')
            self._online_mode = False
            print(f"Loaded {len(self._vggt_features)} precomputed features. Mode: OFFLINE (predict next frame)")
        else:
            print("No precomputed VGGT features found. Mode: ONLINE (load next frame)")

    def _parse_episode_step(self, sample):
        sample_id = sample.get("id", "")
        if "/" not in sample_id:
            return None

        ep_id, step_part = sample_id.rsplit("/", 1)
        match = re.search(r"step[_-]?(\d+)", step_part)
        if match is None:
            return None

        dataset_name = sample.get("dataset_name", "")
        if dataset_name:
            ep_id = f"{dataset_name}:{ep_id}"
        instruction = self._extract_instruction(sample)
        if instruction:
            instr_hash = hashlib.md5(instruction.encode("utf-8")).hexdigest()
            ep_id = f"{ep_id}:{instr_hash}"
        return ep_id, int(match.group(1))

    def _extract_instruction(self, sample):
        try:
            prompt = sample["conversations"][0]["value"]
        except Exception:
            return ""
        marker = "Your task is to "
        if marker not in prompt:
            return ""
        instruction = prompt.split(marker, 1)[1]
        end_marker = "\n You should"
        if end_marker in instruction:
            instruction = instruction.split(end_marker, 1)[0]
        return instruction.strip()

    def _build_episode_index(self):
        self._next_step_map = {}
        self._prev_step_map = {}

        episode_steps = {}
        for idx, sample in enumerate(self.list_data_dict):
            parsed = self._parse_episode_step(sample)
            if parsed is None:
                continue
            ep_id, step_num = parsed
            episode_steps.setdefault(ep_id, []).append((step_num, idx))

        for steps in episode_steps.values():
            steps.sort(key=lambda x: x[0])
            for pos in range(len(steps) - 1):
                curr_idx = steps[pos][1]
                next_idx = steps[pos + 1][1]
                self._next_step_map[curr_idx] = next_idx
                self._prev_step_map[next_idx] = curr_idx

    def _action_label_from_sample(self, sample):
        try:
            action = sample["conversations"][-1]["value"].strip().upper()
        except Exception:
            return -1
        return ACTION_MAP.get(action, -1)

    def _last_image_path(self, sample):
        images = sample.get("images", sample.get("image", []))
        if not isinstance(images, list) or len(images) == 0:
            return None

        image_path = images[-1]
        if not isinstance(image_path, str):
            return None

        image_folder = sample.get("data_path", "")
        if image_folder and not os.path.isabs(image_path):
            image_path = os.path.join(image_folder, image_path)
        return image_path

    def _set_offline_next_feature(self, data_dict, sample):
        image_path = self._last_image_path(sample)
        if image_path is None:
            data_dict["next_vggt_feature"] = None
        else:
            data_dict["next_vggt_feature"] = self._vggt_features.get(image_path, None)

    def _set_online_next_frame(self, data_dict, sample, i):
        image_path = self._last_image_path(sample)
        if image_path is None:
            if "images_vggt" in data_dict and len(data_dict["images_vggt"]) > 0:
                data_dict["next_frame_vggt_img"] = data_dict["images_vggt"][-1].clone()
            else:
                print(f"[DEBUG][FutureNav] next_frame_vggt_img=None: no image fallback, sample {i}")
                data_dict["next_frame_vggt_img"] = None
            return

        try:
            from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
            imgs = load_and_preprocess_images([image_path])
            data_dict["next_frame_vggt_img"] = imgs[0]
        except Exception as e:
            print(f"[DEBUG][FutureNav] next_frame_vggt_img=None: load failed for {image_path}, error: {e}")
            if "images_vggt" in data_dict and len(data_dict["images_vggt"]) > 0:
                data_dict["next_frame_vggt_img"] = data_dict["images_vggt"][-1].clone()
            else:
                data_dict["next_frame_vggt_img"] = None

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        data_dict = super()._get_item(i)

        data_dict["action_label"] = self._action_label_from_sample(self.list_data_dict[i])
        prev_idx = self._prev_step_map.get(i, None)
        data_dict["prev_action_label"] = (
            self._action_label_from_sample(self.list_data_dict[prev_idx])
            if prev_idx is not None
            else -1
        )

        next_idx = self._next_step_map.get(i, None)
        target_sample = self.list_data_dict[next_idx] if next_idx is not None else self.list_data_dict[i]
        has_valid_next = next_idx is not None or data_dict["action_label"] == ACTION_MAP["STOP"]

        if not self._online_mode and self._vggt_features is not None:
            if has_valid_next:
                self._set_offline_next_feature(data_dict, target_sample)
            else:
                data_dict["next_vggt_feature"] = None
        else:
            data_dict["next_vggt_feature"] = None
            if has_valid_next:
                self._set_online_next_frame(data_dict, target_sample, i)
            else:
                data_dict["next_frame_vggt_img"] = None

        return data_dict


@dataclass
class DataCollatorForObsHead(DataCollatorForSupervisedDataset):
    """Collator for ObsHead fields."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)

        batch["action_labels"] = torch.tensor([
            inst.get("action_label", -1) for inst in instances
        ])

        batch["prev_action_labels"] = torch.tensor([
            inst.get("prev_action_label", -1) for inst in instances
        ])

        next_features = []
        for inst in instances:
            feat = inst.get("next_vggt_feature", None)
            next_features.append(feat if feat is not None and isinstance(feat, torch.Tensor) else None)
        if any(f is not None for f in next_features):
            batch["next_vggt_targets"] = next_features

        next_imgs = []
        for inst in instances:
            img = inst.get("next_frame_vggt_img", None)
            next_imgs.append(img if img is not None and isinstance(img, torch.Tensor) else None)
        if any(im is not None for im in next_imgs):
            batch["next_frame_vggt"] = next_imgs

        return batch
