"""
FutureNav R2R navigation agent for VLN-CE evaluation (habitat 0.1.7 compatible).
"""

import os
import copy
import base64
import random
from io import BytesIO
from collections import OrderedDict

import cv2
import numpy as np
import torch
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl_utils import extract_vision_info
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from qwen_vl.model.modeling_qwen3_vl_obshead import Qwen3VLForFutureNav

min_pixels = 28 * 28
max_pixels = 1605632

ACTION_NAME_TO_ENV = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}


class FutureNav_R2R_Agent(Agent):
    """
    FutureNav agent: uses Qwen3-VL to predict discrete navigation actions.
    """

    def __init__(
        self,
        checkpoint_path,
        result_path,
        require_map=True,
        device=None,
        max_history_frames=8,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history_frames = max_history_frames
        self.result_path = result_path
        self.require_map = require_map

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)

        print(f"{'=' * 60}")
        print(f"Initialize FutureNav R2R Agent")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  device: {self.device}")
        print(f"  max_history_frames: {self.max_history_frames}")
        print(f"{'=' * 60}")

        # Load model
        config = AutoConfig.from_pretrained(checkpoint_path)
        if "qwen3" not in checkpoint_path.lower() and getattr(config, "model_type", "") != "qwen3_vl":
            raise ValueError("FutureNav evaluation expects a Qwen3-VL checkpoint.")
        self.model = Qwen3VLForFutureNav.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            attn_implementation="eager",
            mode='evaluation'
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_path, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left"
        )

        self.rgb_list = []
        self.step = 0
        self.reset()

    def reset(self):
        self.rgb_list = []
        self.step = 0
        self.model.past_key_values_vggt = None

    def _prepare_images(self):
        """Select history + current frame images."""
        history_len = len(self.rgb_list) - 1
        if history_len <= self.max_history_frames:
            history_images = self.rgb_list[:history_len]
            images = history_images + [self.rgb_list[-1]]
        else:
            indices = np.linspace(0, history_len, self.max_history_frames + 1, dtype=int)
            images = [self.rgb_list[i] for i in indices]
        return images

    def _call_model(self, images, instruction):
        """Run FutureNav inference, return action string."""
        messages = []
        message = [
            {"role": "system",
             "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
             }
        ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {instruction} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."

        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size

        # Build message content with images
        image_content = []
        for v in images:
            image_content.append({"type": "image", "image": v})
        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
        messages.append(message)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images_vggt = []
        image_inputs = []
        for msg in messages:
            vision_info = extract_vision_info(msg)
            cur_images_vggt = []
            for i, ele in enumerate(vision_info):
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")

                    image = load_and_preprocess_images([image])[0]
                    if i == len(vision_info) - 1:
                        cur_images_vggt.append(image)

                    _, height, width = image.shape
                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = height - (height // patch_size) % merge_size * patch_size
                    image = image[:, :height, :width]
                    image_inputs.append(image)

            images_vggt.append(torch.stack(cur_images_vggt))

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device
        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)

        with torch.no_grad():
            pad_token_id = self.tokenizer.pad_token_id
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=24,
            )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return answers[0]

    def act(self, observations, info, episode_id):
        rgb = observations["rgb"]
        image = Image.fromarray(rgb).convert('RGB')
        self.rgb_list.append(image)

        instruction_text = observations["instruction"]["text"]

        # Prepare images and predict
        images = self._prepare_images()
        action_str = self._call_model(images, instruction_text)

        # Parse action string to env action
        action_str = action_str.strip()
        if action_str in ACTION_NAME_TO_ENV:
            action_id = ACTION_NAME_TO_ENV[action_str]
        else:
            # Try to find a valid action in the output
            for name in ACTION_NAME_TO_ENV:
                if name in action_str:
                    action_id = ACTION_NAME_TO_ENV[name]
                    break
            else:
                action_id = 1  # default MOVE_FORWARD

        self.step += 1
        return {"action": action_id}
