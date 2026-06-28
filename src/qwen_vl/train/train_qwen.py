# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.sampler
from transformers.trainer_utils import get_last_checkpoint
from qwen_vl.data.data_qwen_obshead import LazySupervisedDatasetObsHead, DataCollatorForObsHead

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism

local_rank = None

def rank0_print(*args):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, flush=True)
    elif local_rank in (0, -1, None):
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    llm_module = model.model.language_model
    if model_args.tune_mm_llm:
        for n, p in llm_module.named_parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in llm_module.named_parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

    # vggt is frozen
    for n, p in model.vggt.named_parameters():
        p.requires_grad = False
    # vggt merger is trainable
    vggt_merger = getattr(model, 'merger_vggt', None) or getattr(model, 'merger', None)
    for n, p in vggt_merger.named_parameters():
        p.requires_grad = True
        

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model_type = getattr(config, "model_type", "").lower()
    model_path_lower = model_args.model_name_or_path.lower()
    rank0_print(f"[FutureNav] init model_name_or_path: {model_args.model_name_or_path}")
    rank0_print(f"[FutureNav] init config.model_type: {model_type or '<empty>'}")
    rank0_print(f"[FutureNav] init config.architectures: {getattr(config, 'architectures', None)}")
    rank0_print(f"[FutureNav] output_dir: {training_args.output_dir}")
    rank0_print(f"[FutureNav] dataset_use: {data_args.dataset_use}")

    if "qwen3" not in model_path_lower and "qwen3" not in model_type:
        raise ValueError("FutureNav training expects a Qwen3-VL model.")

    from qwen_vl.model.modeling_qwen3_vl_obshead import Qwen3VLForFutureNav
    setattr(config, "lam", model_args.lam)
    setattr(config, "reference_frame", getattr(model_args, "reference_frame", "first"))
    setattr(config, "forward_loss_weight", getattr(model_args, "forward_loss_weight", 0.1))
    setattr(config, "inverse_loss_weight", getattr(model_args, "inverse_loss_weight", 0.1))
    setattr(config, "gen_loss_weight", getattr(model_args, "gen_loss_weight", 0.1))
    model = Qwen3VLForFutureNav.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        vggt_model_path=model_args.vggt_model_path
    )

    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen3vl"
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Set tokenizer for obs head action label parsing
    if hasattr(model, 'set_tokenizer'):
        model.set_tokenizer(tokenizer)
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        if hasattr(model.visual, 'print_trainable_parameters'):
            model.visual.print_trainable_parameters()
        if hasattr(model.model, 'print_trainable_parameters'):
            model.model.print_trainable_parameters()

    vggt_features_path = getattr(model_args, "vggt_features_path", None)
    train_dataset = LazySupervisedDatasetObsHead(
        tokenizer=tokenizer, data_args=data_args,
        vggt_features_path=vggt_features_path,
    )
    data_collator = DataCollatorForObsHead(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        logging.info("checkpoint found, resume training")
        rank0_print(f"[FutureNav] resume_from_checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        rank0_print("[FutureNav] resume_from_checkpoint: none")
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
