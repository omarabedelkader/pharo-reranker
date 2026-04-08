#!/usr/bin/env python3
"""
Fine-tune a Qwen2.5-Coder model for code completion / FIM using LoRA.

Dataset format (JSONL):
Each row should contain at least:
  - prompt: the input text, e.g. a FIM prompt ending with <|fim_middle|>
  - completion: the target span to generate

Example row:
{"prompt": "<|fim_prefix|>def add(a, b):\n    return <|fim_suffix|>\nprint(add(2,3))\n<|fim_middle|>",
 "completion": "a + b"}

This script trains with loss only on the completion tokens by default.
It also supports inference, where you can enforce generation of at least
10 new tokens via --min-new-tokens 10.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger("qwen25_fim_lora")
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder with LoRA for code completion / FIM")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="Train a LoRA adapter")
    train.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    train.add_argument("--train-file", type=str, required=True)
    train.add_argument("--data-dir", type=str, default=None, help="Directory containing train/validation JSONL files")
    train.add_argument("--validation-file", type=str, default=None)
    train.add_argument("--output-dir", type=str, required=True)
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--per-device-train-batch-size", type=int, default=2)
    train.add_argument("--per-device-eval-batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--num-train-epochs", type=float, default=3.0)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.03)
    train.add_argument("--logging-steps", type=int, default=10)
    train.add_argument("--save-steps", type=int, default=200)
    train.add_argument("--eval-steps", type=int, default=200)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--use-4bit", action="store_true")
    train.add_argument("--no-use-4bit", dest="use_4bit", action="store_false")
    train.set_defaults(use_4bit=True)
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--fp16", action="store_true")
    train.add_argument("--gradient-checkpointing", action="store_true")
    train.add_argument("--train-on-prompt", action="store_true", help="Include prompt tokens in the loss")
    train.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules",
    )
    train.add_argument(
        "--split-ratio",
        type=float,
        default=0.02,
        help="Validation split ratio used only when --validation-file is omitted",
    )

    infer = subparsers.add_parser("infer", help="Run inference with a trained adapter")
    infer.add_argument("--base-model", type=str, required=True)
    infer.add_argument("--adapter-path", type=str, required=True)
    infer.add_argument("--prompt", type=str, default=None)
    infer.add_argument("--prompt-file", type=str, default=None)
    infer.add_argument("--max-new-tokens", type=int, default=32)
    infer.add_argument("--min-new-tokens", type=int, default=10)
    infer.add_argument("--temperature", type=float, default=0.2)
    infer.add_argument("--top-p", type=float, default=0.95)
    infer.add_argument("--top-k", type=int, default=50)
    infer.add_argument("--do-sample", action="store_true")
    infer.add_argument("--use-4bit", action="store_true")
    infer.add_argument("--no-use-4bit", dest="use_4bit", action="store_false")
    infer.set_defaults(use_4bit=True)

    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_jsonl_dataset(train_file: str, validation_file: Optional[str], split_ratio: float, seed: int, data_dir: Optional[str] = None) -> DatasetDict:
    # If data_dir is provided, look for files in that directory
    if data_dir:
        if not os.path.isdir(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Look for train files in the directory
        if not train_file or train_file == "train.jsonl":
            train_path = os.path.join(data_dir, "train.jsonl")
            if not os.path.exists(train_path):
                # Try to find any .jsonl file for training
                jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
                if jsonl_files:
                    train_path = os.path.join(data_dir, jsonl_files[0])
                    LOGGER.info(f"Using {jsonl_files[0]} as train file")
                else:
                    raise ValueError(f"No JSONL files found in {data_dir}")
            train_file = train_path
        
        # Look for validation file
        if not validation_file:
            validation_path = os.path.join(data_dir, "validation.jsonl")
            if os.path.exists(validation_path):
                validation_file = validation_path
                LOGGER.info(f"Found validation file: validation.jsonl")
            else:
                # Try to find other jsonl files for validation
                jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl') and f != os.path.basename(train_file)]
                if jsonl_files:
                    validation_file = os.path.join(data_dir, jsonl_files[0])
                    LOGGER.info(f"Using {jsonl_files[0]} as validation file")
    
    data_files: Dict[str, str] = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file
        ds = load_dataset("json", data_files=data_files)
        return DatasetDict(train=ds["train"], validation=ds["validation"])

    ds = load_dataset("json", data_files={"train": train_file})["train"]
    if len(ds) < 2:
        raise ValueError("Need at least 2 examples to auto-create a validation split.")

    valid_size = max(1, int(len(ds) * split_ratio))
    split = ds.train_test_split(test_size=valid_size, seed=seed, shuffle=True)
    return DatasetDict(train=split["train"], validation=split["test"])


def ensure_required_columns(dataset: Dataset) -> None:
    cols = set(dataset.column_names)
    required = {"prompt", "completion"}
    missing = required - cols
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any
    max_length: int
    train_on_prompt: bool = False

    def _encode_one(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        prompt = str(example["prompt"])
        completion = str(example["completion"])

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        eos_ids: List[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids = [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + completion_ids + eos_ids
        labels = input_ids.copy()

        if not self.train_on_prompt:
            labels[: len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)
        labels[len(prompt_ids) + len(completion_ids) :] = [IGNORE_INDEX] * len(eos_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

            # keep at least one supervised token when truncation happens
            if all(x == IGNORE_INDEX for x in labels):
                last_idx = min(self.max_length, len(prompt_ids + completion_ids)) - 1
                if last_idx >= 0:
                    labels[last_idx] = input_ids[last_idx]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = [self._encode_one(feature) for feature in features]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer pad_token_id is None. Please set tokenizer.pad_token.")

        max_len = max(len(x["input_ids"]) for x in batch)
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            padded_input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            padded_labels.append(x["labels"] + [IGNORE_INDEX] * pad_len)
            padded_attention_mask.append(x["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }


def choose_torch_dtype(fp16: bool, bf16: bool) -> torch.dtype:
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_train_model(model_name: str, use_4bit: bool, dtype: torch.dtype, gradient_checkpointing: bool):
    quantization_config = None
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if use_4bit:
        compute_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    return model


def add_lora(model, args: argparse.Namespace):
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def preprocess_for_debug(dataset: Dataset) -> None:
    first = dataset[0]
    LOGGER.info("Example keys: %s", list(first.keys()))
    LOGGER.info("Example prompt preview: %r", str(first['prompt'])[:240])
    LOGGER.info("Example completion preview: %r", str(first['completion'])[:240])


class LossLoggingTrainer(Trainer):
    def compute_metrics(self, eval_pred):
        return {}


def train(args: argparse.Namespace) -> None:
    setup_logging()
    set_seed(args.seed)

    ds = load_jsonl_dataset(args.train_file, args.validation_file, args.split_ratio, args.seed, args.data_dir)
    ensure_required_columns(ds["train"])
    ensure_required_columns(ds["validation"])
    preprocess_for_debug(ds["train"])

    tokenizer = load_tokenizer(args.model_name)
    dtype = choose_torch_dtype(args.fp16, args.bf16)
    model = load_train_model(args.model_name, args.use_4bit, dtype, args.gradient_checkpointing)
    model = add_lora(model, args)

    data_collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_on_prompt=args.train_on_prompt,
    )

    optim_name = "paged_adamw_8bit" if args.use_4bit else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,
        optim=optim_name,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = LossLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    try:
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        eval_metrics["perplexity"] = float("inf")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    LOGGER.info("Saved adapter and tokenizer to %s", args.output_dir)


@torch.inference_mode()
def infer(args: argparse.Namespace) -> None:
    setup_logging()

    if bool(args.prompt) == bool(args.prompt_file):
        raise ValueError("Provide exactly one of --prompt or --prompt-file")

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = args.prompt

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print("========== PROMPT ==========")
    print(prompt)
    print("========== COMPLETION ==========")
    print(generated_text)


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
