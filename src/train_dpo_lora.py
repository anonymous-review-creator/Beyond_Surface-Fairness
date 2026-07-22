#!/usr/bin/env python3
"""Train LoRA adapters with a stratified PANDA/BiasDPO mixture.

Every optimization batch uses four PANDA-derived examples and one BiasDPO
example, matching the 4:1 mixture described in the paper. Model-specific
presets reflect the retained experiment configuration and can be overridden from
the command line. The script saves adapters only, not base-model weights.

Example:
    python src/train_dpo_lora.py \
        --model-family mistral7b \
        --panda-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
        --biasdpo-jsonl data/Training_Validation_data/Huggingface_ahmedallam_BiasDPO.jsonl \
        --out-dir runs/dpo_mistral7b

Use ``--model-family llama8b`` for the Llama configuration. The selected
model preset controls epochs, learning rate, beta, LoRA rank, and LoRA alpha.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from torch.utils.data import DataLoader, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


PRESETS = {
    "mistral7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "epochs": 4.0,
        "learning_rate": 5e-5,
        "beta": 0.1,
        "lora_r": 16,
        "lora_alpha": 64,
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "epochs": 3.0,
        "learning_rate": 1e-5,
        "beta": 0.2,
        "lora_r": 64,
        "lora_alpha": 256,
    },
}

PANDA_TAG = "panda"
BIAS_DPO_TAG = "biasdpo"
SOURCE_COLUMN = "data_source"
TRAINING_SEED = 42
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA-DPO with a 4:1 data mixture.")
    parser.add_argument("--model-family", choices=sorted(PRESETS), required=True)
    parser.add_argument("--model-id", default=None, help="Optional model id or local path override.")
    parser.add_argument("--panda-jsonl", required=True, help="PANDA-derived preference JSONL.")
    parser.add_argument("--biasdpo-jsonl", required=True, help="BiasDPO preference JSONL.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--panda-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_preference(example: dict[str, Any]) -> dict[str, str] | None:
    aliases = {
        "prompt": ("prompt", "instruction"),
        "chosen": ("chosen", "accepted", "chosen_response"),
        "rejected": ("rejected", "rejected_response"),
    }
    normalized: dict[str, str] = {}
    for target, candidates in aliases.items():
        value = next((example[key] for key in candidates if key in example), None)
        if value is None:
            return None
        normalized[target] = str(value).strip()
    if not all(normalized.values()):
        return None
    return normalized


def load_preference_jsonl(path: str, source: str) -> Dataset:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            normalized = normalize_preference(raw)
            if normalized is not None:
                normalized[SOURCE_COLUMN] = source
                rows.append(normalized)
    if not rows:
        raise ValueError(f"No prompt/chosen/rejected examples found in {path}")
    return Dataset.from_list(rows)


def conversationalize(dataset: Dataset) -> Dataset:
    def convert(example: dict[str, str]) -> dict[str, Any]:
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "chosen": [{"role": "assistant", "content": example["chosen"]}],
            "rejected": [{"role": "assistant", "content": example["rejected"]}],
            SOURCE_COLUMN: example[SOURCE_COLUMN],
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


class StratifiedBatchSampler(Sampler[list[int]]):
    """Draw a fixed source ratio in every batch, recycling the smaller source."""

    def __init__(
        self,
        panda_indices: list[int],
        bias_indices: list[int],
        batch_size: int,
        panda_fraction: float,
        seed: int,
    ) -> None:
        if not panda_indices or not bias_indices:
            raise ValueError("Both PANDA and BiasDPO examples are required")
        if batch_size < 2:
            raise ValueError("Stratified DPO requires --batch-size of at least 2")
        self.panda_indices = panda_indices
        self.bias_indices = bias_indices
        self.batch_size = batch_size
        self.seed = seed
        self.iteration = 0
        self.panda_per_batch = max(1, min(batch_size - 1, round(batch_size * panda_fraction)))
        self.bias_per_batch = batch_size - self.panda_per_batch
        self.num_batches = max(1, (len(panda_indices) + len(bias_indices)) // batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        self.iteration += 1
        rng = random.Random(self.seed + self.iteration)
        panda = self.panda_indices.copy()
        bias = self.bias_indices.copy()
        rng.shuffle(panda)
        rng.shuffle(bias)
        panda_pos = bias_pos = 0

        for _ in range(self.num_batches):
            batch: list[int] = []
            for _ in range(self.panda_per_batch):
                batch.append(panda[panda_pos])
                panda_pos += 1
                if panda_pos == len(panda):
                    rng.shuffle(panda)
                    panda_pos = 0
            for _ in range(self.bias_per_batch):
                batch.append(bias[bias_pos])
                bias_pos += 1
                if bias_pos == len(bias):
                    rng.shuffle(bias)
                    bias_pos = 0
            rng.shuffle(batch)
            yield batch


class StratifiedDPOTrainer(DPOTrainer):
    def __init__(self, *args, panda_fraction: float, **kwargs) -> None:
        self.panda_fraction = panda_fraction
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        sources = self.train_dataset[SOURCE_COLUMN]
        sampler = StratifiedBatchSampler(
            panda_indices=[i for i, source in enumerate(sources) if source == PANDA_TAG],
            bias_indices=[i for i, source in enumerate(sources) if source == BIAS_DPO_TAG],
            batch_size=self.args.per_device_train_batch_size,
            panda_fraction=self.panda_fraction,
            seed=self.args.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def preset_value(args: argparse.Namespace, name: str):
    value = getattr(args, name)
    return PRESETS[args.model_family][name] if value is None else value


def main() -> None:
    args = parse_args()
    if not 0.0 < args.eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be between 0 and 1")
    if not 0.0 < args.panda_fraction < 1.0:
        raise ValueError("--panda-fraction must be between 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required to train the released 7B/8B models")

    set_seed(TRAINING_SEED)
    preset = PRESETS[args.model_family]
    model_id = args.model_id or preset["model_id"]
    epochs = preset_value(args, "epochs")
    learning_rate = preset_value(args, "learning_rate")
    beta = preset_value(args, "beta")
    lora_r = preset_value(args, "lora_r")
    lora_alpha = preset_value(args, "lora_alpha")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panda = load_preference_jsonl(args.panda_jsonl, PANDA_TAG)
    biasdpo = load_preference_jsonl(args.biasdpo_jsonl, BIAS_DPO_TAG)
    panda_split = panda.train_test_split(test_size=args.eval_ratio, seed=TRAINING_SEED)
    bias_split = biasdpo.train_test_split(test_size=args.eval_ratio, seed=TRAINING_SEED)
    train_data = concatenate_datasets([panda_split["train"], bias_split["train"]]).shuffle(
        seed=TRAINING_SEED
    )
    eval_data = concatenate_datasets([panda_split["test"], bias_split["test"]]).shuffle(
        seed=TRAINING_SEED
    )
    train_data = conversationalize(train_data)
    eval_data = conversationalize(eval_data)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_length

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    training_args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        beta=beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=TRAINING_SEED,
    )
    trainer = StratifiedDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
        panda_fraction=args.panda_fraction,
    )

    loader = trainer.get_train_dataloader()
    sampler = loader.batch_sampler
    print(
        "Per-batch mixture: "
        f"PANDA={sampler.panda_per_batch}, BiasDPO={sampler.bias_per_batch}"
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(out_dir / "final_adapter"))
    tokenizer.save_pretrained(out_dir / "final_adapter")

    summary = {
        "model_family": args.model_family,
        "public_model_id": preset["model_id"],
        "model_override_used": args.model_id is not None,
        "train_examples": len(train_data),
        "eval_examples": len(eval_data),
        "train_source_counts": dict(Counter(train_data[SOURCE_COLUMN])),
        "eval_source_counts": dict(Counter(eval_data[SOURCE_COLUMN])),
        "panda_fraction_per_batch": args.panda_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "beta": beta,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": args.lora_dropout,
        "training_seed": TRAINING_SEED,
        "eval_metrics": metrics,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
