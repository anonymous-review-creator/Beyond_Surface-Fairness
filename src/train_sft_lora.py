#!/usr/bin/env python3
"""Train a LoRA adapter with supervised counterfactual examples.

The same entry point supports the Mistral-7B and Llama-3.1-8B models used in
the paper. Input data may use either ``prompt/chosen/rejected`` fields or the
original PANDA fields. Only the preferred (counterfactual) response is used by
the SFT objective. Paths and output locations are supplied at run time; model
weights are never stored in this repository.

Example:
    python src/train_sft_lora.py \
        --model-family mistral7b \
        --train-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
        --out-dir runs/sft_mistral7b

Use ``--model-family llama8b`` for the Llama configuration. Model-specific
LoRA defaults are selected automatically and can be inspected with ``--help``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


PRESETS = {
    "mistral7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_r": 8,
        "lora_alpha": 16,
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_r": 8,
        "lora_alpha": 32,
    },
}

SYSTEM_PROMPT = "You are generating respectful stories and sentences."
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
    parser = argparse.ArgumentParser(description="LoRA-SFT on PANDA counterfactual pairs.")
    parser.add_argument("--model-family", choices=sorted(PRESETS), required=True)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model id or local model directory override.",
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument(
        "--eval-jsonl",
        default=None,
        help="Optional evaluation JSONL. If omitted, --eval-ratio is split from training data.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def panda_prompt(original: str, selected_word: str, target_attribute: str) -> str:
    return (
        "Rewrite the text by replacing demographic references so that "
        f"'{selected_word}' becomes '{target_attribute}'. Preserve the original "
        f"meaning and grammar.\n\nText: {original.strip()}"
    )


def normalize_example(example: dict[str, Any]) -> dict[str, str] | None:
    if all(key in example for key in ("prompt", "chosen")):
        return {
            "prompt": str(example["prompt"]).strip(),
            "chosen": str(example["chosen"]).strip(),
        }

    required = ("original", "perturbed", "selected_word", "target_attribute")
    if all(key in example for key in required):
        return {
            "prompt": panda_prompt(
                str(example["original"]),
                str(example["selected_word"]),
                str(example["target_attribute"]),
            ),
            "chosen": str(example["perturbed"]).strip(),
        }

    if "source" in example and "target" in example:
        return {
            "prompt": panda_prompt(
                str(example["source"]),
                str(example.get("selected_word", "the source attribute")),
                str(example.get("target_attribute", "the target attribute")),
            ),
            "chosen": str(example["target"]).strip(),
        }

    return None


def load_jsonl(path: str) -> Dataset:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            normalized = normalize_example(raw)
            if normalized is not None and normalized["prompt"] and normalized["chosen"]:
                rows.append(normalized)
    if not rows:
        raise ValueError(f"No supported training examples found in {path}")
    return Dataset.from_list(rows)


def format_example(
    tokenizer: AutoTokenizer,
    model_family: str,
    prompt: str,
    chosen: str,
) -> str:
    if model_family == "mistral7b":
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"},
            {"role": "assistant", "content": chosen},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"{SYSTEM_PROMPT}\n\n{prompt}\n\n{chosen}"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be between 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required to train the released 7B/8B models")

    set_seed(TRAINING_SEED)
    preset = PRESETS[args.model_family]
    public_model_id = preset["model_id"]
    model_id = args.model_id or public_model_id
    lora_r = preset["lora_r"] if args.lora_r is None else args.lora_r
    lora_alpha = preset["lora_alpha"] if args.lora_alpha is None else args.lora_alpha
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_raw = load_jsonl(args.train_jsonl)
    if args.eval_jsonl:
        eval_raw = load_jsonl(args.eval_jsonl)
    else:
        split = train_raw.train_test_split(test_size=args.eval_ratio, seed=TRAINING_SEED)
        train_raw, eval_raw = split["train"], split["test"]

    def render(example: dict[str, str]) -> dict[str, str]:
        return {
            "text": format_example(
                tokenizer,
                args.model_family,
                example["prompt"],
                example["chosen"],
            )
        }

    train_data = train_raw.map(render, remove_columns=train_raw.column_names)
    eval_data = eval_raw.map(render, remove_columns=eval_raw.column_names)

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
    training_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=0.3,
        max_length=args.max_length,
        dataset_text_field="text",
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=TRAINING_SEED,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(out_dir / "final_adapter"))
    tokenizer.save_pretrained(out_dir / "final_adapter")

    summary = {
        "model_family": args.model_family,
        "public_model_id": public_model_id,
        "model_override_used": args.model_id is not None,
        "train_examples": len(train_data),
        "eval_examples": len(eval_data),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
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
