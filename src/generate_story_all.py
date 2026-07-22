#!/usr/bin/env python3
"""Generate stories from axis templates and attribute lists.

The script is intentionally model-path agnostic: reviewers can pass either a
Hugging Face model id or a local model directory through --model-id. If a merged
model directory is available, pass it through --merged-dir and --use-merged.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_SUFFIX = (
    " Response in English. You are generating respectful stories and sentences. "
    "Do not look back at any previous generation output."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model outputs from prompt templates.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local model directory.")
    parser.add_argument("--templates", required=True, help="Path to templates.txt.")
    parser.add_argument("--female-list", required=True, help="Path to the female-coded attribute list.")
    parser.add_argument("--male-list", required=True, help="Path to the male-coded attribute list.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--use-merged", action="store_true", help="Load --merged-dir instead of --model-id.")
    parser.add_argument("--merged-dir", default=None, help="Local merged model directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Only load local model/tokenizer files.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-gen-per-item", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    return parser.parse_args()


def torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_list(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_dataset(
    templates: list[str],
    female_items: list[str],
    male_items: list[str],
    n_gen_per_item: int,
    suffix: str,
) -> Dataset:
    prompts, groups, attributes, iters = [], [], [], []
    for group, items in [("F", female_items), ("M", male_items)]:
        for attribute in items:
            for template in templates:
                prompt = template.format(occ=attribute) + suffix
                for i in range(n_gen_per_item):
                    prompts.append(prompt)
                    groups.append(group)
                    attributes.append(attribute)
                    iters.append(i)
    return Dataset.from_dict(
        {"prompt": prompts, "group": groups, "occupation": attributes, "iter": iters}
    )


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(random.randint(0, 2**32 - 1))

    if args.use_merged:
        if not args.merged_dir:
            raise ValueError("--use-merged requires --merged-dir")
        if not Path(args.merged_dir).exists():
            raise FileNotFoundError(f"Merged model directory not found: {args.merged_dir}")
        model_path = args.merged_dir
    else:
        model_path = args.model_id

    templates = load_list(args.templates)
    female_items = load_list(args.female_list)
    male_items = load_list(args.male_list)
    dataset = build_dataset(
        templates=templates,
        female_items=female_items,
        male_items=male_items,
        n_gen_per_item=args.n_gen_per_item,
        suffix=args.suffix,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=torch_dtype(args.dtype),
    )

    def generate_batch(batch):
        outputs = generator(batch["prompt"], return_full_text=False)
        stories = [seq[0]["generated_text"].strip() for seq in outputs]
        return {"story": stories}

    generated = dataset.map(generate_batch, batched=True, batch_size=args.batch_size)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    (
        generated.to_pandas()
        .sort_values(["group", "occupation", "iter"])
        .to_csv(out_csv, index=False, encoding="utf-8")
    )

    print(f"Wrote {len(generated)} generations to {out_csv}")
    print(f"Model source: {model_path}")


if __name__ == "__main__":
    main()
