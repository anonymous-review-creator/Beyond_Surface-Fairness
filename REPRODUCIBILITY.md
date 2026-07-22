# Reproduction Guide

This guide separates lightweight verification of the released results from
GPU-based retraining and regeneration. Commands are run from the repository
root. Model weights and trained adapters are not distributed.

## 1. Environment

Create the pinned environment:

```bash
conda env create -f environment.yml
conda activate llm-bias
```

Llama access requires acceptance of the upstream model license and an
authenticated Hugging Face account. An authorized local model directory can be
passed with `--model-id` and `--local-files-only`.

## 2. Lightweight Metric Reproduction

The released `*_analyze.csv` files already contain row-level tags and manual
validity checks. Recompute one released metric table without loading a model:

```bash
python src/analyze_metrics.py \
  --input-csv data/Occupation/llama/occupation_lm_baseline_analyze.csv \
  --attribute-col occupation \
  --out-dir reproduced/Occupation/llama
```

Regenerate a baseline-versus-SFT plot:

```bash
Rscript src/Visualization/Vis_FS.R \
  results/Metrics_results/Occupation/llama/occupation_lm_baseline_metrics.csv \
  results/Metrics_results/Occupation/llama/occupation_lm_sft_metrics.csv \
  reproduced/llama_occ_SFT_FS.pdf
```

## 3. LoRA-SFT Training

The released PANDA-derived file contains 16,161 preference pairs. SFT uses the
preferred response and creates a deterministic 80/20 train/evaluation split.
Both training scripts use a fixed internal seed of 42 for reproducibility; it
is not exposed as an experiment-tuning argument.

Mistral:

```bash
python src/train_sft_lora.py \
  --model-family mistral7b \
  --train-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
  --out-dir runs/sft_mistral7b
```

Llama:

```bash
python src/train_sft_lora.py \
  --model-family llama8b \
  --train-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
  --out-dir runs/sft_llama8b
```

Both commands save an adapter under `<out-dir>/final_adapter`.

Model-specific SFT defaults:

| Model | LoRA rank | LoRA alpha | Dropout | Epochs | Learning rate |
|---|---:|---:|---:|---:|---:|
| Mistral-7B | 8 | 16 | 0.05 | 2 | `5e-5` |
| Llama-3.1-8B | 8 | 32 | 0.05 | 2 | `5e-5` |

## 4. LoRA-DPO Training

DPO combines 16,161 PANDA-derived pairs with 1,145 BiasDPO pairs. The source
datasets are split independently before each training batch is sampled with
four PANDA-derived examples and one BiasDPO example.

Mistral:

```bash
python src/train_dpo_lora.py \
  --model-family mistral7b \
  --panda-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
  --biasdpo-jsonl data/Training_Validation_data/Huggingface_ahmedallam_BiasDPO.jsonl \
  --out-dir runs/dpo_mistral7b
```

Llama:

```bash
python src/train_dpo_lora.py \
  --model-family llama8b \
  --panda-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
  --biasdpo-jsonl data/Training_Validation_data/Huggingface_ahmedallam_BiasDPO.jsonl \
  --out-dir runs/dpo_llama8b
```

The script prints the realized 4:1 batch composition and saves the final LoRA
adapter and a path-sanitized `run_summary.json`.

Model-specific DPO defaults:

| Model | LoRA rank | LoRA alpha | Dropout | Beta | Epochs | Learning rate |
|---|---:|---:|---:|---:|---:|---:|
| Mistral-7B | 16 | 64 | 0.05 | 0.1 | 4 | `5e-5` |
| Llama-3.1-8B | 64 | 256 | 0.05 | 0.2 | 3 | `1e-5` |

## 5. Generation

The following examples use the Occupation axis. Replace the three template and
attribute-list paths to run another axis.

Baseline generation:

```bash
python src/generate_story_all.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --templates data/Occupation/template_attribution/templates.txt \
  --female-list data/Occupation/template_attribution/female_occupations.txt \
  --male-list data/Occupation/template_attribution/male_occupations.txt \
  --out-csv reproduced/occupation_llama_baseline.csv
```

Generate from a trained adapter by adding:

```bash
--adapter-dir runs/sft_llama8b/final_adapter
```

or:

```bash
--adapter-dir runs/dpo_llama8b/final_adapter
```

Instruction-based generation uses the unmodified base model:

```bash
python src/generate_story_all.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --templates data/Occupation/template_attribution/templates.txt \
  --female-list data/Occupation/template_attribution/female_occupations.txt \
  --male-list data/Occupation/template_attribution/male_occupations.txt \
  --debias-mode ins \
  --out-csv reproduced/occupation_llama_ins.csv
```

Generation defaults are `n=10`, `max_new_tokens=128`, temperature `0.7`,
top-p `0.9`, and repetition penalty `1.1`.

## 6. Annotate Newly Generated Text

New raw generations do not contain lexicon tags or manual validity checks. Run
the released high-confidence identity lexicons first:

```bash
python src/analyze_metrics.py \
  --input-csv reproduced/occupation_llama_baseline.csv \
  --female-lex data/Identity_lexicon/female_identity.txt \
  --male-lex data/Identity_lexicon/male_identity.txt \
  --attribute-col occupation \
  --out-dir reproduced/Occupation/llama
```

Automated lexicon labels reproduce the first annotation stage. Manual review is
still required for malformed generations and context-dependent anomalous cases,
as described in the paper.

## 7. Expected Artifacts

Training directories contain adapter files, tokenizer metadata, evaluation
metrics, and a run summary. They are ignored by Git and should not be committed.
Generated CSVs contain `group`, `occupation`, `iter`, `prompt`, and `story`
columns. Metric recomputation produces an annotated CSV and a corresponding
`*_metrics.csv` table.
