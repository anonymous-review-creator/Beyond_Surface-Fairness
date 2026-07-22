> Anonymous repository for research replication.

# Beyond Surface Fairness

This repository provides the replication materials for studying implicit stereotype associations in large language models. It includes prompt templates, generated model outputs, metric tables, evaluation scripts, and visualization artifacts for four axes:

- `Character`
- `Family_role`
- `Occupation`
- `Study_ability`

The experiments evaluate Llama 3.1 8B and Mistral 7B across baseline and debiased variants. The main outputs are per-generation annotations, aggregate metric tables, StereoSet evaluation results, and FS/SHR visualizations.

## Workflow

The workflow is organized around five stages:

1. Prepare axis-specific prompt templates and attribute lists.
2. Generate identity-free stories or sentences from baseline and debiased model variants.
3. Annotate generated text with female/male/unknown tag counts.
4. Compute implicit association metrics, including Fairness Score (FS) and Stereotype-Hit Rate (SHR).
5. Visualize directional bias and distributional balance across axes, model families, and variants.

## Repository Layout

```text
.
|-- configs/
|   |-- llama8b.yml
|   `-- mistral7b.yml
|-- data/
|   |-- Character/
|   |-- Family_role/
|   |-- Occupation/
|   |-- Study_ability/
|   |-- Identity_lexicon/
|   `-- Training_Validation_data/
|      `-- StereoSet_validation_data.zip
|-- results/
|   |-- Metrics_results/
|   |-- StereoSet_test/
|   |-- Visualization/
|   `-- cross_validation.csv
|-- src/
|   |-- analyze_metrics.py
|   |-- dataloader.py
|   |-- evaluation.py
|   |-- generate_story_all.py
|   |-- train_sft_lora.py
|   |-- train_dpo_lora.py
|   |-- stereoset_test.py
|   `-- Visualization/
|       |-- Vis_FS.R
|       `-- Vis_SHR.R
|-- environment.yml
|-- REPRODUCIBILITY.md
`-- README.md
```

## Configuration Files

`configs/llama8b.yml` and `configs/mistral7b.yml` store model-family settings and experiment parameters, including:

- base model identifier
- merged-model or adapter loading options
- local file loading mode
- generation parameters
- CSV output naming patterns
- LoRA configuration fields
- evaluation dataset settings

These files are intended as reproducibility guides rather than one-command launchers. Before rerunning generation or evaluation, set the model access mode and any local model or adapter paths for the execution environment.

## Data Directory

The `data/` directory contains template files, generated outputs, and training/evaluation data.

### Axis Directories

Each axis directory follows the same high-level structure:

```text
data/<Axis>/
|-- template_attribution/
|-- llama/
`-- mistral/
```

The axis directories are:

- `data/Character/`
- `data/Family_role/`
- `data/Occupation/`
- `data/Study_ability/`

### Template Files

Each `template_attribution/` folder contains axis-specific attribute lists and prompt templates.

Current files include:

```text
data/Character/template_attribution/
|-- female_chara.txt
|-- male_chara.txt
`-- templates.txt

data/Family_role/template_attribution/
|-- female_role.txt
|-- male_role.txt
`-- templates.txt

data/Occupation/template_attribution/
|-- female_occupations.txt
|-- male_occupations.txt
`-- templates.txt

data/Study_ability/template_attribution/
|-- female_ability.txt
|-- male_ability.txt
`-- templates.txt
```

All templates use `{occ}` as the attribute placeholder. The value substituted into `{occ}` comes from the matching female-coded or male-coded attribute list for the selected axis.

### Generated Analysis Tables

The `llama/` and `mistral/` subdirectories contain CSV analysis tables for model outputs. File names encode:

- axis
- model family (`lm` for Llama, `mis` for Mistral)
- variant (`baseline`, `sft`, `ins`, `dpo`)
- output type (`analyze`)

Examples:

```text
data/Character/llama/character_lm_sft_analyze.csv
data/Occupation/mistral/occupation_mis_dpo_analyze.csv
data/Study_ability/llama/ability_lm_baseline_analyze.csv
```

These files contain generated text and per-sample tagging information used for metric computation.

### Training and Validation Data

`data/Training_Validation_data/` contains training data used for debiasing experiments and the StereoSet validation archive used for benchmark evaluation:

```text
data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl
data/Training_Validation_data/Huggingface_ahmedallam_BiasDPO.jsonl
data/Training_Validation_data/StereoSet_validation_data.zip
```

The StereoSet archive contains:

```text
StereoSet_data/dev.json
StereoSet_data/test.json
StereoSet_data/test_terms.txt
```

Large external datasets used in the study may need to be downloaded separately from their original sources, as described in the paper.

The official StereoSet data loader is included as `src/dataloader.py`. It is an unchanged copy of the upstream file at commit `ead7d086a64a192a1eca88e0dd2fd163de375218` from `https://github.com/moinnadeem/StereoSet`, which is distributed under CC BY-SA 4.0.

## Results Directory

The `results/` directory contains generated metrics, visualizations, and benchmark outputs.

### Metric Tables

Metric CSV files are stored under:

```text
results/Metrics_results/
```

Current subdirectories include:

```text
results/Metrics_results/Character/
results/Metrics_results/Family_role/
results/Metrics_results/Occupation/
results/Metrics_results/Study_ability/
```

Metric file names encode the axis, model family, variant, and output type. Examples:

```text
results/Metrics_results/Character/llama/chara_lm_dpo_metrics.csv
results/Metrics_results/Occupation/mistral/occupation_mis_baseline_metrics.csv
results/Metrics_results/Study_ability/llama/ability_lm_sft_metrics.csv
```

These CSV files store the metric values used for reporting and plotting.

### Visualizations

Visualization outputs are stored under:

```text
results/Visualization/
```

The plots are grouped by axis, model family, and metric type:

```text
results/Visualization/<Axis>/<model>/FS/
results/Visualization/<Axis>/<model>/SHR/
```

The FS plots summarize distributional balance. The SHR plots summarize directional stereotype-hit behavior.

### StereoSet Results

StereoSet outputs are stored under:

```text
results/StereoSet_test/
|-- llama_variants_results/
`-- mistral_variants_results/
```

These JSON files contain StereoSet benchmark results for baseline and debiased variants.

## Source Code

`src/generate_story_all.py`

: Generates baseline, adapter-based, merged-model, and instruction-debiased
outputs from prompt templates and attribute lists. INS is implemented through
the `--debias-mode ins` switch in this shared script rather than a duplicated
generation program. Runnable baseline, INS, and adapter examples are included
in the script header.

`src/train_sft_lora.py`

: Trains a LoRA-SFT adapter on preferred PANDA counterfactual responses. One
parameterized script supports both model families and selects the reported
model-specific LoRA settings automatically. A runnable example is included in
the script header.

`src/train_dpo_lora.py`

: Trains a LoRA-DPO adapter from PANDA-derived and BiasDPO preference pairs
using the 4:1 per-batch mixture described in the paper. Model-specific defaults
and a runnable example are documented in the script header.

`src/analyze_metrics.py`

: Annotates generated stories, counts female/male/unknown signals, aggregates results, and computes metrics including FS and SHR.

`src/dataloader.py`

: Loads the official StereoSet intersentence and intrasentence examples for benchmark evaluation.

`src/stereoset_test.py`

: Generates StereoSet predictions for a model or adapter and evaluates the outputs using StereoSet-style scoring.

`src/evaluation.py`

: Computes LM Score, Stereotype Score, and ICAT-style results from prediction files and StereoSet gold data.

`src/Visualization/Vis_FS.R`

: Produces FS visualizations from metric CSV files.

`src/Visualization/Vis_SHR.R`

: Produces SHR directionality visualizations from metric CSV files.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
```

The environment includes PyTorch, Transformers, PEFT, TRL, pandas, NumPy, NLTK, datasets, R, ggplot2, dplyr, and related dependencies.

## Typical Usage

The repository includes generated model outputs and metric tables as CSV files. The lightest way to check the released materials is to recompute metrics from an existing generated-output CSV and regenerate plots from the released metric CSVs.

For the complete sequence from environment setup through training, generation,
annotation, metric computation, and plotting, see `REPRODUCIBILITY.md`.

### 1. Recompute Metrics from Existing Outputs

Run `src/analyze_metrics.py` with one generated-output CSV and the matching axis-specific attribute lists:

```bash
python src/analyze_metrics.py \
  --input-csv data/Occupation/llama/occupation_lm_baseline_analyze.csv \
  --attribute-col occupation \
  --out-dir results/Metrics_results/Occupation/llama
```

The released `*_analyze.csv` files already contain row-level lexicon tags and manual checks. The script reuses those tags, excludes rows marked `INVALID`, retains valid mixed-gender cases as `ANOMALOUS`, and writes the aggregate metric CSV with F/M mean rows. For raw, unannotated generations, also provide identity lexicons through `--female-lex` and `--male-lex`. The same command pattern can be used for the other axes by changing the input CSV, `--attribute-col`, and output directory.

### 2. Regenerate Plots

Use the R scripts in `src/Visualization/` to generate FS and SHR plots from metric CSV files:

```bash
Rscript src/Visualization/Vis_FS.R \
  results/Metrics_results/Occupation/llama/occupation_lm_baseline_metrics.csv \
  results/Metrics_results/Occupation/llama/occupation_lm_sft_metrics.csv \
  results/Visualization/Occupation/llama/FS/llama_occ_SFT_FS.pdf
```

```bash
Rscript src/Visualization/Vis_SHR.R \
  results/Metrics_results/Occupation/llama/occupation_lm_baseline_metrics.csv \
  results/Metrics_results/Occupation/llama/occupation_lm_sft_metrics.csv \
  results/Visualization/Occupation/llama/SHR/llama_occ_SFT_SHR.pdf
```

### 3. Train LoRA Adapters

The SFT script accepts either PANDA source fields or the released
`prompt/chosen/rejected` representation. If a separate evaluation file is not
provided, it creates a deterministic 80/20 split using seed 42.

```bash
python src/train_sft_lora.py \
  --model-family mistral7b \
  --train-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
  --out-dir runs/sft_mistral7b
```

The corresponding Llama run uses `--model-family llama8b`. SFT uses two
epochs, dropout 0.05, batch size 1, gradient accumulation 16, and maximum
length 512. Its model-specific LoRA settings are rank 8/alpha 16 for Mistral
and rank 8/alpha 32 for Llama.

For DPO, the sampler fixes the source composition of every five-example batch
to four PANDA-derived pairs and one BiasDPO pair:

```bash
python src/train_dpo_lora.py \
  --model-family llama8b \
  --panda-jsonl data/Training_Validation_data/Constructed_PANDA_DPO_train.jsonl \
  --biasdpo-jsonl data/Training_Validation_data/Huggingface_ahmedallam_BiasDPO.jsonl \
  --out-dir runs/dpo_llama8b
```

The model-specific DPO configuration uses four epochs, learning rate `5e-5`,
beta `0.1`, rank 16, and alpha 64 for Mistral; Llama uses three total epochs,
learning rate `1e-5`, beta `0.2`, rank 64, and alpha 256. Both use dropout
0.05, batch size 5, gradient
accumulation 16, maximum length 512, and maximum prompt length 256. All values
can be overridden explicitly.

Both scripts save LoRA adapters only. Base-model weights must be obtained under
the original model licenses and are not distributed here.
Llama model access requires acceptance of its upstream license and an
authenticated Hugging Face account; use `--model-id` for an authorized local
copy and `--local-files-only` when all required files are already cached.

### 4. Optional Model-Based Reruns

Full generation and StereoSet reruns require access to the relevant base or locally merged models, suitable hardware, and environment-specific model paths. The repository does not provide model weights.

`src/generate_story_all.py` can be used to regenerate model outputs when a reviewer has configured model access. See the script arguments with:

```bash
python src/generate_story_all.py --help
```

Its defaults match the paper (`n=10`, `max_new_tokens=128`, `temperature=0.7`, `top_p=0.9`, and `repetition_penalty=1.1`). Pass `--debias-mode ins` for the instruction-based condition. The released INS generation appended the fixed instruction to each template; `--instruction-position prefix` is also available for controlled variants. Baseline generation uses the standard mode. An SFT or DPO adapter can be evaluated directly with `--adapter-dir`, while a previously merged checkpoint can be supplied with `--use-merged --merged-dir`.

For example, the released Occupation INS condition can be regenerated with:

```bash
python src/generate_story_all.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --templates data/Occupation/template_attribution/templates.txt \
  --female-list data/Occupation/template_attribution/female_occupations.txt \
  --male-list data/Occupation/template_attribution/male_occupations.txt \
  --debias-mode ins \
  --out-csv reproduced/occupation_llama_ins.csv
```

The same command supports Mistral by changing `--model-id`. No trained adapter
is loaded for INS; it uses the base instruction-tuned model with the fixed
debiasing instruction appended to each experimental template.

For StereoSet evaluation, `src/stereoset_test.py` and `src/evaluation.py` import the included `src/dataloader.py` module. Run the scripts from the repository root so that `src/` is resolved consistently.

## Anonymity and Reproducibility Notes

This repository is intended for anonymous review and research replication. Before sharing a public copy, check that:

- Git history does not contain non-anonymous commit authors, emails, deleted files, or old Office/PDF metadata;
- any Office files included in future releases do not contain author, last-modified-by, company, comment-author, or custom-property metadata;
- PDFs do not contain author fields, local paths, Microsoft sensitivity labels, organization tenant identifiers, or other embedded metadata;
- README text, code comments, configs, and file paths do not contain personal names, institutional names, usernames, emails, or machine-specific paths;
- the repository URL, account name, and branch names do not identify the authors or institution.

For a clean anonymous release, create a new repository from a sanitized working directory rather than reusing a repository with non-anonymous history.
