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
|   `-- Training_Validation_data/
|      `-- StereoSet_validation_data.zip
|-- results/
|   |-- Metrics_results/
|   |-- StereoSet_test/
|   |-- Visualization/
|   `-- cross_validation.csv
|-- src/
|   |-- analyze_metrics.py
|   |-- evaluation.py
|   |-- generate_story_all.py
|   |-- stereoset_test.py
|   `-- Visualization/
|       |-- Vis_FS.R
|       `-- Vis_SHR.R
|-- environment.yml
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

### Third-Party Data Licenses

The third-party datasets in this directory retain their original licenses:

- StereoSet material in `StereoSet_validation_data.zip`: CC BY-SA 4.0.
- PANDA-derived material in `Constructed_PANDA_DPO_train.jsonl`: MIT License.
- BiasDPO material in `Huggingface_ahmedallam_BiasDPO.jsonl`: Apache License 2.0.

See `THIRD_PARTY_NOTICES.md` and the files under `licenses/` for source links,
attribution, license scope, and license terms. These third-party licenses apply
only to the identified third-party materials and do not assign a license to the
repository's original code, generated outputs, or analysis artifacts.

Large external datasets used in the study may need to be downloaded separately from their original sources, as described in the paper.

The scripts `src/evaluation.py` and `src/stereoset_test.py` expect the official StereoSet data-loading utilities. If `dataloader.py` is not present in this repository, obtain it from the official StereoSet evaluation code and place it where Python can import it.

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

: Generates model outputs from prompt templates and attribute lists. The script accepts model, template, attribute-list, and output paths as command-line arguments.

`src/analyze_metrics.py`

: Annotates generated stories, counts female/male/unknown signals, aggregates results, and computes metrics including FS and SHR.

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

The environment includes PyTorch, Transformers, PEFT, TRL, pandas, NumPy, datasets, and related dependencies.

## Typical Usage

The repository includes generated model outputs and metric tables as CSV files. The lightest way to check the released materials is to recompute metrics from an existing generated-output CSV and regenerate plots from the released metric CSVs.

### 1. Recompute Metrics from Existing Outputs

Run `src/analyze_metrics.py` with one generated-output CSV and the matching axis-specific attribute lists:

```bash
python src/analyze_metrics.py \
  --input-csv data/Occupation/llama/occupation_lm_baseline_analyze.csv \
  --female-lex data/Occupation/template_attribution/female_occupations.txt \
  --male-lex data/Occupation/template_attribution/male_occupations.txt \
  --attribute-col occupation \
  --out-dir results/Metrics_results/Occupation/llama
```

The script writes per-sample annotations and aggregate metric CSV files. The same command pattern can be used for the other axes by changing the input CSV, attribute lists, `--attribute-col`, and output directory.

### 2. Regenerate Plots

Use the R scripts in `src/Visualization/` to generate FS and SHR plots from metric CSV files:

```bash
Rscript src/Visualization/Vis_FS.R \
  results/Metrics_results/Occupation/llama/occupation_lm_baseline_metrics.csv \
  results/Metrics_results/Occupation/llama/occupation_lm_sft_metrics.csv \
  results/Visualization/Occupation/llama/FS/llama_occ_SFT_FS.pdf \
  "Occupation FS: Llama baseline vs SFT"
```

```bash
Rscript src/Visualization/Vis_SHR.R \
  results/Metrics_results/Occupation/llama/occupation_lm_baseline_metrics.csv \
  results/Metrics_results/Occupation/llama/occupation_lm_sft_metrics.csv \
  results/Visualization/Occupation/llama/SHR/llama_occ_SFT_SHR.pdf \
  "Occupation ln(SHR): Llama baseline vs SFT"
```

### 3. Optional Model-Based Reruns

Full generation and StereoSet reruns require access to the relevant base or locally merged models, suitable hardware, and environment-specific model paths. The repository does not provide model weights.

`src/generate_story_all.py` can be used to regenerate model outputs when a reviewer has configured model access. See the script arguments with:

```bash
python src/generate_story_all.py --help
```

For StereoSet evaluation, `src/stereoset_test.py` and `src/evaluation.py` require the official StereoSet data-loading utility `dataloader.py`. This file is not included in the repository; obtain it from the original StereoSet evaluation code and place it on the Python path before running the evaluation scripts.

## Anonymity and Reproducibility Notes

This repository is intended for anonymous review and research replication. Before sharing a public copy, check that:

- Git history does not contain non-anonymous commit authors, emails, deleted files, or old Office/PDF metadata;
- any Office files included in future releases do not contain author, last-modified-by, company, comment-author, or custom-property metadata;
- PDFs do not contain author fields, local paths, Microsoft sensitivity labels, organization tenant identifiers, or other embedded metadata;
- README text, code comments, configs, and file paths do not contain personal names, institutional names, usernames, emails, or machine-specific paths;
- the repository URL, account name, and branch names do not identify the authors or institution.

For a clean anonymous release, create a new repository from a sanitized working directory rather than reusing a repository with non-anonymous history.

## License

Original code in this repository is licensed under the Apache License 2.0.
Third-party datasets remain subject to the licenses identified in
`THIRD_PARTY_NOTICES.md` and the corresponding files under `licenses/`.
