<div align="center">
  <h1>Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy</h1>
  <p>
    <strong>Fei Zhang<sup>1</sup>, Zhe Zhao<sup>2</sup>, Haibin Wen<sup>1</sup>, Tianshuo Wei<sup>1</sup>,</strong><br>
    <strong>Zaixi Zhang<sup>3</sup>, Chao Yang<sup>4,*</sup>, and Ye Wei<sup>1,*</sup></strong>
  </p>
  <p>
    <sup>1</sup>City University of Hong Kong &nbsp;&nbsp;
    <sup>2</sup>Stanford University<br>
    <sup>3</sup>Princeton University &nbsp;&nbsp;
    <sup>4</sup>Shanghai Jiaotong University
  </p>
  <p><sup>*</sup>Corresponding authors</p>
</div>

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue.svg)](https://2026.aclweb.org/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Bloom--Eval-orange)](https://huggingface.co/datasets/FeiZhang518/Bloom-Eval-Dataset)
[![Status](https://img.shields.io/badge/Status-Camera--ready-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper **"Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy"**.

This repository is a compact camera-ready public release. It keeps the core materials needed to understand the benchmark design and reproduce topic-level evaluation, including the evaluation scripts, prompt files, and 20 released topics.

## Overview

Bloom-Eval is a six-level benchmark for automatic survey generation (ASG), grounded in Bloom's Taxonomy:

- **Level 1, Memory**: factual recall of entities, claims, and references
- **Level 2, Comprehension**: topical focus, citation faithfulness, and reference balance
- **Level 3, Application**: document organization and academic writing conventions
- **Level 4, Analysis**: structural consistency, section relations, and outline quality
- **Level 5, Evaluation**: critical judgment over conclusions and limitations
- **Level 6, Creation**: novelty of research questions and forward-looking synthesis

The benchmark uses **GRADE** (Generative Rubric Adaptive Differential Evaluation) for rubric-based LLM judging on the metrics that require semantic comparison.

## Framework

![Bloom-Eval framework overview](figs/bloom-eval.png)

*Overview of the Bloom-Eval data pipeline and six-level evaluation hierarchy.*

## Current Release

This camera-ready repository includes:

- evaluation scripts for **16 core metrics**
- prompt templates used by the LLM-based metrics
- `20` released evaluation topics under `data/`
- topic metadata in `data/experimental_topics.csv`
- a convenience script for running all metrics for one topic

## Repository Layout

```text
Bloom-Eval/
├── data/
│   ├── experimental_topics.csv
│   ├── cs_01/ ... cs_10/
│   └── gs_01/ ... gs_10/
├── figs/
├── prompts/
│   ├── level1/ ... level6/
├── scripts/
│   ├── common.py
│   ├── prompt_utils.py
│   └── level1/ ... level6/
├── requirements.txt
├── run_topic_all_metrics.sh
└── README.md
```

Each released topic currently contains a `human/` directory with:

- `content.json`
- `outline.json`
- `reference.json`
- one task/article JSON file named `expert_article.json`

To evaluate a model output for the same topic, create a parallel `llm/` directory with:

- `content.json`
- `outline.json`
- `reference.json`
- `llm_article.json` when a full generated article JSON is available

## Released Topics

The current release contains `20` topics:

- `cs_01` to `cs_10`: Computer Science
- `gs_01` to `gs_10`: General Science

Topic metadata is stored in `data/experimental_topics.csv`, including title, venue, year, and citation count.

## Full Dataset

The complete reference corpus used in Bloom-Eval — **3,519 survey papers** across **60+ academic disciplines** — is available on Hugging Face:

[Bloom-Eval-Dataset](https://huggingface.co/datasets/FeiZhang518/Bloom-Eval-Dataset)

The dataset includes each paper's title, authors, abstract, keywords, full-text sections, references, and citation metadata. It can be downloaded as a single archive or browsed directly on the Hub.

## Metrics Included

The `run_topic_all_metrics.sh` script runs the following 16 core metrics:

### Level 1

- `EFid`: Measures how well the generated survey covers key domain entities and matches their emphasis distribution against the expert survey.
- `FCons`: Measures whether factual claims in the generated survey are supported by the expert reference rather than hallucinated.
- `HIRC`: Measures whether the generated survey includes high-impact foundational references in the field.

### Level 2

- `CF`: Measures whether citation-linked statements faithfully summarize or paraphrase the cited papers.
- `OTC`: Measures how well the generated outline covers the main topics present in the expert survey.
- `TBal`: Measures how evenly the generated survey distributes attention across different research themes.
- `TFSim`: Measures whether the generated survey captures similar research themes and thematic focus compared with the expert survey.

### Level 3

- `DSI`: Measures whether the generated survey contains essential academic sections such as abstract, introduction, conclusion, and references.
- `FAP`: Measures how well the generated survey applies a coherent organizing framework or taxonomy to structure the topic.
- `FMI`: Measures consistency between in-text citation markers and bibliography entries.

### Level 4

- `SCS`: Measures whether the survey structure avoids redundant or overlapping sections across different branches.
- `SCons`: Measures whether the generated outline has similar structural depth and breadth to the expert outline.
- `STS`: Measures the semantic and hierarchical similarity between the generated outline and the expert outline.

### Level 5

- `CAA`: Measures how well the generated survey's critical judgments align with expert-identified limitations, flaws, or comparative evaluations.

### Level 6

- `FNov`: Measures the originality and insightfulness of the conceptual framework proposed by the generated survey.
- `ROQ`: Measures the quality, foresight, and strategic value of the generated survey's future research directions.

## Installation

Create a new conda environment and install the required packages:

```bash
conda create -n bloom-eval python=3.10 -y
conda activate bloom-eval
pip install -r requirements.txt
```

LLM-based metrics require an OpenAI-compatible chat completions API:

```bash
export OPENAI_API_KEY="your_api_key"
```

Optional environment variables and runtime settings:

- `OPENAI_BASE_URL`: API endpoint, default `https://api.openai.com/v1`
- `BLOOM_EVAL_MODEL`: default model override, default `gpt-5-mini`

You can also pass the model and API endpoint directly on the command line. Use `--model` and `--base-url` for `run_topic_all_metrics.sh`; use `--model` and `--base_url` for individual Python metric scripts.

Example API setup:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export BLOOM_EVAL_MODEL="gpt-5-mini"
```

## Input Format

Released topics include only the human reference side. A model output should be placed in a parallel `llm/` directory with the same three required files:

```text
topic/
├── human/
│   ├── content.json
│   ├── outline.json
│   ├── reference.json
│   └── <task>.json
└── llm/
    ├── content.json
    ├── outline.json
    └── reference.json
```

The task JSON is used by `FAP`, `FNov`, and `ROQ`; it should contain at least a `title` field. Some non-LLM metrics operate on markdown text when run directly. The all-metrics script converts `content.json` to temporary markdown files for those metrics.

## Run All Metrics for One Topic

Use `run_topic_all_metrics.sh` when the human reference files and LLM output files are already prepared. If the script does not have executable permission, run it through `bash`:

```bash
bash run_topic_all_metrics.sh \
  --human-dir data/cs_01/human \
  --llm-dir data/cs_01/llm \
  --task-file data/cs_01/human/expert_article.json \
  --llm-article-file data/cs_01/llm/llm_article.json \
  --output-dir results/cs_01/all_metrics \
  --model gpt-5-mini \
  --base-url https://api.openai.com/v1
```

If a topic directory contains both `human/` and `llm/`, you can use:

```bash
bash run_topic_all_metrics.sh \
  --topic-dir data/cs_01 \
  --output-dir results/cs_01/all_metrics \
  --model gpt-5-mini \
  --base-url https://api.openai.com/v1
```

The full run includes API-based metrics, so `OPENAI_API_KEY` must be set even though some individual metrics are non-LLM metrics.

`run_topic_all_metrics.sh` parameters:

| Parameter | Required | Description |
| --- | --- | --- |
| `--topic-dir DIR` | optional | Topic directory containing `human/` and `llm/`. When set, `--human-dir` defaults to `DIR/human` and `--llm-dir` defaults to `DIR/llm`. |
| `--human-dir DIR` | optional | Human reference directory. Default: `data/cs_01/human`. |
| `--llm-dir DIR` | optional | LLM output directory. Default: `data/cs_01/llm`. |
| `--task-file FILE` | recommended | Human-side `expert_article.json` used to provide the topic title/context for `FAP`, `FNov`, and `ROQ`. Default discovery prefers `human/expert_article.json`. |
| `--human-article-file FILE` | optional | Human full-article JSON used only by `FMI`. In the released topics this is the same file as `--task-file`, so it usually does not need to be set separately. Default discovery prefers `human/expert_article.json`. |
| `--llm-article-file FILE` | optional | LLM full-article JSON used only by `FMI` for citation-integrity checking. Default discovery prefers `llm/llm_article.json` when present. |
| `--output-dir DIR` | recommended | Root output directory. Each metric writes to a subdirectory under it. Default: `results/<topic>/all_metrics`. |
| `--python BIN` | optional | Python executable. Default: `python3` or `$PYTHON_BIN` if set. |
| `--model NAME` | recommended for API metrics | Model name passed to API-based metrics. Overrides `$BLOOM_EVAL_MODEL`. |
| `--base-url URL` | recommended for non-default APIs | OpenAI-compatible API base URL. Overrides `$OPENAI_BASE_URL`. |
| `--save-raw-response` | optional | Save raw LLM responses under metric-specific `logs/` directories. |

Required files for a full run:

```text
human/
├── content.json
├── outline.json
├── reference.json
└── expert_article.json

llm/
├── content.json
├── outline.json
├── reference.json
└── llm_article.json
```

## Run a Single Metric

All metric scripts are available under `scripts/level*/run_*.py`. Individual scripts use `--output_dir`, `--save_raw_response`, `--model`, and `--base_url` with underscores. Only API-based metrics need `OPENAI_API_KEY`, `--model`, and `--base_url`.

The examples below assume:

```bash
export OPENAI_API_KEY="your_api_key"
MODEL="gpt-5-mini"
BASE_URL="https://api.openai.com/v1"
HUMAN_DIR="data/cs_01/human"
LLM_DIR="data/cs_01/llm"
OUT_ROOT="results/cs_01/single_metrics"
```

### Level 1

`EFid`:

```bash
python scripts/level1/run_EFid.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --output_dir "$OUT_ROOT/EFid" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

`FCons`:

```bash
python scripts/level1/run_FCons.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --output_dir "$OUT_ROOT/FCons" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

`HIRC`:

```bash
python scripts/level1/run_HIRC.py \
  --reference_file_human "$HUMAN_DIR/reference.json" \
  --reference_file_llm "$LLM_DIR/reference.json" \
  --output_dir "$OUT_ROOT/HIRC"
```

### Level 2

`CF`:

```bash
python scripts/level2/run_CF.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --reference_file_human "$HUMAN_DIR/reference.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --reference_file_llm "$LLM_DIR/reference.json" \
  --output_dir "$OUT_ROOT/CF" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

`OTC`:

```bash
python scripts/level2/run_OTC.py \
  --outline_file_human "$HUMAN_DIR/outline.json" \
  --outline_file_llm "$LLM_DIR/outline.json" \
  --output_dir "$OUT_ROOT/OTC" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

`TBal`:

```bash
python scripts/level2/run_TBal.py \
  --reference_file_human "$HUMAN_DIR/reference.json" \
  --reference_file_llm "$LLM_DIR/reference.json" \
  --output_dir "$OUT_ROOT/TBal"
```

`TFSim`:

```bash
python scripts/level2/run_TFSim.py \
  --reference_file_human "$HUMAN_DIR/reference.json" \
  --reference_file_llm "$LLM_DIR/reference.json" \
  --output_dir "$OUT_ROOT/TFSim" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

### Level 3

`DSI`:

```bash
python scripts/level3/run_DSI.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --output_dir "$OUT_ROOT/DSI"
```

`FAP`:

```bash
python scripts/level3/run_FAP.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --task_file "$HUMAN_DIR/expert_article.json" \
  --output_dir "$OUT_ROOT/FAP" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

`FMI`:

```bash
python scripts/level3/run_FMI.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --article_file_human "$HUMAN_DIR/expert_article.json" \
  --article_file_llm "$LLM_DIR/llm_article.json" \
  --output_dir "$OUT_ROOT/FMI"
```

### Level 4

`SCS`:

```bash
python scripts/level4/run_SCS.py \
  --outline_file_llm "$LLM_DIR/outline.json" \
  --output_dir "$OUT_ROOT/SCS" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

`SCons`:

```bash
python scripts/level4/run_SCons.py \
  --outline_file_human "$HUMAN_DIR/outline.json" \
  --outline_file_llm "$LLM_DIR/outline.json" \
  --output_dir "$OUT_ROOT/SCons"
```

`STS`:

```bash
python scripts/level4/run_STS.py \
  --outline_file_human "$HUMAN_DIR/outline.json" \
  --outline_file_llm "$LLM_DIR/outline.json" \
  --output_dir "$OUT_ROOT/STS"
```

### Level 5

`CAA`:

```bash
python scripts/level5/run_CAA.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --output_dir "$OUT_ROOT/CAA" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

### Level 6

`FNov`:

```bash
python scripts/level6/run_FNov.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --task_file "$HUMAN_DIR/expert_article.json" \
  --output_dir "$OUT_ROOT/FNov" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

`ROQ`:

```bash
python scripts/level6/run_ROQ.py \
  --content_file_human "$HUMAN_DIR/content.json" \
  --content_file_llm "$LLM_DIR/content.json" \
  --task_file "$HUMAN_DIR/expert_article.json" \
  --output_dir "$OUT_ROOT/ROQ" \
  --model "$MODEL" \
  --base_url "$BASE_URL"
```

Add `--save_raw_response` to API-based single-metric commands if you need raw LLM responses for debugging.

## Output

Each metric writes results into the selected output directory. Most metrics include:

- `result.json`: structured metric output
- `report.txt`: human-readable summary

When using `run_topic_all_metrics.sh`, the root output directory also includes:

- `score_summary.md`: a 16-row English Markdown table with only the final score(s) reported for each metric
- `summary.txt`: run metadata, succeeded/failed metric names, and the same final-score table

Additional artifacts are metric-specific and may include intermediate JSON files, generated rubric criteria, scoring files, CSV summaries, `human/` and `llm/` extraction directories, and optional `logs/` files when `--save_raw_response` is enabled.

## Citation

If you use this repository, please cite the Bloom-Eval paper:

```bibtex
@inproceedings{zhang2026bloomeval,
  title     = {Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic
               Survey Generation Based on Bloom's Taxonomy},
  author    = {Zhang, Fei and Zhao, Zhe and Wen, Haibin and Wei, Tianshuo and
               Zhang, Zaixi and Yang, Chao and Wei, Ye},
  booktitle = {ACL},
  year      = {2026},
}
```
