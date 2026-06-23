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
│   ├── level1/ ... level6/
│   └── others/
├── requirements.txt
├── run_topic_all_metrics.sh
└── README.md
```

Each released topic currently contains a `human/` directory with:

- `content.json`
- `outline.json`
- `reference.json`
- one task JSON file named by title or DOI, such as `A Survey on Vision Transformer.json`

To evaluate a model output for the same topic, create a parallel `llm/` directory with:

- `content.json`
- `outline.json`
- `reference.json`

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

- `EFid`
- `FCons`
- `HIRC`

### Level 2

- `CF`
- `OTC`
- `TBal`
- `TFSim`

### Level 3

- `DSI`
- `FAP`
- `FMI`

### Level 4

- `SCS`
- `SCons`
- `STS`

### Level 5

- `CAA`

### Level 6

- `FNov`
- `ROQ`

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LLM-based metrics require an OpenAI-compatible chat completions API:

```bash
export OPENAI_API_KEY="your_api_key"
```

Optional environment variables:

- `OPENAI_BASE_URL`: API endpoint, default `https://api.openai.com/v1`
- `BLOOM_EVAL_MODEL`: default model override, default `gpt-5-mini`

You can also pass `--model` and `--base_url` directly to metric scripts or `run_topic_all_metrics.sh`.

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

Use the convenience script when the human and LLM files are already prepared:

```bash
./run_topic_all_metrics.sh \
  --human-dir data/cs_01/human \
  --llm-dir /path/to/cs_01/llm \
  --task-file "data/cs_01/human/A Survey on Vision Transformer.json" \
  --output-dir results/cs_01_all \
  --model gpt-5-mini
```

If a topic directory contains both `human/` and `llm/`, you can use:

```bash
./run_topic_all_metrics.sh \
  --topic-dir /path/to/cs_01 \
  --output-dir results/cs_01_all
```

Add `--save-raw-response` to save raw LLM responses under metric-specific `logs/` directories.

## Run a Single Metric

All metric scripts are available under `scripts/level*/run_*.py`. Example:

```bash
python scripts/level2/run_OTC.py \
  --outline_file_human data/cs_01/human/outline.json \
  --outline_file_llm /path/to/cs_01/llm/outline.json \
  --output_dir results/otc_cs_01 \
  --model gpt-5-mini
```

Another example for a non-LLM metric:

```bash
python scripts/level3/run_DSI.py \
  --content_file_human /path/to/human_content.md \
  --content_file_llm /path/to/llm_content.md \
  --output_dir results/dsi_example
```

Before running a metric directly, inspect that script's CLI arguments to confirm the exact expected file type and parameter names.

## Output

Each metric writes results into the selected output directory. Most metrics include:

- `result.json`: structured metric output
- `report.txt`: human-readable summary

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
