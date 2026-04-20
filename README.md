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

This repository is a compact camera-ready release. It keeps the evaluation scripts, prompt files, and 20 released topics needed to understand the benchmark and reproduce metric computation on topic-level inputs.

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

## Release Scope

This camera-ready repository includes:

- evaluation scripts for **16 core metrics**
- prompt templates used by the LLM-based metrics
- `20` released evaluation topics under `data/experimental_data`
- topic metadata in `data/experimental_data/experimental_topics.csv`
- a convenience script for running all metrics for one topic

This repository does **not** include the full benchmark curation assets, copyrighted full-text source papers, or the full paper source tree.

## Repository Layout

```text
Bloom-Eval/
├── data/
│   └── experimental_data/
│       ├── experimental_topics.csv
│       ├── cs_01/ ... cs_10/
│       └── gs_01/ ... gs_10/
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
- one task JSON file such as `123_topic_title.json`

To evaluate a model output for the same topic, create a parallel `llm/` directory with:

- `content.json`
- `outline.json`
- `reference.json`

## Released Topics

The current release contains `20` topics:

- `cs_01` to `cs_10`: Computer Science
- `gs_01` to `gs_10`: General Science

Topic metadata is stored in `data/experimental_data/experimental_topics.csv`, including title, venue, year, and citation count.

## Installation

Use Python `3.10+` and install dependencies in a virtual environment:

```bash
cd Bloom-Eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some metrics additionally rely on downloaded embedding models from Hugging Face, especially `TBal`. Run the scripts in an environment with internet access or with those models pre-cached.

## API Configuration

The LLM-based metrics require an OpenAI-compatible chat completion endpoint.

Required environment variable:

```bash
export OPENAI_API_KEY=your_api_key
```

Optional overrides:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export BLOOM_EVAL_MODEL=gpt-5-mini
```

You can also override model and base URL per run with `--model` and `--base-url`.

## Input Format

The evaluation scripts expect topic-level JSON inputs:

- `content.json`: survey content text
- `outline.json`: outline entries
- `reference.json`: cited-paper metadata
- task JSON: topic-specific task description used by `FAP`, `FNov`, and `ROQ`

For a full topic run, the directory should look like:

```text
topic_xx/
├── human/
│   ├── content.json
│   ├── outline.json
│   ├── reference.json
│   └── 123_some_topic.json
└── llm/
    ├── content.json
    ├── outline.json
    └── reference.json
```

## Run All Core Metrics

The easiest way to evaluate one topic is:

```bash
./run_topic_all_metrics.sh \
  --topic-dir /path/to/topic_xx \
  --output-dir results/topic_xx_all
```

Or pass `human/` and `llm/` explicitly:

```bash
./run_topic_all_metrics.sh \
  --human-dir data/experimental_data/cs_01/human \
  --llm-dir /path/to/cs_01/llm \
  --output-dir results/cs_01_all
```

Optional flags:

- `--model NAME`: override the chat model for API-based metrics
- `--base-url URL`: override the OpenAI-compatible endpoint
- `--save-raw-response`: save raw model responses into `logs/`
- `--include-readability`: also run the extra readability script outside the core 16 metrics

The batch script writes one subdirectory per metric plus a `summary.txt` file under the output root.

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

An additional non-core script is available:

- `scripts/others/run_Readability.py`

## Run a Single Metric

All metric scripts are available under `scripts/level*/run_*.py`. Example:

```bash
python scripts/level2/run_OTC.py \
  --outline_file_human data/experimental_data/cs_01/human/outline.json \
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

Each metric writes its results into the selected output directory, typically including:

- `result.json`: structured metric output
- `report.txt`: human-readable summary
- intermediate JSON artifacts for extraction or matching stages
- optional `logs/` files when `--save_raw_response` is enabled

## Notes

- `run_topic_all_metrics.sh` requires `OPENAI_API_KEY` because the full 16-metric run includes API-based metrics.
- `TBal` loads a sentence-transformer embedding model and may take noticeably longer than the lightweight scripts.
- The released topics only contain `human/` inputs. You must provide the corresponding `llm/` outputs to run pairwise evaluation.

## Citation

If you use this repository, please cite the Bloom-Eval paper.
