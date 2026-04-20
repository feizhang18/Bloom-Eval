# Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue.svg)](https://2026.aclweb.org/)
[![Status](https://img.shields.io/badge/Status-Camera--ready-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the paper **"Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy"**.

The repository is currently released as a compact camera-ready version. We keep only the minimum public materials needed to describe the benchmark, expose the released topics, and share the evaluation scripts.

## Project Overview

Bloom-Eval is a six-level hierarchical benchmark for automatic survey generation (ASG). Grounded in Bloom's Taxonomy, it provides fine-grained evaluation across memory, comprehension, application, analysis, evaluation, and creation.

## Framework Overview

![Bloom-Eval framework overview](bloom-eval_10_05.png)

*Figure: Overview of the Bloom-Eval framework, including the data collection pipeline and the six-level evaluation hierarchy.*

## Key Features

### 1. Six-Level Cognitive Evaluation Framework

- **Memory**: accurate recall of entities, core references, and factual statements.
- **Comprehension**: understanding of the research landscape, citation faithfulness, and topical focus.
- **Application**: execution of academic conventions, document structure, and organizing paradigms.
- **Analysis**: hierarchical reasoning, analytical granularity, and structural clarity.
- **Evaluation**: critical judgment over conclusions and limitations in prior work.
- **Creation**: generation of novel conceptual frameworks and future research directions.

### 2. GRADE Evaluation Method

We use **GRADE** (Generative Rubric Adaptive Differential Evaluation), a rubric-based comparative evaluation method for expert-aligned assessment.

### 3. Cross-Domain Benchmark

- `3,506` manually verified expert survey papers in the full benchmark curation pipeline
- `60` academic venues and journals
- `14` scientific domains
- `20` released evaluation topics in this camera-ready repository

## Repository Structure

```text
.
├── dataset/            # Bloom-Eval dataset metadata for the 20 released topics
├── scripts/            # Evaluation scripts, including deterministic metrics and GRADE-related code
├── prompts/            # Prompt templates and prompt release notes
└── README.md
```

## Release Scope

This camera-ready repository currently includes:

- `dataset/experimental_topics.csv`: the 20 released evaluation topics
- `scripts/`: benchmark scripts organized by evaluation level
- `prompts/`: prompt placeholders and release notes

This repository does not include the full paper source tree or copyrighted full-text papers.

## Citation

If you use this repository, please cite the Bloom-Eval paper. Citation metadata is available in `CITATION.cff`.
