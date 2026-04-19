# Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue.svg)](https://2026.aclweb.org/)
[![Status](https://img.shields.io/badge/Status-Camera--ready-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文说明](README_zh.md)

This is the official repository for the paper **"Bloom-Eval: A Hierarchical Evaluation Benchmark for Automatic Survey Generation Based on Bloom's Taxonomy"**. The project is currently in the camera-ready stage, and the code and data release is being organized for public release.

## Project Overview

Bloom-Eval is the first six-level hierarchical benchmark designed specifically for automatic survey generation (ASG). Grounded in Bloom's Taxonomy, it moves beyond flat evaluation and provides a fine-grained framework for diagnosing ASG systems across multiple cognitive dimensions.

## Framework Overview

![Bloom-Eval framework overview](paper/latex/bloom-eval_10_05.png)
*Figure: Overview of the Bloom-Eval framework, including (a) the data collection pipeline and (b) the six-level evaluation hierarchy covering memory, comprehension, application, analysis, evaluation, and creation.*

---

## Key Features

### 1. Six-Level Cognitive Evaluation Framework
Bloom-Eval analyzes ASG systems through six cognitive levels:

- **Memory**: evaluates the accurate recall of domain entities, core references, and factual statements.
- **Comprehension**: evaluates the ability to summarize the research landscape, preserve citation faithfulness, and maintain topical focus.
- **Application**: evaluates the execution of academic formatting conventions, document structure, and organizational paradigms.
- **Analysis**: evaluates hierarchical reasoning, appropriate analytical granularity, and structural clarity.
- **Evaluation**: evaluates critical judgment over existing literature, including conclusions and limitations.
- **Creation**: evaluates the ability to construct novel conceptual frameworks and identify future research directions.

### 2. GRADE Evaluation Method
We introduce **GRADE** (Generative Rubric Adaptive Differential Evaluation), a rubric-based comparative evaluation method:

- **Transparent rubrics**: the evaluator first generates explicit, weighted criteria tailored to the survey topic.
- **Differential scoring**: system outputs are compared against expert-written surveys with textual justifications to improve auditability.

### 3. Large-Scale Cross-Domain Benchmark

- **Scale**: `3,506` manually verified expert survey papers.
- **Coverage**: `60` top-tier academic venues and journals, including conferences such as ACL and ICLR, as well as venues such as *Annual Reviews*.
- **Diversity**: `14` scientific domains and `20` representative evaluation topics.

---

## Current Release Scope

This repository is an initial camera-ready release. The current public version includes:

- benchmark documentation and metric descriptions
- released evaluation topics and metadata
- cleaned research scripts for benchmark construction and evaluation
- sample result files
- paper source files and project figures

Some components are still being finalized for a later, more complete release, including additional prompts, more polished execution pipelines, and expanded metadata.

## Repository Structure

```text
.
├── code/               # Evaluation scripts and experiment utilities
├── configs/            # Example evaluation configurations
├── data/               # Benchmark metadata, samples, and topic files
├── docs/               # Benchmark, metrics, and reproducibility documents
├── paper/              # Paper source files and figures
├── prompts/            # Prompt inventory and release notes
├── results/            # Sample benchmark outputs and result tables
├── scripts/            # Release checking utilities
├── src/                # Minimal Python package and CLI
├── tests/              # Sanity tests
└── README.md
```

## Data and Release Note

The benchmark is derived from published papers across conferences and journals. Unless redistribution rights are explicitly confirmed, this repository should release metadata, identifiers, derived annotations, and evaluation outputs rather than copyrighted full-text PDFs.

## Citation

If you use this repository, please cite the Bloom-Eval paper. Citation metadata is available in `CITATION.cff`.
