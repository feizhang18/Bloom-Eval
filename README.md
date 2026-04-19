# Bloom-Eval

Bloom-Eval is a hierarchical benchmark for evaluating automatic survey generation (ASG) systems with Bloom's Taxonomy. It organizes evaluation into 6 cognitive levels and 16 metrics, covering memory, comprehension, application, analysis, evaluation, and creation.

This repository is prepared as an `initial camera-ready release`. It is suitable for linking from the paper now, while leaving room for a later full release with cleaned prompts, more complete scripts, and expanded metadata.

This repository is structured as a public benchmark release rather than a single model repository. It is designed to hold:

- Benchmark specifications and metric definitions
- Topic metadata and benchmark manifests
- Evaluation configs and prompt inventory
- Sample result schemas and leaderboard formats
- Reproducibility notes and release checklist
- Paper source files and benchmark documentation

The paper reports:

- `3,506` manually verified survey papers
- `60` peer-reviewed venues
- `14` scientific domains
- `20` representative evaluation topics
- `16` metrics across `6` cognitive levels

## Repository Layout

```text
Bloom-Eval/
├── configs/                  # Example evaluation configs
├── code/                     # Sanitized benchmark scripts and experiment utilities
├── data/
│   ├── metadata/             # Benchmark-level manifests and statistics
│   ├── samples/              # Example record schemas
│   └── topics/               # Experimental topic metadata
├── docs/                     # Benchmark, data, metrics, release docs
├── paper/                    # Local copy of paper source and figures
├── prompts/                  # Prompt inventory and prompt release notes
├── results/
│   ├── leaderboard/          # Aggregated result format
│   └── samples/              # Sample benchmark outputs
├── scripts/                  # Small utilities for release checks
├── src/bloom_eval/           # Minimal Python package and CLI
└── tests/                    # Sanity tests for repo metadata
```

## Initial Release Contents

This camera-ready version currently includes:

- benchmark specification and metric documentation
- the 20 evaluation topics used in the paper
- machine-readable main results
- sanitized benchmark scripts under `code/`
- reproducibility notes and release checklist
- paper source and figures

This version intentionally does not redistribute copyrighted full-text papers or raw publisher PDFs.

## What This Open-Source Project Should Contain

- `docs/benchmark.md`: benchmark scope, corpus construction, and evaluation setup
- `docs/metrics.md`: all 16 metrics and their role in the 6-level hierarchy
- `docs/data_format.md`: what data can be released directly and what must remain metadata-only
- `docs/reproducibility.md`: model choices, temperature, embedding model, and evaluation assumptions
- `docs/release_checklist.md`: items that should be completed before the repository is made public
- `data/topics/experimental_topics.csv`: the 20 evaluation topics from the paper
- `results/samples/main_results.csv`: the main table in machine-readable form
- `prompts/`: exact prompt text files for extraction, matching, and GRADE scoring
- `paper/`: the paper source and figures for reference

## Important Data Release Note

The benchmark corpus is built from published papers across conferences, journals, and `Annual Reviews`. Unless redistribution rights are confirmed, this repository should release only:

- metadata
- identifiers
- URLs / DOIs
- hashes
- derived annotations
- evaluation outputs

It should not redistribute copyrighted full-text PDFs by default.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
bloom-eval benchmark-summary
bloom-eval show-topics --limit 5
python scripts/check_release.py
pytest
```

## Code Release Note

The scripts in `code/` are included as a cleaned research release:

- hard-coded API keys were removed
- author-specific absolute paths were replaced with placeholders such as `<EXPERIMENT_ROOT>`
- the scripts reflect the research code structure used to produce benchmark artifacts

They should be treated as release code for inspection and adaptation, not as a polished one-command reproduction package.

## Current Scope

This scaffold gives you a publishable benchmark repository structure immediately. The remaining release-critical items are called out in [docs/release_checklist.md](docs/release_checklist.md), especially:

- adding an explicit open-source license
- exporting exact prompt text from authoring materials
- adding executable evaluation scripts for each metric
- adding released benchmark metadata beyond the 20-topic subset
