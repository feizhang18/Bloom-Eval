# Benchmark Overview

Bloom-Eval is a benchmark for automatic survey generation (ASG) systems. It evaluates generated surveys against expert-written survey papers using a six-level cognitive hierarchy derived from Bloom's Taxonomy:

- Memory
- Comprehension
- Application
- Analysis
- Evaluation
- Creation

## Corpus Construction

According to the paper:

- The benchmark corpus contains `3,506` manually verified survey papers.
- Papers come from `60` peer-reviewed venues.
- Publication years are restricted to `2023-2025`.
- The benchmark spans `14` scientific domains.
- The evaluation set uses `20` representative topics chosen from the most-cited papers.

## Evaluation Philosophy

Bloom-Eval uses a dual-constraint strategy:

- Deterministic algorithms are preferred whenever possible.
- LLMs are restricted to extraction, matching, or constrained judging.

For abstract capabilities, Bloom-Eval uses `GRADE`:

1. Generate an explicit rubric.
2. Score both the generated survey and the human reference.
3. Compute a relative score against the human reference.

## What a Public Release Should Include

- topic manifests
- metric definitions
- prompt text
- result schemas
- evaluation scripts
- baseline outputs
- environment notes
- paper source

