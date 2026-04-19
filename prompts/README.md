# Prompt Inventory

For a full public release, this directory should contain the exact machine-readable prompt text used by Bloom-Eval.

## Prompt Families Mentioned In The Paper

- `level1_entity_extraction`
- `level1_entity_matching`
- `level1_factual_extraction`
- `level1_factual_matching`
- `level2_outline_alignment`
- `level5_critical_matching`
- `level6_roq_rubric_generation_step1a`
- `level6_roq_rubric_generation_step1b`
- `level6_roq_scoring_step2`
- corresponding prompt files for `FAP` and `FNov`

## Recommended File Convention

- one prompt per `.md` or `.yaml` file
- include model role, inputs, outputs, and strict formatting requirements
- version every prompt change

## Current Situation

The current local paper source includes prompt screenshots in the appendix. Before the repository is published, export the prompt text from those authoring materials into this directory.

