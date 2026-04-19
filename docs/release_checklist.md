# Release Checklist

This is the practical list of what the GitHub project should contain before public release.

## Must Have

- repository description and benchmark overview
- explicit open-source license for code. The current `LICENSE` file is a placeholder and must be replaced before publication.
- explicit data license or data usage statement
- citation metadata
- benchmark topic manifest
- machine-readable main results
- reproducibility documentation
- exact prompt text files
- executable evaluation scripts

## Strongly Recommended

- baseline system outputs or pointers to them
- config files for model-backed evaluation
- tests for metadata and schema validation
- CI workflow
- leaderboard aggregation format
- changelog or version tags for releases

## Current Status In This Scaffold

- Included: repo structure, docs, topic manifest, sample results, sample schema, minimal CLI, CI, paper folder
- Missing for full public release: final license choice, exact prompt text exports, full metric implementations, complete benchmark metadata release
