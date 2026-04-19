# Reproducibility

## Reported Evaluation Setup

The paper states the following evaluation choices:

- `gpt-5-mini` with temperature `0.0` for extraction, matching, and scoring tasks
- `gpt-4o` for citation-related modules aligned with prior work
- `nomic-ai/nomic-embed-text-v1` for text embeddings

## Reproducibility Requirements For This Repository

- exact prompt text in machine-readable files
- model names and versions
- temperature and decoding settings
- metric-specific input and output schemas
- benchmark topic manifest
- released result tables in CSV or JSON

## Recommended Future Additions

- pinned environment lockfile
- script-level seeds where applicable
- prompt versioning
- API response caching or cache-key manifests

