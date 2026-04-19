# Leaderboard Format

This directory should store aggregated benchmark results in machine-readable form.

Recommended files:

- `leaderboard.csv`: one row per system
- `detailed_results.json`: per-topic or per-metric breakdown
- `run_metadata.json`: evaluation date, model versions, prompt versions, and config hash

At minimum, each leaderboard row should include:

- `system_name`
- `memory_avg`
- `comprehension_avg`
- `application_avg`
- `analysis_avg`
- `evaluation_avg`
- `creation_avg`
- `overall_avg`

