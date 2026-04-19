# Code Release

This directory contains a sanitized subset of the local research code used to build benchmark artifacts for Bloom-Eval.

## Contents

- `benchmarks/`: metric-specific evaluation scripts grouped by Bloom-Eval level
- `experiment_utils/`: data extraction, reference processing, and experiment preparation utilities

## Important Notes

- Hard-coded API keys were removed.
- Author-specific absolute filesystem paths were replaced with placeholders.
- Some scripts still require local adaptation before execution.
- This is a research release, not yet a polished end-to-end package.

## Placeholder Conventions

- `<EXPERIMENT_ROOT>`: root directory containing the 20 topic experiment folders
- `<OUTPUT_REPORT_PATH>`: output file path for aggregate reports
- `<AUTOSURVEY_DATABASE_JSON>`: local AutoSurvey database path if needed
- `<SURVEYFORGE_DATABASE_JSON>`: local SurveyForge database path if needed

## Expected Environment Variables

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)

## Recommendation

For the camera-ready GitHub link, it is acceptable to expose these scripts exactly as a cleaned research artifact. You do not need to guarantee that every script runs out of the box yet, as long as the repository clearly states the release scope.

