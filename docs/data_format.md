# Data Format

## Release Principle

This benchmark is derived from published survey papers. Before public release, separate:

- releasable metadata and annotations
- non-releasable copyrighted full text

## Recommended Public Data Structure

### `data/topics/`

Store evaluation topic manifests:

- `topic_id`
- `domain`
- `title`
- `venue`
- `year`
- `citation_count`

### `data/metadata/`

Store benchmark-wide metadata:

- venue list
- domain list
- year filters
- corpus counts
- hashes or external identifiers

### `data/samples/`

Store example record schemas:

- generated survey record
- reference list record
- metric output record

## Recommended Non-Public or Rights-Checked Assets

- publisher PDFs
- scraped full text
- proprietary APIs outputs that cannot be redistributed

## Result Record Requirements

Each evaluation record should contain at least:

- `topic_id`
- `domain`
- `system_name`
- `survey_text_path`
- `references`
- `metrics`

