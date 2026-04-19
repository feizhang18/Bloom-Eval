# Upload To GitHub

This repository is already in a state that can be uploaded through the GitHub web UI.

## Recommended Repository Name

- `Bloom-Eval`

## Recommended Short Description

- `Bloom-Eval: a hierarchical benchmark for automatic survey generation based on Bloom's Taxonomy`

## Recommended Visibility

- `Public`

## Suggested README Opening Status

Use the current wording in `README.md`, especially the phrase:

- `initial camera-ready release`

That keeps the repository honest and avoids over-claiming full reproducibility.

## Web Upload Steps

1. Create a new public GitHub repository named `Bloom-Eval`.
2. Do not initialize it with a new README, `.gitignore`, or license on GitHub.
3. In the new repo page, choose `uploading an existing file`.
4. Drag the entire local folder `Bloom-Eval/` into the browser.
5. Wait for GitHub to finish scanning all files.
6. Use a commit message like `Initial camera-ready release`.
7. Click `Commit changes`.

## What Is Safe To Upload In The Current Folder

- documentation
- topic metadata
- result CSV files
- sanitized benchmark scripts
- paper source and figures authored in this project

## What You Should Not Add Before Upload

- unpublished private notes
- API keys
- raw provider responses with secrets
- copyrighted full-text papers you do not have rights to redistribute

## Optional Final Edits Before Clicking Upload

- replace the placeholder `LICENSE` with the final chosen license
- add the final public GitHub URL into the paper if needed
- add exact prompt text under `prompts/` if you finish cleaning them in time
