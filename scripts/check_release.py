from __future__ import annotations

from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "CITATION.cff",
    "docs/benchmark.md",
    "docs/metrics.md",
    "docs/data_format.md",
    "docs/reproducibility.md",
    "docs/release_checklist.md",
    "data/topics/experimental_topics.csv",
    "data/metadata/benchmark_summary.json",
    "results/samples/main_results.csv",
    "prompts/README.md",
    "paper/README.md",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_PATHS if not (repo_root / path).exists()]
    if missing:
        print("missing required release files:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("release scaffold check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
