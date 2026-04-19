from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from bloom_eval.schemas import BenchmarkSummary, Topic


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_benchmark_summary(root: Path | None = None) -> BenchmarkSummary:
    repo_root = root or REPO_ROOT
    summary_path = repo_root / "data" / "metadata" / "benchmark_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return BenchmarkSummary(
        name=payload["name"],
        paper_count=int(payload["paper_count"]),
        venue_count=int(payload["venue_count"]),
        domain_count=int(payload["domain_count"]),
        evaluation_topic_count=int(payload["evaluation_topic_count"]),
    )


def load_topics(root: Path | None = None) -> list[Topic]:
    repo_root = root or REPO_ROOT
    topics_path = repo_root / "data" / "topics" / "experimental_topics.csv"
    topics: list[Topic] = []
    with topics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            topics.append(
                Topic(
                    topic_id=row["topic_id"],
                    domain=row["domain"],
                    title=row["title"],
                    venue=row["venue"],
                    year=int(row["year"]),
                    citations=int(row["citations"]),
                )
            )
    return topics


def validate_result_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = [
        "topic_id",
        "domain",
        "system_name",
        "survey_text_path",
        "references",
        "metrics",
    ]
    for field in required_fields:
        if field not in record:
            errors.append(f"missing field: {field}")

    if "references" in record and not isinstance(record["references"], list):
        errors.append("references must be a list")
    if "metrics" in record and not isinstance(record["metrics"], dict):
        errors.append("metrics must be a dictionary")

    return errors


def cmd_benchmark_summary(_: argparse.Namespace) -> int:
    summary = load_benchmark_summary()
    print(f"name: {summary.name}")
    print(f"papers: {summary.paper_count}")
    print(f"venues: {summary.venue_count}")
    print(f"domains: {summary.domain_count}")
    print(f"evaluation_topics: {summary.evaluation_topic_count}")
    return 0


def cmd_show_topics(args: argparse.Namespace) -> int:
    topics = load_topics()
    for topic in topics[: args.limit]:
        print(f"{topic.topic_id}\t{topic.domain}\t{topic.year}\t{topic.title}")
    return 0


def cmd_validate_result(args: argparse.Namespace) -> int:
    record_path = Path(args.path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    errors = validate_result_record(payload)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("result record is valid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloom-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("benchmark-summary", help="Show benchmark summary.")
    summary_parser.set_defaults(func=cmd_benchmark_summary)

    topics_parser = subparsers.add_parser("show-topics", help="Show evaluation topics.")
    topics_parser.add_argument("--limit", type=int, default=20)
    topics_parser.set_defaults(func=cmd_show_topics)

    validate_parser = subparsers.add_parser("validate-result", help="Validate a result record JSON.")
    validate_parser.add_argument("path")
    validate_parser.set_defaults(func=cmd_validate_result)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

