from __future__ import annotations

import json
from pathlib import Path

from bloom_eval.cli import load_benchmark_summary, load_topics, validate_result_record


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_benchmark_summary() -> None:
    summary = load_benchmark_summary(REPO_ROOT)
    assert summary.name == "Bloom-Eval"
    assert summary.paper_count == 3506
    assert summary.evaluation_topic_count == 20


def test_load_topics() -> None:
    topics = load_topics(REPO_ROOT)
    assert len(topics) == 20
    assert topics[0].topic_id == "cs_001"
    assert topics[-1].topic_id == "gs_010"


def test_validate_result_record_accepts_example() -> None:
    sample_path = REPO_ROOT / "data" / "samples" / "survey_record.example.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    assert validate_result_record(payload) == []

