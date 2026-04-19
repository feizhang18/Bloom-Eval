from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    topic_id: str
    domain: str
    title: str
    venue: str
    year: int
    citations: int


@dataclass(frozen=True)
class BenchmarkSummary:
    name: str
    paper_count: int
    venue_count: int
    domain_count: int
    evaluation_topic_count: int

