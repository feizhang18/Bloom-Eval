import os
import json
import re
import argparse
import time
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sys
from pathlib import Path
from typing import List, Dict
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, resolve_output_dir, to_project_relative, write_json, write_text

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
RANDOM_SEED = 42


def calculate_gini(counts: List[int]) -> float:
    """Compute Gini coefficient to measure topic distribution concentration."""
    counts = np.array(counts, dtype=np.float64)
    if counts.size == 0: return 1.0
    counts = np.sort(counts)
    n = len(counts)
    cum_counts = np.cumsum(counts)
    total_sum = cum_counts[-1]
    if total_sum == 0: return 0.0
    B = np.sum(cum_counts) / total_sum
    return 1 - 2 * B / n + 1 / n

def load_and_prepare_docs(file_path: str) -> List[str]:
    """Extract and clean titles and abstracts from reference_3.json."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        documents = []
        for key, value in data.items():
            if key.startswith('paper_') and isinstance(value, dict):
                inner_ref = next(iter(value.values()), None)
                if inner_ref and isinstance(inner_ref, dict):
                    title = inner_ref.get('searched_title', '')
                    abstract = inner_ref.get('abs', '')
                    if title and abstract and abstract.strip().upper() != 'N/A':
                        full_text = f"{title}. {abstract}"
                        documents.append(re.sub(r'\s+', ' ', full_text).strip())
        return documents
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def analyze_topics(name: str, documents: List[str], embedding_model: SentenceTransformer) -> Dict:
    """Run BERTopic for topic distribution analysis."""
    print(f"--- Analyzing topic distribution for {name} ({len(documents)} documents) ---")
    if len(documents) < 5:
        return {"topic_count": 0, "gini": 1.0, "breadth_score": 0.0, "topic_info": None}

    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=RANDOM_SEED)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))

    topic_model = BERTopic(
        language="english", min_topic_size=2, verbose=False,
        embedding_model=embedding_model, vectorizer_model=vectorizer_model, umap_model=umap_model
    )

    topic_model.fit_transform(documents)
    topic_info = topic_model.get_topic_info()

    valid_topics = topic_info[topic_info['Topic'] != -1]
    topic_counts = valid_topics['Count'].tolist()

    gini = calculate_gini(topic_counts)
    return {
        "topic_count": len(topic_counts),
        "gini": gini,
        "breadth_score": 1 - gini,
        "topic_info": valid_topics
    }


def main():
    parser = argparse.ArgumentParser(description="Reference Topic Balance and Breadth Evaluation Tool")
    parser.add_argument("--reference_file_human", "--human_ref", dest="reference_file_human", type=str, required=True, help="Path to the human-expert reference_3.json")
    parser.add_argument("--reference_file_llm", "--llm_ref", dest="reference_file_llm", type=str, required=True, help="Path to the LLM reference_3.json")
    add_common_arguments(parser, metric_name="tbal", include_model=False)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)

    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    human_docs = load_and_prepare_docs(args.reference_file_human)
    llm_docs = load_and_prepare_docs(args.reference_file_llm)

    human_res = analyze_topics("Human", human_docs, embedding_model)
    llm_res = analyze_topics("LLM", llm_docs, embedding_model)

    report = [
        "=======================================================",
        "      Bloom-Eval Level 2: Topic Balance Report         ",
        "=======================================================",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "\n[1] Human Baseline",
        f"  - Topic count:   {human_res['topic_count']}",
        f"  - Gini:          {human_res['gini']:.4f}",
        f"  - Breadth score: {human_res['breadth_score']:.4f}",
        "\n[2] LLM Generated",
        f"  - Topic count:   {llm_res['topic_count']}",
        f"  - Gini:          {llm_res['gini']:.4f}",
        f"  - Breadth score: {llm_res['breadth_score']:.4f}",
        "\n[3] Gap Analysis",
        f"  - Breadth gap: {llm_res['breadth_score'] - human_res['breadth_score']:.4f}",
        "======================================================="
    ]

    report_text = "\n".join(report)
    print("\n" + report_text)

    report_path = output_dir / "report.txt"
    write_text(report_path, report_text)

    if human_res['topic_info'] is not None:
        human_res['topic_info'].to_csv(output_dir / "human_topics.csv", index=False)
    if llm_res['topic_info'] is not None:
        llm_res['topic_info'].to_csv(output_dir / "llm_topics.csv", index=False)

    write_json(
        output_dir / "result.json",
        build_result_payload(
            metric="TBal",
            inputs={
                "reference_file_human": to_project_relative(Path(args.reference_file_human)),
                "reference_file_llm": to_project_relative(Path(args.reference_file_llm)),
            },
            results={
                "human": {
                    "topic_count": human_res["topic_count"],
                    "gini": human_res["gini"],
                    "breadth_score": human_res["breadth_score"],
                },
                "llm": {
                    "topic_count": llm_res["topic_count"],
                    "gini": llm_res["gini"],
                    "breadth_score": llm_res["breadth_score"],
                },
                "breadth_gap": llm_res["breadth_score"] - human_res["breadth_score"],
            },
            artifacts={"report_file": to_project_relative(report_path)},
        ),
    )

    print(f"\nEvaluation complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
