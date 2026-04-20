import json
import re
import os
import time
import sys
import argparse
from pathlib import Path
from openai import OpenAI
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, List
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, ensure_dir, build_result_payload, call_llm, load_json, parse_llm_json, resolve_output_dir, to_project_relative, write_json, write_text
from prompt_utils import load_prompt

API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
RANDOM_SEED = 42

try:
    TOPIC_MATCHING_PROMPT_TEMPLATE = load_prompt("level2/TFSim_topic_matching.txt")
except Exception as e:
    print(f"Error: Cannot load prompt template, check path: {e}")
    sys.exit(1)


def calculate_ds_score(p: np.ndarray, q: np.ndarray) -> float:
    """Compute composite distribution similarity DS-Score."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    
    hellinger_dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    hellinger_sim = 1 - hellinger_dist
    total_variation_dist = np.sum(np.abs(p - q)) / 2
    total_variation_sim = 1 - total_variation_dist
    
    m = 0.5 * (p + q)
    js_div = 0.5 * (entropy(p, m) + entropy(q, m)) 
    js_div_normalized = js_div / np.log(2)
    js_sim = 1 - js_div_normalized
    
    return (hellinger_sim + total_variation_sim + js_sim) / 3

def get_llm_response(client: OpenAI, model: str, message: List[Dict], log_file: Path | None) -> str:
    print(f"Requesting LLM for topic semantic alignment...")
    try:
        return call_llm(
            client,
            model,
            message[0]["content"],
            log_file,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"Error: API call failed: {e}")
        return "{}"

def load_and_prepare_docs(file_path: str) -> List[str]:
    """Read reference JSON and concatenate title and abstract."""
    if not os.path.exists(file_path):
        print(f"Error: Input file not found: {file_path}")
        return []
    try:
        data = load_json(file_path)
        documents = [
            re.sub(r'\s+', ' ', (v_inner.get('searched_title', '') + ". " + v_inner.get('abs', ''))).strip()
            for k, v in data.items() if k.startswith('paper_') and isinstance(v, dict)
            for v_inner in v.values() if isinstance(v_inner, dict) and v_inner.get('searched_title') and v_inner.get('abs', '').strip().upper() != 'N/A'
        ]
        return documents
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return []

def discover_topics_and_freqs(documents: List[str], embedding_model: SentenceTransformer) -> pd.DataFrame:
    """Use BERTopic to extract topics and frequencies."""
    if len(documents) < 5:
        return pd.DataFrame(columns=['Topic', 'Name', 'Count'])

    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=RANDOM_SEED)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))

    topic_model = BERTopic(
        language="english", min_topic_size=2, verbose=False,
        embedding_model=embedding_model, vectorizer_model=vectorizer_model, umap_model=umap_model
    )
    topic_model.fit_transform(documents)
    topic_info = topic_model.get_topic_info()
    return topic_info[topic_info['Topic'] != -1]


def main():
    parser = argparse.ArgumentParser(description="Reference Topic Coverage and Distribution Similarity Evaluation Tool")
    parser.add_argument("--reference_file_human", "--human_ref", dest="reference_file_human", type=str, required=True, help="Path to the human-expert reference.json")
    parser.add_argument("--reference_file_llm", "--llm_ref", dest="reference_file_llm", type=str, required=True, help="Path to the LLM-generated reference.json")
    add_common_arguments(parser, metric_name="tfsim", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        return

    output_dir = resolve_output_dir(args.output_dir)
    log_dir = ensure_dir(output_dir / "logs") if args.save_raw_response else None

    client = OpenAI(api_key=API_KEY, base_url=args.base_url)

    print("Starting topic distribution evaluation...")
    print("Loading SentenceTransformer embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    human_docs = load_and_prepare_docs(args.reference_file_human)
    llm_docs = load_and_prepare_docs(args.reference_file_llm)

    if not human_docs or not llm_docs:
        print("Error: Missing valid reference content, aborting.")
        return

    print("Extracting latent research topics with BERTopic...")
    expert_topic_info = discover_topics_and_freqs(human_docs, embedding_model)
    llm_topic_info = discover_topics_and_freqs(llm_docs, embedding_model)

    expert_topics = expert_topic_info['Name'].tolist()
    llm_topics = llm_topic_info['Name'].tolist()
    print(f"Found {len(expert_topics)} expert topics and {len(llm_topics)} LLM topics.")

    matches_path = output_dir / "topic_matches.json"
    log_path = (log_dir / f"llm_match_log_{time.strftime('%Y%m%d_%H%M%S')}.json") if log_dir else None

    if os.path.exists(matches_path):
        print("Cached match results found, skipping API call.")
        match_data = load_json(matches_path)
    else:
        if not expert_topics or not llm_topics:
            match_data = {"matched_pairs": []}
        else:
            prompt = TOPIC_MATCHING_PROMPT_TEMPLATE.format(
                expert_topics_json=json.dumps(expert_topics, indent=2),
                llm_topics_json=json.dumps(llm_topics, indent=2)
            )
            response_str = get_llm_response(client, args.model, [{"role": "user", "content": prompt}], log_path)
            try:
                match_data = parse_llm_json(response_str, kind="auto")
                write_json(matches_path, match_data)
            except json.JSONDecodeError:
                match_data = {"matched_pairs": []}

    matched_pairs = match_data.get("matched_pairs", [])
    matched_llm_topics = [pair['llm_topic'] for pair in matched_pairs if 'llm_topic' in pair]
    matched_expert_topics = [pair['expert_topic'] for pair in matched_pairs if 'expert_topic' in pair]

    tp = len(set(matched_llm_topics))
    fp = len(llm_topics) - tp
    fn = len(set(expert_topics) - set(matched_expert_topics))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    ds_score = 0.0
    if matched_pairs and not expert_topic_info.empty and not llm_topic_info.empty:
        expert_counts_map = pd.Series(expert_topic_info.Count.values, index=expert_topic_info.Name).to_dict()
        llm_counts_map = pd.Series(llm_topic_info.Count.values, index=llm_topic_info.Name).to_dict()

        common_expert_topics = sorted(list(set(matched_expert_topics)))
        if common_expert_topics:
            p_counts = [expert_counts_map.get(et, 0) for et in common_expert_topics]
            q_counts = [sum(llm_counts_map.get(pair['llm_topic'], 0) for pair in matched_pairs if pair['expert_topic'] == et) for et in common_expert_topics]

            if sum(p_counts) > 0 and sum(q_counts) > 0:
                ds_score = calculate_ds_score(np.array(p_counts), np.array(q_counts))

    report_lines = [
        "=======================================================",
        "     Bloom-Eval Level 2: Topic & Freq Sim Report       ",
        "=======================================================",
        f"Expert topic clusters: {len(expert_topics)}",
        f"LLM topic clusters:    {len(llm_topics)}",
        "-------------------------------------------------------",
        f"Topic Precision: {precision:.4f} ({precision:.2%})",
        f"Topic Recall:    {recall:.4f} ({recall:.2%})",
        f"Topic F1-Score:  {f1_score:.4f} ({f1_score:.2%})",
        "-------------------------------------------------------",
        f"Distribution Similarity (DS-Score): {ds_score:.4f}",
        "   (Hellinger, Total Variation, Jensen-Shannon average)",
        "======================================================="
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = output_dir / "report.txt"
    write_text(report_path, report_text)
    write_json(
        output_dir / "result.json",
        build_result_payload(
            metric="TFSim",
            inputs={
                "reference_file_human": to_project_relative(Path(args.reference_file_human)),
                "reference_file_llm": to_project_relative(Path(args.reference_file_llm)),
            },
            results={
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "ds_score": ds_score,
                "human_topic_count": len(expert_topics),
                "llm_topic_count": len(llm_topics),
            },
            config={
                "model": args.model,
                "base_url": args.base_url,
            },
            artifacts={
                "report_file": to_project_relative(report_path),
                "topic_matches_file": to_project_relative(matches_path),
            },
        ),
    )

    print(f"\nEvaluation complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
