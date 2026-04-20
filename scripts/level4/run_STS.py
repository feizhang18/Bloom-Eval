import os
import json
import re
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import zss
from sentence_transformers import SentenceTransformer, util
from zss import Node

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, resolve_output_dir, to_project_relative, write_json, write_text


SBERT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"


def clean_title(title: str) -> str:
    return re.sub(r'^\d+(\.\d+)*\s*', '', str(title)).strip()


def load_outline(path: str) -> List[List[Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Outline file is not a list: {path}")
    return data


def normalize_outline(outline_data: List[List[Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for item in outline_data:
        if not isinstance(item, list) or len(item) < 2:
            continue
        normalized.append({
            "level": int(item[0]),
            "title": str(item[1]),
            "clean_title": clean_title(item[1]),
        })
    return normalized


def parse_to_tree(outline_data: List[List[Any]]) -> Node:
    root = Node("root")
    level_parents = {-1: root}
    for item in outline_data:
        if not isinstance(item, list) or len(item) < 2:
            continue
        level = int(item[0])
        title = clean_title(item[1])
        parent_node = level_parents.get(level - 1, root)
        new_node = Node(title)
        parent_node.addkid(new_node)
        level_parents[level] = new_node
    return root


def get_all_topics(node: Node) -> List[str]:
    topics = [node.label] if node.label != "root" else []
    for child in node.children:
        topics.extend(get_all_topics(child))
    return topics


def calculate_structural_similarity(
    tree_expert: Node,
    tree_llm: Node,
    model: SentenceTransformer,
) -> Dict[str, Any]:
    embedding_cache: Dict[str, Any] = {}

    def get_embedding(label: str):
        if label not in embedding_cache:
            embedding_cache[label] = model.encode(label, convert_to_tensor=True)
        return embedding_cache[label]

    def semantic_update_cost(node_a: Node, node_b: Node) -> float:
        emb_a = get_embedding(node_a.label)
        emb_b = get_embedding(node_b.label)
        similarity = util.cos_sim(emb_a, emb_b).item()
        return 1.0 - similarity

    nodes_expert = get_all_topics(tree_expert)
    nodes_llm = get_all_topics(tree_llm)
    if not nodes_expert or not nodes_llm:
        return {
            "semantic_tree_similarity": 0.0,
            "tree_edit_distance": None,
            "distance_upper_bound": len(nodes_expert) + len(nodes_llm),
            "expert_topic_count": len(nodes_expert),
            "llm_topic_count": len(nodes_llm),
        }

    distance = zss.distance(
        tree_expert,
        tree_llm,
        Node.get_children,
        insert_cost=lambda node: 1,
        remove_cost=lambda node: 1,
        update_cost=semantic_update_cost,
    )
    max_dist = len(nodes_expert) + len(nodes_llm)
    score = 1.0 - (distance / max_dist) if max_dist > 0 else 0.0
    return {
        "semantic_tree_similarity": score,
        "tree_edit_distance": distance,
        "distance_upper_bound": max_dist,
        "expert_topic_count": len(nodes_expert),
        "llm_topic_count": len(nodes_llm),
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic Tree Similarity (STS) evaluation tool")
    parser.add_argument("--outline_file_human", "--human_outline", dest="outline_file_human", type=str, required=True, help="Path to human outline.json")
    parser.add_argument("--outline_file_llm", "--llm_outline", dest="outline_file_llm", type=str, required=True, help="Path to LLM outline.json")
    add_common_arguments(parser, metric_name="sts", include_model=False)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    try:
        human_outline = load_outline(args.outline_file_human)
        llm_outline = load_outline(args.outline_file_llm)
    except Exception as e:
        print(f"Error loading input: {e}")
        return

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(SBERT_MODEL_NAME, trust_remote_code=True)

    human_tree = parse_to_tree(human_outline)
    llm_tree = parse_to_tree(llm_outline)
    metrics = calculate_structural_similarity(human_tree, llm_tree, model)

    intermediate = {
        "human_outline": normalize_outline(human_outline),
        "llm_outline": normalize_outline(llm_outline),
        "human_topics": get_all_topics(human_tree),
        "llm_topics": get_all_topics(llm_tree),
        "metrics": metrics,
    }
    intermediate_path = output_dir / "intermediate.json"
    write_json(intermediate_path, intermediate)

    report_lines = [
        "========================================",
        "   Bloom-Eval Level 4: STS Report",
        "========================================",
        f"Human topics: {metrics['expert_topic_count']}",
        f"LLM topics: {metrics['llm_topic_count']}",
        f"Tree edit distance: {metrics['tree_edit_distance']}",
        f"Distance upper bound: {metrics['distance_upper_bound']}",
        f"Semantic Tree Similarity (STS): {metrics['semantic_tree_similarity']:.4f}",
        "========================================",
    ]
    report_text = "\n".join(report_lines)
    report_path = output_dir / "report.txt"
    write_text(report_path, report_text)
    final_path = output_dir / "result.json"
    write_json(
        final_path,
        build_result_payload(
            metric="STS",
            inputs={
                "outline_file_human": to_project_relative(Path(args.outline_file_human)),
                "outline_file_llm": to_project_relative(Path(args.outline_file_llm)),
            },
            results=metrics,
            artifacts={
                "report_file": to_project_relative(report_path),
                "intermediate_file": to_project_relative(intermediate_path),
            },
        ),
    )

    print("\n" + report_text)
    print(f"Intermediate results saved to: {intermediate_path}")
    print(f"Final results saved to: {final_path}")


if __name__ == "__main__":
    main()
