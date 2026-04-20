import os
import re
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from zss import Node

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, load_json, resolve_output_dir, to_project_relative, write_json, write_text


def clean_title(title: str) -> str:
    return re.sub(r'^\d+(\.\d+)*\s*', '', str(title)).strip()


def load_outline(path: str) -> List[List[Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = load_json(path)
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


def get_tree_depth(node: Node) -> int:
    if not node.children:
        return 1
    return 1 + max(get_tree_depth(child) for child in node.children)


def calculate_granularity(tree_expert: Node, tree_llm: Node) -> Dict[str, float]:
    depth_expert = get_tree_depth(tree_expert) - 1
    depth_llm = get_tree_depth(tree_llm) - 1
    count_expert = len(get_all_topics(tree_expert))
    count_llm = len(get_all_topics(tree_llm))

    depth_ratio = (
        min(depth_expert, depth_llm) / max(depth_expert, depth_llm)
        if max(depth_expert, depth_llm) > 0 else 0.0
    )
    breadth_ratio = (
        min(count_expert, count_llm) / max(count_expert, count_llm)
        if max(count_expert, count_llm) > 0 else 0.0
    )
    shape_consistency = float(np.sqrt(depth_ratio * breadth_ratio))
    return {
        "human_depth": depth_expert,
        "llm_depth": depth_llm,
        "human_topic_count": count_expert,
        "llm_topic_count": count_llm,
        "depth_consistency": depth_ratio,
        "breadth_consistency": breadth_ratio,
        "shape_consistency": shape_consistency,
    }


def main():
    parser = argparse.ArgumentParser(description="Structure Consistency (SCons) evaluation tool")
    parser.add_argument("--outline_file_human", "--human_outline", dest="outline_file_human", type=str, required=True, help="Path to human outline.json")
    parser.add_argument("--outline_file_llm", "--llm_outline", dest="outline_file_llm", type=str, required=True, help="Path to LLM outline.json")
    add_common_arguments(parser, metric_name="scons", include_model=False)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    try:
        human_outline = load_outline(args.outline_file_human)
        llm_outline = load_outline(args.outline_file_llm)
    except Exception as e:
        print(f"Error loading input: {e}")
        return

    human_tree = parse_to_tree(human_outline)
    llm_tree = parse_to_tree(llm_outline)
    metrics = calculate_granularity(human_tree, llm_tree)

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
        "   Bloom-Eval Level 4: SCons Report",
        "========================================",
        f"Human depth: {metrics['human_depth']}",
        f"LLM depth: {metrics['llm_depth']}",
        f"Human topics: {metrics['human_topic_count']}",
        f"LLM topics: {metrics['llm_topic_count']}",
        f"Depth consistency (DC): {metrics['depth_consistency']:.4f}",
        f"Breadth consistency (BC): {metrics['breadth_consistency']:.4f}",
        f"Shape consistency (ShapeCons): {metrics['shape_consistency']:.4f}",
        "========================================",
    ]
    report_text = "\n".join(report_lines)
    report_path = output_dir / "report.txt"
    write_text(report_path, report_text)
    final_path = output_dir / "result.json"
    write_json(
        final_path,
        build_result_payload(
            metric="SCons",
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
