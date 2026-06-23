import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, call_llm_for_json, ensure_dir, format_metric_report, load_json, print_metric_summary, resolve_output_dir, to_project_relative, write_json, write_text
from prompt_utils import load_prompt


API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"
SCS_PROMPT_TEMPLATE = load_prompt("level4/SCS_redundancy_detection.txt")


def load_outline(path: str) -> List[List[Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Outline file is not a list: {path}")
    return data


def prepare_outline_for_prompt(outline_data: List[List[Any]]) -> Tuple[str, Dict[str, str], List[str]]:
    formatted_lines = []
    parent_map: Dict[str, str] = {}
    topic_ids: List[str] = []
    level_parents = {-1: "root"}
    id_counters: Dict[str, int] = {}

    for level, title in outline_data:
        level = int(level)
        clean_title = re.sub(r'^\d+(\.\d+)*\s*', '', str(title)).strip()
        parent_id_prefix = level_parents.get(level - 1, "H").replace('.', '_')
        current_counter = id_counters.get(parent_id_prefix, 0) + 1
        id_counters[parent_id_prefix] = current_counter
        unique_id = (
            f"{parent_id_prefix}_{current_counter}"
            if parent_id_prefix != "root"
            else f"H{current_counter}"
        )
        topic_ids.append(unique_id)
        parent_map[unique_id] = level_parents.get(level - 1)
        level_parents[level] = unique_id
        indent = "  " * level
        formatted_lines.append(f"{indent}{unique_id}: {clean_title}")

    return "\n".join(formatted_lines), parent_map, topic_ids


def get_llm_response(client: OpenAI, model: str, prompt: str, query_id: str, raw_response_path: Optional[Path]) -> Dict[str, Any]:
    print(f"--- Running LLM redundancy analysis for '{query_id}'... ---")
    return call_llm_for_json(
        client,
        model,
        prompt,
        raw_response_path,
        temperature=0.0,
        response_format={"type": "json_object"},
    )


def calculate_scs_for_outline(
    client: OpenAI,
    model: str,
    outline_data: List[List[Any]],
    query_id: str,
    raw_response_path: Optional[Path],
) -> Tuple[float, Dict[str, Any]]:
    if not outline_data or len(outline_data) < 2:
        return 1.0, {
            "formatted_outline": "",
            "topic_ids": [],
            "parent_map": {},
            "raw_llm_result": {"redundant_pairs": []},
            "verified_pairs": [],
            "redundant_pairs_count": 0,
            "total_topics": 0,
        }

    formatted_outline, parent_map, topic_ids = prepare_outline_for_prompt(outline_data)
    prompt = SCS_PROMPT_TEMPLATE.format(formatted_outline=formatted_outline)
    llm_result = get_llm_response(client, model, prompt, query_id, raw_response_path)
    llm_pairs = llm_result.get("redundant_pairs", [])

    total_topics = len(topic_ids)
    redundant_count = 0
    verified_pairs = []
    for pair in llm_pairs:
        if len(pair) != 2:
            continue
        id_a, id_b = pair[0], pair[1]
        if (
            parent_map.get(id_a) is not None
            and parent_map.get(id_b) is not None
            and parent_map.get(id_a) != parent_map.get(id_b)
        ):
            redundant_count += 1
            verified_pairs.append([id_a, id_b])

    print(f"LLM identified {len(llm_pairs)} redundant pairs, {redundant_count} cross-branch pairs verified.")
    scs_score = 1.0 - (redundant_count / total_topics) if total_topics > 0 else 1.0
    return scs_score, {
        "formatted_outline": formatted_outline,
        "topic_ids": topic_ids,
        "parent_map": parent_map,
        "prompt": prompt,
        "raw_llm_result": llm_result,
        "verified_pairs": verified_pairs,
        "redundant_pairs_count": redundant_count,
        "total_topics": total_topics,
    }


def main():
    parser = argparse.ArgumentParser(description="Structure Clarity Score (SCS) evaluation tool")
    parser.add_argument("--outline_file_llm", "--llm_outline", dest="outline_file_llm", type=str, required=True, help="Path to LLM outline.json")
    add_common_arguments(parser, metric_name="scs", default_model=DEFAULT_MODEL)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    try:
        llm_outline = load_outline(args.outline_file_llm)
    except Exception as e:
        print(f"Error loading input: {e}")
        sys.exit(1)

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        sys.exit(1)
    client = OpenAI(api_key=API_KEY, base_url=args.base_url)
    query_id = f"single_scs_{time.strftime('%Y%m%d_%H%M%S')}"
    raw_response_path = output_dir / "logs" / "raw_response.json" if args.save_raw_response else None
    if raw_response_path is not None:
        ensure_dir(raw_response_path.parent)
    scs_score, intermediate = calculate_scs_for_outline(client, args.model, llm_outline, query_id, raw_response_path)

    intermediate_path = output_dir / "intermediate.json"
    write_json(intermediate_path, intermediate)
    final_path = output_dir / "result.json"

    report_path = output_dir / "report.txt"
    inputs = {
        "outline_file_llm": to_project_relative(Path(args.outline_file_llm)),
    }
    metrics = {
        "scs_llm": scs_score,
        "total_topics": intermediate["total_topics"],
        "verified_redundant_pairs": intermediate["redundant_pairs_count"],
    }
    config = {
        "model": args.model,
        "base_url": args.base_url,
    }
    report_text = format_metric_report(
        "SCS",
        "Structure Clarity",
        inputs=inputs,
        results=metrics,
        config=config,
    )
    write_text(report_path, report_text)
    write_json(
        final_path,
        build_result_payload(
            metric="SCS",
            inputs=inputs,
            results=metrics,
            config=config,
            artifacts={
                "report_file": to_project_relative(report_path),
                "intermediate_file": to_project_relative(intermediate_path),
                "raw_response_file": to_project_relative(raw_response_path) if raw_response_path else None,
            },
        ),
    )
    print_metric_summary(
        "SCS",
        report_path,
        final_path,
        results=metrics,
        summary_keys=("scs_llm",),
        artifacts={"intermediate": intermediate_path, "raw response": raw_response_path},
    )


if __name__ == "__main__":
    main()
