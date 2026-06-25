import os
import argparse
import sys
import re
from pathlib import Path
from openai import OpenAI
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import DEFAULT_HUMAN_OUTLINE_FILE, DEFAULT_LLM_OUTLINE_FILE, add_common_arguments, build_result_payload, call_llm_for_json, format_metric_report, load_json, print_metric_summary, resolve_output_dir, to_project_relative, write_json, write_text
from prompt_utils import load_prompt

API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"

try:
    OUTLINE_MATCHING_PROMPT_TEMPLATE = load_prompt("level2/OTC_topic_matching.txt")
except Exception as e:
    print(f"Error: Cannot load prompt template, check path: {e}")
    sys.exit(1)


def load_and_flatten_outline(file_path: str) -> List[str]:
    """Load outline from JSON and strip leading numeric prefixes (e.g. '1.1 Introduction' -> 'Introduction')."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []
    try:
        outline_data = load_json(file_path)
        topics = []
        for item in outline_data:
            if isinstance(item, list) and len(item) > 1:
                title = item[1]
                clean_title = re.sub(r'^\d+(\.\d+)*\s*', '', title).strip()
                topics.append(clean_title)
        return topics
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return []

def call_llm_for_matching(client: OpenAI, model: str, expert_topics: List[str], llm_topics: List[str], log_path: Path | None) -> Dict:
    """Call the LLM to perform semantic topic matching."""
    expert_str = "\n".join([f"- {t}" for t in expert_topics])
    llm_str = "\n".join([f"- {t}" for t in llm_topics])
    prompt = OUTLINE_MATCHING_PROMPT_TEMPLATE.format(
        expert_topics_str=expert_str,
        llm_topics_str=llm_str
    )

    print("Calling LLM for semantic comparison...")
    return call_llm_for_json(
        model=model,
        client=client,
        prompt=prompt,
        log_file=log_path,
        temperature=0.0,
        response_format={"type": "json_object"},
    )


def main():
    parser = argparse.ArgumentParser(description="Outline Topic Coverage Evaluation Tool")
    parser.add_argument("--outline_file_human", "--human_file", dest="outline_file_human", type=str, default=DEFAULT_HUMAN_OUTLINE_FILE, help="Path to the human-expert outline.json")
    parser.add_argument("--outline_file_llm", "--llm_file", dest="outline_file_llm", type=str, default=DEFAULT_LLM_OUTLINE_FILE, help="Path to the LLM outline.json")
    add_common_arguments(parser, metric_name="otc", default_model=DEFAULT_MODEL)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    print("Running OTC...")
    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        sys.exit(1)
    client = OpenAI(api_key=API_KEY, base_url=args.base_url)

    expert_topics = load_and_flatten_outline(args.outline_file_human)
    llm_topics = load_and_flatten_outline(args.outline_file_llm)

    if not expert_topics or not llm_topics:
        print("Error: Outline extraction returned empty results.")
        sys.exit(1)

    print(f"Loaded {len(expert_topics)} expert topics and {len(llm_topics)} LLM topics.")

    log_path = output_dir / "logs" / "outline_matching_raw.json" if args.save_raw_response else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    match_data = call_llm_for_matching(client, args.model, expert_topics, llm_topics, log_path)
    matched_pairs = match_data.get("matched_pairs", [])

    matched_llm_headings = {pair.get('llm_heading') for pair in matched_pairs if pair.get('llm_heading')}
    matched_expert_headings = {pair.get('expert_heading') for pair in matched_pairs if pair.get('expert_heading')}

    tp_for_precision = len(matched_llm_headings)
    fp = len(llm_topics) - tp_for_precision

    tp_for_recall = len(matched_expert_headings)
    fn = len(expert_topics) - tp_for_recall

    precision = tp_for_precision / len(llm_topics) if len(llm_topics) > 0 else 0.0
    recall = tp_for_recall / len(expert_topics) if len(expert_topics) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    unmatched_llm = [t for t in llm_topics if t not in matched_llm_headings]
    unmatched_expert = [t for t in expert_topics if t not in matched_expert_headings]

    report_path = output_dir / "report.txt"
    result_path = output_dir / "result.json"
    inputs = {
        "outline_file_human": to_project_relative(Path(args.outline_file_human)),
        "outline_file_llm": to_project_relative(Path(args.outline_file_llm)),
    }
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_pairs_count": len(matched_pairs),
    }
    config = {
        "model": args.model,
        "base_url": args.base_url,
    }
    report_text = format_metric_report(
        "OTC",
        "Outline Topic Coverage",
        inputs=inputs,
        results=metrics,
        config=config,
        sections=[
            ("Matched Topic Pairs", matched_pairs),
            ("Unmatched LLM Topics", unmatched_llm),
            ("Unmatched Human Topics", unmatched_expert),
        ],
    )
    write_text(report_path, report_text)
    write_json(
        result_path,
        build_result_payload(
            metric="OTC",
            inputs=inputs,
            results=metrics,
            config=config,
            artifacts={
                "report_file": to_project_relative(report_path),
                "matched_pairs": matched_pairs,
                "unmatched_llm": unmatched_llm,
                "unmatched_human": unmatched_expert,
            },
        ),
    )
    print_metric_summary("OTC", report_path, result_path, results=metrics, summary_keys=("precision", "recall", "f1_score"))

if __name__ == "__main__":
    main()
