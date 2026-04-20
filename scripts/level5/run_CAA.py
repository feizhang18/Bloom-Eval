import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (
    add_common_arguments,
    build_result_payload,
    ensure_dir,
    resolve_output_dir,
    to_project_relative,
    write_json,
    write_text,
)
from prompt_utils import load_prompt


API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_DELAY = 5

CRITICAL_EXTRACTION_PROMPT_TEMPLATE = load_prompt("level5/CAA_critical_claim_extraction.txt")
CRITICAL_MATCHING_PROMPT_TEMPLATE = load_prompt("level5/CAA_critical_claim_matching.txt")


def load_text_from_json(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], str):
            return " ".join(data)
        return ""
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    purpose: str,
    log_dir: Optional[Path],
) -> str:
    log_path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{purpose}_raw_response.txt" if log_dir else None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8192,
            )
            raw_response = response.choices[0].message.content.strip()
            if log_path is not None:
                write_text(log_path, raw_response)
            return raw_response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                if log_path is not None:
                    write_text(log_path, f"Failed after {MAX_RETRIES} retries.\nLast error: {e}")
                raise


def extract_critical_statements(
    client: OpenAI,
    model: str,
    text: str,
    query_id: str,
    output_path: Path,
    log_dir: Optional[Path],
) -> List[str]:
    prompt = CRITICAL_EXTRACTION_PROMPT_TEMPLATE.format(text=text)
    raw_response = call_llm(client, model, prompt, query_id, log_dir)

    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:-3].strip()

    try:
        result = json.loads(cleaned)
        write_json(output_path, result)
        return result.get("critical_statements", [])
    except json.JSONDecodeError:
        print("Could not parse critical statement extraction result.")
        write_json(output_path, {"critical_statements": []})
        return []


def find_semantic_matches(
    client: OpenAI,
    model: str,
    human_statements: List[str],
    llm_statements: List[str],
    query_id: str,
    output_path: Path,
    log_dir: Optional[Path],
) -> Dict[str, Any]:
    human_str = "\n".join([f"H{i + 1}. {s}" for i, s in enumerate(human_statements)])
    llm_str = "\n".join([f"L{i + 1}. {s}" for i, s in enumerate(llm_statements)])
    prompt = CRITICAL_MATCHING_PROMPT_TEMPLATE.format(
        expert_statements_str=human_str,
        llm_statements_str=llm_str,
    )

    llm_response = call_llm(client, model, prompt, query_id, log_dir)
    try:
        json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if not json_match:
            raise ValueError("Could not find a JSON object in the LLM response.")
        result = json.loads(json_match.group(0))
        if "matched_critical_pairs" not in result or not isinstance(result["matched_critical_pairs"], list):
            result = {"matched_critical_pairs": []}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Could not parse model response as JSON. {e}")
        result = {"matched_critical_pairs": []}

    write_json(output_path, result)
    return result


def calculate_metrics(human_count: int, llm_count: int, matched_count: int) -> Dict[str, float]:
    tp = matched_count
    fp = llm_count - tp
    fn = human_count - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "human_critical_statements": human_count,
        "llm_critical_statements": llm_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Critical Argument Alignment (CAA) evaluation tool")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, required=True, help="Path to human content.json")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, required=True, help="Path to LLM content.json")
    add_common_arguments(parser, metric_name="caa", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        return

    output_dir = resolve_output_dir(args.output_dir)
    raw_log_dir = ensure_dir(output_dir / "logs") if args.save_raw_response else None
    client = OpenAI(api_key=API_KEY, base_url=args.base_url)

    human_text = load_text_from_json(args.content_file_human)
    llm_text = load_text_from_json(args.content_file_llm)
    if not human_text or not llm_text:
        print("Error: Missing or invalid input files.")
        return

    human_extract_path = output_dir / "human_critical_claims.json"
    llm_extract_path = output_dir / "llm_critical_claims.json"
    matching_path = output_dir / "critical_matching.json"

    human_statements = extract_critical_statements(
        client,
        args.model,
        human_text,
        "human_critical_extract",
        human_extract_path,
        raw_log_dir,
    )
    llm_statements = extract_critical_statements(
        client,
        args.model,
        llm_text,
        "llm_critical_extract",
        llm_extract_path,
        raw_log_dir,
    )

    matched_data = find_semantic_matches(
        client,
        args.model,
        human_statements,
        llm_statements,
        "critical_matching",
        matching_path,
        raw_log_dir,
    )

    num_matched = len(matched_data.get("matched_critical_pairs", []))
    metrics = calculate_metrics(len(human_statements), len(llm_statements), num_matched)

    intermediate = {
        "human_critical_statements": human_statements,
        "llm_critical_statements": llm_statements,
        "matched_critical_pairs": matched_data.get("matched_critical_pairs", []),
    }
    intermediate_path = output_dir / "intermediate.json"
    write_json(intermediate_path, intermediate)

    report_lines = [
        "========================================",
        "   Bloom-Eval Level 5: CAA Report",
        "========================================",
        f"Human critical statements: {len(human_statements)}",
        f"LLM critical statements: {len(llm_statements)}",
        f"Matched critical pairs: {num_matched}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1-Score: {metrics['f1_score']:.4f}",
        f"(TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']})",
        "========================================",
    ]
    report_text = "\n".join(report_lines)
    report_path = output_dir / "report.txt"
    write_text(report_path, report_text)

    write_json(
        output_dir / "result.json",
        build_result_payload(
            metric="CAA",
            inputs={
                "content_file_human": to_project_relative(Path(args.content_file_human)),
                "content_file_llm": to_project_relative(Path(args.content_file_llm)),
            },
            results=metrics,
            config={
                "model": args.model,
                "base_url": args.base_url,
            },
            artifacts={
                "report_file": to_project_relative(report_path),
                "intermediate_file": to_project_relative(intermediate_path),
                "human_claims_file": to_project_relative(human_extract_path),
                "llm_claims_file": to_project_relative(llm_extract_path),
                "matching_file": to_project_relative(matching_path),
            },
        ),
    )

    print("\n" + report_text)
    print(f"Intermediate results saved to: {intermediate_path}")
    print(f"Final results saved to: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
