import os
import time
import argparse
import sys
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Any, Optional

# sys.path setup to import prompt_utils
# Assumes this script is located under Bloom-Eval/scripts/level1/
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (
    add_common_arguments,
    build_result_payload,
    call_llm_for_json,
    ensure_dir,
    load_json,
    resolve_output_dir,
    save_json,
    to_project_relative,
    write_json,
)
from prompt_utils import load_prompt

API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"

PROMPT_EXTRACT = load_prompt("level1/FCons_claim_extraction.txt")
PROMPT_MATCH = load_prompt("level1/FCons_claim_matching.txt")

def step1_extract_claims(client: OpenAI, model: str, text_content: str, output_path: Path, log_path: Optional[Path]) -> List[str]:
    """Stage 1: Extract factual statements from text."""
    if os.path.exists(output_path):
        data = load_json(output_path)
        return data.get("factual_statements", data.get("actual_claims", []))

    prompt = PROMPT_EXTRACT.replace("{text}", text_content)
    result = call_llm_for_json(client, model, prompt, log_path)

    statements = result.get("factual_statements", result.get("actual_claims", []))
    save_json({"factual_statements": statements}, output_path)
    return statements

def step2_match_claims(client: OpenAI, model: str, expert_list: List[str], llm_list: List[str], output_path: Path, log_path: Optional[Path]) -> List[Dict]:
    """Stage 2: Use LLM to determine semantic equivalence between two sets of factual statements."""
    if os.path.exists(output_path):
        return load_json(output_path).get("matched_pairs", [])

    expert_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(expert_list)])
    llm_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(llm_list)])

    prompt = PROMPT_MATCH.replace("{expert_statements_str}", expert_str)
    prompt = prompt.replace("{llm_statements_str}", llm_str)

    result = call_llm_for_json(client, model, prompt, log_path)
    matched_pairs = result.get("matched_pairs", [])

    save_json({"matched_pairs": matched_pairs}, output_path)
    return matched_pairs

def step3_calculate_and_report(expert_list: List[str], llm_list: List[str], matched_pairs: List[Dict], report_path: Path) -> Dict[str, float]:
    """Stage 3: Compute precision, recall, F1 and generate the final report."""
    total_expert = len(expert_list)
    total_llm = len(llm_list)
    tp = len(matched_pairs)
    fp = total_llm - tp
    fn = total_expert - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    report = [
        "=======================================================",
        "     Bloom-Eval Level 1: Factual Claims Evaluation     ",
        "=======================================================",
        f"Expert factual statements: {total_expert}",
        f"LLM factual statements:    {total_llm}",
        "-------------------------------------------------------",
        f"TP (Matched):  {tp}",
        f"FP (Extra):    {fp}",
        f"FN (Missed):   {fn}",
        "-------------------------------------------------------",
        "Final Metrics:",
        f"  -> Precision: {precision:.4f} ({precision:.2%})",
        f"  -> Recall:    {recall:.4f} ({recall:.2%})",
        f"  -> F1-Score:  {f1_score:.4f} ({f1_score:.2%})",
        "======================================================="
    ]

    report_text = "\n".join(report)
    print("\n" + report_text)

    write_text(report_path, report_text)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_expert": total_expert,
        "total_llm": total_llm,
    }

def main():
    parser = argparse.ArgumentParser(description="Factual Claims End-to-End Evaluation Pipeline")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, required=True, help="Path to the LLM-generated content.json")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, required=True, help="Path to the human-expert content.json")
    add_common_arguments(parser, metric_name="fcons", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        return

    output_dir = resolve_output_dir(args.output_dir)
    client = OpenAI(api_key=API_KEY, base_url=args.base_url)
    ts = time.strftime("%Y%m%d_%H%M%S")

    dir_human = ensure_dir(output_dir / "human")
    dir_llm = ensure_dir(output_dir / "llm")
    dir_logs = ensure_dir(output_dir / "logs") if args.save_raw_response else None

    print(f"Starting factual claims evaluation pipeline...")
    print(f"Output directory: {output_dir}")

    human_data = load_json(args.content_file_human)
    llm_data = load_json(args.content_file_llm)
    human_text = human_data[0] if isinstance(human_data, list) else human_data
    llm_text = llm_data[0] if isinstance(llm_data, list) else llm_data

    # Stage 1: Extract factual claims
    print("\n>>> Stage 1: Factual Claim Extraction")
    human_claims = step1_extract_claims(client, args.model, human_text, dir_human / "factual_claims.json", (dir_logs / f"ext_human_{ts}.txt") if dir_logs else None)
    llm_claims = step1_extract_claims(client, args.model, llm_text, dir_llm / "factual_claims.json", (dir_logs / f"ext_llm_{ts}.txt") if dir_logs else None)
    print(f"  - Human expert claims: {len(human_claims)}")
    print(f"  - LLM claims:          {len(llm_claims)}")

    if not human_claims or not llm_claims:
        print("\nError: One side produced no factual statements; cannot proceed with matching.")
        return

    # Stage 2: Semantic alignment
    print("\n>>> Stage 2: Semantic Alignment")
    matched_pairs = step2_match_claims(client, args.model, human_claims, llm_claims, output_dir / "matched_pairs.json", (dir_logs / f"match_{ts}.txt") if dir_logs else None)

    # Stage 3: Compute metrics and generate report
    print("\n>>> Stage 3: Metric Calculation")
    report_path = output_dir / "report.txt"
    metrics = step3_calculate_and_report(human_claims, llm_claims, matched_pairs, report_path)
    write_json(
        output_dir / "result.json",
        build_result_payload(
            metric="FCons",
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
                "human_dir": to_project_relative(dir_human),
                "llm_dir": to_project_relative(dir_llm),
            },
        ),
    )

    print(f"\nPipeline complete.")

if __name__ == "__main__":
    main()
