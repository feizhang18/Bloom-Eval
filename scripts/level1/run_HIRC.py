import os
import json
import argparse
import sys
from pathlib import Path
from thefuzz import fuzz
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, resolve_output_dir, to_project_relative, write_json, write_text
CITATION_THRESHOLD = 50    # only expert refs with citations > 50 are considered "core"
SIMILARITY_THRESHOLD = 90  # fuzzy match similarity threshold


def load_references_from_file(filepath: str) -> list:
    """Load and parse the JSON structure to extract reference list with citation counts."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Cannot parse JSON file: {filepath}")
        return None

    references = []
    for i in range(1, data.get("reference_num", 0) + 1):
        paper_key = f"paper_{i}_info"
        ref_key = f"reference_{i}"

        paper_info = data.get(paper_key, {})
        ref_info = paper_info.get(ref_key, {})

        title = ref_info.get("searched_title")

        citations = ref_info.get("citation_count", 0)
        try:
            citations = int(citations) if str(citations).isdigit() else 0
        except (ValueError, TypeError):
            citations = 0

        if title:
            references.append({"title": title, "citations": citations})

    return references


def calculate_coverage(human_refs: List[Dict], llm_refs: List[Dict]) -> Dict:
    """Compute core reference coverage metrics and return detailed matched pairs."""
    set_a_titles = [ref["title"] for ref in llm_refs]
    set_b_titles = [ref["title"] for ref in human_refs if ref["citations"] > CITATION_THRESHOLD]

    matched_pairs = []
    unmatched_b_titles = list(set_b_titles)

    for a_title in set_a_titles:
        best_match = None
        highest_similarity = 0
        
        for b_title in unmatched_b_titles:
            similarity = fuzz.ratio(a_title.lower(), b_title.lower())
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = b_title
        
        if highest_similarity >= SIMILARITY_THRESHOLD:
            matched_pairs.append({
                "llm_title": a_title,
                "human_core_title": best_match,
                "similarity": highest_similarity
            })
            unmatched_b_titles.remove(best_match)

    tp = len(matched_pairs)
    fp = len(set_a_titles) - tp
    fn = len(set_b_titles) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_llm_refs": len(set_a_titles),
            "total_human_core_refs": len(set_b_titles)
        },
        "matched_pairs": matched_pairs
    }


def main():
    parser = argparse.ArgumentParser(description="Core Reference Coverage Evaluation Tool")
    parser.add_argument("--reference_file_human", type=str, required=True, help="Path to the human-expert reference.json")
    parser.add_argument("--reference_file_llm", type=str, required=True, help="Path to the LLM reference.json")
    add_common_arguments(parser, metric_name="hirc", include_model=False)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    print("Starting core reference coverage evaluation...")
    print(f"Human reference file: {args.reference_file_human}")
    print(f"LLM reference file:   {args.reference_file_llm}")

    human_refs = load_references_from_file(args.reference_file_human)
    llm_refs = load_references_from_file(args.reference_file_llm)

    if human_refs is None or llm_refs is None:
        print("\nAborted: cannot read input files.")
        return

    results = calculate_coverage(human_refs, llm_refs)
    metrics = results["metrics"]

    report_lines = [
        "=======================================================",
        "     Bloom-Eval: Core Reference Coverage Report        ",
        "=======================================================",
        f"Expert core references (citations > {CITATION_THRESHOLD}): {metrics['total_human_core_refs']}",
        f"LLM total references: {metrics['total_llm_refs']}",
        "-------------------------------------------------------",
        f"TP (Matched core refs): {metrics['tp']}",
        f"FP (Extra/non-core):    {metrics['fp']}",
        f"FN (Missed core refs):  {metrics['fn']}",
        "-------------------------------------------------------",
        "Metrics:",
        f"  -> Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})",
        f"  -> Recall:    {metrics['recall']:.4f} ({metrics['recall']:.2%})",
        f"  -> F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']:.2%})",
        "======================================================="
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = output_dir / "report.txt"
    result_path = output_dir / "result.json"
    write_text(report_path, report_text)
    write_json(
        result_path,
        build_result_payload(
            metric="HIRC",
            inputs={
                "reference_file_human": to_project_relative(Path(args.reference_file_human)),
                "reference_file_llm": to_project_relative(Path(args.reference_file_llm)),
            },
            results=results["metrics"],
            artifacts={
                "matched_pairs": results["matched_pairs"],
                "report_file": to_project_relative(report_path),
            },
            config={
                "citation_threshold": CITATION_THRESHOLD,
                "similarity_threshold": SIMILARITY_THRESHOLD,
            },
        ),
    )

    print(f"\nDone. Results saved to: {output_dir}")
    print("  - report.txt")
    print("  - result.json")

if __name__ == "__main__":
    main()
