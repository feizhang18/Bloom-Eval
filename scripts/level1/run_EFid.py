import os
import json
import time
import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
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
    write_text,
)
from prompt_utils import load_prompt

# ==========================================
# 0. Global configuration and prompt template loading
# ==========================================
API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"

PROMPT_EXTRACT = load_prompt("level1/EFid_entity_extraction.txt")
PROMPT_NORMALIZE = load_prompt("level1/EFid_entity_normalization.txt")
PROMPT_MATCH = load_prompt("level1/EFid_entity_resolution.txt")


# ==========================================
# 2. Pipeline stages
# ==========================================
def step1_extract(client, model: str, text_content: str, output_path: Path, log_path: Optional[Path]) -> Dict:
    if os.path.exists(output_path):
        return load_json(output_path)
    prompt = PROMPT_EXTRACT.replace("{text}", text_content)
    result = call_llm_for_json(client, model, prompt, log_path)
    for k in ["methods_models", "datasets", "evaluation_metrics"]:
        result.setdefault(k, [])
    save_json(result, output_path)
    return result

def step2_normalize(client, model: str, raw_entities: Dict, output_path: Path, log_path: Optional[Path]) -> Dict:
    if os.path.exists(output_path):
        return load_json(output_path)
    prompt = PROMPT_NORMALIZE.replace("{json_data}", json.dumps(raw_entities, ensure_ascii=False))
    result = call_llm_for_json(client, model, prompt, log_path)
    for k in ["methods_models_map", "datasets_map", "evaluation_metrics_map"]:
        result.setdefault(k, {})
    save_json(result, output_path)
    return result

def step3_count(raw_data: Dict, mapping_data: Dict, output_path: Path) -> Dict:
    if os.path.exists(output_path):
        return load_json(output_path)
    final_counts = {"methods_models": {}, "datasets": {}, "evaluation_metrics": {}}
    for category in final_counts.keys():
        raw_entities = raw_data.get(category, [])
        entity_map = mapping_data.get(f"{category}_map", {})
        category_results = {}
        for alias in raw_entities:
            if alias in entity_map:
                canonical = entity_map[alias]
                if canonical not in category_results:
                    category_results[canonical] = {"total_count": 0, "aliases": defaultdict(int)}
                category_results[canonical]["total_count"] += 1
                category_results[canonical]["aliases"][alias] += 1
        
        for can, data in category_results.items():
            data["aliases"] = dict(sorted(data["aliases"].items(), key=lambda i: i[1], reverse=True))
        final_counts[category] = dict(sorted(category_results.items(), key=lambda i: i[1]['total_count'], reverse=True))
    save_json(final_counts, output_path)
    return final_counts

def step4_match(client, model: str, human_counts: Dict, llm_counts: Dict, output_path: Path, log_path: Optional[Path]) -> Dict:
    if os.path.exists(output_path):
        return load_json(output_path)
    expert_simple = {k: list(v.keys()) for k, v in human_counts.items()}
    llm_simple = {k: list(v.keys()) for k, v in llm_counts.items()}
    
    prompt = PROMPT_MATCH.replace("{expert_data_json}", json.dumps(expert_simple, ensure_ascii=False))
    prompt = prompt.replace("{llm_data_json}", json.dumps(llm_simple, ensure_ascii=False))
    
    result = call_llm_for_json(client, model, prompt, log_path)
    save_json(result, output_path)
    return result

# ==========================================
# 3. Final metric calculation
# ==========================================
def calculate_metrics(human_counts: Dict, llm_counts: Dict, matched_data: Dict, report_path: Path) -> Dict[str, Any]:
    all_human, all_llm, all_matched = {}, {}, []
    for cat in ['methods_models', 'datasets', 'evaluation_metrics']:
        all_human.update(human_counts.get(cat, {}))
        all_llm.update(llm_counts.get(cat, {}))
        all_matched.extend(matched_data.get(cat, []))

    # Recognition performance
    tp = len(all_matched)
    fp = len(all_llm) - tp
    fn = len(all_human) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Distribution similarity
    sim_scores = {"jensen_shannon_sim": 0.0, "hellinger_sim": 0.0, "total_variation_sim": 0.0}
    if tp > 0:
        h_arr = np.array([all_human[p['expert_main_name']]['total_count'] for p in all_matched])
        l_arr = np.array([all_llm[p['llm_main_name']]['total_count'] for p in all_matched])
        if h_arr.sum() > 0 and l_arr.sum() > 0:
            p_dist, q_dist = h_arr / h_arr.sum(), l_arr / l_arr.sum()
            m = 0.5 * (p_dist + q_dist)
            eps = 1e-10
            jsd = 0.5 * (entropy(p_dist + eps, m + eps) + entropy(q_dist + eps, m + eps))
            hd = np.sqrt(np.sum((np.sqrt(p_dist) - np.sqrt(q_dist)) ** 2)) / np.sqrt(2)
            tvd = 0.5 * np.sum(np.abs(p_dist - q_dist))
            
            sim_scores["jensen_shannon_sim"] = 1 - (jsd / np.log(2))
            sim_scores["hellinger_sim"] = 1 - hd
            sim_scores["total_variation_sim"] = 1 - tvd

    # Generate report
    report = [
        "========================================",
        "     Bloom-Eval Level 1 Evaluation      ",
        "========================================",
        "\n[1] Entity Recognition Performance",
        f"TP (Matched): {tp} | FP (LLM Extra): {fp} | FN (Missed): {fn}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1-Score:  {f1_score:.4f}",
        "\n[2] Distribution Similarity",
        f"Jensen-Shannon Sim:  {sim_scores['jensen_shannon_sim']:.4f}",
        f"Hellinger Sim:       {sim_scores['hellinger_sim']:.4f}",
        f"Total Variation Sim: {sim_scores['total_variation_sim']:.4f}",
        "========================================"
    ]
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    write_text(report_path, report_text)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        **sim_scores,
    }


# ==========================================
# 4. Main execution pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation Pipeline")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, required=True, help="Path to the LLM-generated survey content.json")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, required=True, help="Path to the human-expert survey content.json")
    add_common_arguments(parser, metric_name="efid", default_model=DEFAULT_MODEL)
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

    print(f"Starting automated evaluation pipeline...")
    print(f"Output directory: {output_dir}")

    human_data = load_json(args.content_file_human)
    llm_data = load_json(args.content_file_llm)
    human_text = human_data[0] if isinstance(human_data, list) else human_data
    llm_text = llm_data[0] if isinstance(llm_data, list) else llm_data

    # Stage 1: Entity Extraction
    print("\n>>> Stage 1: Entity Extraction")
    h_raw = step1_extract(client, args.model, human_text, dir_human / "all_entity.json", (dir_logs / f"ext_human_{ts}.txt") if dir_logs else None)
    l_raw = step1_extract(client, args.model, llm_text, dir_llm / "all_entity.json", (dir_logs / f"ext_llm_{ts}.txt") if dir_logs else None)

    # Stage 2: Entity Normalization
    print("\n>>> Stage 2: Entity Normalization")
    h_map = step2_normalize(client, args.model, h_raw, dir_human / "entity_normalization.json", (dir_logs / f"norm_human_{ts}.txt") if dir_logs else None)
    l_map = step2_normalize(client, args.model, l_raw, dir_llm / "entity_normalization.json", (dir_logs / f"norm_llm_{ts}.txt") if dir_logs else None)

    # Stage 3: Frequency Counting
    print("\n>>> Stage 3: Frequency Counting")
    h_counts = step3_count(h_raw, h_map, dir_human / "final_counts.json")
    l_counts = step3_count(l_raw, l_map, dir_llm / "final_counts.json")

    # Stage 4: Cross-domain Entity Matching
    print("\n>>> Stage 4: Cross-domain Entity Matching")
    matched = step4_match(client, args.model, h_counts, l_counts, output_dir / "entity_matching.json", (dir_logs / f"match_{ts}.txt") if dir_logs else None)

    # Stage 5: Compute Evaluation Metrics
    print("\n>>> Stage 5: Computing Evaluation Metrics")
    report_path = output_dir / "report.txt"
    metrics = calculate_metrics(h_counts, l_counts, matched, report_path)
    write_json(
        output_dir / "result.json",
        build_result_payload(
            metric="EFid",
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

    print(f"\nPipeline complete. All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
