import os
import csv
import json
import time
import argparse
import threading
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, call_llm, load_json, resolve_output_dir, to_project_relative, write_json, write_text
from prompt_utils import load_prompt

API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"
MAX_CONCURRENT_THREADS = 10

try:
    NLI_PROMPT_TEMPLATE = load_prompt("level2/CF_citation_nli.txt")
except Exception as e:
    print(f"Error: Cannot load prompt template, check path: {e}")
    sys.exit(1)

class CitationEvaluator:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.lock = threading.Lock()
        self.successful_requests = 0
        self.failed_requests = 0

    def _get_llm_response(self, claim: str, source: str, result_list: list, index: int):
        prompt = NLI_PROMPT_TEMPLATE.replace('[CLAIM]', claim).replace('[SOURCE]', source)
        try:
            answer = call_llm(
                self.client,
                self.model,
                prompt,
                temperature=0.0,
                max_tokens=5,
                verbose=False,
            )
            result_list[index] = 1 if 'yes' in answer.lower() else 0
            with self.lock:
                self.successful_requests += 1
        except Exception:
            result_list[index] = 0
            with self.lock:
                self.failed_requests += 1

    def run_evaluation(self, data_rows: list) -> List[Dict]:
        num_rows = len(data_rows)
        if num_rows == 0: return []
        threads = []
        scores = [0] * num_rows
        
        with tqdm(total=num_rows, desc="LLM NLI scoring", leave=False) as pbar:
            for i, row in enumerate(data_rows):
                claim = row.get('sentence', '')
                source = row.get('abstract', '')
                if not claim or not source:
                    pbar.update(1)
                    continue
                
                while len([t for t in threads if t.is_alive()]) >= MAX_CONCURRENT_THREADS:
                    time.sleep(0.1)

                thread = threading.Thread(target=self._get_llm_response, args=(claim, source, scores, i))
                threads.append(thread)
                thread.start()
                pbar.update(1)

            for thread in threads:
                thread.join()

        detailed_results = []
        for i, row in enumerate(data_rows):
            new_row = row.copy()
            new_row['is_supported'] = scores[i]
            detailed_results.append(new_row)
        return detailed_results


def iter_numbered_reference_entries(ref_data: Dict) -> List[tuple[str, Dict]]:
    """Return (reference number, details) pairs from legacy and flattened schemas."""
    entries = []

    if "reference_num" in ref_data:
        for i in range(1, ref_data.get("reference_num", 0) + 1):
            paper_key, ref_key = f"paper_{i}_info", f"reference_{i}"
            details = ref_data.get(paper_key, {}).get(ref_key)
            if isinstance(details, dict):
                entries.append((str(i), details))
        return entries

    def reference_index(item: tuple[str, Dict]) -> int:
        key, _ = item
        try:
            return int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    for key, details in sorted(ref_data.items(), key=reference_index):
        if key.startswith("reference_") and isinstance(details, dict):
            entries.append((key.rsplit("_", 1)[1], details))

    return entries


def extract_and_group_sentences(paper_path: str, ref_path: str, output_csv: str) -> bool:
    """Stage 1: Split sentences from paper and reference files and unpack citation markers."""
    if os.path.exists(output_csv):
        return True

    try:
        paper_data = load_json(paper_path)
        ref_data = load_json(ref_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return False

    citation_info = {}
    for num, details in iter_numbered_reference_entries(ref_data):
        title = details.get("title") or details.get("searched_title")
        abstract = details.get("abs", "")
        if abstract == "N/A":
            abstract = ""
        if title and title != "N/A":
            citation_info[num] = {"title": title, "abstract": abstract}

    full_text = ""
    if isinstance(paper_data, dict) and 'context' in paper_data:
        full_text = " ".join(paper_data.get('context', {}).values())
    elif isinstance(paper_data, list):
        full_text = str(paper_data[0])

    full_text_cleaned = re.sub(r'-\n', '', full_text)
    full_text_cleaned = re.sub(r'\s+', ' ', full_text_cleaned)

    try:
        sentences = nltk.sent_tokenize(full_text_cleaned)
    except LookupError:
        print("First run: downloading nltk punkt data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        sentences = nltk.sent_tokenize(full_text_cleaned)

    output_data = []
    sentence_id_counter = 0
    marker_pattern = re.compile(r'\[([\d,\s;]+)\]')

    for sentence in sentences:
        clean_sentence = sentence.strip()
        markers = marker_pattern.findall(clean_sentence)
        if not markers: continue

        sentence_id_counter += 1
        all_citation_numbers = set()
        for marker_group in markers:
            for num in re.split(r'[,;]\s*', marker_group):
                if num.strip().isdigit(): all_citation_numbers.add(num.strip())

        for num in sorted(list(all_citation_numbers), key=int):
            if num in citation_info:
                output_data.append({
                    'sentence_id': sentence_id_counter,
                    'sentence': clean_sentence,
                    'references': f"[{num}] {citation_info[num]['title']}",
                    'abstract': citation_info[num]['abstract']
                })

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence_id', 'sentence', 'references', 'abstract'])
        writer.writeheader()
        writer.writerows(output_data)
    return True

def calculate_metrics(csv_path: str) -> Dict:
    """Stage 3: Compute Precision, Recall and F1 from the evaluated CSV."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty: return {"recall": 0.0, "precision": 0.0, "f1_score": 0.0}
        df['sentence_id'] = pd.to_numeric(df['sentence_id'], errors='coerce')
        df['is_supported'] = pd.to_numeric(df['is_supported'], errors='coerce').fillna(0).astype(int)
        df = df.dropna(subset=['sentence_id'])
    except Exception: return {"recall": 0.0, "precision": 0.0, "f1_score": 0.0}

    grouped = df.groupby('sentence_id')
    total_claims = len(grouped)
    supported_claims_count = sum(1 for _, group in grouped if group['is_supported'].any())
    recall = supported_claims_count / total_claims if total_claims > 0 else 0.0

    supported_rows = df[df['is_supported'] == 1].copy()
    precise_references_count = 0
    total_references_count = len(df)
    
    if not supported_rows.empty:
        support_counts_in_group = grouped['is_supported'].transform('sum')
        supported_rows['support_counts_in_group'] = support_counts_in_group[supported_rows.index]
        precise_references_count = (supported_rows['support_counts_in_group'] == 1).sum()

    precision = precise_references_count / total_references_count if total_references_count > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"recall": recall, "precision": precision, "f1_score": f1_score}

def process_single_pipeline(evaluator: CitationEvaluator, paper_path: str, ref_path: str, output_dir: Path, prefix: str) -> Dict:
    """Run the full pipeline for a single input."""
    print(f"\n[{prefix.upper()}] Processing citation pipeline...")

    grouped_csv = output_dir / f"{prefix}_sentences_grouped.csv"
    evaluated_csv = output_dir / f"{prefix}_nli_evaluated_results.csv"

    if not extract_and_group_sentences(paper_path, ref_path, str(grouped_csv)):
        return None
    print(f"  Stage 1 done: sentence extraction -> {grouped_csv.name}")

    if not os.path.exists(evaluated_csv):
        with open(grouped_csv, 'r', encoding='utf-8') as f:
            data_rows = list(csv.DictReader(f))
        print(f"  Stage 2: running LLM NLI scoring ({len(data_rows)} citation pairs)...")
        results = evaluator.run_evaluation(data_rows)
        with open(evaluated_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        print(f"  Stage 2: cached results found, skipping API calls -> {evaluated_csv.name}")

    metrics = calculate_metrics(str(evaluated_csv))
    print(f"  Stage 3 done: metrics computed.")
    return metrics

# ==========================================
# 3. Main entry point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Citation Quality Comparative Evaluation Tool")
    parser.add_argument("--content_file_llm", "--llm_paper", dest="content_file_llm", type=str, required=True, help="Path to the LLM content.json / paper.json")
    parser.add_argument("--reference_file_llm", "--llm_ref", dest="reference_file_llm", type=str, required=True, help="Path to the LLM reference.json")
    parser.add_argument("--content_file_human", "--human_paper", dest="content_file_human", type=str, required=True, help="Path to the human-expert content.json / paper.json")
    parser.add_argument("--reference_file_human", "--human_ref", dest="reference_file_human", type=str, required=True, help="Path to the human-expert reference.json")
    add_common_arguments(parser, metric_name="cf", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        return

    output_dir = resolve_output_dir(args.output_dir)
    evaluator = CitationEvaluator(API_KEY, args.base_url, args.model)

    print("="*50)
    print("Bloom-Eval Level 2: Citation Quality Evaluation")
    print("="*50)

    llm_metrics = process_single_pipeline(evaluator, args.content_file_llm, args.reference_file_llm, output_dir, "llm")
    human_metrics = process_single_pipeline(evaluator, args.content_file_human, args.reference_file_human, output_dir, "human")

    if not llm_metrics or not human_metrics:
        print("\nError: pipeline failed.")
        return

    report_lines = [
        "=======================================================",
        "      Bloom-Eval Level 2: Citation Final Report        ",
        "=======================================================",
        "\n[Human Baseline]",
        f"  - Recall:    {human_metrics['recall']:.4f} ({human_metrics['recall']:.2%})",
        f"  - Precision: {human_metrics['precision']:.4f} ({human_metrics['precision']:.2%})",
        f"  - F1-Score:  {human_metrics['f1_score']:.4f} ({human_metrics['f1_score']:.2%})",
        "\n[LLM Generated]",
        f"  - Recall:    {llm_metrics['recall']:.4f} ({llm_metrics['recall']:.2%})",
        f"  - Precision: {llm_metrics['precision']:.4f} ({llm_metrics['precision']:.2%})",
        f"  - F1-Score:  {llm_metrics['f1_score']:.4f} ({llm_metrics['f1_score']:.2%})",
        "======================================================="
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = output_dir / "report.txt"
    write_text(report_path, report_text)
    write_json(
        output_dir / "result.json",
        build_result_payload(
            metric="CF",
            inputs={
                "content_file_human": to_project_relative(Path(args.content_file_human)),
                "reference_file_human": to_project_relative(Path(args.reference_file_human)),
                "content_file_llm": to_project_relative(Path(args.content_file_llm)),
                "reference_file_llm": to_project_relative(Path(args.reference_file_llm)),
            },
            results={
                "human": human_metrics,
                "llm": llm_metrics,
            },
            config={
                "model": args.model,
                "base_url": args.base_url,
                "max_concurrent_threads": MAX_CONCURRENT_THREADS,
            },
            artifacts={
                "report_file": to_project_relative(report_path),
            },
        ),
    )

    print(f"\nEvaluation complete. All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
