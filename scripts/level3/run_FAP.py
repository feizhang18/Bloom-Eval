import os
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (
    DEFAULT_HUMAN_ARTICLE_FILE,
    DEFAULT_HUMAN_CONTENT_FILE,
    DEFAULT_LLM_CONTENT_FILE,
    add_common_arguments,
    build_result_payload,
    build_log_path,
    call_llm_with_retry,
    ensure_dir,
    format_metric_report,
    load_json,
    load_json_field_text,
    parse_llm_json,
    print_metric_summary,
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

class FrameworkApplicationEvaluator:
    def __init__(self, api_key, base_url, model, save_raw_response=False):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.save_raw_response = save_raw_response
        self.criteria_prompt_tmpl = load_prompt("level3/FAP_criteria_generation.txt")
        self.scoring_prompt_tmpl = load_prompt("level3/FAP_scoring.txt")

    def generate_criteria(self, task_prompt: str, output_path: Path, log_dir: Path) -> list:
        print("Generating evaluation criteria...")
        prompt = self.criteria_prompt_tmpl.format(task_prompt=task_prompt)
        res = call_llm_with_retry(
            self.client,
            self.model,
            prompt,
            build_log_path(log_dir if self.save_raw_response else None, "criteria_gen", suffix="_raw.txt"),
            temperature=0.0,
            max_tokens=128000,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
        )
        criteria = parse_llm_json(res, kind="array")
        write_json(output_path, criteria)
        return criteria

    def perform_scoring(self, criteria: list, task_prompt: str, human_text: str, llm_text: str, output_path: Path, log_dir: Path) -> dict:
        print("Scoring comparative framework application...")
        criteria_json = json.dumps([{"criterion": c["criterion"], "explanation": c["explanation"]} for c in criteria], indent=2, ensure_ascii=False)
        prompt = self.scoring_prompt_tmpl.format(
            task_prompt=task_prompt, article_1_text=llm_text, article_2_text=human_text,
            criteria_json_string=criteria_json
        )
        res = call_llm_with_retry(
            self.client,
            self.model,
            prompt,
            build_log_path(log_dir if self.save_raw_response else None, "scoring", suffix="_raw.txt"),
            temperature=0.0,
            max_tokens=128000,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
        )
        scores = parse_llm_json(res, kind="object")
        write_json(output_path, scores)
        return scores

    def calculate_metrics(self, scores_dict: dict, criteria_list: list) -> dict:
        print("Computing weighted scores...")
        weights = {c['criterion']: c['weight'] for c in criteria_list}
        sum_l, sum_h, total_w = 0.0, 0.0, 0.0
        
        for item in scores_dict.get('framework_application', []):
            w = weights.get(item['criterion'], 0)
            sum_l += float(item['article_1_score']) * w
            sum_h += float(item['article_2_score']) * w
            total_w += w
        
        avg_l = sum_l / total_w if total_w > 0 else 0
        avg_h = sum_h / total_w if total_w > 0 else 0
        fap_score = avg_l / (avg_l + avg_h) if (avg_l + avg_h) > 0 else 0.0
        return {"llm_avg": avg_l, "human_avg": avg_h, "fap_score": fap_score}

def load_text(path):
    if not os.path.exists(path): return None
    data = load_json(path)
    return "\n".join(data) if isinstance(data, list) else str(data)

def get_task_title(path):
    if not os.path.exists(path): return None
    return load_json_field_text(path, "title")

def main():
    parser = argparse.ArgumentParser(description="Framework Application (FAP) end-to-end evaluation tool")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, default=DEFAULT_LLM_CONTENT_FILE, help="Path to LLM content.json")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, default=DEFAULT_HUMAN_CONTENT_FILE, help="Path to human content.json")
    parser.add_argument("--task_file", type=str, default=DEFAULT_HUMAN_ARTICLE_FILE, help="Path to task prompt JSON (containing title)")
    add_common_arguments(parser, metric_name="fap", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        sys.exit(1)
    output_dir = resolve_output_dir(args.output_dir)
    log_dir = ensure_dir(output_dir / "logs") if args.save_raw_response else output_dir / "logs"
    evaluator = FrameworkApplicationEvaluator(API_KEY, args.base_url, args.model, args.save_raw_response)

    llm_content = load_text(args.content_file_llm)
    human_content = load_text(args.content_file_human)
    task_title = get_task_title(args.task_file)

    if not all([llm_content, human_content, task_title]):
        print("Error: Missing or invalid input files.")
        sys.exit(1)

    try:
        c_path = output_dir / "criteria.json"
        criteria = evaluator.generate_criteria(task_title, c_path, log_dir)

        s_path = output_dir / "scoring_raw.json"
        raw_scores = evaluator.perform_scoring(criteria, task_title, human_content, llm_content, s_path, log_dir)

        metrics = evaluator.calculate_metrics(raw_scores, criteria)

        report_path = output_dir / "report.txt"
        res_path = output_dir / "result.json"
        inputs = {
            "content_file_human": to_project_relative(Path(args.content_file_human)),
            "content_file_llm": to_project_relative(Path(args.content_file_llm)),
            "task_file": to_project_relative(Path(args.task_file)),
        }
        config = {
            "model": args.model,
            "base_url": args.base_url,
        }
        report_text = format_metric_report(
            "FAP",
            "Framework Application",
            inputs=inputs,
            results=metrics,
            config=config,
        )
        write_text(report_path, report_text)
        write_json(
            res_path,
            build_result_payload(
                metric="FAP",
                inputs=inputs,
                results=metrics,
                config=config,
                artifacts={
                    "report_file": to_project_relative(report_path),
                    "criteria_file": to_project_relative(c_path),
                    "scoring_file": to_project_relative(s_path),
                },
            ),
        )
        print_metric_summary("FAP", report_path, res_path, results=metrics, summary_keys=("fap_score", "llm_avg", "human_avg"))

    except Exception as e:
        print(f"Error: Pipeline interrupted: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
