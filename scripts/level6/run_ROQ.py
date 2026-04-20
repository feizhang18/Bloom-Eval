import os
import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (
    add_common_arguments,
    build_result_payload,
    build_log_path,
    call_llm_with_retry,
    ensure_dir,
    load_json,
    load_json_field_text,
    parse_llm_json,
    resolve_output_dir,
    to_project_relative,
    write_json,
)
from prompt_utils import load_prompt


API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_DELAY = 5

CRITERIA_GENERATION_PROMPT_TEMPLATE = load_prompt("level6/ROQ_criteria_generation.txt")
SCORING_PROMPT_TEMPLATE = load_prompt("level6/ROQ_scoring.txt")


class ResearchHeuristicsEvaluator:
    def __init__(self, api_key: str, base_url: str, model: str, save_raw_response: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.save_raw_response = save_raw_response

    def generate_criteria(
        self,
        task_prompt: str,
        output_path: Path,
        log_dir: Path,
    ) -> Optional[List[Dict[str, Any]]]:
        print("--- Stage 1: Generating Evaluation Criteria for Research Heuristics ---")
        try:
            prompt = CRITERIA_GENERATION_PROMPT_TEMPLATE.format(task_prompt=task_prompt)
            llm_response = call_llm_with_retry(
                self.client,
                self.model,
                prompt,
                build_log_path(log_dir if self.save_raw_response else None, "criteria_generation"),
                temperature=0.0,
                max_retries=MAX_RETRIES,
                retry_delay=RETRY_DELAY,
                failure_log_message=f"Failed after {MAX_RETRIES} retries.",
            )
            criteria_list = parse_llm_json(llm_response, kind="array")
            write_json(output_path, criteria_list)
            print(f"Criteria saved to '{os.path.basename(output_path)}'.")
            return criteria_list
        except Exception as e:
            print(f"ERROR in Stage 1: {e}")
            return None

    def perform_comparative_scoring(
        self,
        criteria: List[Dict[str, Any]],
        task_prompt: str,
        human_survey: str,
        llm_survey: str,
        output_path: Path,
        log_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        print("\n--- Stage 2: Performing Comparative Scoring ---")
        try:
            criteria_for_prompt = [
                {"criterion": c["criterion"], "explanation": c["explanation"]}
                for c in criteria
            ]
            criteria_json_string = json.dumps(criteria_for_prompt, indent=2, ensure_ascii=False)
            scoring_prompt = SCORING_PROMPT_TEMPLATE.format(
                task_prompt=task_prompt,
                article_1_text=llm_survey,
                article_2_text=human_survey,
                criteria_json_string=criteria_json_string,
            )
            llm_response = call_llm_with_retry(
                self.client,
                self.model,
                scoring_prompt,
                build_log_path(log_dir if self.save_raw_response else None, "comparative_scoring"),
                temperature=0.0,
                max_retries=MAX_RETRIES,
                retry_delay=RETRY_DELAY,
                failure_log_message=f"Failed after {MAX_RETRIES} retries.",
            )
            scores_dict = parse_llm_json(llm_response, kind="object", replace_nbsp=True)
            write_json(output_path, scores_dict)
            print(f"Comparative scores saved to '{os.path.basename(output_path)}'.")
            return scores_dict
        except Exception as e:
            print(f"ERROR in Stage 2: {e}")
            return None

    def calculate_final_score(
        self,
        scores: Dict[str, Any],
        criteria: List[Dict[str, Any]],
    ) -> Optional[Dict[str, float]]:
        print("\n--- Stage 3: Calculating Final Score ---")
        try:
            criteria_map = {item["criterion"]: item["weight"] for item in criteria}
            total_weighted_score_1 = 0.0
            total_weighted_score_2 = 0.0
            total_weight = 0.0

            scored_items = scores.get("research_heuristics", [])
            if not scored_items:
                raise KeyError("'research_heuristics' key not found in scores.")

            for item in scored_items:
                weight = criteria_map.get(item["criterion"])
                if weight is None:
                    continue
                total_weighted_score_1 += float(item["article_1_score"]) * weight
                total_weighted_score_2 += float(item["article_2_score"]) * weight
                total_weight += weight

            avg_score_1 = total_weighted_score_1 / total_weight if total_weight > 0 else 0.0
            avg_score_2 = total_weighted_score_2 / total_weight if total_weight > 0 else 0.0
            final_score = avg_score_1 / (avg_score_1 + avg_score_2) if (avg_score_1 + avg_score_2) > 0 else 0.0

            result = {
                "llm_survey_weighted_avg": avg_score_1,
                "human_survey_weighted_avg": avg_score_2,
                "research_heuristics_score": final_score,
            }
            print("Final score calculation complete.")
            return result
        except Exception as e:
            print(f"ERROR in Stage 3: {e}")
            return None


def read_survey_content(file_path: str) -> Optional[str]:
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        content_list = load_json(file_path)
        return "\n".join(content_list) if isinstance(content_list, list) else None
    except (json.JSONDecodeError, TypeError):
        return None


def find_task_prompt_file(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        if re.match(r"^\d+_.+\.json$", filename):
            return os.path.join(directory, filename)
    return None


def get_task_prompt(file_path: str) -> Optional[str]:
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        title = load_json_field_text(file_path, "title")
        if not title:
            print(f"WARNING: Could not find 'title' field in '{os.path.basename(file_path)}'.")
            return None
        return title
    except (json.JSONDecodeError, AttributeError):
        print(f"WARNING: Could not parse or process task file '{os.path.basename(file_path)}'.")
        return None


def resolve_task_file(human_file: str, explicit_task_file: Optional[str]) -> Optional[str]:
    if explicit_task_file:
        return explicit_task_file
    human_dir = os.path.dirname(human_file)
    return find_task_prompt_file(human_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROQ")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, required=True, help="Path to human content.json")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, required=True, help="Path to LLM content.json")
    parser.add_argument(
        "--task_file",
        type=str,
        default=None,
        help="Path to task definition JSON; if not provided, auto-searches for 'digits_*.json' in the human_file directory",
    )
    add_common_arguments(parser, metric_name="roq", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Please set OPENAI_API_KEY in the environment.")
        return

    output_dir = resolve_output_dir(args.output_dir)

    human_content = read_survey_content(args.content_file_human)
    llm_content = read_survey_content(args.content_file_llm)
    task_file = resolve_task_file(args.content_file_human, args.task_file)
    task_prompt = get_task_prompt(task_file) if task_file else None

    if not all([human_content, llm_content, task_prompt]):
        print("ERROR: Missing or invalid input files.")
        if not human_content:
            print(f"   - Human content issue: {args.content_file_human}")
        if not llm_content:
            print(f"   - LLM content issue: {args.content_file_llm}")
        if not task_prompt:
            if task_file:
                print(f"   - Task file issue: {task_file}")
            else:
                print(f"   - Task file not found under: {os.path.dirname(args.content_file_human)}")
        return

    evaluator = ResearchHeuristicsEvaluator(api_key=API_KEY, base_url=args.base_url, model=args.model, save_raw_response=args.save_raw_response)
    log_dir = ensure_dir(output_dir / "logs") if args.save_raw_response else output_dir / "logs"
    criteria_path = output_dir / "research_heuristics_criteria.json"
    scoring_path = output_dir / "comparative_scores_heuristics.json"
    final_result_path = output_dir / "result.json"
    intermediate_path = output_dir / "intermediate.json"
    report_path = output_dir / "report.txt"

    criteria = evaluator.generate_criteria(task_prompt, criteria_path, log_dir)
    if not criteria:
        return

    raw_scores = evaluator.perform_comparative_scoring(
        criteria,
        task_prompt,
        human_content,
        llm_content,
        scoring_path,
        log_dir,
    )
    if not raw_scores:
        return

    final_result = evaluator.calculate_final_score(raw_scores, criteria)
    if not final_result:
        return

    write_json(
        final_result_path,
        build_result_payload(
            metric="ROQ",
            inputs={
                "content_file_human": to_project_relative(Path(args.content_file_human)),
                "content_file_llm": to_project_relative(Path(args.content_file_llm)),
                "task_file": to_project_relative(Path(task_file)) if task_file else None,
            },
            results=final_result,
            config={
                "model": args.model,
                "base_url": args.base_url,
            },
            artifacts={
                "criteria_file": to_project_relative(criteria_path),
                "scoring_file": to_project_relative(scoring_path),
                "intermediate_file": to_project_relative(intermediate_path),
                "report_file": to_project_relative(report_path),
            },
        ),
    )

    write_json(
        intermediate_path,
        {
            "task_prompt": task_prompt,
            "criteria": criteria,
            "raw_scores": raw_scores,
        },
    )

    report_lines = [
        "========================================",
        " Bloom-Eval Level 6: ROQ Report",
        "========================================",
        f"Task prompt: {task_prompt}",
        f"LLM weighted average: {final_result['llm_survey_weighted_avg']:.4f}",
        f"Human weighted average: {final_result['human_survey_weighted_avg']:.4f}",
        f"Research heuristics score: {final_result['research_heuristics_score']:.4f}",
        "========================================",
    ]
    report_text = "\n".join(report_lines)
    write_text(report_path, report_text)

    print("\n" + report_text)
    print(f"Intermediate results saved to: {intermediate_path}")
    print(f"Final results saved to: {final_result_path}")


if __name__ == "__main__":
    main()
