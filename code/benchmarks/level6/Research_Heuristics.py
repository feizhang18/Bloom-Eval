import os
import json
import time
import re
import glob
import numpy as np
from openai import OpenAI
from typing import Dict, List, Optional

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 替换为您的 agicto API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")            # 您的 agicto API 端点
LLM_MODEL = "qwen3.5-397b-a17b"
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# --- 批量处理与报告配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = [1, 2, 3, 11, 12, 13]  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'surveyforge'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'


CRITERIA_GENERATION_PROMPT_TEMPLATE = """
<system_role>
You are an experienced research mentor and academic editor, specializing in identifying high-impact future research directions. You excel at deconstructing the abstract concept of "good research questions" into concrete, weighted criteria tailored to a specific research field.
</system_role>
<user_prompt>
**Background**: We are evaluating the quality of the "future research directions" section of a survey paper. The survey was written to address the following research task:
<task>{task_prompt}</task>

**Core Evaluation Dimension: Research Outlook Quality**
This metric evaluates the quality of the proposed directions for future research. The assessment focuses on the novelty, insightfulness, and feasibility of the new questions or hypotheses, reflecting the ability to identify promising avenues for future inquiry.

<Instruction>
**Your Goal**: For the Research Outlook Quality dimension, develop a detailed, specific, and logically sound set of evaluation criteria tailored to the research `<task>`. You must:
1.  **Analyze the Task & Dimension**: Based on the `<task>`, identify what constitutes insightful and high-impact future research questions for this specific topic.
2.  **Formulate Task-Specific Criteria**: Propose criteria to evaluate the quality of future research directions.
3.  **Provide Rationale**: For each criterion, provide a concise explanation (`explanation`).
4.  **Assign Weights**: Assign a weight (`weight`) to each criterion, ensuring the sum of all weights is exactly **1.0**.

**Core Requirements**:
1.  **Task-Centric**: Your criteria must be directly linked to the nuances of the `<task>`.
2.  **Sufficient Justification**: The `<analysis>` must explain your reasoning for the criteria and weights.
3.  **Standard Output Format**: Strictly follow the example format.
</Instruction>

<example>
<task>"A comprehensive survey on Vision Transformers."</task>
<output>
<analysis>
For a Vision Transformer survey, strong future research directions must move beyond generic suggestions. Key criteria should assess whether the suggestions are grounded in the identified limitations (e.g., data-hungriness, computational cost) and whether they propose specific, feasible research avenues. Therefore, "Specificity and Feasibility" and "Grounding in Literature Gaps" are weighted most heavily.
</analysis>
<json_output>
[
 {{
    "criterion": "Specificity and Feasibility",
    "explanation": "Assesses if the proposed research questions are concrete and actionable, rather than vague, high-level suggestions. A good direction is one that a researcher could immediately start designing an experiment for.",
    "weight": 0.3
 }},
 {{
    "criterion": "Grounding in Literature Gaps",
    "explanation": "Evaluates whether the proposed directions logically follow from the limitations, challenges, and open problems identified in the main body of the survey.",
    "weight": 0.3
 }},
 {{
    "criterion": "Potential Impact and Insightfulness",
    "explanation": "Measures the potential of the proposed research to significantly advance the field, open new sub-fields, or resolve major theoretical or practical bottlenecks.",
    "weight": 0.25
 }},
 {{
    "criterion": "Novelty of Questions",
    "explanation": "Assesses whether the research directions are fresh and forward-looking, rather than restating well-known, incremental next steps.",
    "weight": 0.15
 }}
]
</json_output>
</output>
</example>

Now, please perform this task for the following research prompt:
<task>{task_prompt}</task>
</user_prompt>
"""

SCORING_PROMPT_TEMPLATE = """
<system_role>
You are a rigorous, meticulous, and objective academic reviewer. You are skilled at deeply comparing the "future research directions" sections of two research reviews based on specific evaluation criteria, providing precise scores with clear justifications.
</system_role>
<user_prompt>
**Task Background**
You need to evaluate the quality of the proposed future research directions in two survey papers. The surveys were written to address the following research task:
<task>{task_prompt}</task>

**Articles to be Evaluated**
<article_1>{article_1_text}</article_1>
<article_2>{article_2_text}</article_2>

**Evaluation Criteria: Research Outlook Quality**
Now, you must evaluate and compare the future research directions proposed in these two articles on a criterion-by-criterion basis according to the following **list of criteria**.
<criteria_list>{criteria_json_string}</criteria_list>

<Instruction>
**Your Task**
Strictly following **each criterion** in the `<criteria_list>`, compare and evaluate the performance of `<article_1>` and `<article_2>` on that criterion. Your analysis should focus ONLY on the sections discussing conclusions, future work, challenges, and future prospects.

**Scoring Rubric**
For each criterion, score both articles on a continuous scale from 0 to 10.

**Output Format Requirement**
Please **strictly** follow the `<output_format>` below. Ensure the final output is a single, valid JSON object that can be parsed directly.
</Instruction>

<output_format>
{{
    "research_heuristics": [
        {{
            "criterion": "[Text of the first evaluation criterion]",
            "analysis": "[Comparative analysis]",
            "article_1_score": [0-10 continuous score],
            "article_2_score": [0-10 continuous score]
        }},
        ...
    ]
}}
</output_format>

Now, based on the evaluation criteria, please assess the two articles and provide a detailed comparative analysis and scores as required.
</user_prompt>
"""
# --- 3. CORE IMPLEMENTATION ---

class ResearchHeuristicsEvaluator:
    """
    A class to orchestrate the three-stage evaluation of the "Research Heuristics" metric.
    """
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _call_llm(self, prompt: str, purpose: str, log_dir: str) -> str:
        """Helper function to call the LLM API, save the raw response, and handle retries."""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filepath = os.path.join(log_dir, f"{timestamp}_{purpose}_raw_response.txt")

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                raw_response = response.choices[0].message.content.strip()
                with open(log_filepath, 'w', encoding='utf-8') as f:
                    f.write(raw_response)
                print(f"✅ Raw LLM response for '{purpose}' saved.")
                return raw_response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached. Failing.")
                    with open(log_filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Failed after {MAX_RETRIES} retries.\nLast error: {e}")
                    raise

    def generate_criteria(self, task_prompt: str, output_path: str, log_dir: str) -> Optional[list]:
        """Stage 1: Generate evaluation criteria."""
        print("--- Stage 1: Generating Evaluation Criteria for Research Heuristics ---")
        try:
            prompt = CRITERIA_GENERATION_PROMPT_TEMPLATE.format(task_prompt=task_prompt)
            llm_response = self._call_llm(prompt, "criteria_generation", log_dir)
            json_match = re.search(r'\[\s*\{.*\}\s*\]', llm_response, re.DOTALL)
            if not json_match: raise ValueError("Could not find a valid JSON list in the response.")
            criteria_list = json.loads(json_match.group(0))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(criteria_list, f, indent=4, ensure_ascii=False)
            print(f"✅ Criteria saved to '{os.path.basename(output_path)}'.")
            return criteria_list
        except Exception as e:
            print(f"❌ ERROR in Stage 1: {e}")
            return None

    def perform_comparative_scoring(self, criteria: list, task_prompt: str, human_survey: str, llm_survey: str, output_path: str, log_dir: str) -> Optional[dict]:
            """Stage 2: Perform comparative scoring."""
            print("\n--- Stage 2: Performing Comparative Scoring ---")
            try:
                criteria_for_prompt = [{"criterion": c["criterion"], "explanation": c["explanation"]} for c in criteria]
                criteria_json_string = json.dumps(criteria_for_prompt, indent=2, ensure_ascii=False)
                scoring_prompt = SCORING_PROMPT_TEMPLATE.format(
                    task_prompt=task_prompt, article_1_text=llm_survey, article_2_text=human_survey,
                    criteria_json_string=criteria_json_string
                )
                llm_response = self._call_llm(scoring_prompt, "comparative_scoring", log_dir)
                
                # --- 修复开始 ---
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if not json_match: 
                    raise ValueError("Could not find a JSON object in the scoring response.")
                
                json_str = json_match.group(0)
                # 关键修复：替换不间断空格
                json_str = json_str.replace('\xa0', ' ').strip()
                
                scores_dict = json.loads(json_str)
                # --- 修复结束 ---

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(scores_dict, f, indent=4, ensure_ascii=False)
                print(f"✅ Comparative scores saved to '{os.path.basename(output_path)}'.")
                return scores_dict
            except Exception as e:
                print(f"❌ ERROR in Stage 2: {e}")
                if 'json_str' in locals():
                    print(f"DEBUG: Failed JSON string snippet: {json_str[:100]}...")
                return None
            
    def calculate_final_score(self, scores: dict, criteria: list) -> Optional[dict]:
        """Stage 3: Calculate the final weighted and normalized score."""
        print("\n--- Stage 3: Calculating Final Score ---")
        try:
            criteria_map = {item['criterion']: item['weight'] for item in criteria}
            total_weighted_score_1, total_weighted_score_2, total_weight = 0.0, 0.0, 0.0
            scored_items = scores.get('research_heuristics', [])
            if not scored_items: raise KeyError("'research_heuristics' key not found in scores.")
            for item in scored_items:
                weight = criteria_map.get(item['criterion'])
                if weight is not None:
                    total_weighted_score_1 += float(item['article_1_score']) * weight
                    total_weighted_score_2 += float(item['article_2_score']) * weight
                    total_weight += weight
            
            avg_score_1 = total_weighted_score_1 / total_weight if total_weight > 0 else 0
            avg_score_2 = total_weighted_score_2 / total_weight if total_weight > 0 else 0
            final_score = avg_score_1 / (avg_score_1 + avg_score_2) if (avg_score_1 + avg_score_2) > 0 else 0.0
            
            print("✅ Final score calculation complete.")
            return {
                "llm_survey_weighted_avg": avg_score_1,
                "human_survey_weighted_avg": avg_score_2,
                "research_heuristics_score": final_score
            }
        except Exception as e:
            print(f"❌ ERROR in Stage 3: {e}")
            return None

# --- 4. 辅助函数 ---

def read_survey_content(file_path: str) -> Optional[str]:
    if not file_path or not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content_list = json.load(f)
        return "\n".join(content_list) if isinstance(content_list, list) else None
    except (json.JSONDecodeError, TypeError): return None

def find_task_prompt_file(directory: str) -> Optional[str]:
    """In a directory, find the JSON file that starts with a number and an underscore."""
    if not os.path.isdir(directory): return None
    for filename in os.listdir(directory):
        if re.match(r'^\d+_.+\.json$', filename):
            return os.path.join(directory, filename)
    return None

def get_task_prompt(file_path: str) -> Optional[str]:
    """(MODIFIED) Extracts only the title from the task file."""
    if not file_path or not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        title = data.get("title", "")
        if title:
            return title
        else:
            print(f"  - WARNING: Could not find 'title' field in '{os.path.basename(file_path)}'.")
            return None
    except (json.JSONDecodeError, AttributeError):
        print(f"  - WARNING: Could not parse or process task file '{os.path.basename(file_path)}'.")
        return None

# --- 5. 主执行流程 ---

def main():
    """Main function to iterate through experiments, run evaluations, and generate a report."""
    print("🚀 Starting Research Heuristics Batch Evaluation Pipeline...")
    all_scores = []
    full_report_lines = ["="*25 + " Research Heuristics Benchmark Report " + "="*25]
    
    evaluator = ResearchHeuristicsEvaluator(api_key=API_KEY, base_url=BASE_URL, model=LLM_MODEL)

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} Processing Experiment ID: {exp_id} {'='*30}")
        
        # 1. Define paths for the current experiment
        human_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human')
        llm_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME)
        output_dir = os.path.join(llm_dir, 'bench', 'level6', 'research_heuristics_evaluation')
        
        human_survey_path = os.path.join(human_dir, 'content.json')
        llm_survey_path = os.path.join(llm_dir, 'content.json')
        task_prompt_path = find_task_prompt_file(human_dir)
        
        criteria_path = os.path.join(output_dir, 'research_heuristics_criteria_qwen3.5-397b-a17b.json')
        scoring_path = os.path.join(output_dir, 'comparative_scores_heuristics_qwen3.5-397b-a17b.json')
        final_score_path = os.path.join(output_dir, 'final_heuristics_score_qwen3.5-397b-a17b.json')
        log_dir = os.path.join(output_dir, 'llm_raw_responses')

        # 2. Smart skipping
        if os.path.exists(final_score_path):
            print(f"⏭️  Skipping: Final score file already exists.")
            try:
                with open(final_score_path, 'r', encoding='utf-8') as f: final_result = json.load(f)
                all_scores.append(final_result['research_heuristics_score'])
                report_str = f"\n{'='*25} Experiment ID: {exp_id} Result (from cache) {'='*25}\nResearch Heuristics Score: {final_result['research_heuristics_score']:.4f}"
                full_report_lines.append(report_str)
                print(report_str)
                continue
            except (KeyError, FileNotFoundError, json.JSONDecodeError):
                 print("   - WARNING: Could not load cached score. Re-processing.")

        # 3. Load inputs
        human_content = read_survey_content(human_survey_path)
        llm_content = read_survey_content(llm_survey_path)
        task_prompt = get_task_prompt(task_prompt_path)

        if not all([human_content, llm_content, task_prompt]):
            print("❌ ERROR: Missing one or more required input files. Cannot proceed.")
            if not human_content: print(f"   - Check: {human_survey_path}")
            if not llm_content: print(f"   - Check: {llm_survey_path}")
            if not task_prompt: print(f"   - Check for task file (e.g., '200_...json') in: '{human_dir}'")
            continue

        # 4. Run evaluation pipeline
        os.makedirs(output_dir, exist_ok=True)
        criteria = evaluator.generate_criteria(task_prompt, criteria_path, log_dir)
        if not criteria: continue
        
        raw_scores = evaluator.perform_comparative_scoring(criteria, task_prompt, human_content, llm_content, scoring_path, log_dir)
        if not raw_scores: continue
        
        final_result = evaluator.calculate_final_score(raw_scores, criteria)
        if not final_result: continue
        
        with open(final_score_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        
        all_scores.append(final_result['research_heuristics_score'])
        report_str = f"""
{'='*25} Experiment ID: {exp_id} Result {'='*25}
Research Heuristics Score: {final_result['research_heuristics_score']:.4f}
(LLM Avg Score: {final_result['llm_survey_weighted_avg']:.2f}, Human Avg Score: {final_result['human_survey_weighted_avg']:.2f})
"""
        full_report_lines.append(report_str)
        print(report_str)

    # --- Final Statistical Analysis ---
    if all_scores:
        summary_lines = [f"\n\n{'='*28} Final Statistical Summary {'='*28}"]
        summary_lines.append(f"Based on {len(all_scores)} successfully processed experiments.")
        
        summary_lines.append("\n--- Average and Variance of Research Heuristics Score ---")
        header = f"{'Metric':<28} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        mean_val = np.mean(all_scores)
        var_val = np.var(all_scores)
        summary_lines.append(f"{'Research Heuristics Score':<28} | {mean_val:<12.4f} | {var_val:<12.4f}")
        
        full_report_lines.extend(summary_lines)

    # --- Save the complete report ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"\n\n✅ Complete report successfully saved to: {OUTPUT_REPORT_PATH}")
    except IOError as e:
        print(f"\n❌ ERROR: Could not write the report file. {e}")

    print(f"\n{'='*30} 🎉 All tasks complete! {'='*30}")

if __name__ == '__main__':
    main()
