import os
import json
import time
import threading
from openai import OpenAI
from tqdm import tqdm
import re
import glob
import numpy as np
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
METHOD_NAME = 'LLMxMapReduce_V2'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# <<< MODIFIED: Prompt for "Framework Application" criteria generation
CRITERIA_GENERATION_PROMPT_TEMPLATE = """
<system_role>
You are an expert in research methodology and academic writing, specializing in the structure and organization of high-quality literature reviews. You have a deep understanding of how theoretical frameworks can be used to provide structure, coherence, and analytical depth to a survey paper.
</system_role>

<user_prompt>
**Background**: We are evaluating the structural quality of a survey paper. The survey was written to address the following research task:

<task>
{task_prompt}
</task>

**Core Evaluation Dimension: Framework Application**
This metric assesses the ability of a model to apply a recognized framework to organize and construct a review.

<Instruction>
**Your Goal**: For the **Framework Application** dimension, develop a detailed, specific, and logically sound set of evaluation criteria tailored to the research `<task>`. You must:

1.  **Analyze the Task & Dimension**: Based on the `<task>`, identify what constitutes a well-structured and framework-driven review for this specific topic. Consider established frameworks in the field if applicable.
2.  **Formulate Task-Specific Criteria**: Propose criteria to evaluate the quality of the framework's application.
3.  **Provide Rationale**: For each criterion, provide a concise explanation (`explanation`).
4.  **Assign Weights**: Assign a weight (`weight`) to each criterion, ensuring the sum of all weights is exactly **1.0**.

**Core Requirements**:
1.  **Task-Centric**: Your criteria must be directly linked to the nuances of organizing a review for the `<task>`.
2.  **Sufficient Justification**: The `<analysis>` must explain your reasoning for the criteria and weights.
3.  **Standard Output Format**: Strictly follow the example format.
</Instruction>

<example>
<task>
"A comprehensive survey on Reinforcement Learning from Human Feedback (RLHF)."
</task>
<output>
<analysis>
For an RLHF survey, a strong structure is crucial for clarity. The review should not be just a flat list of papers. A recognized framework, such as a chronological evolution, a taxonomy of methods (e.g., based on feedback type), or a problem-solution structure, is essential. Therefore, "Clarity and Relevance of Framework" and "Consistent Application" are weighted most heavily, as they determine the fundamental readability and coherence of the survey.
</analysis>
<json_output>
[
  {{
    "criterion": "Clarity and Relevance of Framework",
    "explanation": "Assesses whether the paper explicitly identifies and explains a clear, logical framework for organizing the review. The chosen framework must be appropriate for the topic of RLHF.",
    "weight": 0.35
  }},
  {{
    "criterion": "Consistent Application",
    "explanation": "Evaluates if the review consistently follows the stated framework throughout the paper. Sections and subsections should logically map to the framework's components.",
    "weight": 0.35
  }},
  {{
    "criterion": "Analytical Depth",
    "explanation": "Measures whether the framework is used not just for organization, but also to facilitate deeper analysis, such as comparing and contrasting different approaches or identifying trends.",
    "weight": 0.3
  }}
]
</json_output>
</output>
</example>

Now, please perform this task for the following research prompt:
<task>
{task_prompt}
</task>
</user_prompt>
"""
# <<< MODIFIED: Prompt for "Framework Application" scoring
SCORING_PROMPT_TEMPLATE = """
<system_role>
You are a rigorous, meticulous, and objective academic reviewer. You are skilled at analyzing the structure and organization of research reviews and comparing them based on specific evaluation criteria.
</system_role>

<user_prompt>
**Task Background**
You need to evaluate how well two survey papers apply a theoretical framework to organize their content. The surveys were written to address the following research task:
<task>
{task_prompt}
</task>

**Articles to be Evaluated**
<article_1>
{article_1_text}
</article_1>

<article_2>
{article_2_text}
</article_2>

**Evaluation Criteria: Framework Application**
Now, you must evaluate and compare the two articles on a criterion-by-criterion basis according to the following **list of criteria**.

<criteria_list>
{criteria_json_string}
</criteria_list>

<Instruction>
**Your Task**
Strictly following **each criterion** in the `<criteria_list>`, compare and evaluate the performance of `<article_1>` and `<article_2>`. Your analysis should consider the entire structure of the articles, including the introduction, main body, and conclusion, to assess how the framework is applied.

**Scoring Rubric**
For each criterion, score both articles on a continuous scale from 0 to 10.

**Output Format Requirement**
Please **strictly** follow the `<output_format>` below. Ensure the final output is a single, valid JSON object that can be parsed directly.
</Instruction>

<output_format>
{{
    "framework_application": [
        {{
            "criterion": "[Text of the first evaluation criterion]",
            "analysis": "[Comparative analysis of how each article meets this criterion]",
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

# --- 3. 核心实现 ---

class FrameworkApplicationEvaluator:
    """
    一个用于编排“框架应用”指标三阶段评估的类。
    """
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _call_llm(self, prompt: str, purpose: str, log_dir: str) -> str:
        """辅助函数，用于调用LLM API，保存原始响应并处理重试。"""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = f"{timestamp}_{purpose}_raw_response.txt"
        log_filepath = os.path.join(log_dir, log_filename)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                raw_response = response.choices[0].message.content.strip()
                with open(log_filepath, 'w', encoding='utf-8') as f:
                    f.write(raw_response)
                print(f"✅ LLM原始响应 '{purpose}' 已保存至 '{log_filename}'")
                return raw_response
            except Exception as e:
                print(f"尝试 {attempt + 1} 失败: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print("已达到最大重试次数。失败。")
                    with open(log_filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Failed after {MAX_RETRIES} retries.\nLast error: {e}")
                    raise

    def generate_criteria(self, task_prompt: str, output_path: str, log_dir: str) -> Optional[list]:
        """阶段 1: 生成评估标准。"""
        print("--- 阶段 1: 生成框架应用评估标准 ---")
        try:
            prompt = CRITERIA_GENERATION_PROMPT_TEMPLATE.format(task_prompt=task_prompt)
            llm_response = self._call_llm(prompt, "criteria_generation", log_dir)
            json_match = re.search(r'\[\s*\{.*\}\s*\]', llm_response, re.DOTALL)
            if not json_match: raise ValueError("在标准生成响应中找不到有效的JSON列表 `[]`。")
            
            criteria_list = json.loads(json_match.group(0))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(criteria_list, f, indent=4, ensure_ascii=False)
            print(f"✅ 标准已成功生成并保存至 '{os.path.basename(output_path)}'。")
            return criteria_list
        except Exception as e:
            print(f"❌ 阶段 1 出错: {e}")
            return None

    def perform_comparative_scoring(self, criteria: list, task_prompt: str, human_survey: str, llm_survey: str, output_path: str, log_dir: str) -> Optional[dict]:
        """阶段 2: 执行比较评分。"""
        print("\n--- 阶段 2: 执行比较评分 ---")
        try:
            criteria_for_prompt = [{"criterion": c["criterion"], "explanation": c["explanation"]} for c in criteria]
            criteria_json_string = json.dumps(criteria_for_prompt, indent=2, ensure_ascii=False)
            scoring_prompt = SCORING_PROMPT_TEMPLATE.format(
                task_prompt=task_prompt, article_1_text=llm_survey, article_2_text=human_survey,
                criteria_json_string=criteria_json_string
            )
            llm_response = self._call_llm(scoring_prompt, "comparative_scoring", log_dir)
            
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match: raise ValueError("在评分响应中找不到JSON结构。")
            
            scores_dict = json.loads(json_match.group(0))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scores_dict, f, indent=4, ensure_ascii=False)
            print(f"✅ 比较分数已成功生成并保存至 '{os.path.basename(output_path)}'。")
            return scores_dict
        except Exception as e:
            print(f"❌ 阶段 2 出错: {e}")
            return None

    def calculate_final_score(self, scores: dict, criteria: list) -> Optional[dict]:
        """阶段 3: 计算最终的加权和归一化分数。"""
        print("\n--- 阶段 3: 计算最终分数 ---")
        try:
            criteria_map = {item['criterion']: item['weight'] for item in criteria}
            total_weighted_score_1, total_weighted_score_2, total_weight = 0.0, 0.0, 0.0
            
            scored_items = scores.get('framework_application', [])
            if not scored_items: raise KeyError("在评分结果中找不到'framework_application'键。")

            for item in scored_items:
                weight = criteria_map.get(item['criterion'])
                if weight is not None:
                    total_weighted_score_1 += float(item['article_1_score']) * weight
                    total_weighted_score_2 += float(item['article_2_score']) * weight
                    total_weight += weight
            
            if abs(total_weight - 1.0) > 1e-6 and total_weight > 0:
                print(f"⚠️ 警告: 评分标准总权重为 {total_weight}，而不是1.0。")

            avg_score_1 = total_weighted_score_1 / total_weight if total_weight > 0 else 0
            avg_score_2 = total_weighted_score_2 / total_weight if total_weight > 0 else 0
            final_score = avg_score_1 / (avg_score_1 + avg_score_2) if (avg_score_1 + avg_score_2) > 0 else 0.0

            print("✅ 最终分数计算完成。")
            return {
                "llm_survey_weighted_avg": avg_score_1,
                "human_survey_weighted_avg": avg_score_2,
                "framework_application_score": final_score
            }
        except Exception as e:
            print(f"❌ 阶段 3 出错: {e}")
            return None

# --- 4. 辅助函数 ---

def read_survey_content(file_path: str) -> Optional[str]:
    if not file_path or not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        return "\n".join(content_list) if isinstance(content_list, list) else None
    except (json.JSONDecodeError, TypeError):
        return None

def find_task_prompt_file(directory: str) -> Optional[str]:
    """
    在目录中查找以 '数字_' 开头的JSON文件。
    """
    if not os.path.isdir(directory):
        return None
    # 遍历目录中的所有条目
    for filename in os.listdir(directory):
        # 使用正则表达式匹配 "一个或多个数字" + "_" + "任何字符" + ".json"
        if re.match(r'^\d+_.+\.json$', filename):
            return os.path.join(directory, filename)
    return None

def get_task_prompt(file_path: str) -> Optional[str]:
    """
    (已修改) 从任务文件中仅提取标题作为任务提示。
    """
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 只获取 'title' 字段
        title = data.get("title", "")
        
        if title:
            return title
        else:
            print(f"  - 警告: 在 '{os.path.basename(file_path)}' 中找不到 'title' 字段。")
            return None
            
    except (json.JSONDecodeError, AttributeError):
        print(f"  - 警告: 无法解析或处理任务文件 '{os.path.basename(file_path)}'。")
        return None

# --- 5. 主执行流程 ---

def main():
    """主函数，负责遍历所有实验文件夹、调用处理函数、进行统计并生成报告。"""
    print("🚀 开始框架应用批量评估任务...")
    all_scores = []
    full_report_lines = ["="*25 + " 框架应用基准测试报告 " + "="*25]
    
    evaluator = FrameworkApplicationEvaluator(api_key=API_KEY, base_url=BASE_URL, model=LLM_MODEL)

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        # 1. 动态构建路径
        human_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human')
        llm_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME)
        output_dir = os.path.join(llm_dir, 'bench', 'level3', 'framework_application_evaluation')
        
        human_survey_path = os.path.join(human_dir, 'content.json')
        llm_survey_path = os.path.join(llm_dir, 'content.json')
        # <<< 修改点: 调用新的查找函数
        task_prompt_path = find_task_prompt_file(human_dir)
        
        # 输出文件路径
        criteria_path = os.path.join(output_dir, 'framework_application_criteria_qwen3.5-397b-a17b.json')
        scoring_path = os.path.join(output_dir, 'comparative_scores_qwen3.5-397b-a17b.json')
        final_score_path = os.path.join(output_dir, 'final_framework_application_score_qwen3.5-397b-a17b.json')
        log_dir = os.path.join(output_dir, 'llm_raw_responses_qwen3.5-397b-a17b')

        # 2. 智能跳过
        if os.path.exists(final_score_path):
            print(f"⏭️  跳过: 最终分数文件已存在。")
            try:
                with open(final_score_path, 'r', encoding='utf-8') as f:
                    final_result = json.load(f)
                all_scores.append(final_result['framework_application_score'])
                report_str = f"""
{'='*25} 实验ID: {exp_id} 评估结果 (从缓存加载) {'='*25}
框架应用分数: {final_result['framework_application_score']:.4f}
"""
                full_report_lines.append(report_str)
                print(report_str)
                continue
            except (KeyError, FileNotFoundError, json.JSONDecodeError):
                 print("   - 警告: 无法加载缓存分数，将重新处理。")

        # 3. 加载输入数据
        human_content = read_survey_content(human_survey_path)
        llm_content = read_survey_content(llm_survey_path)
        task_prompt = get_task_prompt(task_prompt_path)

        if not all([human_content, llm_content, task_prompt]):
            print("❌ 错误: 缺少一个或多个必要的输入文件，无法处理。")
            if not human_content: print(f"   - 检查: {human_survey_path}")
            if not llm_content: print(f"   - 检查: {llm_survey_path}")
            if not task_prompt: print(f"   - 检查任务文件: 在 '{human_dir}' 中未找到 '数字_...' 格式的 .json 文件")
            continue

        # 4. 执行评估流程
        os.makedirs(output_dir, exist_ok=True)
        criteria = evaluator.generate_criteria(task_prompt, criteria_path, log_dir)
        if not criteria: continue
        
        raw_scores = evaluator.perform_comparative_scoring(criteria, task_prompt, human_content, llm_content, scoring_path, log_dir)
        if not raw_scores: continue
        
        final_result = evaluator.calculate_final_score(raw_scores, criteria)
        if not final_result: continue
        
        with open(final_score_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        
        all_scores.append(final_result['framework_application_score'])
        report_str = f"""
{'='*25} 实验ID: {exp_id} 评估结果 {'='*25}
框架应用分数: {final_result['framework_application_score']:.4f}
(LLM 平均分: {final_result['llm_survey_weighted_avg']:.2f}, Human 平均分: {final_result['human_survey_weighted_avg']:.2f})
"""
        full_report_lines.append(report_str)
        print(report_str)

    # --- 最终统计分析 ---
    if all_scores:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_scores)} 个实验结果的统计分析")
        
        summary_lines.append("\n--- 框架应用分数的平均值与方差 ---")
        header = f"{'Metric':<28} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        mean_val = np.mean(all_scores)
        var_val = np.var(all_scores)
        summary_lines.append(f"{'Framework Application Score':<28} | {mean_val:<12.4f} | {var_val:<12.4f}")
        
        full_report_lines.extend(summary_lines)

    # --- 保存完整报告到文件 ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"\n\n✅ 完整报告已成功保存至: {OUTPUT_REPORT_PATH}")
    except IOError as e:
        print(f"\n❌ 错误: 无法写入报告文件。{e}")

    print(f"\n{'='*30} 🎉 所有任务已完成! {'='*30}")

if __name__ == '__main__':
    main()

