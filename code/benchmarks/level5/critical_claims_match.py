# -*- coding: utf-8 -*-
"""
LLM 语义匹配评估器

本脚本通过调用大语言模型（LLM），批量评估 LLM 生成的“批判性声明”
与人类专家撰写的“批判性声明”之间的语义匹配程度。

核心流程：
1. 遍历指定的多个实验文件夹 (如 1, 2, ..., 20)。
2. 在每个文件夹中，加载人类专家和 LLM 各自提取的批判性声明列表。
3. 构建一个详细的 Prompt，要求大模型找出两个列表中语义等效的声明对。
4. 调用 LLM API，获取匹配结果。
5. （新功能）在控制台实时打印 LLM 的原始回答。
6. 解析 LLM 返回的 JSON 结果，并计算精确率、召回率和 F1 分数。
7. 将所有实验结果汇总，计算平均值和方差，并生成一份完整的评估报告。
"""
import os
import json
import time
import re
import numpy as np
from openai import OpenAI
from typing import Dict, List, Optional

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 替换为您的 agicto API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")            # 您的 agicto API 端点
LLM_MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# --- 批量处理与报告配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'LLMxMapReduce_V2'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# --- 2. 提示词定义 ---
CRITICAL_MATCHING_PROMPT_TEMPLATE = """
# ROLE
You are an expert in Natural Language Understanding and semantic analysis, specializing in academic and research-oriented texts.
# TASK
Your goal is to meticulously identify all pairs of **semantically equivalent "critical statements"** from two provided lists: `expert_critical_statements` and `llm_critical_statements`.
# DEFINITION OF "SEMANTICALLY EQUIVALENT CRITICAL STATEMENT"
Two critical statements are semantically equivalent if they express the **same core evaluation, analysis, limitation, research gap, or future direction**, even if the wording is different.
- **Good Match (Equivalent):**
  - Statement 1: "A key drawback of this model is its significant computational overhead."
  - Statement 2: "The model is computationally expensive, which limits its practical use."
- **Bad Match (Not Equivalent):**
  - Statement 1: "The model struggles with small objects."
  - Statement 2: "The model requires extensive training data."
# RULES
1.  **Strictness is Paramount:** Only match statements if you are highly confident they convey the identical critique or suggestion.
2.  **One-to-One Matching:** Each statement from one list can be matched to at most one statement from the other list.
3.  **Focus on Meaning, Not Phrasing:** Ignore differences in wording if the core critical point is the same.
---
### INPUT LISTS
#### expert_critical_statements
{expert_statements_str}
#### llm_critical_statements
{llm_statements_str}
---
# OUTPUT RULES
- Your output **MUST** be a single, valid JSON object.
- The JSON object must contain a single key: `"matched_critical_pairs"`.
- The value must be a **LIST** of objects, each with two keys: `"expert_critical_statement"` and `"llm_critical_statement"`.
- The values must be the full, original strings of the statements you have matched.
- If no matches are found, return an empty list: `{{"matched_critical_pairs": []}}`.
"""

# --- 3. 核心实现 ---

def call_llm(client: OpenAI, prompt: str, purpose: str, log_dir: str) -> str:
    """辅助函数，用于调用LLM API，打印并保存原始响应，同时处理重试。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filepath = os.path.join(log_dir, f"{timestamp}_{purpose}_raw_response.txt")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8192
            )
            raw_response = response.choices[0].message.content.strip()

            # --- 新增代码：打印大模型的原始回答 ---
            print("\n" + "-"*15 + " LLM Raw Response " + "-"*15)
            print(raw_response)
            print("-" * (30 + len(" LLM Raw Response ")) + "\n")
            # --- 修改结束 ---

            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write(raw_response)
            print(f"✅ LLM原始响应 '{purpose}' 已保存。")
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

def find_semantic_matches(client: OpenAI, expert_statements: List[str], llm_statements: List[str], query_id: str, final_output_path: str, log_dir: str) -> Dict:
    """使用大模型来寻找两个列表之间的语义匹配对。"""
    expert_str = "\n".join([f"E{i+1}. {s}" for i, s in enumerate(expert_statements)])
    llm_str = "\n".join([f"L{i+1}. {s}" for i, s in enumerate(llm_statements)])
    prompt = CRITICAL_MATCHING_PROMPT_TEMPLATE.format(expert_statements_str=expert_str, llm_statements_str=llm_str)
    
    llm_response = call_llm(client, prompt, query_id, log_dir)
    
    try:
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if not json_match: raise ValueError("在LLM响应中找不到JSON结构。")
        
        result = json.loads(json_match.group(0))
        
        # 验证并保存
        if 'matched_critical_pairs' in result and isinstance(result['matched_critical_pairs'], list):
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ 匹配结果已成功保存至 '{os.path.basename(final_output_path)}'")
            return result
        else:
            print("错误: LLM返回的JSON格式不符合预期。")
            return {"matched_critical_pairs": []}
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ 错误：无法将模型的最终回答解析为JSON。 {e}")
        return {"matched_critical_pairs": []}

# --- 4. 辅助函数 ---

def load_statements_from_json(file_path: str) -> Optional[List[str]]:
    """从JSON文件中加载批判性声明列表。"""
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("critical_statements", [])
    except json.JSONDecodeError:
        return []

def calculate_metrics(expert_list_len: int, llm_list_len: int, matched_pairs_len: int) -> Dict[str, float]:
    """根据匹配结果计算P/R/F1分数。"""
    tp = matched_pairs_len
    fp = llm_list_len - tp
    fn = expert_list_len - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1_score": f1_score, "tp": tp, "fp": fp, "fn": fn}

# --- 5. 主执行流程 ---

def main():
    """主函数，负责遍历所有实验文件夹、调用处理函数、进行统计并生成报告。"""
    print("🚀 开始批判性声明匹配批量评估任务...")
    all_results = []
    full_report_lines = ["="*25 + " 批判性声明匹配基准测试报告 " + "="*25]
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        # 1. 动态构建路径
        human_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'bench', 'level5', 'critical_claims_extraction.json')
        llm_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'bench', 'level5', 'critical_claims_extraction.json')
        output_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'bench', 'level5')
        output_path = os.path.join(output_dir, 'critical_matching_result.json')
        log_dir = os.path.join(output_dir, 'llm_outputs_critical_matching')

        # 2. 检查和加载输入
        expert_list = load_statements_from_json(human_path)
        llm_list = load_statements_from_json(llm_path)
        
        if expert_list is None or llm_list is None:
            print("❌ 错误: 缺少一个或多个输入文件，无法处理。")
            if expert_list is None: print(f"   - 缺失: {human_path}")
            if llm_list is None: print(f"   - 缺失: {llm_path}")
            continue

        # 3. 智能跳过或执行匹配
        if os.path.exists(output_path):
            print(f"⏭️  跳过API调用: 目标文件 '{os.path.basename(output_path)}' 已存在。")
            with open(output_path, 'r', encoding='utf-8') as f:
                matched_data = json.load(f)
        else:
            print(f"加载成功: 专家声明 {len(expert_list)} 条, LLM声明 {len(llm_list)} 条。")
            print("\n--- 调用大模型进行语义匹配 ---")
            os.makedirs(output_dir, exist_ok=True)
            query_id = f"exp{exp_id}_critical_matching"
            matched_data = find_semantic_matches(client, expert_list, llm_list, query_id, output_path, log_dir)

        # 4. 计算并记录指标
        num_matched = len(matched_data.get("matched_critical_pairs", []))
        metrics = calculate_metrics(len(expert_list), len(llm_list), num_matched)
        all_results.append(metrics)
        
        # 5. 准备单次报告
        report_str = f"""
{'='*25} 实验ID: {exp_id} 评估结果 {'='*25}
精确率 (Precision): {metrics['precision']:.4f}
召回率 (Recall):    {metrics['recall']:.4f}
F1-Score:        {metrics['f1_score']:.4f}
(TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']})
"""
        full_report_lines.append(report_str)
        print(report_str)

    # 6. 最终统计分析
    if all_results:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_results)} 个实验结果的统计分析")
        metric_keys = ["precision", "recall", "f1_score"]
        summary_lines.append("\n--- 各项指标的平均值与方差 ---")
        header = f"{'Metric':<12} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        for key in metric_keys:
            values = [res[key] for res in all_results]
            mean_val = np.mean(values)
            var_val = np.var(values)
            summary_lines.append(f"{key:<12} | {mean_val:<12.4f} | {var_val:<12.4f}")
        
        full_report_lines.extend(summary_lines)
    
    # 7. 保存完整报告
    try:
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"\n\n✅ 完整报告已成功保存至: {OUTPUT_REPORT_PATH}")
    except IOError as e:
        print(f"\n❌ 错误: 无法写入报告文件。{e}")

    print(f"\n{'='*30} 🎉 所有任务已完成! {'='*30}")

if __name__ == "__main__":
    main()
