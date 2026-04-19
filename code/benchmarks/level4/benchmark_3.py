import json
import re
import os
import time
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any, Tuple

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 替换为您的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)
METHOD_NAME = 'human'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>' # 修改了报告文件名以作区分

SCS_PROMPT_TEMPLATE = """
# ROLE
You are a meticulous structural logic judge. Your task is to analyze an academic paper's outline and identify any semantically redundant section headings that appear in different branches of the structure.

# TASK
Your goal is to identify all pairs of section headings from the provided outline that are semantically equivalent or cover the same core topic.

# CORE CONSTRAINT
Focus on identifying redundancy **ACROSS DIFFERENT BRANCHES** of the outline. Do not flag sibling sections that are naturally distinct parts of a larger topic.
- **GOOD MATCH (Redundant):** A heading "2.1. Method A" and another heading "3.2. Details of Method A" are redundant because they discuss the same topic in different main sections.
- **BAD MATCH (Not Redundant):** Headings "2.1. Method A" and "2.2. Method B" are NOT redundant. They are distinct sub-topics under the same parent and should not be matched.

# INPUT FORMAT
You will be given an outline where each heading has a unique ID (e.g., H1, H2, H2_1) and is presented in a numbered, indented format to show its position in the hierarchy.

# OUTPUT RULES
- Your output **MUST** be a single, valid JSON object.
- The JSON object must contain one key: `"redundant_pairs"`.
- The value for this key must be a list of lists/tuples, where each inner list/tuple contains the two **unique IDs** of the headings you have identified as redundant.
- Example: `{{"redundant_pairs": [["H2_1", "H3_1"], ["H4", "H5_2"]]}}`
- If no redundant pairs are found, return an empty list: `{{"redundant_pairs": []}}`.
- Do not add any explanations or extra text.

### OUTLINE TO ANALYZE
{formatted_outline}
"""

# --- 3. 核心功能函数 (无变化) ---

def get_llm_response(client: OpenAI, prompt: str, query_id: str) -> Dict:
    """调用LLM API并返回解析后的JSON。"""
    print(f"--- 正在为 '{query_id}' 调用LLM进行冗余分析... ---")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"❌ LLM调用或JSON解析失败: {e}")
        return {"redundant_pairs": []}

def prepare_outline_for_prompt(outline_data: List[List[Any]]) -> Tuple[str, Dict[str, str], List[str]]:
    """将大纲数据格式化为带ID的字符串，并创建父子关系图。"""
    formatted_lines = []
    parent_map = {}
    topic_ids = []
    level_parents = {-1: "root"}
    id_counters = {}
    for level, title in outline_data:
        level = int(level) # <-- Add this line to fix the error
        clean_title = re.sub(r'^\d+(\.\d+)*\s*', '', title).strip()
        parent_id_prefix = level_parents.get(level - 1, "H").replace('.', '_')
        current_counter = id_counters.get(parent_id_prefix, 0) + 1
        id_counters[parent_id_prefix] = current_counter
        unique_id = f"{parent_id_prefix}_{current_counter}" if parent_id_prefix != "root" else f"H{current_counter}"
        topic_ids.append(unique_id)
        parent_map[unique_id] = level_parents.get(level - 1)
        level_parents[level] = unique_id
        indent = "  " * level
        formatted_lines.append(f"{indent}{unique_id}: {clean_title}")
    return "\n".join(formatted_lines), parent_map, topic_ids

def calculate_scs_for_outline(client: OpenAI, outline_data: List[List[Any]], query_id: str) -> float:
    """为一个大纲计算SCS分数。"""
    if not outline_data or len(outline_data) < 2:
        return 1.0
    formatted_outline, parent_map, topic_ids = prepare_outline_for_prompt(outline_data)
    prompt = SCS_PROMPT_TEMPLATE.format(formatted_outline=formatted_outline)
    llm_result = get_llm_response(client, prompt, query_id)
    llm_pairs = llm_result.get("redundant_pairs", [])
    N_total = len(topic_ids)
    N_redundant = 0
    verified_pairs = []
    for pair in llm_pairs:
        if len(pair) == 2:
            id_a, id_b = pair[0], pair[1]
            if parent_map.get(id_a) is not None and parent_map.get(id_b) is not None and parent_map.get(id_a) != parent_map.get(id_b):
                N_redundant += 1
                verified_pairs.append(pair)
    print(f"LLM识别到 {len(llm_pairs)} 对冗余, 后端校验后剩余 {N_redundant} 对跨分支冗余。")
    scs_score = 1.0 - (N_redundant / N_total) if N_total > 0 else 1.0
    return scs_score

# --- 4. 主执行流程 (已修改) ---
def main():
    """主函数，负责遍历所有实验文件夹、仅计算LLM V2的SCS并生成报告。"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"🚀 开始大纲结构清晰度 (SCS) 批量评估任务 (仅处理 {METHOD_NAME})...")
    
    all_llm_scores = []
    full_report_lines = [f"={'='*25} 大纲结构清晰度 (SCS) 基准测试报告 (仅处理 {METHOD_NAME}) {'='*25}"]

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        # --- 修改点: 定义输出目录和缓存文件路径 ---
        output_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'bench', 'level4')
        llm_outline_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'outline.json')
        llm_scs_cache_path = os.path.join(output_dir, 'scs_score.json') # 缓存文件

        scs_llm_score = None
        
        # --- 新增功能: 缓存机制 ---
        if os.path.exists(llm_scs_cache_path):
            try:
                with open(llm_scs_cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                scs_llm_score = cached_data['scs_score']
                print(f"⏭️  发现缓存文件，直接加载分数: {scs_llm_score:.4f}")
            except (json.JSONDecodeError, KeyError):
                print(f"⚠️ 缓存文件 '{llm_scs_cache_path}' 损坏，将重新计算。")
                scs_llm_score = None
        
        # 如果没有从缓存加载分数，则执行计算
        if scs_llm_score is None:
            try:
                os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在
                with open(llm_outline_path, 'r', encoding='utf-8') as f:
                    llm_data = json.load(f)
                
                query_id = f"exp{exp_id}_{METHOD_NAME}_scs"
                scs_llm_score = calculate_scs_for_outline(client, llm_data, query_id)
                
                # 保存结果到缓存文件
                with open(llm_scs_cache_path, 'w', encoding='utf-8') as f:
                    json.dump({"scs_score": scs_llm_score}, f, indent=2)
                print(f"✅ 计算完成，分数已保存至缓存: {llm_scs_cache_path}")

            except (FileNotFoundError, json.JSONDecodeError) as e:
                error_msg = f"❌ 错误: 无法加载 '{llm_outline_path}'，跳过实验ID {exp_id}。 {e}"
                print(error_msg)
                full_report_lines.append(f"\n实验ID {exp_id}: {error_msg}")
                continue # 跳到下一个循环

        all_llm_scores.append(scs_llm_score)
        report_str = f"【结构清晰度 (SCS)】: AI Outline ({METHOD_NAME}) = {scs_llm_score:.4f}"
        
        full_report_lines.append(f"\n{'='*25} 实验ID: {exp_id} 评估结果 {'='*25}")
        full_report_lines.append(report_str)
        print("\n" + report_str)

    # --- 最终统计分析 (已简化) ---
    if all_llm_scores:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_llm_scores)} 个 '{METHOD_NAME}' 实验结果的统计分析")
        
        summary_lines.append("\n--- SCS 分数的平均值与方差 ---")
        header = f"{'Metric':<15} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        mean_val = np.mean(all_llm_scores)
        var_val = np.var(all_llm_scores)
        summary_lines.append(f"{'scs_llm':<15} | {mean_val:<12.4f} | {var_val:<12.4f}")
        
        full_report_lines.extend(summary_lines)

    # --- 保存完整报告到文件 ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"\n\n✅ 完整报告已成功保存至: {OUTPUT_REPORT_PATH}")
    except IOError as e:
        print(f"❌ 错误: 无法写入报告文件。{e}")

    print(f"\n{'='*30} 🎉 任务已完成! {'='*30}")

if __name__ == "__main__":
    main()