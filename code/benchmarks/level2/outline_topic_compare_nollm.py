import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any
import re
import numpy as np

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "") # 替换为您的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

# --- 批量处理配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21) # 遍历 1 到 20 号文件夹
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# --- 2. 核心API调用函数 ---
def get_llm_response(client: OpenAI, message: List[Dict], query_id: str, final_output_path: str, log_output_dir: str) -> str:
    """
    从LLM获取响应，将最终答案保存到固定路径，并将日志保存到指定目录。
    """
    answer_dir = os.path.join(log_output_dir, "answer_logs")
    os.makedirs(answer_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    identifier = f"{query_id}_{timestamp}"
    answer_log_file = os.path.join(answer_dir, f"answer_{identifier}.json")

    print(f"\n{'='*20} 正在执行查询: {query_id} {'='*20}\n")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=message,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        answer_content = response.choices[0].message.content
        print("--- LLM 返回的匹配结果 (JSON) ---")
        print(answer_content)
        
        try:
            parsed_json = json.loads(answer_content)
            # 1. 保存最终结果到固定的输出路径
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            # 2. 同时将原始回答保存到日志文件
            with open(answer_log_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 匹配结果已保存至: {final_output_path}")
        except json.JSONDecodeError:
            error_txt_file = final_output_path.replace('.json', '_error.txt')
            with open(error_txt_file, 'w', encoding='utf-8') as f: f.write(answer_content)
            print(f"\n⚠️ 警告: LLM返回的不是有效的JSON，原始文本已保存至 {error_txt_file}")

        return answer_content

    except Exception as e:
        print(f"\n❌ API调用过程中发生错误: {e}")
        return "{}"

# --- 3. 大纲处理与Prompt工程 ---
def load_and_flatten_outline(file_path: str) -> List[str]:
    """从JSON文件加载大纲，并返回一个扁平化的、清理过的标题列表。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            outline_data = json.load(f)
        topics = []
        for item in outline_data:
            if isinstance(item, list) and len(item) > 1:
                title = item[1]
                clean_title = re.sub(r'^\d+(\.\d+)*\s*', '', title).strip()
                topics.append(clean_title)
        return topics
    except Exception:
        return []

def build_outline_matching_prompt(expert_topics: List[str], llm_topics: List[str]) -> str:
    """为LLM构建用于匹配两个大纲主题的综合性Prompt。"""
    expert_topics_str = "\n".join([f"- {topic}" for topic in expert_topics])
    llm_topics_str = "\n".join([f"- {topic}" for topic in llm_topics])
    prompt = f"""
# ROLE
You are an expert academic researcher specializing in scientific literature analysis. Your task is to meticulously compare two outlines for a survey paper and identify all pairs of section headings that refer to the same core topic.
# TASK DEFINITION
Analyze the two lists of section headings provided below: `EXPERT_HEADINGS` and `LLM_HEADINGS`. Identify all pairs of headings that are semantically equivalent. A match occurs if a heading from the LLM list is a direct synonym, a clear paraphrase, or covers the same conceptual ground as a heading from the Expert list.
# EXAMPLES
- If `EXPERT_HEADINGS` has "Historical Development" and `LLM_HEADINGS` has "The Rise of Transformers", they are a match.
- If `EXPERT_HEADINGS` has "Conclusion" and `LLM_HEADINGS` has "Summary and Future Work", they are a match.
- If `EXPERT_HEADINGS` has "Core Mechanisms" and `LLM_HEADINGS` has "Applications", they are NOT a match as they cover different concepts.
# INPUT DATA
### EXPERT_HEADINGS
{expert_topics_str}
### LLM_HEADINGS
{llm_topics_str}
# OUTPUT RULES
Your response MUST be a single, valid JSON object.
This object must contain only one key: `"matched_pairs"`.
The value for this key must be a list of objects. Each object in the list represents one matched pair and must have exactly two keys:
1. `"expert_heading"`: The heading from the `EXPERT_HEADINGS` list.
2. `"llm_heading"`: The corresponding heading from the `LLM_HEADINGS` list.
If no matches are found, return an empty list: `[]`.
DO NOT include any explanations or extra text outside the JSON code block.
"""
    return prompt

# --- 4. 主执行流程 ---
def main():
    if not API_KEY or "xxxxxxxx" in API_KEY:
        print("❌ 错误: 请在代码第9行设置您的有效API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("🚀 开始大纲匹配批量处理任务...")
    
    all_results = []
    full_report_lines = ["="*25 + " 大纲主题覆盖率基准测试报告 " + "="*25]

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        # 1. 动态构建路径
        human_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'outline.json')
        llm_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'outline.json')
        output_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level2', 'outline_matching.json')
        log_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level2', 'llm_outputs_outline_matching')

        # 2. 加载基础数据
        expert_topics = load_and_flatten_outline(human_path)
        llm_topics = load_and_flatten_outline(llm_path)

        if not expert_topics or not llm_topics:
            print(f"❌ 错误: 缺少一个或多个输入大纲文件，无法处理。")
            if not expert_topics: print(f"   - 缺失: {human_path}")
            if not llm_topics: print(f"   - 缺失: {llm_path}")
            continue
        
        # 3. 获取匹配数据 (调用LLM或从文件加载)
        if os.path.exists(output_path):
            print(f"⏭️  跳过API调用: 目标文件 'outline_matching.json' 已存在。直接加载用于计算。")
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    match_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                 print(f"   - 警告: 无法加载已存在的JSON文件，将跳过此实验。")
                 continue
        else:
            print(f"✅ 成功加载 {len(expert_topics)} 个专家主题和 {len(llm_topics)} 个LLM主题。")
            prompt = build_outline_matching_prompt(expert_topics, llm_topics)
            messages = [{"role": "user", "content": prompt}]
            query_id = f"exp{exp_id}_outline_matching"
            response_json_str = get_llm_response(client, messages, query_id, output_path, log_dir)
            try:
                match_data = json.loads(response_json_str)
            except (json.JSONDecodeError, AttributeError):
                match_data = {"matched_pairs": []}
        
        # 4. 计算并记录指标
        # --- Start of Replacement Block ---

        # 4. 计算并记录指标
        matched_pairs = match_data.get("matched_pairs", [])
        if not isinstance(matched_pairs, list): matched_pairs = []

        # 从匹配对中提取唯一的llm主题和专家主题
        matched_llm_headings = {pair['llm_heading'] for pair in matched_pairs if 'llm_heading' in pair}
        matched_expert_headings = {pair['expert_heading'] for pair in matched_pairs if 'expert_heading' in pair}

        # 重新计算TP, FP, FN
        # 对于Precision: TP是成功匹配的llm主题的唯一数量
        tp_for_precision = len(matched_llm_headings)
        fp = len(llm_topics) - tp_for_precision

        # 对于Recall: TP是成功匹配的expert主题的唯一数量
        tp_for_recall = len(matched_expert_headings)
        fn = len(expert_topics) - tp_for_recall

        # 计算指标，确保分母不为0
        precision = tp_for_precision / len(llm_topics) if len(llm_topics) > 0 else 0
        recall = tp_for_recall / len(expert_topics) if len(expert_topics) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # --- End of Replacement Block ---
        # 存储结果用于最终统计
        all_results.append({'precision': precision, 'recall': recall, 'f1_score': f1_score})

        # 准备单次报告并添加到总报告中
        individual_report = f"""
{'='*25} 实验ID: {exp_id} 覆盖率报告 {'='*25}
精确率 (Precision): {precision:.4f}
召回率 (Recall):    {recall:.4f}
F1-Score:         {f1_score:.4f}
"""
        full_report_lines.append(individual_report)
        print(individual_report)

    # --- 5. 最终统计分析 ---
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
    
    # --- 6. 保存完整报告到文件 ---
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

