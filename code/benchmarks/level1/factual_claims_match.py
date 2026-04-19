import os
import json
import time
from openai import OpenAI
import numpy as np
from typing import Dict, List, Any

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "") # 替换为你的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

# --- 批量处理与输出配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'
EXPERIMENT_IDS = range(1, 21) # 遍历 1 到 20 号文件夹

# --- 2. 核心API调用函数 ---
def get_streaming_output_with_reasoning(client: OpenAI, message: List[Dict], query_id: str, final_output_path: str, log_output_dir: str) -> str:
    """
    获取大模型流式输出，将最终答案保存到固定路径，并将日志保存到指定目录。
    """
    reasoning_dir = os.path.join(log_output_dir, "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    identifier = f"{query_id}_{timestamp}"
    reasoning_file = os.path.join(reasoning_dir, f"reasoning_{identifier}.txt")
    
    print(f"\n{'='*20} 正在执行查询: {query_id} {'='*20}\n")
    print("--- 模型思考过程 (Streaming) ---\n")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=message, stream=True, temperature=0.0
        )
        reasoning_content, answer_content = "", ""
        is_answering = False

        for chunk in response:
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            chunk_reasoning = getattr(delta, "reasoning_content", None)
            if chunk_reasoning and not is_answering:
                print(chunk_reasoning, end="", flush=True)
                reasoning_content += chunk_reasoning
            chunk_answer = getattr(delta, "content", None)
            if chunk_answer:
                if not is_answering:
                    print("\n" + "=" * 20 + " 最终回答 (JSON) " + "=" * 20 + "\n")
                    is_answering = True
                print(chunk_answer, end="", flush=True)
                answer_content += chunk_answer
        
        print("\n\n--- 流式输出结束 ---")
        with open(reasoning_file, 'w', encoding='utf-8') as f: f.write(reasoning_content)
        
        final_json_str = answer_content.strip()
        try:
            if final_json_str.startswith("```json"):
                final_json_str = final_json_str[7:-3].strip()
            parsed_json = json.loads(final_json_str)
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            error_txt_file = final_output_path.replace('.json', '_error.txt')
            with open(error_txt_file, 'w', encoding='utf-8') as f: f.write(answer_content)
            print(f"\n警告: 回答不是有效的JSON。原始文本已保存至 {error_txt_file}")

        print(f"\n思考过程日志已保存至: {reasoning_file}")
        return final_json_str
    except Exception as e:
        print(f"\nAPI调用或流式处理时发生错误: {e}")
        return "{}"

# --- 3. 核心功能: 构建Prompt并调用API进行匹配 ---
def find_semantic_matches_with_llm(client: OpenAI, expert_statements: List[str], llm_statements: List[str], query_id: str, final_output_path: str, log_output_dir: str) -> Dict:
    """使用大模型来寻找两个列表之间的语义匹配对。"""
    expert_statements_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(expert_statements)])
    llm_statements_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(llm_statements)])

    prompt = f"""
# ROLE
You are an expert in Natural Language Understanding and semantic analysis. Your task is to act as a meticulous fact-checker.
# TASK
Your goal is to identify all pairs of semantically equivalent factual statements from two provided lists: `expert_factual_claims` and `llm_factual_claims`.
# DEFINITION OF "SEMANTICALLY EQUIVALENT"
Two statements are semantically equivalent if and only if they assert the exact same core fact.
- **Good Match (Equivalent):** "ViT was developed by Google researchers in 2020." vs. "In 2020, researchers at Google created the Vision Transformer (ViT)."
- **Bad Match (Not Equivalent):** "BERT uses an encoder." vs. "GPT uses a decoder." (Different facts).
# RULES
1.  **Strictness is Key:** Only match statements if you are highly confident they convey the identical meaning. If there is any factual discrepancy (e.g., different numbers, dates, or subtle meanings), do not match them.
2.  **One-to-One Matching:** Each statement from one list should be matched to at most one statement from the other list. Find the best possible pairings.
3.  **Ignore Wording:** Do not be distracted by different phrasing or word order if the core fact is the same.
---
### INPUT LISTS
#### expert_factual_claims
{expert_statements_str}
#### llm_factual_claims
{llm_statements_str}
---
# OUTPUT RULES
- Your output **MUST** be a single, valid JSON object and nothing else.
- The JSON object must contain a single key: `"matched_pairs"`.
- The value of `"matched_pairs"` must be a **LIST** of JSON objects.
- Each object in the list must have exactly two keys: `"expert_factual_claims"` and `"llm_factual_claims"`.
- The values for these keys must be the full, original strings of the statements that you have matched.
- If no matches are found, return an empty list: `{{"matched_pairs": []}}`.
- **DO NOT** add any explanations, comments, or text outside the final JSON code block.
"""
    messages = [{"role": "user", "content": prompt}]
    final_json_str = get_streaming_output_with_reasoning(client, messages, query_id, final_output_path, log_output_dir)
    
    try:
        if final_json_str.strip().startswith("```json"):
            final_json_str = final_json_str.strip()[7:-3].strip()
        result = json.loads(final_json_str)
        if 'matched_pairs' in result and isinstance(result['matched_pairs'], list):
            return result
        else:
            print("错误: LLM返回的JSON格式不符合预期。")
            return {"matched_pairs": []}
    except json.JSONDecodeError:
        print(f"\n错误：无法将模型的最终回答解析为JSON。")
        return {"matched_pairs": []}

# --- 4. 辅助与计算函数 ---
def find_input_file(base_dir: str) -> str:
    """
    在指定目录中查找 'actual_claims_extraction.json' 或 'factual_claims_extraction.json'。
    返回找到的第一个文件的完整路径，如果都找不到则返回 None。
    """
    possible_filenames = ['actual_claims_extraction.json', 'factual_claims_extraction.json']
    for filename in possible_filenames:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return path
    return None

def load_statements_from_json(file_path: str) -> List[str]:
    """从JSON文件中加载事实陈述列表。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容两种可能的key
            return data.get("factual_statements", data.get("actual_claims", []))
    except (FileNotFoundError, json.JSONDecodeError):
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
    if not API_KEY or "xxxxxxxx" in API_KEY:
        print("错误：请在代码第11行设置您的有效API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    all_results = []
    full_report_lines = ["="*25 + " 事实声明匹配基准测试报告 " + "="*25]

    print("🚀 开始事实声明匹配批量处理任务...")
    
    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        # 1. 动态构建路径 (已修改)
        human_base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'bench', 'level1')
        llm_base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level1')
        
        expert_path = find_input_file(human_base_dir)
        llm_path = find_input_file(llm_base_dir)

        output_path = os.path.join(llm_base_dir, 'matched_pairs.json')
        log_dir = os.path.join(llm_base_dir, 'llm_outputs_statement_matching')

        # 2. 检查前置条件 (已修改)
        if not expert_path or not llm_path:
            print(f"❌ 错误: 缺少一个或多个输入文件，无法处理。")
            if not expert_path: print(f"   - 在目录 {human_base_dir} 中未找到 'actual_claims_extraction.json' 或 'factual_claims_extraction.json'")
            if not llm_path: print(f"   - 在目录 {llm_base_dir} 中未找到 'actual_claims_extraction.json' 或 'factual_claims_extraction.json'")
            continue

        expert_list = load_statements_from_json(expert_path)
        llm_list = load_statements_from_json(llm_path)

        if os.path.exists(output_path):
            print(f"⏭️  跳过: 目标文件 'matched_pairs.json' 已存在。正在直接加载并计算指标。")
            with open(output_path, 'r', encoding='utf-8') as f:
                matched_data = json.load(f)
        elif not expert_list or not llm_list:
            print(f"❌ 错误: 一个或多个输入文件为空或格式错误，无法处理。")
            if not expert_list: print(f"   - 文件为空或格式错误: {expert_path}")
            if not llm_list: print(f"   - 文件为空或格式错误: {llm_path}")
            continue
        else:
            print(f"加载成功: 专家陈述 {len(expert_list)} 条, LLM陈述 {len(llm_list)} 条。")
            print("\n--- 调用大模型进行语义匹配 ---")
            os.makedirs(log_dir, exist_ok=True)
            query_id = f"exp{exp_id}_statement_matching"
            matched_data = find_semantic_matches_with_llm(client, expert_list, llm_list, query_id, output_path, log_dir)
            print(f"\n--- 匹配结果已保存到 --- \n{output_path}")

        # 3. 计算并记录指标
        num_matched = len(matched_data.get("matched_pairs", []))
        metrics = calculate_metrics(len(expert_list), len(llm_list), num_matched)
        all_results.append(metrics)
        
        # 4. 准备单次报告
        report_str = f"""
{'='*25} 实验ID: {exp_id} {'='*25}
TP (匹配成功数): {metrics['tp']}
FP (LLM独有/幻觉数): {metrics['fp']}
FN (LLM遗漏数): {metrics['fn']}
-----------------------------------------------------
精确率 (Precision): {metrics['precision']:.4f}
召回率 (Recall):    {metrics['recall']:.4f}
F1-Score:          {metrics['f1_score']:.4f}
"""
        full_report_lines.append(report_str)
        print(report_str)

    # 5. 最终统计分析
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
    
    # 6. 保存完整报告
    try:
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"\n\n✅ 完整报告已成功保存至: {OUTPUT_REPORT_PATH}")
    except IOError as e:
        print(f"\n❌ 错误: 无法写入报告文件。{e}")

if __name__ == "__main__":
    main()


