import json
from thefuzz import fuzz
import os
import numpy as np
from typing import Dict, List

# --- 1. 全局配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# 阈值配置
CITATION_THRESHOLD = 50
SIMILARITY_THRESHOLD = 80  # 匹配相似度阈值为80%

# --- 2. 辅助函数 ---

def load_references_from_file(filepath: str) -> list:
    """
    加载并解析特定的JSON结构，以提取参考文献列表。
    每个参考文献是一个包含其标题和引用计数的字典。
    """
    if not os.path.exists(filepath):
        # 此函数现在返回None以表示失败，而不是打印错误
        return None
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"警告: 无法解析JSON文件 {filepath}")
        return None

    references = []
    # JSON的键格式为 "paper_1_info", "paper_2_info", etc.
    for i in range(1, data.get("reference_num", 0) + 1):
        paper_key = f"paper_{i}_info"
        ref_key = f"reference_{i}"
        
        paper_info = data.get(paper_key, {})
        ref_info = paper_info.get(ref_key, {})
        
        title = ref_info.get("searched_title")
        # 将引用计数规范化为整数
        citations = ref_info.get("citation_count", 0)
        try:
            # 处理 "N/A" 或其他非数字值
            citations = int(citations) if str(citations).isdigit() else 0
        except (ValueError, TypeError):
            citations = 0
        
        if title:
            references.append({"title": title, "citations": citations})
            
    return references

# --- 3. 核心计算逻辑 ---

def calculate_coverage_for_experiment(human_refs: List[Dict], llm_refs: List[Dict]) -> Dict:
    """
    为单次实验计算核心参考文献覆盖率指标。
    """
    # 集合A: LLM生成的所有参考文献标题
    set_a_titles = [ref["title"] for ref in llm_refs]
    # 集合B: 引用次数 > 阈值的专家参考文献标题
    set_b_titles = [ref["title"] for ref in human_refs if ref["citations"] > CITATION_THRESHOLD]

    matched_pairs = []
    unmatched_b_titles = list(set_b_titles)

    for a_title in set_a_titles:
        best_match = None
        highest_similarity = 0
        
        for b_title in unmatched_b_titles:
            similarity = fuzz.ratio(a_title.lower(), b_title.lower())
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = b_title
        
        if highest_similarity >= SIMILARITY_THRESHOLD:
            matched_pairs.append({
                "llm_title": a_title,
                "human_core_title": best_match,
                "similarity": highest_similarity
            })
            unmatched_b_titles.remove(best_match)

    tp = len(matched_pairs)
    fp = len(set_a_titles) - tp
    fn = len(set_b_titles) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp, "fp": fp, "fn": fn
    }

# --- 4. 主执行流程 ---
def main():
    """
    主函数，遍历所有实验，计算指标，进行统计分析，并保存报告。
    """
    all_results = []
    full_report_lines = ["="*25 + " 核心参考文献覆盖率基准测试报告 " + "="*25]
    
    print("🚀 开始核心参考文献覆盖率批量计算任务...")

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        human_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'reference_3.json')
        llm_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'reference_3.json')

        human_refs = load_references_from_file(human_path)
        llm_refs = load_references_from_file(llm_path)

        if human_refs is None or llm_refs is None:
            print("❌ 错误: 缺少一个或多个输入文件，跳过此实验。")
            if human_refs is None: print(f"   - 缺失或错误: {human_path}")
            if llm_refs is None: print(f"   - 缺失或错误: {llm_path}")
            continue

        metrics = calculate_coverage_for_experiment(human_refs, llm_refs)
        all_results.append(metrics)

        # 准备单次实验报告
        individual_report = f"""
{'='*25} 实验ID: {exp_id} 覆盖率报告 {'='*25}
精确率 (Precision): {metrics['precision']:.4f}
召回率 (Recall):    {metrics['recall']:.4f}
F1-Score:         {metrics['f1_score']:.4f}
(TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']})
"""
        full_report_lines.append(individual_report)
        print(individual_report)

    # --- 最终统计分析 ---
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

    # --- 保存完整报告到文件 ---
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
