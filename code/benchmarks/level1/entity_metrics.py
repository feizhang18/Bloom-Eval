import os
import json
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from typing import Dict, List, Any

# --- 1. 全局配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
OUTPUT_FILE_PATH = '<OUTPUT_REPORT_PATH>'
EXPERIMENT_IDS = range(1, 21)  # 处理 1 到 20 号文件夹

# --- 2. 核心计算函数 (无变化) ---
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    epsilon = 1e-10
    kl_p_m = entropy(p + epsilon, m + epsilon)
    kl_q_m = entropy(q + epsilon, m + epsilon)
    return 0.5 * (kl_p_m + kl_q_m)

def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def calculate_distribution_metrics(human_counts, llm_counts):
    if not human_counts or not llm_counts: return None
    p_counts = np.array(human_counts)
    q_counts = np.array(llm_counts)
    if p_counts.sum() == 0 or q_counts.sum() == 0: return None
    p_dist = p_counts / p_counts.sum()
    q_dist = q_counts / q_counts.sum()
    jsd = jensen_shannon_divergence(p_dist, q_dist)
    hd = hellinger_distance(p_dist, q_dist)
    tvd = total_variation_distance(p_dist, q_dist)
    # 移除了 Wasserstein Distance 的计算
    return {"jensen_shannon_dist": jsd, "hellinger_dist": hd, "total_variation_dist": tvd, "entity_count": len(p_dist)}

def calculate_normalized_similarity_scores(dist_metrics):
    if dist_metrics is None:
        return {"jensen_shannon_sim": 1.0, "hellinger_sim": 1.0, "total_variation_sim": 1.0}
    jsd_dist_normalized = dist_metrics['jensen_shannon_dist'] / np.log(2)
    hd_dist_normalized = dist_metrics['hellinger_dist']
    tvd_dist_normalized = dist_metrics['total_variation_dist']
    # 移除了 Wasserstein Similarity 的计算
    return {
        "jensen_shannon_sim": 1 - jsd_dist_normalized,
        "hellinger_sim": 1 - hd_dist_normalized,
        "total_variation_sim": 1 - tvd_dist_normalized,
    }

# --- 3. 单个实验处理函数 ---
def process_single_experiment(exp_id: int):
    """
    处理单个实验文件夹，计算所有指标并返回结果字典和输出字符串。
    """
    output_lines = [f"\n{'='*25} 实验ID: {exp_id} {'='*25}"]
    
    human_json_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'bench', 'level1', 'final_counts.json')
    llm_json_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level1', 'final_counts.json')
    matched_json_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level1', 'compare_same.json')

    try:
        with open(human_json_path, 'r', encoding='utf-8') as f: human_data = json.load(f)
        with open(llm_json_path, 'r', encoding='utf-8') as f: llm_data = json.load(f)
        with open(matched_json_path, 'r', encoding='utf-8') as f: matched_data = json.load(f)
    except FileNotFoundError as e:
        error_msg = f"❌ 错误: 找不到文件 {e.filename}。跳过此实验。"
        output_lines.append(error_msg)
        return None, "\n".join(output_lines)

    metrics = {}
    categories = ['methods_models', 'datasets', 'evaluation_metrics']
    all_human_entities, all_llm_entities, all_matched_pairs = {}, {}, []

    for category in categories:
        all_human_entities.update(human_data.get(category, {}))
        all_llm_entities.update(llm_data.get(category, {}))
        all_matched_pairs.extend(matched_data.get(category, []))

    # 1. 实体识别性能
    tp = len(all_matched_pairs)
    fp = len(all_llm_entities) - tp
    fn = len(all_human_entities) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.update({'precision': precision, 'recall': recall, 'f1_score': f1_score})
    output_lines.append("\n--- 实体识别性能 ---")
    output_lines.append(f"TP: {tp}, FP: {fp}, FN: {fn}")
    output_lines.append(f"准确率 (Precision): {precision:.4f}")
    output_lines.append(f"召回率 (Recall):    {recall:.4f}")
    output_lines.append(f"F1-Score:          {f1_score:.4f}")

    # 2. 实体分布相似度
    if tp > 0:
        try:
            human_counts = [all_human_entities[p['expert_main_name']]['total_count'] for p in all_matched_pairs]
            llm_counts = [all_llm_entities[p['llm_main_name']]['total_count'] for p in all_matched_pairs]
            dist_metrics = calculate_distribution_metrics(human_counts, llm_counts)
            sim_scores = calculate_normalized_similarity_scores(dist_metrics)
        except KeyError as e:
            output_lines.append(f"\n❌ 错误: 找不到键 {e}。无法计算分布相似度。")
            sim_scores = {k: 0.0 for k in ["jensen_shannon_sim", "hellinger_sim", "total_variation_sim", "wasserstein_1d_sim"]}
    else:
        output_lines.append("\n没有任何共同实体，无法计算分布相似度。")
        sim_scores = {k: 0.0 for k in ["jensen_shannon_sim", "hellinger_sim", "total_variation_sim"]}
    
    metrics.update(sim_scores)
    avg_sim = np.mean(list(sim_scores.values()))
    metrics['average_similarity'] = avg_sim

    output_lines.append("\n--- 实体分布相似度 (0-1, 越高越好) ---")
    output_lines.append(f"Jensen-Shannon Similarity: {sim_scores['jensen_shannon_sim']:.4f}")
    output_lines.append(f"Hellinger Similarity:      {sim_scores['hellinger_sim']:.4f}")
    output_lines.append(f"Total Variation Similarity:{sim_scores['total_variation_sim']:.4f}")
    # 移除了 Wasserstein Similarity 的打印
    output_lines.append(f"-------------------------------------------")
    output_lines.append(f"平均相似度 (前三种):         {avg_sim:.4f}")
    
    return metrics, "\n".join(output_lines)

# --- 4. 主执行流程 ---
def main():
    """
    主函数：遍历所有实验，计算指标，进行统计分析，并保存完整报告。
    """
    all_results = []
    full_report_lines = ["="*20 + " 实体识别与分布相似度基准测试报告 " + "="*20]

    # --- 阶段一: 逐个计算 ---
    print("🚀 开始逐个计算每个实验文件夹的指标...")
    for exp_id in EXPERIMENT_IDS:
        print(f"   -> 正在处理实验ID: {exp_id}")
        metrics, report_str = process_single_experiment(exp_id)
        full_report_lines.append(report_str)
        if metrics:
            all_results.append(metrics)
    
    # --- 阶段二: 统计分析 ---
    print("\n📊 开始对所有结果进行统计分析...")
    if not all_results:
        summary_str = "未成功处理任何实验，无法生成统计摘要。"
        print(summary_str)
        full_report_lines.append(summary_str)
    else:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_results)} 个成功处理的实验结果")
        
        # 将结果从字典列表转换为指标字典的列表
        metric_keys = all_results[0].keys()
        metrics_data = {key: [res[key] for res in all_results] for key in metric_keys}
        
        summary_lines.append("\n--- 各项指标的平均值与方差 ---")
        header = f"{'Metric':<28} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        for key, values in metrics_data.items():
            mean_val = np.mean(values)
            var_val = np.var(values)
            summary_lines.append(f"{key:<28} | {mean_val:<12.4f} | {var_val:<12.4f}")
        
        full_report_lines.extend(summary_lines)
        print("   -> 统计分析完成。")

    # --- 阶段三: 保存报告 ---
    print(f"\n💾 正在将完整报告保存到文件...")
    try:
        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        os.makedirs(output_dir, exist_ok=True)
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"✅ 报告已成功保存至: {OUTPUT_FILE_PATH}")
    except IOError as e:
        print(f"❌ 错误: 无法写入报告文件。{e}")

if __name__ == "__main__":
    main()

