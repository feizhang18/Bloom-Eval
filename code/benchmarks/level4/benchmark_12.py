import json
import re
import os
import time
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from zss import Node
import zss
from typing import List, Dict, Any, Tuple

# --- 1. 全局配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'human'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

SBERT_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'

# --- 2. 核心功能函数 ---

def parse_to_tree(outline_data: List[List[Any]]) -> Node:
    """将 [[level, title], ...] 格式的扁平列表解析为 zss 树结构。"""
    root = Node("root")
    level_parents = {-1: root}
    for level, title in outline_data:
        clean_title = re.sub(r'^\d+(\.\d+)*\s*', '', title).strip()
        parent_node = level_parents.get(level - 1, root) # 找不到父节点则挂在根下
        new_node = Node(clean_title)
        parent_node.addkid(new_node)
        level_parents[level] = new_node
    return root

def get_all_topics(node: Node) -> List[str]:
    """递归获取树中所有节点的标签（主题）。"""
    topics = [node.label] if node.label != "root" else []
    for child in node.children:
        topics.extend(get_all_topics(child))
    return topics

def calculate_structural_similarity(tree_expert: Node, tree_llm: Node, model: SentenceTransformer) -> float:
    """使用带语义感知的树编辑距离计算结构相似度 (STS)。"""
    embedding_cache = {}
    def get_embedding(label):
        if label not in embedding_cache:
            embedding_cache[label] = model.encode(label, convert_to_tensor=True)
        return embedding_cache[label]
    
    def semantic_update_cost(node_a: Node, node_b: Node) -> float:
        emb_a = get_embedding(node_a.label)
        emb_b = get_embedding(node_b.label)
        similarity = util.cos_sim(emb_a, emb_b).item()
        return 1.0 - similarity

    nodes_expert = get_all_topics(tree_expert)
    nodes_llm = get_all_topics(tree_llm)
    if not nodes_expert or not nodes_llm: return 0.0

    distance = zss.distance(
        tree_expert, tree_llm, Node.get_children,
        insert_cost=lambda node: 1, remove_cost=lambda node: 1,
        update_cost=semantic_update_cost
    )
    max_dist = len(nodes_expert) + len(nodes_llm)
    return 1.0 - (distance / max_dist) if max_dist > 0 else 0

def get_tree_depth(node: Node) -> int:
    """计算树的最大深度。"""
    if not node.children: return 1
    return 1 + max(get_tree_depth(child) for child in node.children)

def calculate_shape_consistency(tree_expert: Node, tree_llm: Node) -> Dict[str, float]:
    """
    计算形状一致性 (ShapeCons)，包括子指标和最终总分。
    """
    depth_expert = get_tree_depth(tree_expert) - 1
    depth_llm = get_tree_depth(tree_llm) - 1
    count_expert = len(get_all_topics(tree_expert))
    count_llm = len(get_all_topics(tree_llm))
    
    # Depth Consistency (DC)
    dc = min(depth_expert, depth_llm) / max(depth_expert, depth_llm) if max(depth_expert, depth_llm) > 0 else 0
    # Breadth Consistency (BC)
    bc = min(count_expert, count_llm) / max(count_expert, count_llm) if max(count_expert, count_llm) > 0 else 0
    
    # ShapeCons Score (Geometric Mean)
    shape_cons_score = np.sqrt(dc * bc)
    
    return {
        "depth_consistency": dc, 
        "breadth_consistency": bc,
        "shape_consistency": shape_cons_score
    }


def process_single_experiment(exp_id: int, sbert_model: SentenceTransformer) -> Tuple[Dict, str]:
    """处理单个实验文件夹并返回指标和报告字符串。"""
    human_outline_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'outline.json')
    llm_outline_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'outline.json')

    scores = {}
    report_lines = []

    try:
        with open(human_outline_path, 'r', encoding='utf-8') as f: expert_data = json.load(f)
        with open(llm_outline_path, 'r', encoding='utf-8') as f: llm_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        error_msg = f"❌ 错误: 无法加载输入文件。{e}"
        return None, error_msg

    tree_expert = parse_to_tree(expert_data)
    tree_llm = parse_to_tree(llm_data)

    # --- 计算指标 ---
    
    # 1. Semantic Tree Similarity (STS)
    scores['semantic_tree_similarity'] = calculate_structural_similarity(tree_expert, tree_llm, sbert_model)
    
    # 2. Shape Consistency (ShapeCons)
    shape_scores = calculate_shape_consistency(tree_expert, tree_llm)
    scores.update(shape_scores)

    # --- 生成报告 ---
    report_lines.append(f"【指标一：语义树相似度 (STS)】: {scores['semantic_tree_similarity']:.4f}")
    report_lines.append(f"【指标二：形状一致性 (ShapeCons)】: {scores['shape_consistency']:.4f}")
    report_lines.append(f"    - (子指标) 深度一致性 (DC): {scores['depth_consistency']:.4f}")
    report_lines.append(f"    - (子指标) 广度一致性 (BC): {scores['breadth_consistency']:.4f}")
    
    return scores, "\n".join(report_lines)

# --- 3. 主执行流程 ---
def main():
    """主函数，负责遍历所有实验、计算指标、进行统计并生成报告。"""
    print("🚀 开始大纲结构质量批量评估任务 (STS & ShapeCons)...")
    all_results = []
    full_report_lines = ["="*25 + " 大纲结构质量基准测试报告 (STS & ShapeCons) " + "="*25]
    
    print("--- 正在初始化句向量模型 (这可能需要一些时间)... ---")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, trust_remote_code=True)
    # 为zss树的所有节点添加递归辅助函数
    def get_all_nodes(self):
        nodes = [self]
        for child in self.children: nodes.extend(get_all_nodes(child))
        return nodes
    Node.get_all_nodes = get_all_nodes
    print("--- 模型初始化完成 ---\n")

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        scores, report_str = process_single_experiment(exp_id, sbert_model)
        
        full_report_lines.append(f"\n{'='*25} 实验ID: {exp_id} 评估结果 {'='*25}")
        full_report_lines.append(report_str)
        print(report_str)

        if scores:
            all_results.append(scores)

    # --- 最终统计分析 ---
    if all_results:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_results)} 个实验结果的统计分析")
        
        # 确保 all_results 不为空
        if all_results:
            metric_keys = all_results[0].keys()
            summary_lines.append("\n--- 各项指标的平均值与方差 ---")
            header = f"{'Metric':<28} | {'Average':<12} | {'Variance':<12}"
            summary_lines.append(header)
            summary_lines.append("-" * len(header))
            
            for key in metric_keys:
                values = [res[key] for res in all_results]
                mean_val = np.mean(values)
                var_val = np.var(values)
                summary_lines.append(f"{key:<28} | {mean_val:<12.4f} | {var_val:<12.4f}")
        
        full_report_lines.extend(summary_lines)

    # --- 保存完整报告到文件 ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"\n\n✅ 完整报告已成功保存至: {OUTPUT_REPORT_PATH}")
    except IOError as e:
        print(f"❌ 错误: 无法写入报告文件。{e}")

    print(f"\n{'='*30} 🎉 所有任务已完成! {'='*30}")

if __name__ == "__main__":
    main()