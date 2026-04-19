import json
import re
import pandas as pd
import os
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict
from umap import UMAP  # <--- 1. 添加 umap 库的导入

# --- 1. 全局配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'surveyx'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# --- 模型配置 ---
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1" # 使用与之前脚本一致的嵌入模型
RANDOM_SEED = 42  # <--- 2. 定义一个全局随机种子以确保可复现性

# --- 2. 核心功能函数 ---

def calculate_gini(counts: List[int]) -> float:
    """
    根据一个代表分布的列表来计算基尼系数。
    """
    counts = np.array(counts, dtype=np.float64)
    if counts.size == 0: return 1.0  # 没有主题，视为极度集中
    counts = np.sort(counts)
    n = len(counts)
    cum_counts = np.cumsum(counts)
    total_sum = cum_counts[-1]
    if total_sum == 0: return 0.0  # 所有主题都为空
    B = np.sum(cum_counts) / total_sum
    gini = 1 - 2 * B / n + 1 / n
    return gini

def load_and_prepare_docs(file_path: str) -> List[str]:
    """
    从指定的JSON文件加载参考文献，并将标题和摘要合并为文档列表。
    """
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        documents = []
        for key, value in data.items():
            if key.startswith('paper_') and isinstance(value, dict):
                inner_ref = next(iter(value.values()), None)
                if inner_ref and isinstance(inner_ref, dict):
                    title = inner_ref.get('searched_title', '')
                    abstract = inner_ref.get('abs', '')
                    if title and abstract and abstract.strip().upper() != 'N/A':
                        full_text = title + ". " + abstract
                        documents.append(re.sub(r'\s+', ' ', full_text).strip())
        return documents
    except (json.JSONDecodeError, AttributeError):
        return []

def analyze_survey_topics(survey_name: str, documents: List[str], embedding_model: SentenceTransformer) -> Dict:
    """
    使用BERTopic对给定的文档列表进行主题建模和分析。
    """
    print(f"--- 正在为 '{survey_name}' 进行主题发现... ---")
    if len(documents) < 5:
        print(f"文献数量过少 ({len(documents)}篇)，跳过主题建模。\n")
        return {
            "Survey": survey_name, "文献总数": len(documents), "发现主题数": 0,
            "Gini系数": 1.0, "广度得分 (1-Gini)": 0.0
        }
    try:
        # <--- 3. 修改此函数以确保BERTopic结果可复现 ---
        # a. 创建一个带有固定随机种子的UMAP模型实例
        umap_model = UMAP(n_neighbors=5, 
                          n_components=5, 
                          min_dist=0.0, 
                          metric='cosine', 
                          random_state=RANDOM_SEED)

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        
        # b. 将配置好的umap_model传递给BERTopic
        topic_model = BERTopic(
            language="english", min_topic_size=2, verbose=False,
            embedding_model=embedding_model, 
            vectorizer_model=vectorizer_model,
            umap_model=umap_model
        )
        # ---------------------------------------------------

        topics, _ = topic_model.fit_transform(documents)
        print("主题建模完成！")
        
        topic_info = topic_model.get_topic_info()
        topic_counts = topic_info[topic_info['Topic'] != -1]['Count'].tolist()
        num_topics = len(topic_counts)
        gini = calculate_gini(topic_counts)
        breadth_score = 1 - gini
        
        return {
            "Survey": survey_name, "文献总数": len(documents), "发现主题数": num_topics,
            "Gini系数": gini, "广度得分 (1-Gini)": breadth_score
        }
    except Exception as e:
        print(f"❌ 主题建模过程中发生错误: {e}")
        return None

# --- 3. 主执行流程 ---
def main():
    print("🚀 开始参考文献主题广度批量评估任务...")
    all_human_results = []
    all_llm_results = []
    full_report_lines = ["="*25 + " 参考文献主题广度基准测试报告 " + "="*25]
    
    # 预加载嵌入模型以节省时间
    print("正在预加载嵌入模型...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    print("模型加载完成。")

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        human_refs_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'reference_3.json')
        llm_refs_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'reference_3.json')

        human_docs = load_and_prepare_docs(human_refs_path)
        llm_docs = load_and_prepare_docs(llm_refs_path)

        if not human_docs or not llm_docs:
            print("❌ 错误: 缺少一个或多个输入文件中的有效参考文献。")
            if not human_docs: print(f"   - 检查: {human_refs_path}")
            if not llm_docs: print(f"   - 检查: {llm_refs_path}")
            continue

        human_result = analyze_survey_topics(f"ID:{exp_id} (Human)", human_docs, embedding_model)
        llm_result = analyze_survey_topics(f"ID:{exp_id} (LLM)", llm_docs, embedding_model)

        if human_result and llm_result:
            all_human_results.append(human_result)
            all_llm_results.append(llm_result)
            
            individual_report = f"""
{'='*25} 实验ID: {exp_id} 主题广度报告 {'='*25}
Human (专家):
  - 发现主题数: {human_result['发现主题数']}
  - 广度得分 (1-Gini): {human_result['广度得分 (1-Gini)']:.4f}
LLM ({METHOD_NAME}):
  - 发现主题数: {llm_result['发现主题数']}
  - 广度得分 (1-Gini): {llm_result['广度得分 (1-Gini)']:.4f}
"""
            full_report_lines.append(individual_report)
            print(individual_report)

    # --- 最终统计分析 ---
    if all_human_results and all_llm_results:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_human_results)} 个实验结果的统计分析")
        
        human_scores = [res['广度得分 (1-Gini)'] for res in all_human_results]
        llm_scores = [res['广度得分 (1-Gini)'] for res in all_llm_results]
        
        summary_lines.append("\n--- 广度得分 (1-Gini) 的平均值与方差 ---")
        header = f"{'Method':<12} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        summary_lines.append(f"{'Human':<12} | {np.mean(human_scores):<12.4f} | {np.var(human_scores):<12.4f}")
        summary_lines.append(f"{METHOD_NAME:<12} | {np.mean(llm_scores):<12.4f} | {np.var(llm_scores):<12.4f}")
        
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