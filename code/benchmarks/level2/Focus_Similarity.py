import json
import re
import os
import time
from openai import OpenAI
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, List, Any
from umap import UMAP

# --- 1. 全局配置 (保持不变) ---
API_KEY = os.getenv("OPENAI_API_KEY", "") # 替换为您的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)
METHOD_NAME = 'LLMxMapReduce_V2'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'
LLM_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
RANDOM_SEED = 42

# --- 2. 辅助函数 (保持不变) ---
def calculate_ds_score(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    hellinger_dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    hellinger_sim = 1 - hellinger_dist
    total_variation_dist = np.sum(np.abs(p - q)) / 2
    total_variation_sim = 1 - total_variation_dist
    m = 0.5 * (p + q)
    js_div = 0.5 * (entropy(p, m) + entropy(q, m)) 
    js_div_normalized = js_div / np.log(2)
    js_sim = 1 - js_div_normalized
    ds_score = (hellinger_sim + total_variation_sim + js_sim) / 3
    return ds_score

def get_llm_response(client: OpenAI, message: List[Dict], query_id: str, final_output_path: str, log_output_dir: str) -> str:
    answer_dir = os.path.join(log_output_dir, "answer_logs")
    os.makedirs(answer_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    identifier = f"{query_id}_{timestamp}"
    answer_log_file = os.path.join(answer_dir, f"answer_{identifier}.json")
    print(f"\n{'='*20} 正在执行LLM查询: {query_id} {'='*20}")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=message, temperature=0.0, response_format={"type": "json_object"}
        )
        answer_content = response.choices[0].message.content
        print("--- LLM返回了有效的JSON响应 ---")
        parsed_json = json.loads(answer_content)
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        with open(answer_log_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        print(f"✅ LLM响应已保存至: {final_output_path}")
        return answer_content
    except Exception as e:
        print(f"❌ API调用过程中发生错误: {e}")
        return "{}"

def load_and_prepare_docs(file_path: str) -> List[str]:
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        documents = [
            re.sub(r'\s+', ' ', (v_inner.get('searched_title', '') + ". " + v_inner.get('abs', ''))).strip()
            for k, v in data.items() if k.startswith('paper_') and isinstance(v, dict)
            for v_inner in v.values() if isinstance(v_inner, dict) and v_inner.get('searched_title') and v_inner.get('abs', '').strip().upper() != 'N/A'
        ]
        return documents
    except (json.JSONDecodeError, AttributeError): return []

def discover_topics_and_freqs(documents: List[str], embedding_model: SentenceTransformer) -> pd.DataFrame:
    if len(documents) < 5: return pd.DataFrame(columns=['Topic', 'Name', 'Count'])
    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=RANDOM_SEED)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
    topic_model = BERTopic(language="english", min_topic_size=2, verbose=False, embedding_model=embedding_model, vectorizer_model=vectorizer_model, umap_model=umap_model)
    _, _ = topic_model.fit_transform(documents)
    topic_info = topic_model.get_topic_info()
    return topic_info[topic_info['Topic'] != -1]

def match_topics_with_llm(client: OpenAI, expert_topics: List[str], llm_topics: List[str], query_id: str, final_output_path: str, log_output_dir: str) -> Dict:
    """使用LLM对两组主题进行匹配。"""
    prompt = f"""
# ROLE & OBJECTIVE
You are a highly specialized academic research assistant. Your sole objective is to analyze two lists of research topics, `EXPERT_TOPICS` and `LLM_TOPICS`, and identify all pairs that are semantically equivalent.

# TASK DEFINITION
- A topic from `LLM_TOPICS` is considered a match to a topic from `EXPERT_TOPICS` if they describe the same core research area, methodology, or concept.
- The matching should be robust to differences in wording, ordering of keywords, and minor variations in scope.
- One expert topic can be matched by multiple LLM topics if they all refer to the same concept. Similarly, one LLM topic can match multiple expert topics.

# EXAMPLES OF STRONG MATCHES
- Expert: "0_rendering_quality_image" vs. LLM: "1_nerf_rendering_scenes" -> This is a strong match as both relate to rendering in computer graphics.
- Expert: "2_optimization_training_speed" vs. LLM: "0_efficient_training_models" -> This is a strong match as both relate to efficient model training.

# INPUT DATA
### EXPERT_TOPICS
{json.dumps(expert_topics, indent=2)}

### LLM_TOPICS
{json.dumps(llm_topics, indent=2)}

# OUTPUT RULES (VERY IMPORTANT)
- Your entire response MUST be a single, valid JSON object.
- The JSON object must have one top-level key: `"matched_pairs"`.
- The value of `"matched_pairs"` must be a list of JSON objects.
- Each object in the list represents a single valid match and must contain exactly two keys: `"expert_topic"` and `"llm_topic"`.
- If you find no semantically equivalent pairs, you MUST return an empty list for the `"matched_pairs"` key, like this: `{{"matched_pairs": []}}`.
- Do NOT include any explanations, apologies, or text outside of the final JSON object.
"""
    messages = [{"role": "user", "content": prompt}]
    response_str = get_llm_response(client, messages, query_id, final_output_path, log_output_dir)
    try:
        return json.loads(response_str)
    except (json.JSONDecodeError, AttributeError):
        return {"matched_pairs": []}

# --- 3. 主执行流程 (已重构) ---
def main():
    print("🚀 开始参考文献主题覆盖度批量评估任务...")
    all_results = []
    full_report_lines = ["="*25 + " 参考文献主题覆盖度基准测试报告 " + "="*25]
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("正在预加载嵌入模型...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    print("模型加载完成。")

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        # 1. 定义路径
        human_refs_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'reference_3.json')
        llm_refs_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'reference_3.json')
        output_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'bench', 'level2')
        matches_path = os.path.join(output_dir, 'matches.json')
        log_dir = os.path.join(output_dir, 'llm_outputs_ref_topic_matching')
        os.makedirs(output_dir, exist_ok=True)

        # 2. 本地计算：加载文档并进行主题建模 (每次都运行以获取频率)
        print("--- 正在加载文档并运行BERTopic进行本地主题分析...")
        human_docs = load_and_prepare_docs(human_refs_path)
        llm_docs = load_and_prepare_docs(llm_refs_path)

        if not human_docs or not llm_docs:
            print(f"❌ 错误: ID {exp_id} 缺少有效参考文献，跳过。")
            continue

        expert_topic_info = discover_topics_and_freqs(human_docs, embedding_model)
        llm_topic_info = discover_topics_and_freqs(llm_docs, embedding_model)
        
        expert_topics = expert_topic_info['Name'].tolist()
        llm_topics = llm_topic_info['Name'].tolist()
        print(f"--- 本地分析完成: 发现 {len(expert_topics)} 个专家主题, {len(llm_topics)} 个生成主题。")

        # 3. 智能跳过API调用：检查 matches.json 是否存在
        if os.path.exists(matches_path):
            print(f"✅ API调用跳过: 'matches.json' 文件已存在, 直接加载。")
            with open(matches_path, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
        else:
            if not expert_topics or not llm_topics:
                print("警告: 一方或双方未能发现足够的主题，无法调用API进行比较。")
                match_data = {"matched_pairs": []}
            else:
                query_id = f"exp{exp_id}_ref_topic_matching"
                match_data = match_topics_with_llm(client, expert_topics, llm_topics, query_id, matches_path, log_dir)

        # 4. 计算所有指标 (现在总能获取到频率信息)
        # F1 Score 计算
        matched_llm_topics = [pair['llm_topic'] for pair in match_data.get("matched_pairs", [])]
        tp = len(set(matched_llm_topics))
        fp = len(llm_topics) - tp
        matched_expert_topics = [pair['expert_topic'] for pair in match_data.get("matched_pairs", [])]
        fn = len(set(expert_topics) - set(matched_expert_topics))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # DS Score 计算
        ds_score = 0.0
        matched_pairs = match_data.get("matched_pairs", [])
        if matched_pairs and not expert_topic_info.empty and not llm_topic_info.empty:
            expert_counts_map = pd.Series(expert_topic_info.Count.values, index=expert_topic_info.Name).to_dict()
            llm_counts_map = pd.Series(llm_topic_info.Count.values, index=llm_topic_info.Name).to_dict()
            common_expert_topics = sorted(list(set(matched_expert_topics)))
            if common_expert_topics:
                p_counts = [expert_counts_map.get(et, 0) for et in common_expert_topics]
                q_counts = [sum(llm_counts_map.get(pair['llm_topic'], 0) for pair in matched_pairs if pair['expert_topic'] == et) for et in common_expert_topics]
                if sum(p_counts) > 0 and sum(q_counts) > 0:
                    ds_score = calculate_ds_score(np.array(p_counts), np.array(q_counts))

        # 5. 记录和报告结果
        all_results.append({'precision': precision, 'recall': recall, 'f1_score': f1_score, 'ds_score': ds_score})
        individual_report = f"""
{'='*25} 实验ID: {exp_id} 主题覆盖度报告 {'='*25}
准确率 (Precision): {precision:.4f}
召回率 (Recall):     {recall:.4f}
F1-Score:           {f1_score:.4f}
DS-Score:           {ds_score:.4f}
"""
        full_report_lines.append(individual_report)
        print(individual_report.strip())

    # --- 最终统计与保存 (保持不变) ---
    if all_results:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_results)} 个实验结果的统计分析")
        metric_keys = ["precision", "recall", "f1_score", "ds_score"]
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