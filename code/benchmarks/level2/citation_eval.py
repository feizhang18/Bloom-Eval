import os
import csv
import time
import threading
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 替换为您的 agicto API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")            # 您的 agicto API 端点
LLM_MODEL = "gpt-4o-2024-08-06"
MAX_CONCURRENT_THREADS = 10

# --- 批量处理与报告配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'human'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# --- 2. 提示词定义 ---
NLI_PROMPT_TEMPLATE = """
---
Claim:
[CLAIM]
---
Source:
[SOURCE]
---
Is the Claim faithful to the Source?
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.
Only reply with 'Yes' or 'No':
"""

# --- 3. 核心代码实现 (已重构) ---

class CitationEvaluator:
    """
    一个用于评估引文质量的类。
    """
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.lock = threading.Lock()
        self.successful_requests = 0
        self.failed_requests = 0

    def _get_llm_response(self, claim: str, source: str, result_list: list, index: int):
        """
        向大模型发送单个请求并获取'Yes'或'No'的判断。
        """
        prompt = NLI_PROMPT_TEMPLATE.replace('[CLAIM]', claim).replace('[SOURCE]', source)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            answer = response.choices[0].message.content.strip().lower()
            result_list[index] = 1 if 'yes' in answer else 0
            with self.lock:
                self.successful_requests += 1
        except Exception as e:
            result_list[index] = 0
            with self.lock:
                self.failed_requests += 1

    def run_initial_nli_evaluation(self, data_rows: list) -> List[Dict]:
        """
        对CSV的每一行执行初始的 "句子-单个引用" NLI判断。
        """
        num_rows = len(data_rows)
        if num_rows == 0:
            return []

        threads = []
        scores = [0] * num_rows
        
        with tqdm(total=num_rows, desc="Running Initial NLI Evaluation") as pbar:
            for i, row in enumerate(data_rows):
                claim = row.get('sentence', '')
                source = row.get('abstract', '')

                if not claim or not source:
                    scores[i] = 0
                    pbar.update(1)
                    continue
                
                while len([t for t in threads if t.is_alive()]) >= MAX_CONCURRENT_THREADS:
                    time.sleep(0.1)

                thread = threading.Thread(target=self._get_llm_response, args=(claim, source, scores, i))
                threads.append(thread)
                thread.start()
                pbar.update(1)

            for thread in threads:
                thread.join()

        detailed_results = []
        for i, row in enumerate(data_rows):
            new_row = row.copy()
            new_row['is_supported'] = scores[i]
            detailed_results.append(new_row)
            
        print(f"\nInitial NLI API Requests - Success: {self.successful_requests}, Failed: {self.failed_requests}")
        return detailed_results

# --- 4. 最终指标计算函数 (核心修改) ---

def calculate_final_metrics_from_csv(csv_path: str) -> Dict:
    """
    从预计算的CSV文件中，使用优化的逻辑正确计算召回率和精确率，无需额外API调用。
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {"recall": 0.0, "precision": 0.0, "f1_score": 0.0}
        
        # 确保关键列存在且类型正确
        df['sentence_id'] = pd.to_numeric(df['sentence_id'], errors='coerce')
        df['is_supported'] = pd.to_numeric(df['is_supported'], errors='coerce').fillna(0).astype(int)
        df = df.dropna(subset=['sentence_id'])
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到指标计算所需的CSV文件 '{csv_path}'")
        return {}
    except Exception as e:
        print(f"❌ 读取或处理CSV时出错: {e}")
        return {}

    # 按 sentence_id 分组
    grouped = df.groupby('sentence_id')
    
    # --- 4.1 正确计算召回率 (Recall) ---
    total_claims = len(grouped)
    supported_claims_count = 0
    for _, group in grouped:
        if group['is_supported'].any():
            supported_claims_count += 1
    
    recall = supported_claims_count / total_claims if total_claims > 0 else 0.0

    # --- 4.2 正确计算精确率 (Precision) - 优化版 ---
    precise_references_count = 0
    total_references_count = len(df)

    # 筛选出所有被支持的行，并计算它们在各自句子分组中的支持总数
    supported_rows = df[df['is_supported'] == 1].copy()
    if not supported_rows.empty:
        # transform会返回一个与原始DataFrame同样大小的Series，值为每个组的计算结果
        support_counts_in_group = grouped['is_supported'].transform('sum')
        supported_rows['support_counts_in_group'] = support_counts_in_group[supported_rows.index]
        
        # 如果一个引用被支持(is_supported=1)，且它所在句子的支持总数也为1，那么它就是精确的
        precise_references_count = (supported_rows['support_counts_in_group'] == 1).sum()

    precision = precise_references_count / total_references_count if total_references_count > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"recall": recall, "precision": precision, "f1_score": f1_score}


# --- 5. 主执行流程 (已修改) ---

def process_single_experiment(exp_id: int, evaluator: CitationEvaluator) -> Dict:
    """
    处理单个实验文件夹的核心逻辑。
    """
    base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME)
    input_csv_path = os.path.join(base_dir, 'sentences_grouped_for_precision_eval.csv')
    # 这是存储初始NLI判断结果的中间文件
    nli_results_csv_path = os.path.join(base_dir, 'single_citation_evaluation_results.csv')

    # 步骤 A: 确保初始NLI结果文件存在
    if not os.path.exists(nli_results_csv_path):
        print(f"🔍 未找到预计算的NLI结果, 将运行初始评估...")
        if not os.path.exists(input_csv_path):
            print(f"❌ 错误: 找不到输入文件 '{input_csv_path}'。")
            return {}
        try:
            with open(input_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data_rows = list(reader)
            print(f"✅ 从 '{os.path.basename(input_csv_path)}' 读取 {len(data_rows)} 行数据。")
        except Exception as e:
            print(f"❌ 读取输入CSV时出错: {e}")
            return {}
        
        # 运行初始NLI判断
        detailed_results = evaluator.run_initial_nli_evaluation(data_rows)
        
        # 保存结果以备将来使用
        if detailed_results:
            try:
                with open(nli_results_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
                    writer.writeheader()
                    writer.writerows(detailed_results)
                print(f"✅ 初始NLI结果已保存至: '{os.path.basename(nli_results_csv_path)}'")
            except Exception as e:
                print(f"❌ 保存NLI结果时出错: {e}")
                return {}
    else:
        print(f"⏭️  跳过API调用: 找到已存在的NLI结果文件 '{os.path.basename(nli_results_csv_path)}'。")

    # 步骤 B: 从NLI结果文件计算最终指标
    print(f"📊 开始根据 '{os.path.basename(nli_results_csv_path)}' 计算最终指标...")
    final_metrics = calculate_final_metrics_from_csv(nli_results_csv_path)
    
    return final_metrics


def main():
    """
    主函数，负责遍历所有实验文件夹、调用处理函数、进行统计并生成报告。
    """
    print(f"🚀 开始引文质量批量评估任务 (使用最终优化版计算逻辑)...")
    all_results = []
    full_report_lines = ["="*25 + " 引文质量基准测试报告 (最终优化版) " + "="*25]
    
    evaluator = CitationEvaluator(api_key=API_KEY, base_url=BASE_URL, model=LLM_MODEL)

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        metrics = process_single_experiment(exp_id, evaluator)
        
        if metrics:
            all_results.append(metrics)
            individual_report = f"""
{'='*25} 实验ID: {exp_id} 评估结果 {'='*25}
引文召回率 (Citation Recall):      {metrics['recall']:.4f}
引文精确率 (Citation Precision):    {metrics['precision']:.4f}
F1 分数 (F1-Score):              {metrics['f1_score']:.4f}
"""
            full_report_lines.append(individual_report)
            print(individual_report)
    
    # --- 最终统计分析 ---
    if all_results:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_results)} 个有效实验结果的统计分析")
        
        metric_keys = ["recall", "precision", "f1_score"]
        summary_lines.append("\n--- 各项指标的平均值与标准差 ---")
        header = f"{'Metric':<12} | {'Average':<12} | {'Std Dev':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * (len(header) + 2))
        
        for key in metric_keys:
            values = [res[key] for res in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary_lines.append(f"{key:<12} | {mean_val:<12.4f} | {std_val:<12.4f}")
        
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