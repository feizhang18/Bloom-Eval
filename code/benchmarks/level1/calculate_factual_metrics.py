import json
import os

# --- 1. 配置文件路径 ---
# 请确保这些路径指向你的实际文件

# 输入文件
EXPERT_JSON_PATH = '<EXPERIMENT_ROOT>/1/human/bench/level1/factual_claim.json'
LLM_JSON_PATH = '<EXPERIMENT_ROOT>/1/surveyx/bench/level1/factual_claim.json'
MATCHED_JSON_PATH = '<EXPERIMENT_ROOT>/1/surveyx/bench/level1/factual_claims_match.json'  # 这是由上一步大模型匹配后生成的JSON文件

# --- 2. 辅助函数：安全地加载JSON文件 ---

def load_json_data(file_path: str) -> dict:
    """
    从指定路径加载JSON文件，并处理可能发生的错误。
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确，无法解析JSON -> {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生未知错误 -> {file_path}: {e}")
        return None

# --- 3. 主计算与打印函数 ---

def calculate_and_print_metrics(expert_path: str, llm_path: str, matched_path: str):
    """
    根据输入的三个JSON文件路径，计算并打印事实一致性指标。
    """
    # 加载所有需要的数据
    expert_data = load_json_data(expert_path)
    llm_data = load_json_data(llm_path)
    matched_data = load_json_data(matched_path)

    # 检查文件是否成功加载
    if expert_data is None or llm_data is None or matched_data is None:
        print("\n由于文件加载失败，计算已终止。")
        return

    # 从加载的数据中提取所需的信息
    try:
        expert_statements = expert_data["factual_statements"]
        llm_statements = llm_data["factual_statements"]
        matched_pairs = matched_data["matched_pairs"]
    except KeyError as e:
        print(f"\n错误: JSON文件中缺少预期的键: {e}。请检查文件内容是否正确。")
        return

    # 获取各项计数
    total_expert_statements = len(expert_statements)
    total_llm_statements = len(llm_statements)
    
    # 计算 TP, FP, FN
    # TP (True Positives): 匹配成功的数量
    tp = len(matched_pairs)
    
    # FP (False Positives): LLM提出但未匹配成功的数量 (幻觉或不相关)
    fp = total_llm_statements - tp
    
    # FN (False Negatives): 专家提出但LLM未能匹配的数量 (遗漏)
    fn = total_expert_statements - tp

    # 计算精确率, 召回率, F1-Score，并处理分母为0的情况
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 打印最终的评估报告
    print("\n\n" + "="*20 + " 事实性陈述评估报告 " + "="*20)
    print(f"专家事实陈述总数: {total_expert_statements}")
    print(f"LLM事实陈述总数:  {total_llm_statements}")
    print("-" * 55)
    print(f"TP (真正例 / 匹配成功数): {tp}")
    print(f"FP (假正例 / 幻觉或不相关数): {fp}")
    print(f"FN (假负例 / 遗漏数): {fn}")
    print("-" * 55)
    print("评估指标:")
    print(f"  -> 精确率 (Precision): {precision:.4f}  ({precision:.2%})")
    print(f"  -> 召回率 (Recall):    {recall:.4f}  ({recall:.2%})")
    print(f"  -> F1-Score:          {f1_score:.4f}  ({f1_score:.2%})")
    print("=" * 55)
    print("\n指标解读:")
    print("  - 精确率: 在LLM提取的所有事实中，正确的比例有多高？(越高，幻觉越少)")
    print("  - 召回率: 在所有专家认定的事实中，LLM覆盖了多少？(越高，遗漏越少)")
    print("  - F1-Score: 精确率和召回率的综合平衡分数。")
    print("\n")


# --- 4. 执行入口 ---
if __name__ == "__main__":
    calculate_and_print_metrics(
        expert_path=EXPERT_JSON_PATH,
        llm_path=LLM_JSON_PATH,
        matched_path=MATCHED_JSON_PATH
    )