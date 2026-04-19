import json
from typing import Dict, Any

# --- 1. 配置区域 ---
# 请在此处修改为您的实际文件路径

# 专家编写的原始大纲文件路径
EXPERT_OUTLINE_PATH = '<EXPERIMENT_ROOT>/1/human/outline.json'

# AI生成的原始大纲文件路径
LLM_OUTLINE_PATH = '<EXPERIMENT_ROOT>/1/surveyx/outline.json'

# 包含LLM匹配结果的JSON文件路径
MATCHED_PAIRS_PATH = '<EXPERIMENT_ROOT>/1/surveyx/bench/level2/compare_outline_topic.json'

# --- 2. 核心功能函数 ---

def count_topics_from_outline(file_path: str) -> int:
    """
    从原始大纲JSON文件中读取并计算标题的总数。
    假定输入格式为: [[level, title], [level, title], ...]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                # 标题总数就是列表的长度
                return len(data)
            else:
                print(f"⚠️ 警告: 文件 {file_path} 的格式不是预期的列表格式。")
                return 0
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {file_path} 不是一个有效的JSON文件。")
        return 0
    except Exception as e:
        print(f"❌ 读取文件 {file_path} 时发生未知错误: {e}")
        return 0

def count_matched_pairs(file_path: str) -> int:
    """
    从匹配结果JSON文件中读取并计算成功匹配的标题对数量 (TP)。
    假定输入格式为: {"matched_pairs": [...]}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 使用.get()安全地获取键值，如果键不存在则返回空列表
            matched_pairs = data.get("matched_pairs", [])
            if isinstance(matched_pairs, list):
                # 成功匹配的数量就是列表中元素的数量
                return len(matched_pairs)
            else:
                print(f"⚠️ 警告: 文件 {file_path} 中 'matched_pairs' 的值不是列表。")
                return 0
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {file_path} 不是一个有效的JSON文件。")
        return 0
    except Exception as e:
        print(f"❌ 读取文件 {file_path} 时发生未知错误: {e}")
        return 0

# --- 3. 主执行流程 ---

def main():
    """
    主函数，执行所有计算和报告生成步骤。
    """
    print("--- 开始计算大纲覆盖率指标 ---")

    # 步骤 1: 从各文件获取基础计数值
    total_expert_topics = count_topics_from_outline(EXPERT_OUTLINE_PATH)
    total_llm_topics = count_topics_from_outline(LLM_OUTLINE_PATH)
    tp = count_matched_pairs(MATCHED_PAIRS_PATH) # True Positives (真正例)

    # 如果任何一个文件读取失败，则终止程序
    if total_expert_topics == 0 or total_llm_topics == 0:
        print("\n--- 由于无法读取原始大纲文件，计算终止 ---")
        return

    # 步骤 2: 计算FP和FN
    # False Positives (假正例): AI大纲中有，但匹配不上的标题
    fp = total_llm_topics - tp
    # False Negatives (假负例): 专家大纲中有，但AI未能匹配的标题
    fn = total_expert_topics - tp

    # 步骤 3: 计算核心指标 (Precision, Recall, F1-Score)
    # 精确率: 匹配上的 / AI生成的所有标题
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # 召回率: 匹配上的 / 专家定义的所有标题
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1分数: 精确率和召回率的调和平均数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 步骤 4: 打印格式化的报告
    print("\n" + "="*25 + " 评估报告 " + "="*25)
    print(f"\n【基础数据】")
    print(f"  - 专家大纲中的总标题数: {total_expert_topics}")
    print(f"  - AI大纲中的总标题数:   {total_llm_topics}")
    print("-" * 62)
    print(f"  - 匹配成功的标题对 (TP): {tp}")
    print(f"  - AI多出的标题 (FP):     {fp}")
    print(f"  - AI遗漏的标题 (FN):     {fn}")

    print("\n【最终指标分数】")
    print(f"  - 🎯 精确率 (Precision): {precision:.2%}")
    print(f"  - 🎯 召回率 (Recall):    {recall:.2%}")
    print(f"  - 🏆 F1-Score:          {f1_score:.2%}")
    print("="*62)
    print("\n--- 计算完成 ---")

if __name__ == "__main__":
    main()
