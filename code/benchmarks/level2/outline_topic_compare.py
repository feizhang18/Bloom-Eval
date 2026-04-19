import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any

# --- 1. 配置区域 ---

# !!! 重要：请在此处填写您的有效API密钥
API_KEY = os.getenv("OPENAI_API_KEY", "") # 替换为您的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") # 或您偏好的 API 端点

# 集中化配置管理
CONFIG = {
    "model": "gpt-5-mini",
    "root_output_dir": "<EXPERIMENT_ROOT>/1/autosurvey/bench/level2/llm_outputs_outline_matching",
    "base_filename": "<EXPERIMENT_ROOT>/1/autosurvey/bench/level2/outline_matching_result"
}

# [--- 文件路径配置 ---]
# 脚本将会在当前目录下寻找这两个文件
EXPERT_OUTLINE_PATH = '<EXPERIMENT_ROOT>/1/human/outline.json'
LLM_OUTLINE_PATH = '<EXPERIMENT_ROOT>/1/autosurvey/outline.json'
# [---------------------]


# --- 2. 核心API调用函数 ---

def get_llm_response(client: OpenAI, message: List[Dict], query_id: str, config: Dict) -> str:
    """
    从LLM获取响应，保存推理过程和最终答案。
    """
    model = config['model']
    root_output_dir = config['root_output_dir']
    answer_dir = os.path.join(root_output_dir, "answer")
    os.makedirs(answer_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    identifier = f"{timestamp}_{query_id}"
    answer_file = os.path.join(answer_dir, f"{config['base_filename']}_answer_{identifier}.json")

    print(f"\n{'='*20} 正在执行查询: {query_id} {'='*20}\n")

    try:
        # 对于此任务，一次性返回JSON的非流式调用更直接
        response = client.chat.completions.create(
            model=model,
            messages=message,
            temperature=0.0,
            response_format={"type": "json_object"}, # 强制LLM输出JSON
        )

        answer_content = response.choices[0].message.content
        print("--- LLM 返回的匹配结果 (JSON) ---")
        print(answer_content)
        
        # 保存答案
        try:
            parsed_json = json.loads(answer_content)
            with open(answer_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 匹配结果已保存至: {answer_file}")
        except json.JSONDecodeError:
            answer_file_txt = answer_file.replace('.json', '.txt')
            with open(answer_file_txt, 'w', encoding='utf-8') as f:
                f.write(answer_content)
            print(f"\n⚠️ 警告: LLM返回的不是有效的JSON，已作为文本保存至 {answer_file_txt}")

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
                # 智能清理标题：移除开头的数字和点、空格
                clean_title = re.sub(r'^\d+(\.\d+)*\s*', '', title).strip()
                topics.append(clean_title)
        return topics
    except Exception as e:
        print(f"❌ 读取或解析文件 {file_path} 时出错: {e}")
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

# --- 4. 主执行流程与报告生成 ---

def main():
    if not API_KEY or "xxxxxxxx" in API_KEY:
        print("❌ 错误: 请在代码第13行设置您的有效API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 步骤 1: 加载并准备大纲数据
    print("--- 步骤 1: 正在加载并解析大纲文件 ---")
    expert_topics = load_and_flatten_outline(EXPERT_OUTLINE_PATH)
    llm_topics = load_and_flatten_outline(LLM_OUTLINE_PATH)

    if not expert_topics or not llm_topics:
        print("❌ 无法加载一个或两个大纲文件，程序终止。")
        return
        
    print(f"✅ 成功加载 {len(expert_topics)} 个专家主题和 {len(llm_topics)} 个LLM主题。")

    # 步骤 2: 构建Prompt并调用LLM
    print("\n--- 步骤 2: 正在调用LLM进行主题匹配 ---")
    prompt = build_outline_matching_prompt(expert_topics, llm_topics)
    messages = [{"role": "user", "content": prompt}]
    
    response_json_str = get_llm_response(client, messages, "outline_matching", CONFIG)
    
    try:
        match_data = json.loads(response_json_str)
        # 确保 "matched_pairs" 键存在且其值为列表
        matched_pairs = match_data.get("matched_pairs", [])
        if not isinstance(matched_pairs, list):
            print(f"❌ 错误: LLM返回的 'matched_pairs' 不是一个列表。")
            matched_pairs = []
    except (json.JSONDecodeError, AttributeError):
        print("\n❌ 错误: 无法解析LLM的响应。无法计算指标。")
        return

    # 步骤 3: 基于LLM的响应计算指标
    print("\n--- 步骤 3: 正在计算覆盖率指标 ---")
    tp = len(matched_pairs)
    fp = len(llm_topics) - tp
    fn = len(expert_topics) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 步骤 4: 显示最终报告
    print("\n\n" + "="*25 + " 大纲主题覆盖率报告 " + "="*25)
    
    print("\n【✅ LLM 匹配成功的标题对】")
    if matched_pairs:
        for i, pair in enumerate(matched_pairs, 1):
            print(f"  {i}. 专家: '{pair.get('expert_heading', 'N/A')}' <=> LLM: '{pair.get('llm_heading', 'N/A')}'")
    else:
        print("  LLM未能匹配任何主题。")

    # 找出并显示未匹配的主题以供诊断
    matched_llm_topics = {pair['llm_heading'] for pair in matched_pairs}
    unmatched_llm_topics = [topic for topic in llm_topics if topic not in matched_llm_topics]
    
    matched_expert_topics = {pair['expert_heading'] for pair in matched_pairs}
    unmatched_expert_topics = [topic for topic in expert_topics if topic not in matched_expert_topics]

    print("\n【❌ AI提出但专家未提及的主题 (FP)】")
    if unmatched_llm_topics:
        for topic in unmatched_llm_topics:
            print(f"  - {topic}")
    else:
        print("  无。")

    print("\n【❌ AI遗漏的专家主题 (FN)】")
    if unmatched_expert_topics:
        for topic in unmatched_expert_topics:
            print(f"  - {topic}")
    else:
        print("  无。")
        
    print("\n" + "-"*30 + " 最终分数 " + "-"*30)
    print(f"  - 真正例 (TP - 匹配上的主题数):    {tp}")
    print(f"  - 假正例 (FP - AI多出来的主题数):   {fp}")
    print(f"  - 假负例 (FN - AI遗漏的主题数):   {fn}")
    print("-" * 74)
    print(f"  - 精确率 (Precision):             {precision:.2%}")
    print(f"  - 召回率 (Recall):                {recall:.2%}")
    print(f"  - F1-Score:                       {f1_score:.2%}")
    print("="*75)

if __name__ == "__main__":
    import re
    main()