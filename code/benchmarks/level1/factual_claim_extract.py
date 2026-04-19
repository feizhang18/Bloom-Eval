import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "") # 替换为你的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

# --- 2. 核心API调用函数 (已重构) ---
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
            if chunk_reasoning:
                if not is_answering: print(chunk_reasoning, end="", flush=True)
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

# --- 3. 核心功能: 构建Prompt并调用API (已重构) ---
def extract_factual_statements(client: OpenAI, text: str, query_id: str, final_output_path: str, log_output_dir: str) -> List[str]:
    """
    使用专门的提示词提取事实性声明。
    """
    prompt = f"""
# ROLE
You are a meticulous, detail-oriented research assistant tasked with extracting all **Factual Statements** from a given academic article.
# TASK DEFINITION
A "Factual Statement" is a complete sentence that is **verifiable, objective, contains no subjective evaluation, and its factual content can be directly understood from the current text without consulting external citations.**
It typically describes:
- **Definitions**: e.g., "A radiance field is a function..."
- **Methods**: e.g., "The model is trained using the Adam optimizer..."
- **Specific Data or History**: e.g., "The Transformer architecture was proposed in 2017."

Please DO NOT extract the following:
- **Statements reliant on external citations**: Claims whose validity can only be verified by reading a cited reference. This is the most important rule.
- **Subjective evaluations or opinions**: e.g., "A key limitation of this approach is...", "This is a promising direction..."
- **Future outlooks or suggestions**: e.g., "Future work should focus on..."
- **Structural descriptions of the article**: e.g., "Section 3 describes our architecture..."
---
### POSITIVE EXAMPLES (These are the types of sentences you SHOULD extract)
1. "A radiance field is a function that maps a 5D coordinate (spatial location and viewing direction) to a color and density."
2. "The model is trained using the Adam optimizer with a learning rate of 1e-4."
3. "The Transformer architecture was first proposed in the 2017 paper 'Attention Is All You Need'."
---
### NEGATIVE EXAMPLES (Do NOT extract these sentences)
1. "However, a key limitation of this approach is its significant memory footprint, making it impractical for consumer-grade hardware." (This is an evaluation/limitation)
2. "Handling dynamic and deforming objects remains a largely unexplored challenge." (This identifies a research gap/analysis)
3. "Future work should focus on integrating a simultaneous localization and mapping (SLAM) component." (This is a future direction/suggestion)
4. "As demonstrated by Smith et al. [25], this phenomenon is caused by quantum entanglement." (This claim's validity depends on external reference [25] and should not be extracted.)
---
# TASK
Follow these steps carefully:
1.  First, read through the entire article below and identify a preliminary list of all potential "Factual Statements".
2.  Next, perform a **mandatory final review** of your preliminary list. For each sentence in your list, you must ask yourself: **"Does this sentence contain a citation marker (e.g., [9], [60]) or mention a specific author's name (e.g., 'Smith et al.')?"**
3.  If the answer is YES, you **MUST** delete that sentence from your list. This final filtering step is the most critical part of your task.
4.  Finally, present the fully cleaned and filtered list according to the output rules.
# OUTPUT RULES
- Your output **MUST** be a single, valid JSON object.
- The JSON object must contain only one key: `"factual_statements"`.
- The value for this key must be a **list of strings**, where each string is a complete, standalone factual statement you extracted after the final review.
- If no factual statements remain after filtering, return an empty list: `[]`.
- **DO NOT** add any explanations or extra text outside the JSON code block.
---
### ARTICLE TO ANALYZE
{text}
---
"""
    messages = [{"role": "user", "content": prompt}]
    final_json_str = get_streaming_output_with_reasoning(client, messages, query_id, final_output_path, log_output_dir)
    
    try:
        if final_json_str.strip().startswith("```json"):
            final_json_str = final_json_str.strip()[7:-3].strip()
        result = json.loads(final_json_str)
        return result.get("factual_statements", [])
    except json.JSONDecodeError:
        print(f"\n错误: 未能将模型的最终回答解析为JSON。")
        return []

# --- 4. 辅助函数 ---
def load_text_from_json(file_path: str) -> str:
    """从JSON文件加载文本内容。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], str):
                return data[0]
        return ""
    except Exception as e:
        print(f"读取或解析文件 {file_path} 时出错: {e}")
        return ""

# --- 5. 主执行流程 ---
def main():
    if not API_KEY or "sk-" not in API_KEY:
        print("错误: 请在代码第9行设置有效的API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # --- 批量处理配置 ---
    EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
    EXPERIMENT_IDS = range(2, 21) # 遍历 2, 3, ..., 20
    METHOD_NAME = 'LLMxMapReduce_V2'
    # --- 结束配置 ---

    print(f"🚀 开始事实声明批量提取任务...")
    success, skipped, errors = 0, 0, 0
    
    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        # 1. 动态构建路径
        base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME)
        input_path = os.path.join(base_dir, 'content.json')
        output_dir = os.path.join(base_dir, 'bench', 'level1')
        final_output_path = os.path.join(output_dir, 'actual_claims_extraction.json')
        log_dir = os.path.join(output_dir, "llm_outputs_factual_extraction")

        # 2. 检查前置条件
        if os.path.exists(final_output_path):
            print(f"⏭️  跳过: 目标文件 '{os.path.basename(final_output_path)}' 已存在。")
            skipped += 1
            continue
        if not os.path.exists(input_path):
            print(f"❌ 错误: 输入文件 '{os.path.basename(input_path)}' 未找到。")
            errors += 1
            continue
        
        # 3. 加载并执行
        article_text = load_text_from_json(input_path)
        if not article_text:
            print(f"❌ 错误:未能从 '{os.path.basename(input_path)}' 加载任何文本内容。")
            errors += 1
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        query_id = f"exp{exp_id}_{METHOD_NAME}_fact_extract"
        
        extract_factual_statements(client, article_text, query_id, final_output_path, log_dir)

        if os.path.exists(final_output_path):
            print(f"\n✅ 成功处理实验ID: {exp_id}。结果已保存。")
            success += 1
        else:
            print(f"\n❌ 处理实验ID {exp_id} 时出错，未生成输出文件。")
            errors += 1

    # 4. 打印最终总结
    print(f"\n\n{'='*30} 🎉 所有任务已完成! {'='*30}")
    print("📊 最终统计:")
    print(f"   - ✅ 成功处理: {success} 个文件夹")
    print(f"   - ⏭️ 跳过处理: {skipped} 个文件夹")
    print(f"   - ❌ 发生错误: {errors} 个文件夹")
    print("="*72)

if __name__ == "__main__":
    main()
