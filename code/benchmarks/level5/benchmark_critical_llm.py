import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any

# --- 1. 全局配置 ---
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

# --- 批量处理配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'LLMxMapReduce_V2'          # 目标子文件夹

# --- 2. 核心API调用函数 (已重构) ---
def get_streaming_output_with_reasoning(client: OpenAI, message: List[Dict], query_id: str, final_output_path: str, log_output_dir: str) -> str:
    """
    获取大模型流式输出，将最终答案保存到固定路径，并将日志保存到指定目录。
    """
    reasoning_dir = os.path.join(log_output_dir, "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)
    
    # 日志文件仍然保留时间戳以避免冲突
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    identifier = f"{query_id}_{timestamp}"
    reasoning_file = os.path.join(reasoning_dir, f"reasoning_{identifier}.txt")

    print(f"\n{'='*20} 正在执行查询: {query_id} {'='*20}\n")
    print("--- 思考过程 (Streaming) ---\n")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=message,
            stream=True,
            temperature=0
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
        
        # 清理并保存最终的JSON结果到固定路径
        cleaned_answer = answer_content.strip()
        if cleaned_answer.startswith("```json"):
            cleaned_answer = cleaned_answer[7:-3].strip()

        try:
            parsed_json = json.loads(cleaned_answer)
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            error_txt_file = final_output_path.replace('.json', '_error.txt')
            with open(error_txt_file, 'w', encoding='utf-8') as f: f.write(answer_content)
            print(f"\n警告：JSON解析失败，原始回答已保存到: {error_txt_file}")
            
        print(f"\n思考内容已保存到: {reasoning_file}")
        
        return cleaned_answer # 返回清理后的字符串
        
    except Exception as e:
        print(f"\nAPI调用或流式处理时发生错误: {e}")
        return "{}"

# --- 3. 核心功能：构建Prompt并调用API (已重构) ---
def extract_critical_statements(client: OpenAI, text: str, query_id: str, final_output_path: str, log_output_dir: str) -> List[str]:
    """
    使用少样本Prompt和流式函数，提取批判性声明。
    """
    prompt = f"""
You are a meticulous and insightful academic researcher. Your task is to identify and extract ALL complete sentences that function as 'critical statements' from the provided article.
A "critical statement" is a sentence that **evaluates, analyzes, identifies limitations, points out research gaps, discusses trade-offs, or suggests future research directions based on a problem.** It is NOT a simple factual description, a statement of results, or a definition.
--- EXAMPLES of Critical Statements (These are what you should extract) ---
1. "However, a key limitation of this approach is its significant memory footprint, making it impractical for consumer-grade hardware."
2. "While significant progress has been made in static scenes, handling dynamic and deforming objects remains a largely unexplored challenge."
3. "The reliance on pre-calibrated camera poses is a major drawback; future work should therefore focus on integrating a simultaneous localization and mapping (SLAM) component."
4. "Although these compression techniques reduce storage size, they often introduce noticeable artifacts, revealing a trade-off between efficiency and visual fidelity."
--- EXAMPLES of Non-Critical Statements (Do NOT extract these) ---
1. "The model is trained using the Adam optimizer with a learning rate of 1e-4." (Factual description)
2. "On the Tanks and Temples benchmark, our method achieves a PSNR of 35.2 dB." (Statement of results)
3. "A radiance field is a function that maps a 5D coordinate to a color and density." (Definition)
4. "Section 3 describes our proposed architecture, while Section 4 presents the experimental results." (Paper structure)
5. "This paper provides a comprehensive review of recent methods." (Do not extract sentences that describe the scope, purpose, or structure of the article itself.)
--- TASK ---
Now, please analyze the following article and provide the final list of extracted sentences.
RULES FOR OUTPUT:
- Your output MUST be a single, valid JSON object.
- The JSON object must contain one key: "critical_statements".
- The value for this key must be a list of strings, where each string is a complete critical sentence you extracted.
- If no critical statements are found, return an empty list: [].
--- ARTICLE TEXT START ---
{text}
--- ARTICLE TEXT END ---
"""
    messages = [{"role": "user", "content": prompt}]
    
    final_json_str = get_streaming_output_with_reasoning(client, messages, query_id, final_output_path, log_output_dir)
    
    try:
        if final_json_str.strip().startswith("```json"):
            final_json_str = final_json_str.strip()[7:-3].strip()
        result = json.loads(final_json_str)
        return result.get("critical_statements", [])
    except json.JSONDecodeError:
        print(f"\n无法解析最终的JSON答案。")
        return []

# --- 4. 辅助函数 ---
def load_text_from_json(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], str):
            return " ".join(data) # 将列表中的所有段落合并
        return ""
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return ""

# --- 5. 主执行流程 ---
def main():
    if not API_KEY or "sk-" not in API_KEY:
        print("错误：请在代码中或通过环境变量设置您的API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"🚀 开始批判性声明批量提取任务...")
    success, skipped, errors = 0, 0, 0

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        # 1. 动态构建路径
        base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME)
        input_path = os.path.join(base_dir, 'content.json')
        output_dir = os.path.join(base_dir, 'bench', 'level5')
        final_output_path = os.path.join(output_dir, 'critical_claims_extraction.json')
        log_dir = os.path.join(output_dir, "llm_outputs_critical")

        # 2. 智能跳过
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
            print(f"❌ 错误: 未能从 '{os.path.basename(input_path)}' 加载任何文本内容。")
            errors += 1
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        query_id = f"exp{exp_id}_{METHOD_NAME}_critical_extract"
        
        statements = extract_critical_statements(client, article_text, query_id, final_output_path, log_dir)

        if os.path.exists(final_output_path):
            print(f"\n✅ 成功处理实验ID: {exp_id}。提取到 {len(statements)} 条声明。")
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

