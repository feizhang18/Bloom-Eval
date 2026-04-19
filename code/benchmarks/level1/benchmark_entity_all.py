import os
import json
import time
from openai import OpenAI
from typing import Dict, List

# --- 1. 配置区域 ---

# 注意：请确保您的API密钥已设置为环境变量 OPENAI_API_KEY，或在此处直接填写
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 替换为你的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

# --- 2. 核心API调用函数 ---

def get_streaming_output_with_reasoning(client: OpenAI, message: List[Dict], query_id: str, log_output_dir: str) -> str:
    """
    获取大模型流式输出，保存详细日志，并返回最终的JSON字符串。
    """
    reasoning_dir = os.path.join(log_output_dir, "reasoning")
    answer_dir = os.path.join(log_output_dir, "answer")

    os.makedirs(reasoning_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    identifier = f"{query_id}_{timestamp}"

    reasoning_file = os.path.join(reasoning_dir, f"reasoning_{identifier}.txt")
    answer_file = os.path.join(answer_dir, f"answer_{identifier}.json")

    print(f"\n{'='*20} 正在向LLM发送查询: {query_id} {'='*20}\n")
    print("--- 模型思考过程 (Streaming) ---\n")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=message,
            stream=True,
            temperature=0.0
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        for chunk in response:
            if not chunk.choices:
                continue
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

        with open(reasoning_file, 'w', encoding='utf-8') as f:
            f.write(reasoning_content)
        
        # 保存原始回答用于调试
        temp_answer_file = answer_file
        try:
            # 清理可能的Markdown代码块标记
            cleaned_answer = answer_content.strip()
            if cleaned_answer.startswith("```json"):
                cleaned_answer = cleaned_answer[7:-3].strip()
            # 尝试解析以确认是JSON
            json.loads(cleaned_answer)
            with open(temp_answer_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_answer)
        except json.JSONDecodeError:
            temp_answer_file = temp_answer_file.replace('.json', '_error.txt')
            with open(temp_answer_file, 'w', encoding='utf-8') as f:
                f.write(answer_content)

        print(f"\n思考日志已保存到: {reasoning_file}")
        print(f"原始回答已保存到: {temp_answer_file}")

        return answer_content

    except Exception as e:
        print(f"\nAPI调用或流式处理时发生错误: {e}")
        return "{}"

# --- 3. 提取实体智能体 (Prompt 无变化) ---

def run_extractor(client: OpenAI, text: str, query_id: str, log_output_dir: str) -> Dict[str, List[str]]:
    """
    智能体1：构建Prompt，调用LLM，并解析返回的实体JSON。
    """
    prompt = f"""
    You are a meticulous text scanner. Your task is to extract EVERY occurrence of technical entities from the academic survey text below. You will act like a machine, scanning the text from beginning to end and listing entities as you find them.

    **CRITICAL INSTRUCTIONS:**
    1.  **DO NOT DEDUPLICATE**: If an entity like "3DGS" appears 10 times, you MUST list it 10 times. If "3D Gaussian Splatting" appears 5 times, you MUST list it 5 times. List every single instance you find.
    2.  **EXTRACT EXACT PHRASES**: Extract the exact wording as it appears in the text. Do not normalize or change the entities.
    3.  **CATEGORIZE EACH INSTANCE**: For each entity instance you find, classify it into one of the three categories below.

    **Categories to extract:**
    1.  `methods_models`: Any named technique, algorithm, framework, architecture, or specific system (e.g., "3D Gaussian Splatting", "NeRF", "SfM").
    2.  `datasets`: Standardized data collections or benchmarks (e.g., "Neuman", "Stereo Blur").
    3.  `evaluation_metrics`: Quantitative metrics used to measure performance (e.g., "PSNR", "Chamfer Distance").

    **OUTPUT FORMAT:**
    You MUST return the output as a single, valid JSON object. Do not add any explanatory text before or after the JSON. The lists in the JSON should contain every single occurrence of the entities.

    **JSON format example:**
    If the text says "We use PSNR. Our method improves PSNR.", the output should be:
    {{
      "methods_models": [],
      "datasets": [],
      "evaluation_metrics": ["PSNR", "PSNR"]
    }}

    Now, analyze the text below and extract all entity occurrences:
    --- TEXT START ---
    {text}
    --- TEXT END ---
    """
    messages = [{"role": "user", "content": prompt}]
    final_json_str = get_streaming_output_with_reasoning(client, messages, query_id, log_output_dir)

    try:
        if final_json_str.strip().startswith("```json"):
            final_json_str = final_json_str.strip()[7:-3]
        entities = json.loads(final_json_str)
        # 确保所有必需的键都存在，防止后续代码出错
        for key in ["methods_models", "datasets", "evaluation_metrics"]:
            entities.setdefault(key, [])
        return entities
    except json.JSONDecodeError:
      print(f"Extractor Agent Error: 无法解析最终的JSON答案。将返回空结果。")
      return {"methods_models": [], "datasets": [], "evaluation_metrics": []}

# --- 4. 辅助函数 ---

def load_text_from_json(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data[0] if data and isinstance(data, list) else ""
    except Exception as e:
        print(f"错误：读取文件 {file_path} 失败: {e}")
        return ""

# --- 5. 主执行流程 (已重构) ---
def main():
    """
    主函数，遍历所有指定的实验文件夹并执行实体提取任务。
    """
    if "sk-" not in API_KEY:
        print("❌ 错误：请在代码第9行设置您的有效API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # --- 批量处理配置 ---
    EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
    EXPERIMENT_IDS = range(2, 21)  # 遍历 3, 4, ..., 20
    METHOD_NAME = 'LLMxMapReduce_V2'
    # --- 结束配置 ---

    print(f"🚀 开始批量提取实体任务，目标文件夹: {EXPERIMENT_IDS.start}-{EXPERIMENT_IDS.stop - 1} 中的 '{METHOD_NAME}'")
    
    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        # 1. 动态构建路径
        method_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME)
        content_path = os.path.join(method_dir, 'content.json')
        output_dir = os.path.join(method_dir, 'bench', 'level1')
        final_output_path = os.path.join(output_dir, 'all_entity.json')
        log_output_dir = os.path.join(output_dir, "benchmark_entity_llm_outputs_all_occurrences")

        # 2. 检查前置条件
        if not os.path.exists(content_path):
            print(f"🟡 跳过：源文件 'content.json' 在 {method_dir} 中不存在。")
            continue
        if os.path.exists(final_output_path):
            print(f"⏭️ 跳过：目标文件 'all_entity.json' 已存在于 {output_dir}")
            continue

        # 3. 加载源文本
        text_content = load_text_from_json(content_path)
        if not text_content:
            print(f"🟡 跳过：'content.json' 文件为空或格式不正确。")
            continue

        # 4. 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 5. 执行提取
        query_id = f"exp{exp_id}_{METHOD_NAME}"
        predicted_entities = run_extractor(client, text_content, query_id, log_output_dir)

        # 6. 保存最终结果到 'all_entity.json'
        try:
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(predicted_entities, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 实体提取成功！")
            print(f"   最终结果已保存至: {final_output_path}")
            for category, entities in predicted_entities.items():
                print(f"     - 类别 '{category}' 提取到 {len(entities)} 个实体实例。")
        except Exception as e:
            print(f"❌ 错误：保存最终结果到 {final_output_path} 时失败: {e}")

    print(f"\n\n{'='*30} 🎉 所有提取任务已完成! {'='*30}")

if __name__ == "__main__":
    main()
