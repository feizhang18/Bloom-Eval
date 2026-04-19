import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any

# --- 1. 配置区域 ---

# 注意：请确保您的API密钥已设置为环境变量，或在此处直接填写
API_KEY = os.getenv("OPENAI_API_KEY", "") # 替换为你的 API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

# --- 2. 核心API调用函数 (已修改) ---

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
            temperature=0.0
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
            with open(error_txt_file, 'w', encoding='utf-8') as f:
                f.write(answer_content)
            print(f"\n警告：JSON解析失败，原始回答已保存到: {error_txt_file}")
            
        print(f"\n思考内容已保存到: {reasoning_file}")
        
        return cleaned_answer # 返回清理后的字符串
        
    except Exception as e:
        print(f"\nAPI调用或流式处理时发生错误: {e}")
        return "{}"

# --- 3. 实体归一化智能体 (已修改) ---

def run_normalizer(client: OpenAI, raw_entities: Dict[str, List[str]], query_id: str, final_output_path: str, log_output_dir: str) -> Dict[str, Any]:
    """
    智能体：接收实体字典，调用LLM进行归一化，并控制结果的保存。
    """
    json_data_str = json.dumps(raw_entities, indent=2, ensure_ascii=False)

    prompt_template = """
You are an expert in analyzing scientific literature. Your first step is to infer the specific research domain from the provided list of technical entities. Then, you will act as a specialist in THAT domain to perform the entity normalization task below.
Your task is to process a list of technical entities extracted from an academic paper. You must filter out invalid or generic terms and create a mapping from each valid alias to a single, standardized "Canonical Name".
**CRITICAL INSTRUCTIONS:**
1.  **DO NOT COUNT:** Your output must not contain any numbers representing counts or frequencies. Your only job is to create a mapping.
2.  **FILTER AND DISCARD:** You must ignore and completely discard the following types of entries. They should NOT appear in your final output:
    - **Generic Technical Terms:** e.g., `encoder`, `decoder`, `backbone network`, `convolutional layers`, `self-attention mechanism`.
    - **Descriptive Phrases:** e.g., `positive sample`, `feature fusion module`, `fast inference image classification`.
    - **Mathematical Variables/Symbols:** e.g., `\\(f _ {{ q }}\\)`.  
    - **Broad Field Names:** e.g., `artificial intelligence (AI)`, `computer vision (CV)`.
3.  **NORMALIZE AND MAP:**
    - For all remaining **valid, specific entities**, group together those that refer to the same concept (e.g., acronyms and full names).
    - For each group, choose a single, consistent **Canonical Name**. (e.g., use "Vision Transformer" as the canonical name for both "ViT" and "vision transformer").
    - Create a mapping where the key is the original entity (the alias) and the value is its assigned Canonical Name.
**OUTPUT FORMAT:**
You MUST return the output as a single, valid JSON object. Do not add any explanatory text before or after the JSON. The object must contain three keys: `methods_models_map`, `datasets_map`, and `evaluation_metrics_map`. Each value should be a dictionary representing the alias-to-canonical-name mapping.
**JSON Format Example:**
If the input list for `methods_models` is `["ViT", "Vision Transformer", "encoder", "CNN", "CNNs"]`, the corresponding output should be:
{{
  "methods_models_map": {{
    "ViT": "Vision Transformer",
    "Vision Transformer": "Vision Transformer",
    "CNN": "Convolutional Neural Network",
    "CNNs": "Convolutional Neural Network"
  }}
}}
(Notice: "encoder" was discarded and does not appear in the output.)
Now, process the JSON data I provide you.
--- JSON DATA START ---
{json_data}
--- JSON DATA END ---
"""
    prompt = prompt_template.format(json_data=json_data_str)
    messages = [{"role": "user", "content": prompt}]
    
    # 将输出路径和日志路径传递给API调用函数
    final_json_str = get_streaming_output_with_reasoning(client, messages, query_id, final_output_path, log_output_dir)

    try:
        normalized_map = json.loads(final_json_str)
        for key in ["methods_models_map", "datasets_map", "evaluation_metrics_map"]:
            normalized_map.setdefault(key, {}) # 使用 setdefault 简化代码
        return normalized_map
    except json.JSONDecodeError:
        print(f"Normalizer Agent Error: 无法解析最终的JSON答案。将返回空结果。")
        return {"methods_models_map": {}, "datasets_map": {}, "evaluation_metrics_map": {}}

# --- 4. 辅助函数 (无变化) ---

def load_entities_from_json(file_path: str) -> Dict[str, List[str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{file_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 时发生未知错误: {e}")
        return None

# --- 5. 主执行流程 (已重构) ---
def main():
    if "sk-" not in API_KEY:
        print("❌ 错误：请在代码第9行设置您的有效API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # --- 批量处理配置 ---
    EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
    EXPERIMENT_IDS = range(1, 21) 
    METHOD_NAME = 'LLMxMapReduce_V2'          # 目标子文件夹
    # --- 结束配置 ---

    print(f"🚀 开始实体归一化任务, 目标文件夹: {EXPERIMENT_IDS.start}-{EXPERIMENT_IDS.stop - 1} 中的 '{METHOD_NAME}'")
    
    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        # 1. 动态构建所有需要的路径
        base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'bench', 'level1')
        input_path = os.path.join(base_dir, 'all_entity.json')
        final_output_path = os.path.join(base_dir, 'difference_entity.json')
        log_output_dir = os.path.join(base_dir, "benchmark_entity_normalization_outputs")

        # 2. 检查前置条件，实现智能跳过
        if not os.path.exists(input_path):
            print(f"🟡 跳过：输入文件 'all_entity.json' 在 {base_dir} 中不存在。")
            continue
        if os.path.exists(final_output_path):
            print(f"⏭️ 跳过：目标文件 'difference_entity.json' 已存在于 {base_dir}")
            continue

        # 3. 加载源实体文件
        raw_entities = load_entities_from_json(input_path)
        if not raw_entities:
            print(f"🟡 跳过：无法从 '{input_path}' 加载实体。")
            continue

        # 4. 确保日志目录存在
        os.makedirs(log_output_dir, exist_ok=True)
        
        # 5. 执行归一化
        query_id = f"exp{exp_id}_{METHOD_NAME}_normalizer"
        normalized_map = run_normalizer(client, raw_entities, query_id, final_output_path, log_output_dir)

        # 6. 打印总结信息 (保存操作已在 run_normalizer 内部完成)
        print(f"\n✅ 归一化处理成功！")
        print(f"   最终结果已保存至: {final_output_path}")
        for category, entity_map in normalized_map.items():
            print(f"     - 类别 '{category}' 生成了 {len(entity_map)} 个映射条目。")

    print(f"\n\n{'='*30} 🎉 所有归一化任务已完成! {'='*30}")

if __name__ == "__main__":
    main()
