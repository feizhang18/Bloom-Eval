import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any

# --- 1. 全局配置 ---

# !!! 重要：请确保您的API密钥已设置为环境变量 OPENAI_API_KEY !!!
API_KEY = os.getenv("OPENAI_API_KEY", "") # <--- 不推荐：直接在此处填写密钥
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-5-mini"

# --- 2. 核心API调用函数 (已重构) ---

def get_streaming_output_with_reasoning(client: OpenAI, message: List[Dict], query_id: str, final_output_path: str, log_output_dir: str) -> str:
    """
    获取大模型流式输出，将最终答案保存到固定路径，并将日志保存到指定目录。
    """
    reasoning_dir = os.path.join(log_output_dir, "reasoning")
    answer_dir = os.path.join(log_output_dir, "answer_logs") # 原始回答日志目录
    os.makedirs(reasoning_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    identifier = f"{query_id}_{timestamp}"
    
    reasoning_file = os.path.join(reasoning_dir, f"reasoning_{identifier}.txt")
    answer_log_file = os.path.join(answer_dir, f"answer_{identifier}.json")

    print(f"\n{'='*20} 正在执行查询: {query_id} {'='*20}\n")
    print("--- 思考过程 (Streaming) ---\n")

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
        if final_json_str.startswith("```json"):
            final_json_str = final_json_str[7:-3].strip()
        
        try:
            parsed_json = json.loads(final_json_str)
            # 1. 保存最终结果到固定的输出路径
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            # 2. 同时将原始回答保存到日志文件
            with open(answer_log_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            txt_log_file = answer_log_file.replace('.json', '_error.txt')
            with open(txt_log_file, 'w', encoding='utf-8') as f: f.write(answer_content)
            print(f"回答不是有效的JSON，原始文本已保存到: {txt_log_file}")
            answer_log_file = txt_log_file

        print(f"\n思考内容已保存到: {reasoning_file}")
        print(f"原始回答日志已保存到: {answer_log_file}")
        
        return final_json_str
    except Exception as e:
        print(f"\nAPI调用或流式处理时发生错误: {e}")
        return "{}"

# --- 3. 实体解析智能体 (已重构) ---

def run_entity_resolver(client: OpenAI, expert_json: Dict[str, Any], llm_json: Dict[str, Any], query_id: str, final_output_path: str, log_output_dir: str) -> Dict[str, Any]:
    """
    智能体：利用LLM在两个JSON对象之间进行实体解析。
    """
    prompt = f"""
# Role
You are an expert in Entity Resolution for scientific literature. Your first step is to analyze the content of the `expert_data` and `llm_data` JSON objects to identify the specific academic domain they belong to. Then, you will act as a specialist in THAT domain to perform the matching task. You are highly familiar with the names, abbreviations, and common variants of models, datasets, and evaluation metrics found in your specialized domain.
Your task is to accurately identify the common entities between the two provided JSON objects (`expert_data` and `llm_data`). For each common entity you identify, you must provide its corresponding main name from both of the original JSON files.
# Core Requirements
1.  **Definition**: A "common entity" refers to the same underlying concept existing in both JSON files, **even if their main names or aliases have different spellings or variations**.
2.  **Method**: You must use the entity's main name, its list of aliases, and your specialized domain knowledge to determine if they refer to the same concept.
3.  **Confidence**: Match entities only when you are highly confident they are the same.
4.  **Categorization**: Matching must occur *within* the same top-level category (e.g., a `methods_models` from expert_data can only match a `methods_models` from llm_data).
# Examples of Correct Matching
- An entity named "Convolutional Neural Network" (with an alias "CNN") in `expert_data` should be matched with an entity named "CNN" (with an alias "convolutional neural networks") in `llm_data`.
- An entity named "IPT" (with an alias "Image Processing Transformer (IPT)") in `expert_data` should be matched with "Image Processing Transformer" (with an alias "IPT") in `llm_data`.
# Input Data Format
I am providing two JSON objects: `expert_data` and `llm_data`. Each contains three categories: `methods_models`, `datasets`, and `evaluation_metrics`.
# Output Requirements
1.  **Format**: The output must be a **single, valid JSON object**.
2.  **No Extra Text**: **Do not** add any explanations, comments, or additional text outside of the final JSON code block.
3.  **Structure**: The output JSON must contain the same three top-level keys: `methods_models`, `datasets`, and `evaluation_metrics`.
4.  **Content**: Under each key, provide a list of matched pairs. Each element in the list must be an object with exactly two keys:
    * `expert_main_name`: The main name of the entity from `expert_data`.
    * `llm_main_name`: The main name of the entity from `llm_data`.
5.  **Accuracy**: If an entity exists in only one file, it must not be included. If a category has no matches, the list for that category should be empty (`[]`).
---
### INPUT DATA
#### expert_data
{json.dumps(expert_json, indent=2, ensure_ascii=False)}
#### llm_data
{json.dumps(llm_json, indent=2, ensure_ascii=False)}
---
Now, perform the entity resolution and provide the final JSON output.
"""
    messages = [{"role": "user", "content": prompt}]
    final_json_str = get_streaming_output_with_reasoning(client, messages, query_id, final_output_path, log_output_dir)

    try:
        entities = json.loads(final_json_str)
        for key in ["methods_models", "datasets", "evaluation_metrics"]:
            entities.setdefault(key, []) # 使用 setdefault 简化代码
        return entities
    except json.JSONDecodeError:
        print(f"Resolver Agent Error: 无法解析最终的JSON答案。返回一个空结构。")
        return {"methods_models": [], "datasets": [], "evaluation_metrics": []}

# --- 4. 主执行流程 (已重构) ---

def main():
    """
    主执行函数，负责遍历所有文件夹并调用处理函数。
    """
    if not API_KEY or not API_KEY.startswith("sk-"):
        print("错误：API密钥未设置或格式无效。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # --- 批量处理配置 ---
    EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
    EXPERIMENT_IDS = range(1, 21)
    # --- 结束配置 ---

    print("🚀 开始实体解析批量处理任务...")
    success_count, skipped_count, error_count = 0, 0, 0
    
    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        # 1. 动态构建所有路径
        expert_data_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'human', 'bench', 'level1', 'final_counts.json')
        llm_data_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level1', 'final_counts.json')
        output_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level1', 'compare_same.json')
        log_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), 'LLMxMapReduce_V2', 'bench', 'level1', 'benchmark_entity_llm_resolution')

        # 2. 检查前置条件
        if os.path.exists(output_path):
            print(f"⏭️ 跳过：结果文件 'compare_same.json' 已存在。")
            skipped_count += 1
            continue
        if not os.path.exists(expert_data_path) or not os.path.exists(llm_data_path):
            print(f"❌ 错误：缺少一个或多个输入文件，无法处理。")
            if not os.path.exists(expert_data_path): print(f"   - 缺失: {expert_data_path}")
            if not os.path.exists(llm_data_path): print(f"   - 缺失: {llm_data_path}")
            error_count += 1
            continue
            
        # 3. 加载数据
        try:
            with open(expert_data_path, 'r', encoding='utf-8') as f: expert_data = json.load(f)
            with open(llm_data_path, 'r', encoding='utf-8') as f: llm_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ 错误：无法解析JSON文件 {e.doc_path}。")
            error_count += 1
            continue
            
        # 4. 执行解析
        query_id = f"exp{exp_id}_resolve_expert_vs_llm"
        run_entity_resolver(client, expert_data, llm_data, query_id, output_path, log_dir)
        
        # 5. 确认输出文件是否已创建
        if os.path.exists(output_path):
            print(f"✅ 成功处理实验ID: {exp_id}。结果已保存。")
            success_count += 1
        else:
            print(f"❌ 处理实验ID: {exp_id} 时发生错误，未生成输出文件。")
            error_count += 1
            
    # 6. 打印最终总结
    print(f"\n\n{'='*30} 🎉 所有任务已完成! {'='*30}")
    print("📊 最终统计:")
    print(f"   - ✅ 成功处理: {success_count} 个文件夹")
    print(f"   - ⏭️ 跳过处理: {skipped_count} 个文件夹")
    print(f"   - ❌ 处理失败: {error_count} 个文件夹")
    print("="*72)

if __name__ == "__main__":
    main()
