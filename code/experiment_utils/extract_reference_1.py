import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any

# --- 1. Configuration Area ---

# !!! IMPORTANT: Please enter your valid API key here.
# Ensure your API key is set as an environment variable or fill it in directly.
API_KEY = os.getenv("OPENAI_API_KEY", "") # Replace with your API Key
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# LLM Model Configuration
# Note: GPT-4 models are generally more reliable for producing correct JSON formats.
MODEL_NAME = "gpt-5-mini"

# --- Directory and Method Configuration ---
EXPERIMENT_ROOT = "<EXPERIMENT_ROOT>"
# Update the range if you have more or fewer experiment folders
EXPERIMENT_IDS = range(2, 21)  # Folders 2 to 20
METHOD_NAMES = ["LLMxMapReduce_V2"]
# ["autosurvey", "surveyforge", "surveyx", "LLMxMapReduce_V2"]

# --- 2. Core API and Prompt Functions ---

def get_llm_output(client: OpenAI, message: List[Dict], output_json_path: str) -> str:
    """
    Calls the LLM API, handles the streaming output, and saves the final JSON result.
    """
    reasoning_dir = os.path.join(os.path.dirname(output_json_path), "reasoning_references")
    os.makedirs(reasoning_dir, exist_ok=True)
    
    base_name = os.path.basename(output_json_path).replace('.json', '')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reasoning_file = os.path.join(reasoning_dir, f"{base_name}_reasoning_{timestamp}.txt")
    
    print("\n--- Parsing References ---")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
            stream=True,
            temperature=0.0,
        )

        full_response_content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_content = chunk.choices[0].delta.content
                print(chunk_content, end="", flush=True)
                full_response_content += chunk_content
        
        print("\n--- End of Parsing ---")
        
        # Save the full response as a reasoning log
        with open(reasoning_file, 'w', encoding='utf-8') as f:
            f.write(full_response_content)
        
        # Clean and parse the JSON from the response
        final_json_str = full_response_content.strip()
        try:
            # Remove potential Markdown code block markers
            if final_json_str.startswith("```json"):
                final_json_str = final_json_str[7:-3].strip()
            
            parsed_json = json.loads(final_json_str)

            if not isinstance(parsed_json, dict) or "references" not in parsed_json or "total_references" not in parsed_json:
                 raise json.JSONDecodeError("Model did not return a valid object with 'references' and 'total_references' keys.", final_json_str, 0)

            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            print(f"✅ Successfully parsed and saved references to: {output_json_path}")
            return final_json_str

        except json.JSONDecodeError as e:
            output_txt_path = output_json_path.replace('.json', '_error.txt')
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(full_response_content)
            print(f"⚠️ 警告: 模型响应不是有效的JSON对象。详情: {e}")
            print(f"   原始输出已保存为文本: {output_txt_path}")
            return "{}"

    except Exception as e:
        print(f"\n❌ API调用期间发生错误: {e}")
        return "{}"

def build_reference_parser_prompt(reference_list: List[str]) -> str:
    """
    Builds the prompt for parsing a list of reference strings into a structured JSON object.
    """
    references_text = "\n".join(reference_list)

    prompt = f"""
# ROLE
You are an expert data extraction bot specializing in parsing academic citations. Your task is to convert a list of raw reference strings into a structured JSON format.

# TASK DEFINITION
- **Parse Each Entry**: For each numbered reference string in the input, you must extract four key pieces of information:
  1.  **title**: The main title of the work.
  2.  **year**: The year of publication.
  3.  **authors**: A list of all authors. If "et al." is present, it should be included as an item in the authors list.
  4.  **publication**: All remaining publication details (e.g., journal name, volume, page numbers, publisher).
- **Count the Total**: You must count the total number of references provided in the input.
- **Strict Formatting**: Your final output must strictly adhere to the JSON object structure specified below.

# OUTPUT RULES
- Your output **MUST** be a single, valid JSON **object**.
- The object must have a top-level key `"total_references"` with an integer value for the total count.
- The object must have a second top-level key `"references"` with a value that is a JSON **array** (a list).
- Each element in the `"references"` array must be an object with exactly these four keys: `"title"`, `"year"`, `"authors"` (with a list of strings as its value), and `"publication"`.
- **DO NOT** add any explanations or extraneous text outside of the final JSON code block. Your response must start with `{{` and end with `}}`.

---
### EXAMPLE

#### INPUT:
```
[1] F. Rosenblatt, The Perceptron, a Perceiving and Recognizing Automaton Project Para. Buffalo, New York, USA: Cornell Aeronautical Lab., 1957.
[2] J. Orbach, “Principles of neurodynamics. perceptrons and the theory of brain mechanisms,” Arch. General Psychiatry, vol. 7, no. 3, pp. 218–219, 1962.
[3] Y. LeCun et al., “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp 2278–2324, 1998.
```

#### CORRECT OUTPUT:
```json
{{
  "total_references": 3,
  "references": [
    {{
      "title": "The Perceptron, a Perceiving and Recognizing Automaton Project Para",
      "year": "1957",
      "authors": [
        "F. Rosenblatt"
      ],
      "publication": "Buffalo, New York, USA: Cornell Aeronautical Lab."
    }},
    {{
      "title": "Principles of neurodynamics. perceptrons and the theory of brain mechanisms",
      "year": "1962",
      "authors": [
        "J. Orbach"
      ],
      "publication": "Arch. General Psychiatry, vol. 7, no. 3, pp. 218–219"
    }},
    {{
      "title": "Gradient-based learning applied to document recognition",
      "year": "1998",
      "authors": [
        "Y. LeCun et al."
      ],
      "publication": "Proc. IEEE, vol. 86, no. 11, pp 2278–2324"
    }}
  ]
}}
```
---

# REFERENCES TO PARSE
{references_text}
"""
    return prompt

def process_references_file(client: OpenAI, reference_path: str, output_path: str):
    """
    Loads references from a single file, calls the LLM for parsing, and saves the result.
    """
    print(f"📄 正在处理: {reference_path}")
    
    try:
        with open(reference_path, 'r', encoding='utf-8') as f:
            reference_list = json.load(f)
            if not isinstance(reference_list, list) or not reference_list:
                print(f"   - ❗️ 警告: {reference_path} 为空或格式不是预期的列表。跳过。")
                return
    except FileNotFoundError:
        print(f"   - ❗️ 错误: 在 {reference_path} 未找到参考文献文件。跳过。")
        return
    except (json.JSONDecodeError, IndexError) as e:
        print(f"   - ❗️ 错误: 无法读取或解析 {reference_path}: {e}。跳过。")
        return

    prompt = build_reference_parser_prompt(reference_list)
    messages = [{"role": "user", "content": prompt}]
    
    get_llm_output(client, messages, output_path)

# --- 3. Main Execution Flow ---

def main():
    """
    Scans all specified experiment directories and processes `reference.json` files that
    are missing a formatted counterpart.
    """
    if not API_KEY or "sk-" not in API_KEY:
        print("❌ 错误: 请在第13行设置您的有效API密钥。")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    total_processed = 0
    total_skipped = 0

    print("🚀 开始扫描并重新格式化 SurveyBench 参考文献文件...")
    print("="*70)

    for exp_id in EXPERIMENT_IDS:
        for method in METHOD_NAMES:
            method_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), method)
            reference_path = os.path.join(method_dir, "reference.json")
            output_path = os.path.join(method_dir, "reference_1.json")

            if not os.path.exists(reference_path):
                continue

            if os.path.exists(output_path):
                print(f"⏭️  跳过: 格式化后的参考文献文件已存在于 {method_dir}")
                total_skipped += 1
            else:
                process_references_file(client, reference_path, output_path)
                total_processed += 1
                # Add a brief delay to avoid hitting API rate limits
                time.sleep(2)

    print("\n" + "="*70)
    print("🎉 所有任务完成!")
    print(f"📊 总结:")
    print(f"   - 新生成的参考文献文件: {total_processed}")
    print(f"   - 跳过的文件 (已存在): {total_skipped}")

if __name__ == "__main__":
    main()
