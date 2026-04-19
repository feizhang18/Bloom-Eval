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
MODEL_NAME = "gpt-4o-mini"

# --- Directory and Method Configuration ---
EXPERIMENT_ROOT = "<EXPERIMENT_ROOT>"
# Update the range if you have more or fewer experiment folders
EXPERIMENT_IDS = range(2, 21)  # Folders 1 to 20
METHOD_NAMES = ["LLMxMapReduce_V2"]
#["autosurvey", "surveyforge", "surveyx", "LLMxMapReduce_V2"]
# --- 2. Core API and Prompt Functions ---

def get_llm_output(client: OpenAI, message: List[Dict], output_json_path: str) -> str:
    """
    Calls the LLM API, handles the streaming output, and saves the final JSON result.
    """
    reasoning_dir = os.path.join(os.path.dirname(output_json_path), "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)
    
    base_name = os.path.basename(output_json_path).replace('.json', '')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reasoning_file = os.path.join(reasoning_dir, f"{base_name}_reasoning_{timestamp}.txt")
    
    print("\n--- Generating Outline ---")

    try:
        # The `response_format` parameter is removed to allow for a JSON array output.
        # The prompt is now solely responsible for enforcing the format.
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
        
        print("\n--- End of Generation ---")
        
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

            # Ensure the parsed content is a list
            if not isinstance(parsed_json, list):
                 raise json.JSONDecodeError("Model did not return a list/array.", final_json_str, 0)

            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=4, ensure_ascii=False)
            print(f"✅ Successfully generated and saved outline to: {output_json_path}")
            return final_json_str

        except json.JSONDecodeError as e:
            output_txt_path = output_json_path.replace('.json', '_error.txt')
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(full_response_content)
            print(f"⚠️ WARNING: Model response was not valid JSON array. Details: {e}")
            print(f"   Original output saved as text to: {output_txt_path}")
            return "[]"

    except Exception as e:
        print(f"\n❌ An error occurred during the API call: {e}")
        return "[]"

def build_outline_prompt(text_content: str) -> str:
    """
    Builds the detailed few-shot prompt for extracting a hierarchical outline as a JSON array.
    """
    prompt = f"""
# ROLE
You are an expert research assistant specializing in academic document analysis. Your task is to extract the hierarchical outline (table of contents) from the provided academic paper content.

# TASK DEFINITION
- **Extract Headings**: Identify all section and subsection headings in the text.
- **Identify Levels**: Determine the hierarchical level of each heading.
  - The main title (usually the first heading, often without a number) is **Level 0**.
  - Headings like '1', '2.', etc., are **Level 1**.
  - Headings like '2.1', '3.2', etc., are **Level 2**.
  - Headings like '3.1.1', '3.6.4', etc., are **Level 3**.
- **Preserve Text**: The heading text, including any numbers, must be preserved exactly as it appears in the source.
- **Ignore Non-Headings**: Do not extract abstracts, body paragraphs, tables, or figure captions. Only extract lines that function as structural headings.

# OUTPUT RULES
- Your output **MUST** be a single, valid JSON **array** (a list).
- The list must contain inner lists, where each inner list has exactly two elements in this order: `[level, "heading_text"]`.
  1. An **integer** for the heading level (0, 1, 2, 3, ...).
  2. A **string** containing the full, original heading text.
- **DO NOT** wrap the array in a JSON object (e.g., `{{"outline": [...]}}`). The output must start with `[` and end with `]`.

---
### DETAILED EXAMPLE OF CORRECT OUTPUT FORMAT

```json
[
    [
        0,
        "A Survey on Vision Transformer"
    ],
    [
        1,
        "1 INTRODUCTION"
    ],
    [
        1,
        "2 FORMULATION OF TRANSFORMER"
    ],
    [
        2,
        "2.1 Self-Attention"
    ],
    [
        2,
        "2.2 Other Key Concepts in Transformer"
    ],
    [
        1,
        "3 VISION TRANSFORMER"
    ],
    [
        2,
        "3.1 Backbone for Representation Learning"
    ],
    [
        3,
        "3.1.1 Pure Transformer"
    ],
    [
        3,
        "3.1.2 Transformer With Convolution"
    ],
    [
        3,
        "3.1.3 Self-Supervised Representation Learning"
    ],
    [
        3,
        "3.1.4 Discussions"
    ],
    [
        2,
        "3.2 High/Mid-Level Vision"
    ],
    [
        3,
        "3.2.1 Generic Object Detection"
    ],
    [
        3,
        "3.2.2 Segmentation"
    ],
    [
        3,
        "3.2.3 Pose Estimation"
    ],
    [
        3,
        "3.2.4 Other Tasks"
    ],
    [
        3,
        "3.2.5 Discussions"
    ],
    [
        2,
        "3.3 Low-Level Vision"
    ],
    [
        3,
        "3.3.1 Image Generation"
    ],
    [
        3,
        "3.3.2 Image Processing"
    ],
    [
        2,
        "3.4 Video Processing"
    ],
    [
        3,
        "3.4.1 High-Level Video Processing"
    ],
    [
        3,
        "3.4.2 Low-Level Video Processing"
    ],
    [
        3,
        "3.4.3 Discussions"
    ],
    [
        2,
        "3.5 Multi-Modal Tasks"
    ],
    [
        2,
        "3.6 Efficient Transformer"
    ],
    [
        3,
        "3.6.1 Pruning and Decomposition"
    ],
    [
        3,
        "3.6.2 Knowledge Distillation"
    ],
    [
        3,
        "3.6.3 Quantization"
    ],
    [
        3,
        "3.6.4 Compact Architecture Design"
    ],
    [
        1,
        "4 CONCLUSION AND DISCUSSIONS"
    ],
    [
        2,
        "4.1 Challenges"
    ],
    [
        2,
        "4.2 Future Prospects"
    ]
]
```
---

# ARTICLE TO ANALYZE
{text_content}
"""
    return prompt

def process_single_file(client: OpenAI, content_path: str, outline_path: str):
    """
    Loads content from a single file, calls the LLM to generate an outline, and saves it.
    """
    print(f"📄 Processing: {content_path}")
    
    try:
        with open(content_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
            if not isinstance(content_list, list) or not content_list:
                print(f"   - ❗️ WARNING: {content_path} is empty or not in the expected list format. Skipping.")
                return
            article_text = content_list[0]
    except FileNotFoundError:
        print(f"   - ❗️ ERROR: Content file not found at {content_path}. Skipping.")
        return
    except (json.JSONDecodeError, IndexError) as e:
        print(f"   - ❗️ ERROR: Could not read or parse {content_path}: {e}. Skipping.")
        return

    prompt = build_outline_prompt(article_text)
    messages = [{"role": "user", "content": prompt}]
    
    get_llm_output(client, messages, outline_path)

# --- 3. Main Execution Flow ---

def main():
    """
    Scans all specified experiment directories and processes files that are missing an outline.
    """
    if not API_KEY or "YOUR_API_KEY" in API_KEY:
        print("❌ ERROR: Please set your valid API key on line 13.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    total_processed = 0
    total_skipped = 0

    print("🚀 Starting to scan and process SurveyBench experiment folders...")
    print("="*70)

    for exp_id in EXPERIMENT_IDS:
        for method in METHOD_NAMES:
            method_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), method)
            content_path = os.path.join(method_dir, "content.json")
            outline_path = os.path.join(method_dir, "outline.json")

            if not os.path.exists(content_path):
                continue

            if os.path.exists(outline_path):
                print(f"⏭️  Skipping: Outline already exists for {method_dir}")
                total_skipped += 1
            else:
                process_single_file(client, content_path, outline_path)
                total_processed += 1
                # Add a brief delay to avoid hitting API rate limits
                time.sleep(2)

    print("\n" + "="*70)
    print("🎉 All tasks completed!")
    print(f"📊 Summary:")
    print(f"   - New outlines generated: {total_processed}")
    print(f"   - Files skipped (outline already existed): {total_skipped}")

if __name__ == "__main__":
    main()
