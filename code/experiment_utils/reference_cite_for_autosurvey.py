import json
import requests
import re
from difflib import SequenceMatcher
import time
import os

# --- 1. 配置 ---
# !!! 重要: 请确保这里的API密钥是有效的 !!!

API_KEY = 'b41436d20ae60c07dc9a3f7ebad3f016cce58a94'
SIMILARITY_THRESHOLD = 0.8 # 匹配相似度阈值，高于80%则认为匹配成功

# --- 2. 辅助函数 (这部分无需修改) ---

def calculate_similarity(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def search_google_scholar(query_title):
    """
    使用 Serper.dev 搜索并返回引用数和状态信息。
    """
    if not query_title:
        return 0, "Failed (Input reference has no 'title')"

    url = "https://google.serper.dev/scholar"
    headers = {'X-API-KEY': API_KEY, 'Content-Type': 'application/json'}
    payload = json.dumps({"q": query_title})
    
    print(f"  Searching: {query_title}")
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code != 200:
            return 0, f"Failed (HTTP Error {response.status_code})"
            
        results = response.json()
        organic_results = results.get("organic", [])
        if not organic_results:
            return 0, "Failed (No results found)"

        best_match, max_similarity = None, 0.0
        for result in organic_results:
            if not isinstance(result, dict): continue
            result_title = result.get('title', '').strip()
            if result_title:
                similarity = calculate_similarity(query_title.lower(), result_title.lower())
                if similarity > max_similarity:
                    max_similarity, best_match = similarity, result
        
        if best_match and max_similarity >= SIMILARITY_THRESHOLD:
            status = f"Success (Similarity: {max_similarity:.2%})"
            citation_count = best_match.get("citedBy", 0)
            return citation_count, status
        elif best_match:
            status = f"Failed (Best match similarity {max_similarity:.2%} is below threshold)"
            return 0, status
        else:
            return 0, "Failed (No valid results with titles found)"
            
    except Exception as e:
        print(f"  Error: {e}")
        return 0, f"Failed (Error: {e})"

# --- 3. 主程序 ---

def main():
    base_path = '<EXPERIMENT_ROOT>/5/autosurvey/'
    # 假设您的输入文件名为 reference_1.json
    input_file_name = 'reference.json' 
    output_file_name = 'reference_3.json'
    
    input_file = os.path.join(base_path, input_file_name)
    output_file = os.path.join(base_path, output_file_name)

    if not os.path.exists(input_file):
        print(f"错误: 输入文件未找到 '{input_file}'")
        return

    print(f"读取文件: '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
        
    # --- 【核心修改】 ---
    # 解析您提供的 "paper_X_info" 格式，而不是寻找 "references" 列表
    references = []
    # 通过对键进行数字排序，确保处理顺序是 paper_1, paper_2, paper_10...
    sorted_keys = sorted(
        [k for k in input_data.keys() if k.startswith("paper_")],
        key=lambda k: int(re.search(r'\d+', k).group())
    )
    
    for key in sorted_keys:
        # value 的结构是 {"reference_X": {...}}
        value = input_data[key] 
        # list(value.values())[0] 直接提取出里面的论文信息字典
        if value and isinstance(value, dict):
            paper_details = list(value.values())[0]
            references.append(paper_details)
    # --- 修改结束 ---
    
    if not references:
        print("错误：在输入文件中没有找到任何 'paper_X_info' 格式的有效数据。")
        return

    print(f"找到 {len(references)} 篇参考文献，开始处理...")
    print(f"输出将保存至: '{output_file}'.\n")
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_references_info = json.load(f)
        print("检测到已有进度文件，将从上次中断处继续...")
    else:
        all_references_info = {'reference_num': len(references)}

    for i, ref in enumerate(references, 1):
        paper_key = f"paper_{i}_info"
        ref_key = f"reference_{i}"
        
        if paper_key in all_references_info and all_references_info.get(paper_key, {}).get(ref_key, {}).get("scholar_status", "").startswith("Success"):
            print(f"--- ({i}/{len(references)}) 跳过已处理 ---")
            continue
        
        # --- 【核心修改】 ---
        # 从 "searched_title" 键获取标题，而不是 "title"
        ref_title = ref.get('searched_title', 'No Title Found in Input')
        print(f"--- ({i}/{len(references)}) 正在处理: \"{ref_title}\" ---")

        citation_count, status = search_google_scholar(ref_title)
        
        # 复制原始信息，并添加新字段
        paper_info = ref.copy() 
        paper_info['scholar_status'] = status
        paper_info['citation_count'] = citation_count

        if status.startswith("Success"):
            print(f"  ✓ 匹配成功! 引用数: {citation_count}")
        else:
            print(f"  ✗ 未能成功获取引用. 状态: {status}")

        all_references_info[paper_key] = {ref_key: paper_info}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_references_info, f, indent=4, ensure_ascii=False)
        
        print(f"  > 进度已更新至 '{output_file}'.")
        print("-" * 50 + "\n")
        
        time.sleep(1.2)

    print(f"处理完成. 最终结果已保存至 '{output_file}'.")

if __name__ == "__main__":
    main()
