import json
import requests
import re
from difflib import SequenceMatcher
import time
import os
from typing import Dict, List, Optional

# --- 1. 全局配置 ---
# !!! 重要: 请确保这里的API密钥是有效的 !!!
API_KEY = 'b41436d20ae60c07dc9a3f7ebad3f016cce58a94' # 请替换为您的 Serper API 密钥
SIMILARITY_THRESHOLD = 0.8  # 匹配相似度阈值为80%

# --- 单个文件夹处理配置 ---
# !!! 重要: 请将此路径修改为您要处理的单个文件夹的绝对路径 !!!
# 例如: '<EXPERIMENT_ROOT>/1/human'
TARGET_DIRECTORY = '<EXPERIMENT_ROOT>/19/LLMxMapReduce_V2' 

# --- 2. 辅助函数 (无修改) ---

def calculate_similarity(a, b):
    """计算两个字符串的相似度。"""
    return SequenceMatcher(None, a, b).ratio()

def parse_arxiv_id(url):
    """从URL中解析ArXiv ID。"""
    if url and 'arxiv.org' in url:
        match = re.search(r'/(?:abs|pdf)/([\d\.]+(?:v\d+)?)', url)
        if match:
            return match.group(1)
    return "N/A"

def search_google_scholar(reference: Dict) -> (Optional[Dict], float, str):
    """使用 Serper.dev 搜索并返回最匹配的结果、相似度和状态。"""
    query_title = reference.get('title')
    if not query_title:
        return None, 0.0, "Failed (Input reference has no 'title')"

    url = "https://google.serper.dev/scholar"
    headers = {'X-API-KEY': API_KEY, 'Content-Type': 'application/json'}
    payload = json.dumps({"q": query_title})
    
    print(f"  Searching: {query_title}")
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=20)
        if response.status_code != 200:
            return None, 0.0, f"Failed (HTTP {response.status_code})"
            
        results = response.json()
        organic_results = results.get("organic", [])
        if not organic_results:
            return None, 0.0, "Failed (No results found)"

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
            return best_match, max_similarity, status
        elif best_match:
            status = f"Failed (Best match similarity {max_similarity:.2%} is below threshold)"
            return None, max_similarity, status
        else:
            return None, 0.0, "Failed (No valid results with titles)"
            
    except Exception as e:
        print(f"  Error: {e}")
        return None, 0.0, f"Failed (Error: {e})"

# --- 3. 核心处理逻辑 (已修改) ---

def process_single_directory(base_path: str):
    """
    处理单个目录中的'reference_1.json'文件，并生成'reference_2.json'。
    """
    if not os.path.isdir(base_path):
        print(f"❌ 错误: 目标路径 '{base_path}' 不是一个有效的文件夹。")
        return "error"
        
    input_file = os.path.join(base_path, 'reference_1.json')
    output_file = os.path.join(base_path, 'reference_3.json')

    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件未找到于 '{input_file}'")
        return "error"

    print(f"正在读取: '{os.path.basename(input_file)}'...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ 错误: JSON文件格式无效: '{input_file}'")
        return "error"
        
    # --- 主要修改点 ---
    # 直接从根对象的 "references" 键获取文献列表
    references = input_data.get('references', [])
    
    if not isinstance(references, list) or not references:
        print("🟡 警告: 文件中没有找到参考文献列表，或 'references' 键不存在/为空。")
        return "skipped"
    
    print(f"找到 {len(references)} 篇参考文献。开始处理...")
    
    all_references_info = {'reference_num': len(references)}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                all_references_info = json.load(f)
            print("从已保存的进度中恢复...")
        except (json.JSONDecodeError, IOError):
            print("🟡 警告: 无法读取已存在的输出文件，将重新开始。")

    processed_in_this_run = 0
    for i, ref in enumerate(references, 1):
        paper_key = f"paper_{i}_info"
        ref_key = f"reference_{i}"
        
        if paper_key in all_references_info and all_references_info.get(paper_key, {}).get(ref_key, {}).get("scholar_status", "").startswith("Success"):
            continue

        processed_in_this_run += 1
        ref_title = ref.get('title', 'No Title Found in Input')
        print(f"--- 正在处理第 {i}/{len(references)} 篇: \"{ref_title}\" ---")

        best_match, similarity, status = search_google_scholar(ref)
        paper_info = {"searched_title": ref_title, "scholar_status": status}

        if best_match:
            print(f"  ✓ 找到匹配项! (相似度: {similarity:.2%})")
            pub_info_str = best_match.get("publicationInfo", "")
            authors_list = []
            publication = "N/A"
            
            if isinstance(pub_info_str, str) and ' - ' in pub_info_str:
                parts = pub_info_str.split(' - ', 1)
                authors_str = parts[0]
                authors_list = [author.strip() for author in authors_str.split(',') if author.strip()]
                publication = parts[1].strip()
            # --- 修复逻辑开始 ---
            # 先安全地获取 citedBy 的值
            cited_by_value = best_match.get("citedBy")
            citation_count = 0 # 默认值

            # 判断它的类型
            if isinstance(cited_by_value, dict):
                # 如果是字典，就从里面获取 total
                citation_count = cited_by_value.get("total", 0)
            elif isinstance(cited_by_value, int):
                # 如果直接是整数，就直接使用
                citation_count = cited_by_value
            
            paper_info.update({
                "arxiv_id": parse_arxiv_id(best_match.get("link")),
                "url": best_match.get("link", "N/A"),
                "date": str(best_match.get("year", "N/A")),
                "abs": best_match.get("snippet", "N/A"),
                "authors": authors_list,
                "publication": publication,
                "citation_count": citation_count # 使用我们安全获取到的值
            })

        else:
            print(f"  ✗ 未找到匹配项。状态: {status}")

        all_references_info[paper_key] = {ref_key: paper_info}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_references_info, f, indent=4, ensure_ascii=False)
        
        print(f"  > '{os.path.basename(output_file)}' 已更新。")
        time.sleep(1.2)

    if processed_in_this_run == 0:
        print("所有参考文献均已处理完毕。")
        return "skipped"

    print(f"\n处理完成。最终结果已保存至 '{os.path.basename(output_file)}'。")
    return "success"

# --- 4. 主执行流程 (无修改) ---
def main():
    """
    主函数，处理在 TARGET_DIRECTORY 中指定的单个文件夹。
    """
    if not API_KEY or API_KEY == 'Your_API_Key_Here':
        print("❌ 错误: 请在代码第9行设置您的有效Serper API密钥。")
        return
        
    if not TARGET_DIRECTORY or TARGET_DIRECTORY == '/path/to/your/target/folder':
        print("❌ 错误: 请在代码第14行设置您要处理的目标文件夹 'TARGET_DIRECTORY'。")
        return

    print(f"🚀 开始处理文件夹: {TARGET_DIRECTORY}")
    print("="*70)
    
    status = process_single_directory(TARGET_DIRECTORY)
    
    print("="*70)
    if status == "success":
        print("🎉 任务成功完成!")
    elif status == "skipped":
        print("⏭️  任务跳过 (所有文献可能已处理完毕)。")
    else: # status == "error"
        print("❌ 任务因错误而终止。请检查上面的错误信息。")

if __name__ == "__main__":
    main()