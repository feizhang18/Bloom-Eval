import json
import requests
import re
from difflib import SequenceMatcher
import time

# --- 配置 ---
# 请将 'YOUR_SERPAPI_KEY' 替换为您的 SerpApi 密钥
API_KEY = 'b84e414864e68835ad5dced661d30585b8df7231e43cd18c003eeec392feda9e'
INPUT_FILE = '<EXPERIMENT_ROOT>/1/LLMxMapReduce_V2/json/reference_1.json'
OUTPUT_FILE = '<EXPERIMENT_ROOT>/1/LLMxMapReduce_V2/json/reference_2.json'
# 相似度阈值，只有当标题相似度高于此值时才认为匹配成功
SIMILARITY_THRESHOLD = 0.9

def calculate_similarity(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def parse_arxiv_id(url):
    """从 URL 中解析 arXiv ID"""
    if url and 'arxiv.org' in url:
        match = re.search(r'/(?:abs|pdf)/([\d\.]+(?:v\d+)?)', url)
        if match:
            return match.group(1)
    return "N/A"

def search_google_scholar(reference):
    """使用 SerpApi 搜索 Google Scholar 并返回最佳匹配结果"""
    query = f"\"{reference['title']}\""
    if reference['authors']:
        query += f" author:\"{reference['authors'][0]}\""

    params = {"engine": "google_scholar", "q": query, "api_key": API_KEY}

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        results = response.json()

        if "organic_results" not in results or not results["organic_results"]:
            return None, 0.0, "Failed (No results found)"

        best_match, max_similarity = None, 0.0
        for result in results["organic_results"]:
            similarity = calculate_similarity(reference['title'].lower(), result.get('title', '').lower())
            if similarity > max_similarity:
                max_similarity, best_match = similarity, result

        if best_match and max_similarity >= SIMILARITY_THRESHOLD:
            status = f"Success (Similarity: {max_similarity:.2%})"
            return best_match, max_similarity, status
        else:
            status = f"Failed (Best match similarity {max_similarity:.2%} is below threshold)"
            return None, max_similarity, status

    except requests.exceptions.RequestException as e:
        return None, 0.0, f"Failed (API Request Error: {e})"

def main():
    """主函数，处理整个流程"""
    # --- 1. 初始化阶段：创建骨架 JSON 文件 ---
    print(f"Initializing output file '{OUTPUT_FILE}'...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    references = input_data.get('references', [])
    all_references_info = {'reference_num': len(references)}

    # 创建包含所有待处理条目的初始结构
    for i, ref in enumerate(references, 1):
        all_references_info[f"paper_{i}_info"] = {
            f"reference_{i}": {
                "searched_title": ref['title'],
                "scholar_status": "Pending"
            }
        }
    
    # 将初始化的骨架写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_references_info, f, indent=4, ensure_ascii=False)
    
    print("Initialization complete. Starting processing...\n")
    time.sleep(1) # 短暂暂停以便用户看到消息

    # --- 2. 处理和实时更新阶段 ---
    for i, ref in enumerate(references, 1):
        print(f"--- Processing reference {i}/{len(references)}: \"{ref['title']}\" ---")

        best_match, similarity, status = search_google_scholar(ref)
        
        # --- 新增：实时打印搜索结果 ---
        if best_match:
            print(f"  [MATCH FOUND] Similarity: {similarity:.2%}")
            print(f"  > Found Title: \"{best_match.get('title', 'N/A')}\"")
            print(f"  > Abstract: \"{best_match.get('snippet', 'N/A')}\"")
        else:
            print(f"  [NO MATCH] Status: {status}")

        # 准备要更新到文件中的数据
        paper_info = {
            "searched_title": ref['title'],
            "scholar_status": status
        }

        if best_match:
            pub_info = best_match.get("publication_info", {})
            inline_links = best_match.get("inline_links", {})
            cited_by = inline_links.get("cited_by", {})
            
            year_match = re.search(r'\b(19|20)\d{2}\b', pub_info.get("summary", ""))
            year = year_match.group(0) if year_match else ref.get('year', 'N/A')

            publication = "N/A"
            summary_text = pub_info.get("summary", "")
            if " - " in summary_text:
                publication = summary_text.rsplit(' - ', 1)[-1].strip()
            
            if publication == "N/A" or not publication:
                 publication = ref.get("publication", "N/A")

            paper_info.update({
                "arxiv_id": parse_arxiv_id(best_match.get("link")),
                "title": best_match.get("title", "N/A"),
                "url": best_match.get("link", "N/A"),
                "date": year,
                "abs": best_match.get("snippet", "N/A"),
                "authors": [author['name'] for author in pub_info.get("authors", [])] or ref['authors'],
                "publication": publication,
                "citation_count": cited_by.get("total", 0)
            })
        
        # --- 更新内存中的字典，并立即写回文件 ---
        all_references_info[f"paper_{i}_info"][f"reference_{i}"] = paper_info
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_references_info, f, indent=4, ensure_ascii=False)
        
        print(f"  > '{OUTPUT_FILE}' has been updated.")
        print("-" * (len(ref['title']) + 30) + "\n")
        time.sleep(1) # 添加延迟，避免 API 调用过于频繁

    print(f"Processing complete. Final results are in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
