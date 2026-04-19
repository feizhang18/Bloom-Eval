import json
import requests
import re
from difflib import SequenceMatcher
import time
import os
from typing import Dict, Optional
import arxiv # <--- 引入新的库

# --- 1. 全局配置 ---
SIMILARITY_THRESHOLD = 0.8  # 标题匹配相似度阈值为80%

# --- 单个文件夹处理配置 ---
# !!! 重要: 请将此路径修改为您要处理的单个文件夹的绝对路径 !!!
# 例如: '<EXPERIMENT_ROOT>/20/surveyx'
TARGET_DIRECTORY = '<EXPERIMENT_ROOT>/20/LLMxMapReduce_V2' 

# --- 2. 辅助函数 ---

def calculate_similarity(a, b):
    """计算两个字符串的相似度。"""
    return SequenceMatcher(None, a, b).ratio()

def search_arxiv(reference: Dict) -> (Optional[arxiv.Result], float, str):
    """
    使用 arxiv 库搜索并返回最匹配的论文对象、相似度和状态。
    """
    query_title = reference.get('title')
    if not query_title:
        return None, 0.0, "Failed (Input reference has no 'title')"

    print(f"  Searching ArXiv: {query_title}")
    
    try:
        # 使用 arxiv 库进行搜索，精确匹配标题 (ti:)
        # 我们获取5个结果，然后从中找到最相似的那个，以提高准确性
        search = arxiv.Search(
            query=f'ti:"{query_title}"',
            max_results=5
        )
        
        results = list(search.results())

        if not results:
            return None, 0.0, "Failed (No results found on ArXiv)"

        best_match, max_similarity = None, 0.0
        for result in results:
            # 清理标题中的换行符和多余空格
            result_title = ' '.join(result.title.strip().split())
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
        # 捕获包括网络错误在内的所有异常
        print(f"  Error during ArXiv search: {e}")
        return None, 0.0, f"Failed (Error: {e})"

# --- 3. 核心处理逻辑 ---

def process_single_directory(base_path: str):
    """
    处理单个目录中的'reference_1.json'文件，并通过ArXiv API生成'reference_3.json'。
    """
    if not os.path.isdir(base_path):
        print(f"❌ 错误: 目标路径 '{base_path}' 不是一个有效的文件夹。")
        return "error"
        
    input_file = os.path.join(base_path, 'reference_1.json')
    # 输出文件统一命名为 reference_3.json
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
        
    references = input_data.get('references', [])
    if not references:
        print("🟡 警告: 文件中没有找到参考文献，或'references'键不存在。")
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

        # 使用基于 `arxiv` 库的新搜索函数
        best_match, similarity, status = search_arxiv(ref)
        paper_info = {"searched_title": ref_title, "scholar_status": status}

        if best_match: # best_match 现在是一个 arxiv.Result 对象
            print(f"  ✓ 找到匹配项! (相似度: {similarity:.2%})")
            
            # 从 arxiv.Result 对象中提取信息
            authors = [author.name for author in best_match.authors]
            
            paper_info.update({
                "arxiv_id": best_match.get_short_id(),
                "url": best_match.entry_id, # 这是摘要页面的URL
                "date": str(best_match.published), # 将datetime对象转为字符串
                "abs": ' '.join(best_match.summary.strip().split()),
                "authors": authors,  # 使用从ArXiv获取的作者
                "publication": best_match.journal_ref or "arXiv",
                "citation_count": None # ArXiv API不提供引用数，设置为null
            })
        else:
            print(f"  ✗ 未找到匹配项。状态: {status}")

        all_references_info[paper_key] = {ref_key: paper_info}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_references_info, f, indent=4, ensure_ascii=False)
        
        print(f"  > '{os.path.basename(output_file)}' 已更新。")
        # 遵守ArXiv API使用规则，避免请求过快导致连接被重置
        time.sleep(3.1) 

    if processed_in_this_run == 0:
        print("所有参考文献均已处理完毕。")
        return "skipped"

    print(f"\n处理完成。最终结果已保存至 '{os.path.basename(output_file)}'。")
    return "success"

# --- 4. 主执行流程 ---
def main():
    """
    主函数，处理在 TARGET_DIRECTORY 中指定的单个文件夹。
    """
    print(f"🚀 开始通过 ArXiv API 处理文件夹: {TARGET_DIRECTORY}")
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
