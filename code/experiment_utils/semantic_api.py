import json
import time
import os
from semanticscholar import SemanticScholar

def get_references_json_fill_mode(api_key: str, paper_title: str, output_filename: str = 'reference_3.json'):
    """
    使用 API 密钥，根据论文标题获取其所有参考文献的详细信息。
    该函数会先创建一个包含所有参考文献骨架的 JSON 文件，然后逐条填充并实时保存。

    参数:
    api_key (str): 你的 Semantic Scholar API 密钥。
    paper_title (str): 你要查询的论文的完整标题。
    output_filename (str): 输出的 JSON 文件名。
    """
    if not api_key:
        print("错误：请通过参数或环境变量提供您的 Semantic Scholar API 密钥。")
        return

    # 1. 使用 API 密钥初始化 Semantic Scholar 客户端
    s2 = SemanticScholar(api_key=api_key, timeout=30)
    print(f"正在使用 API 密钥搜索目标论文: '{paper_title}'...")

    try:
        # 2. 根据标题搜索目标论文
        search_results = s2.search_paper(paper_title, limit=1)
        if not search_results:
            print(f"错误：找不到标题为 '{paper_title}' 的论文。")
            return

        main_paper = search_results[0]
        print(f"成功找到论文: '{main_paper.title}', ID: {main_paper.paperId}")

        # 3. 获取目标论文的参考文献列表
        main_paper_details = s2.get_paper(main_paper.paperId, fields=['references', 'references.title'])
        references = main_paper_details.references
        
        if not references:
            print(f"论文 '{main_paper.title}' 没有可获取的参考文献。")
            return
        
        ref_count = len(references)
        print(f"共找到 {ref_count} 篇参考文献。")

        # 4. 创建骨架 JSON 结构
        output_data = {"reference_num": ref_count}
        for i, ref in enumerate(references):
            paper_key = f"paper_{i+1}_info"
            reference_key = f"reference_{i+1}"
            output_data[paper_key] = {
                reference_key: {
                    "searched_title": ref.title,
                    "scholar_status": "Pending", # 初始状态为待处理
                    "arxiv_id": "N/A",
                    "url": "N/A",
                    "date": "N/A",
                    "abs": "N/A",
                    "authors": [],
                    "publication": "N/A",
                    "citation_count": 0
                }
            }
        
        # 5. 先将骨架 JSON 写入文件
        print(f"正在创建骨架文件 '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print("骨架文件创建成功，开始逐条填充详细信息...")

        # 6. 遍历每一篇参考文献，获取详细信息并实时更新 JSON 文件
        for i, ref in enumerate(references):
            print(f"\n正在处理第 {i+1}/{ref_count} 篇: '{ref.title}'")
            paper_key = f"paper_{i+1}_info"
            reference_key = f"reference_{i+1}"
            
            # 为了防止 API 请求过于频繁，可以加入短暂的延时
            if i > 0 and i % 15 == 0:
                print("...为防止API超速，暂停5秒...")
                time.sleep(5)

            try:
                # 获取参考文献的详细信息
                ref_details = s2.get_paper(ref.paperId, fields=[
                    'title', 'abstract', 'year', 'authors', 
                    'venue', 'url', 'externalIds', 'citationCount'
                ])

                authors = [author['name'] for author in ref_details.authors] if ref_details.authors else []
                display_authors = [authors[0], "et al."] if len(authors) > 1 else authors
                arxiv_id = ref_details.externalIds.get('ArXiv', 'N/A') if ref_details.externalIds else 'N/A'

                # 更新字典中的信息
                filled_info = {
                    "searched_title": ref.title,
                    "scholar_status": "Success (Similarity: 100.00%)",
                    "arxiv_id": arxiv_id,
                    "url": ref_details.url,
                    "date": str(ref_details.year) if ref_details.year else "N/A",
                    "abs": ref_details.abstract if ref_details.abstract else "N/A",
                    "authors": display_authors,
                    "publication": ref_details.venue if ref_details.venue else "N/A",
                    "citation_count": ref_details.citationCount if ref_details.citationCount is not None else 0
                }
                output_data[paper_key][reference_key] = filled_info
                print(f"  -> 成功获取。")

            except Exception as e:
                # 如果获取失败，也在 JSON 中记录失败状态
                output_data[paper_key][reference_key]['scholar_status'] = f"Failed ({e})"
                print(f"  -> 获取失败，原因: {e}")
            
            # 7. 每次更新后，都将整个字典写回文件
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n全部处理完成！最终结果已保存在文件 '{output_filename}' 中。")

    except Exception as e:
        print(f"发生严重错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 请通过环境变量提供你的 Semantic Scholar API 密钥
    MY_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    
    # 2. 请在这里替换为你需要查询的论文的准确标题
    target_paper_title = "Random Quantum Circuits"
    
    # 3. 运行主函数，输出文件将是 reference_3.json
    get_references_json_fill_mode(api_key=MY_API_KEY, paper_title=target_paper_title)
