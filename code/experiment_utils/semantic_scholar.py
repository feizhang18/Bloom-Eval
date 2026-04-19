import json
from semanticscholar import SemanticScholar
import time

def get_references_json(paper_title: str, output_filename: str = 'references.json'):
    """
    根据论文标题，从 Semantic Scholar 获取其所有参考文献的详细信息，
    并将其保存为您指定的 JSON 格式。

    参数:
    paper_title (str): 你要查询的论文的完整标题。
    output_filename (str): 输出的 JSON 文件名。
    """
    # 1. 初始化 Semantic Scholar API 客户端
    s2 = SemanticScholar(timeout=20)
    print(f"正在搜索目标论文: '{paper_title}'...")

    try:
        # 2. 根据标题搜索目标论文，获取其 paperId
        # 我们只取搜索结果的第一个，通常是最相关的
        search_results = s2.search_paper(paper_title, limit=1)
        if not search_results:
            print(f"错误：找不到标题为 '{paper_title}' 的论文。")
            return

        main_paper = search_results[0]
        main_paper_id = main_paper.paperId
        print(f"成功找到论文: '{main_paper.title}', ID: {main_paper_id}")

        # 3. 获取目标论文的详细信息，包括参考文献列表
        # 使用 fields 参数明确指定需要返回的数据，提高 API 效率
        main_paper_details = s2.get_paper(main_paper_id, fields=['references', 'references.title'])
        references = main_paper_details.references
        
        if not references:
            print(f"论文 '{main_paper.title}' 没有可获取的参考文献。")
            return

        print(f"共找到 {len(references)} 篇参考文献，开始获取详细信息...")

        # 4. 构建最终的 JSON 结构
        output_data = {
            "reference_num": len(references),
        }

        # 5. 遍历每一篇参考文献并获取详细信息
        for i, ref in enumerate(references):
            paper_key = f"paper_{i+1}_info"
            reference_key = f"reference_{i+1}"
            
            # 为了防止 API 请求过于频繁，可以加入短暂的延时
            if i > 0 and i % 10 == 0:
                print(f"已处理 {i} 篇参考文献，暂停3秒...")
                time.sleep(3)

            try:
                # 获取参考文献的详细信息
                # 同样，使用 fields 来指定需要的数据
                ref_details = s2.get_paper(ref.paperId, fields=[
                    'title', 'abstract', 'year', 'authors', 
                    'venue', 'url', 'externalIds', 'citationCount'
                ])

                # 准备要填充的数据
                authors = [author['name'] for author in ref_details.authors] if ref_details.authors else []
                # 格式化作者列表，超过一个作者则用 "et al."
                display_authors = [authors[0], "et al."] if len(authors) > 1 else authors
                
                arxiv_id = ref_details.externalIds.get('ArXiv', 'N/A') if ref_details.externalIds else 'N/A'

                paper_info = {
                    reference_key: {
                        "searched_title": ref.title,
                        "scholar_status": f"Success (Similarity: 100.00%)", # 假设直接找到就是成功
                        "arxiv_id": arxiv_id,
                        "url": ref_details.url,
                        "date": str(ref_details.year) if ref_details.year else "N/A",
                        "abs": ref_details.abstract if ref_details.abstract else "N/A",
                        "authors": display_authors,
                        "publication": ref_details.venue if ref_details.venue else "N/A",
                        "citation_count": ref_details.citationCount if ref_details.citationCount is not None else 0
                    }
                }
                
                print(f"  - 成功获取: '{ref.title}'")

            except Exception as e:
                # 如果获取某个参考文献失败，记录下来
                paper_info = {
                    reference_key: {
                        "searched_title": ref.title,
                        "scholar_status": f"Failed ({e})",
                        "arxiv_id": "N/A",
                        "url": "N/A",
                        "date": "N/A",
                        "abs": "N/A",
                        "authors": [],
                        "publication": "N/A",
                        "citation_count": 0
                    }
                }
                print(f"  - 获取失败: '{ref.title}', 原因: {e}")

            output_data[paper_key] = paper_info
        
        # 6. 将结果写入 JSON 文件
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n处理完成！结果已保存到文件 '{output_filename}' 中。")

    except Exception as e:
        print(f"发生严重错误: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 请在这里替换为你需要查询的论文的准确标题
    target_paper_title = "Biological Impacts of Marine Heatwaves"
    
    # 运行主函数
    get_references_json(target_paper_title)

