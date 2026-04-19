import json
import os
from urllib.parse import urlparse

def process_autosurvey_file(input_path: str, output_path: str):
    """
    处理 'autosurvey' 方法生成的 survey_results.json 文件。
    """
    print(f"正在以 [autosurvey] 模式处理: {input_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"🟡 跳过：输入文件不存在。")
        return False

    # 读取并解析JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 自动查找包含 '_paper_info' 的键
    paper_info_key = next((key for key in data if key.endswith('_paper_info')), None)
    if not paper_info_key:
        print("❌ 错误：在源JSON文件中未找到 '_paper_info' 键。")
        return False

    source_papers = data[paper_info_key]
    new_data = {}
    
    # 遍历并转换数据
    for i, (paper_id, details) in enumerate(source_papers.items(), 1):
        paper_key = f"paper_{i}_info"
        ref_key = f"reference_{i}"
        new_data[paper_key] = {
            ref_key: {
                "searched_title": details.get("title", "N/A"),
                "scholar_status": "Success (Converted from source)", 
                "arxiv_id": details.get("id", "N/A"),
                "url": details.get("url", "N/A"),
                "date": details.get("date", "N/A"),
                "abs": details.get("abs", "N/A"),
                "authors": details.get("authors", []),
                "publication": "arxiv.org" if "arxiv.org" in details.get("url", "") else "N/A",
                "citation_count": "N/A"
            }
        }
    
    new_data["reference_num"] = len(source_papers)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 文件转换成功！总共处理了 {len(source_papers)} 篇论文。")
    print(f"   结果已保存至：{output_path}")
    return True

def process_surveyforge_file(input_path: str, output_path: str, exp_id: int):
    """
    处理 'surveyforge' 方法生成的 survey_{id}_references_found.json 文件。
    """
    print(f"正在以 [surveyforge] 模式处理: {input_path}")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"🟡 跳过：输入文件不存在。")
        return False
    
    # 读取并解析JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 根据实验ID动态构建关键字段名
    paper_info_key = f"survey_{exp_id}_paper_info"
    if paper_info_key not in data:
        print(f"❌ 错误：在源JSON文件中未找到预期的键 '{paper_info_key}'。")
        return False
        
    source_papers = data[paper_info_key]
    new_data = {}

    # 遍历并转换数据
    for i, (paper_id, details) in enumerate(source_papers.items(), 1):
        paper_key = f"paper_{i}_info"
        ref_key = f"reference_{i}"
        
        publication = "N/A"
        if details.get("url"):
            try:
                publication = urlparse(details["url"]).netloc
            except Exception:
                pass # 解析失败则保持 N/A

        new_data[paper_key] = {
            ref_key: {
                "searched_title": details.get("searched_title", "N/A"),
                "scholar_status": details.get("scholar_status", "N/A"),
                "arxiv_id": details.get("id", "N/A"),
                "title": details.get("title", "").strip(),
                "url": details.get("url", "N/A"),
                "date": details.get("date", "N/A"),
                "abs": details.get("abs", "").strip(),
                "authors": details.get("authors", []),
                "publication": publication,
                "citation_count": details.get("citation_count", 0)
            }
        }
        
    new_data["reference_num"] = len(source_papers)

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 文件转换成功！总共处理了 {len(source_papers)} 篇论文。")
    print(f"   结果已保存至：{output_path}")
    return True

def main():
    """
    主函数，根据 METHOD_NAME 的配置，循环处理所有实验文件夹。
    """
    # --- 配置区 ---
    # !!! 只需修改这里，即可切换处理 'autosurvey' 或 'surveyforge' !!!
    METHOD_NAME = "surveyforge"  # 可选项: "autosurvey" 或 "surveyforge"
    
    EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
    EXPERIMENT_IDS = range(2, 21) # 处理 2 到 20 的文件夹
    # --- 结束配置 ---
    
    successful_conversions = 0
    failed_or_skipped = 0

    print(f"🚀 开始批量转换任务，目标方法: [{METHOD_NAME}]")
    print("="*60)
    
    for exp_id in EXPERIMENT_IDS:
        print(f"\n--- 正在处理实验ID: {exp_id} ---")
        base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME)
        
        try:
            if METHOD_NAME == "autosurvey":
                input_path = os.path.join(base_dir, 'survey_results.json')
                output_path = os.path.join(base_dir, 'reference.json')
                if process_autosurvey_file(input_path, output_path):
                    successful_conversions += 1
                else:
                    failed_or_skipped += 1

            elif METHOD_NAME == "surveyforge":
                input_path = os.path.join(base_dir, f'survey_{exp_id}_references_found.json')
                output_path = os.path.join(base_dir, f'reference_{3}.json')
                if process_surveyforge_file(input_path, output_path, exp_id):
                    successful_conversions += 1
                else:
                    failed_or_skipped += 1
            else:
                print(f"❌ 配置错误: 未知的 METHOD_NAME '{METHOD_NAME}'。")
                break # 停止执行
        
        except Exception as e:
            print(f"❌ 处理实验ID {exp_id} 时发生严重错误: {e}")
            failed_or_skipped += 1


    print("\n" + "="*60)
    print("🎉 所有任务已完成！")
    print("\n--- 任务总结 ---")
    print(f"  成功转换的文件数: {successful_conversions}")
    print(f"  失败或跳过的文件数: {failed_or_skipped}")
    print("="*60)

# --- 主程序入口 ---
if __name__ == "__main__":
    main()

