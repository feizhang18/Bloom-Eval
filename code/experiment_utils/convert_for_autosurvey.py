import json
import os

def process_single_file(input_path: str, output_path: str):
    """
    读取单个 survey_results.json 文件，转换其格式，并保存为 reference.json。
    这是一个被主函数调用的辅助函数。

    Args:
        input_path (str): 输入的 survey_results.json 文件路径。
        output_path (str): 输出的 reference.json 文件路径。
    """
    print("-" * 60)
    print(f"正在处理输入文件: {input_path}")

    try:
        # 1. 检查输入文件是否存在，不存在则跳过
        if not os.path.exists(input_path):
            print(f"🟡 跳过：输入文件不存在。")
            return False # 返回 False 表示处理失败或跳过

        # 2. 读取输入的JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. 找到包含论文信息的主键 (key)
        # 这个逻辑会自动查找以 '_paper_info' 结尾的键，增加了代码的通用性
        paper_info_key = None
        for key in data.keys():
            if key.endswith('_paper_info'):
                paper_info_key = key
                break
        
        if not paper_info_key:
            print("❌ 错误：在源JSON文件中未找到包含论文信息的键 (例如 '..._paper_info')。")
            return False

        source_papers = data[paper_info_key]

        # 4. 初始化新的JSON结构
        new_data = {}
        paper_counter = 0

        # 5. 遍历并转换每一篇论文
        for i, (paper_id, paper_details) in enumerate(source_papers.items(), 1):
            paper_counter += 1
            
            # 创建新的论文信息结构
            new_paper_info = {
                f"reference_{i}": {
                    "searched_title": paper_details.get("title", "N/A"),
                    "scholar_status": "Success (Converted from source)", 
                    "arxiv_id": paper_details.get("id", "N/A"),
                    "url": paper_details.get("url", "N/A"),
                    "date": paper_details.get("date", "N/A"),
                    "abs": paper_details.get("abs", "N/A"),
                    "authors": paper_details.get("authors", []),
                    "publication": "arxiv.org" if "arxiv.org" in paper_details.get("url", "") else "N/A",
                    "citation_count": "N/A" # 源文件无此字段，设为默认值
                }
            }
            
            # 将转换后的论文信息添加到新数据中
            new_data[f"paper_{i}_info"] = new_paper_info

        # 6. 更新论文总数
        new_data["reference_num"] = paper_counter
        
        # 7. 将转换后的数据写入输出文件
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            # 使用 indent=4 参数使输出的JSON文件格式化，易于阅读
            json.dump(new_data, f, ensure_ascii=False, indent=4)

        print(f"✅ 文件转换成功！总共处理了 {paper_counter} 篇论文。")
        print(f"   结果已保存至：{output_path}")
        return True # 返回 True 表示处理成功

    except json.JSONDecodeError:
        print(f"❌ 错误：无法解析JSON文件。请检查 '{input_path}' 的格式是否正确。")
        return False
    except Exception as e:
        print(f"❌ 处理过程中发生未知错误：{e}")
        return False

def main():
    """
    主函数，循环处理从 2 到 20 的所有实验文件夹。
    """
    # --- 配置区 ---
    experiment_root = '<EXPERIMENT_ROOT>'
    # 定义要处理的实验ID范围，range(2, 21) 代表 2, 3, ..., 20
    experiment_ids = range(3, 21)
    # --- 结束配置 ---
    
    successful_conversions = 0
    failed_or_skipped = 0

    print("🚀 开始批量转换任务...")
    
    for exp_id in experiment_ids:
        # 动态构建输入和输出文件路径
        input_path = os.path.join(experiment_root, str(exp_id), 'autosurvey', 'survey_results.json')
        output_path = os.path.join(experiment_root, str(exp_id), 'autosurvey', 'reference.json')
        
        # 调用核心处理函数
        if process_single_file(input_path, output_path):
            successful_conversions += 1
        else:
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
