import json
import os

# 1. 定义文件路径
# 包含 "reference" 列表的JSON文件路径
survey_json_path = r"<EXPERIMENT_ROOT>/20/autosurvey/Everything You Wanted to Know about Deep Eutectic Solvents but Were Afraid to Be Told.json"

# ArXiv论文数据库的路径
db_file_path = r"<AUTOSURVEY_DATABASE_JSON>"

# 2. 从文件加载初始的 survey JSON 数据
try:
    with open(survey_json_path, "r", encoding='utf-8') as f:
        initial_data = json.load(f)
except FileNotFoundError:
    print(f"错误: 文件未找到 {survey_json_path}")
    # 如果文件不存在，则退出或使用默认空数据
    initial_data = {}
except json.JSONDecodeError:
    print(f"错误: 文件 {survey_json_path} 不是有效的JSON格式。")
    initial_data = {}


# 3. 统计reference中的条目个数
# 确保 'reference' 键存在 (修正了此处的逻辑)
reference_dict = initial_data.get("reference", {})
reference_num = len(reference_dict)

# 获取所有需要查找的论文ID
reference_ids_to_find = set(reference_dict.values())

# 4. 从数据库文件中读取并筛选论文信息
extracted_paper_info = {}
if os.path.exists(db_file_path):
    try:
        with open(db_file_path, "r", encoding='utf-8') as f:
            arxiv_db = json.load(f)
        
        # 遍历数据库中的每一篇论文
        for key, paper_details in arxiv_db.get("cs_paper_info", {}).items():
            # 检查这篇论文的ID是否在我们想要查找的ID列表中
            if paper_details.get("id") in reference_ids_to_find:
                # 如果是，就将这篇论文的信息添加到我们的结果中
                extracted_paper_info[paper_details["id"]] = paper_details

    except json.JSONDecodeError:
        print(f"错误: 数据库文件 {db_file_path} 不是有效的JSON格式。")
    except Exception as e:
        print(f"读取数据库文件时发生未知错误: {e}")
else:
    print(f"错误: 数据库文件未找到 {db_file_path}")


# 5. 构建最终的JSON对象
final_json_data = {
    "reference_num": reference_num,
    "survey_Everything You Wanted to Know about Deep_paper_info": extracted_paper_info
}

# 6. 打印最终生成的JSON
# 使用indent=4进行格式化输出，ensure_ascii=False确保中文等字符正常显示
print(json.dumps(final_json_data, indent=4, ensure_ascii=False))

# 您也可以选择将结果保存到新文件中
output_filename = "<EXPERIMENT_ROOT>/20/autosurvey/survey_results.json"
with open(output_filename, "w", encoding='utf-8') as f:
    json.dump(final_json_data, f, indent=4, ensure_ascii=False)
print(f"结果已保存到 {output_filename}")
