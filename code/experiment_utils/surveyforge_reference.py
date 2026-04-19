import json
import re
import os
from thefuzz import process, fuzz

# --- 1. 配置区域 ---

# ArXiv论文数据库JSON文件的路径
DATABASE_PATH = r'<SURVEYFORGE_DATABASE_JSON>'

# 要处理的文献目录路径模板
# 我们将使用 {i} 作为占位符来循环遍历2到20
BASE_REFERENCE_DIR_TEMPLATE = r'<EXPERIMENT_ROOT>/{i}/surveyforge/'

# 设置一个匹配度阈值，只有相似度高于此值的才被认为是成功匹配
SIMILARITY_THRESHOLD = 80 # 0-100之间，建议90以上以确保准确性

# --- 2. 数据加载与预处理 ---

def clean_title(title):
    """清理参考文献标题，移除 "[数字]" 前缀和首尾空格"""
    # 使用正则表达式移除如 "[1]", "[23]" 等格式的前缀
    cleaned = re.sub(r'^\[\d+\]\s*', '', title)
    return cleaned.strip().lower() # 转换为小写以进行不区分大小写的比较

def load_paper_database(path):
    """加载庞大的论文数据库并进行预处理，为快速搜索做准备"""
    print(f"正在加载大型论文数据库: {path}")
    print("这个过程可能需要一些时间，请稍候...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 数据库的实际内容在 "cs_paper_info" 键下
        paper_dict = data.get("cs_paper_info", {})
        
        # 创建一个从清理后的标题到完整论文信息的映射，以及一个所有标题的列表
        title_to_paper_map = {}
        all_titles = []

        for paper in paper_dict.values():
            if 'title' in paper:
                # 同样对数据库中的标题进行清理，以提高匹配率
                cleaned_db_title = paper['title'].strip().lower()
                title_to_paper_map[cleaned_db_title] = paper
                all_titles.append(cleaned_db_title)
        
        print(f"数据库加载完成，共处理 {len(all_titles)} 篇论文。")
        return title_to_paper_map, all_titles
    except FileNotFoundError:
        print(f"错误：数据库文件未找到 at '{path}'")
        return None, None
    except json.JSONDecodeError:
        print(f"错误：数据库文件 '{path}' 不是有效的JSON格式。")
        return None, None

# --- 3. 主处理逻辑 ---

def process_single_reference_file(references_path, title_map, all_db_titles):
    """
    处理单个参考文献JSON文件。
    这个函数会被主循环多次调用。
    """
    if not os.path.exists(references_path):
        print(f"警告：文件不存在，跳过: {references_path}")
        return

    print(f"\n{'='*20} 开始处理文件: {references_path} {'='*20}")

    # 加载参考文献列表
    try:
        with open(references_path, 'r', encoding='utf-8') as f:
            reference_titles = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：参考文献文件 '{references_path}' 不是有效的JSON格式。跳过此文件。")
        return
        
    found_papers_info = {}
    found_count = 0

    # 遍历每一篇参考文献
    for ref_title in reference_titles:
        cleaned_ref_title = clean_title(ref_title)
        
        # 使用 fuzzywuzzy 进行模糊搜索
        best_match = process.extractOne(
            cleaned_ref_title, 
            all_db_titles, 
            scorer=fuzz.token_sort_ratio
        )
        
        if best_match and best_match[1] >= SIMILARITY_THRESHOLD:
            matched_title, score = best_match
            paper_data = title_map[matched_title].copy()
            
            paper_data["searched_title"] = ref_title
            paper_data["scholar_status"] = f"Success (Similarity: {score:.2f}%)"
            
            found_papers_info[paper_data['id']] = paper_data
            found_count += 1
            print(f"  [成功] 找到: '{ref_title}' (匹配度: {score:.2f}%)")
        else:
            if best_match:
                 print(f"  [失败] 未找到: '{ref_title}' (最佳匹配 '{best_match[0]}' 相似度仅 {best_match[1]:.2f}%, 未达到阈值 {SIMILARITY_THRESHOLD}%)")
            else:
                 print(f"  [失败] 未找到: '{ref_title}' (数据库中无任何匹配项)")

    # --- 生成结果并输出 ---
    
    # 从路径中动态提取 survey 序号
    try:
        survey_name = os.path.basename(os.path.dirname(os.path.dirname(references_path)))
    except Exception:
        survey_name = "unknown"


    final_output = {
        "reference_num": len(reference_titles),
        f"survey_{survey_name}_paper_info": found_papers_info
    }

    # 将结果保存到与reference.json相同的目录下
    output_dir = os.path.dirname(references_path)
    output_filename = os.path.join(output_dir, f"survey_{survey_name}_references_found.json")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"\n--- 文件处理完成 ---")
    print(f"结果已保存到: {output_filename}")
    
    total_references = len(reference_titles)
    print(f"搜索到的数量: {found_count} / {total_references}")
    print(f"是否全部成功搜索到: {'是' if found_count == total_references else '否'}")


def main():
    """主函数：执行整个批量查找和生成过程"""
    # 首先，一次性加载大型数据库
    title_map, all_db_titles = load_paper_database(DATABASE_PATH)
    if title_map is None:
        print("数据库加载失败，程序终止。")
        return

    # 循环处理序号为 2 到 20 的文件
    for i in range(2, 21):
        # 构建当前要处理的 reference.json 文件的完整路径
        current_dir = BASE_REFERENCE_DIR_TEMPLATE.format(i=i)
        references_path = os.path.join(current_dir, 'reference.json')
        
        # 调用函数处理这个文件
        process_single_reference_file(references_path, title_map, all_db_titles)
    
    print(f"\n{'='*25} 所有任务已完成 {'='*25}")

# --- 运行主程序 ---
if __name__ == "__main__":
    main()
