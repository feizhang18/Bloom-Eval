import json
import re
import csv
import nltk

# --- 新增代码：明确告诉NLTK数据在哪里 ---
# 这样可以确保无论在什么环境下运行，NLTK都能找到您手动下载的数据包
# Configure NLTK data locally if needed, for example via NLTK_DATA.
# ------------------------------------

# --- 1. 配置: 请确保文件路径正确 ---
# 请将这些路径更改为您计算机上的实际文件路径
PAPER_JSON_PATH = '<EXPERIMENT_ROOT>/1/autosurvey/2_A_Comprehensive_Survey_on_Vision_Transformers.json'
REFERENCES_JSON_PATH = '<EXPERIMENT_ROOT>/1/autosurvey/reference_3.json'
# 使用新的输出文件名，以体现其“分组”特性
OUTPUT_CSV_PATH = '<EXPERIMENT_ROOT>/1/autosurvey/bench/level2/sentences_grouped_for_precision_eval.csv'

def build_citation_info_from_structured_json(ref_data):
    """
    (修改版) 从结构化的JSON文件中构建引用编号到其详细信息（标题和摘要）的映射表。
    """
    print("--- Building citation info map (title and abstract) from structured JSON file... ---")
    citation_info = {}
    total_papers = ref_data.get("reference_num", 0)

    for i in range(1, total_papers + 1):
        paper_num_str = str(i)
        paper_info_key = f"paper_{paper_num_str}_info"
        reference_key = f"reference_{paper_num_str}"

        try:
            # 获取引用信息
            reference_details = ref_data[paper_info_key][reference_key]
            title = reference_details.get("searched_title")
            # 获取摘要信息，如果不存在或为"N/A"，则设为空字符串
            abstract = reference_details.get("abs", "N/A")
            if abstract == "N/A":
                abstract = "" # 将"N/A"转换为空字符串，使CSV更整洁

            if title and title != "N/A":
                citation_info[paper_num_str] = {
                    "title": title,
                    "abstract": abstract
                }
        except KeyError:
            print(f"Warning: Could not find info for reference number {paper_num_str}")

    print(f"✅ Successfully created a map with info for {len(citation_info)} citations.")
    return citation_info

def main():
    """
    主函数，加载数据、处理并生成最终的“分组”CSV文件，以便于计算精确率。
    """
    # --- Step 1: 加载JSON文件 ---
    print("--- Loading JSON files... ---")
    try:
        with open(PAPER_JSON_PATH, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        with open(REFERENCES_JSON_PATH, 'r', encoding='utf-8') as f:
            references_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the file paths are correct.")
        return

    # --- Step 2: 创建引用信息映射表 (包含标题和摘要) ---
    citation_info_map = build_citation_info_from_structured_json(references_data)

    # --- Step 3: 提取并“清洗”文本 ---
    print("--- Extracting and CLEANING text... ---")
    full_text = " ".join(paper_data.get('context', {}).values())
    full_text_cleaned = re.sub(r'-\n', '', full_text)
    full_text_cleaned = re.sub(r'\s+', ' ', full_text_cleaned)

    # ==========================================================
    # ==      REPLACE THE NLTK SECTION WITH THIS BLOCK        ==
    # ==========================================================
    try:
        # First, try to use the tokenizer. If it works, do nothing.
        sentences = nltk.sent_tokenize(full_text_cleaned)
    except LookupError:
        # If it fails, it means the data is missing.
        print("NLTK 'punkt' tokenizer not found. Attempting to download...")
        try:
            nltk.download('punkt')
            print("Download successful. Retrying tokenization...")
            # Retry tokenizing after download
            sentences = nltk.sent_tokenize(full_text_cleaned)
        except Exception as e:
            print(f"Automatic download failed: {e}")
            print("Please try downloading manually in a Python terminal:")
            print(">>> import nltk")
            print(">>> nltk.download('punkt')")
            return # Exit the script if download fails
    # ==========================================================

    print(f"Found {len(sentences)} sentences to process after cleaning.")
    # ==========================================================
    # ==      Step 4: (核心修改) 查找句子并按 sentence_id 分组     ==
    # ==========================================================
    output_data = []
    citation_marker_pattern = re.compile(r'\[([\d,\s;]+)\]')
    sentence_id_counter = 0 # <--- 新增：句子ID计数器

    print("--- Grouping sentences and references by sentence_id... ---")
    for sentence in sentences:
        clean_sentence = sentence.strip()
        markers = citation_marker_pattern.findall(clean_sentence)

        if not markers:
            continue

        # 如果句子中包含引用，分配一个新的ID
        sentence_id_counter += 1
        
        all_citation_numbers = set()
        for marker_group in markers:
            numbers = re.split(r'[,;]\s*', marker_group)
            for num in numbers:
                if num.strip().isdigit():
                    all_citation_numbers.add(num.strip())

        # 为这个句子的每一个引用创建一条记录
        for num in sorted(list(all_citation_numbers), key=int):
            info = citation_info_map.get(num)
            if info:
                title = info.get("title", "")
                abstract = info.get("abstract", "")
                formatted_reference = f"[{num}] {title}"
                
                # 添加到输出列表，包含新的 sentence_id
                output_data.append({
                    'sentence_id': sentence_id_counter, # <--- 新增
                    'sentence': clean_sentence,
                    'references': formatted_reference,
                    'abstract': abstract
                })

    # --- Step 5: 将分组后的结果写入新的CSV文件 ---
    if not output_data:
        print("Warning: No sentences with valid citations were found. The output CSV will be empty.")
        return

    print(f"--- Writing {len(output_data)} rows to {OUTPUT_CSV_PATH}... ---")
    # 新增: 在fieldnames中加入 'sentence_id' 并置于首位
    fieldnames = ['sentence_id', 'sentence', 'references', 'abstract']
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)

    print(f"🎉 Success! The file '{OUTPUT_CSV_PATH}' has been created with the new grouped format.")

if __name__ == '__main__':
    main()