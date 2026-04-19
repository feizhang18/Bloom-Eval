import json
import re
import os
import sys
# 会覆盖为 citation = NULL


# ==============================================================================
# 1. 辅助函数：清理文件名
# ==============================================================================
def sanitize_filename(title):
    if not title: return "untitled"
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
    safe_title = re.sub(r'\s+', '_', safe_title)
    safe_title = re.sub(r'\.+', '.', safe_title)
    return safe_title[:100]

# ==============================================================================
# 2. 核心提取逻辑
# ==============================================================================

# --- 方法一：从 .md 文件提取 (首选方案) ---
def extract_references_from_md(md_path):
    if not os.path.exists(md_path): return None
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'(?:^|\n)#* *[Rr]eferences\s*(.*?)(?=\n#+\s*|$)', content, re.DOTALL)
        if not match: return None
        references_text = match.group(1).strip()
        if not references_text: return None

        bracketed_refs = re.findall(r'(\[.*?\][^\[]*)', references_text, re.DOTALL)
        if bracketed_refs:
            cleaned_refs = [item.strip() for item in bracketed_refs if item.strip()]
            if cleaned_refs: return cleaned_refs
        
        paragraphs = re.split(r'\n\s*\n', references_text)
        if len(paragraphs) > 1:
            paragraph_refs = [re.sub(r'\s*\n\s*', ' ', p.strip()) for p in paragraphs if p.strip()]
            if paragraph_refs: return paragraph_refs

        line_refs = [line.strip() for line in references_text.splitlines() if line.strip()]
        if line_refs: return line_refs
        return None
    except Exception as e:
        print(f"    - ERROR: 读取或解析 .md 文件时出错 {md_path}: {e}", file=sys.stderr)
        return None

# --- 方法二：从 _content_list.json 文件提取 (强大的备用方案) ---
def extract_references_from_json(content_list):
    start_index = -1
    for i, item in enumerate(content_list):
        if item.get("type") == "text" and "REFERENCES" in item["text"].upper():
            start_index = i
            break
    
    if start_index == -1:
        return None

    # 从找到 "REFERENCES" 的那个 item 开始，拼接所有后续的 text item
    full_references_text = ""
    
    # 1. 处理包含 "REFERENCES" 的初始 item
    initial_text = content_list[start_index]["text"]
    # 移除 "REFERENCES" 标题行本身
    text_after_header = re.sub(r'.*REFERENCES.*\n?', '', initial_text, flags=re.IGNORECASE)
    full_references_text += text_after_header.strip() + "\n"

    # 2. 拼接后续所有 text item，直到遇到下一个主要章节标题
    for i in range(start_index + 1, len(content_list)):
        item = content_list[i]
        if item.get("type") != "text":
            continue
        
        # 停止条件，避免吞掉附录
        current_text = item["text"].strip()
        if item.get("text_level") == 1 and any(keyword in current_text.upper() for keyword in ["APPENDIX", "ACKNOWLEDGEMENTS"]):
            break
            
        full_references_text += current_text + "\n"
    
    # 3. 按行分割最终拼接好的文本
    references = [line.strip() for line in full_references_text.splitlines() if line.strip()]
    return references if references else None

def extract_paper_info(input_json_path, paper_id, publication_name):
    result = {
        "id": str(paper_id), "label": "survey paper", "publication": publication_name,
        "title": None, "date": None, "authors": [], "citation_count": None,
        "arxiv_id": None, "doi_id": None, "context": {}, "reference": []
    }

    basename_full = os.path.basename(input_json_path)
    basename = basename_full.replace('_content_list.json', '')
    current_dir = os.path.dirname(input_json_path)
    
    # --- 参考文献提取逻辑 ---
    final_references = []
    
    # 1. 优先尝试从 .md 提取
    md_path = os.path.join(current_dir, f"{basename}.md")
    md_references = extract_references_from_md(md_path)

    # 2. 检查 .md 结果
    if md_references and len(md_references) >= 5:
        final_references = md_references
    else:
        # 3. .md 失败或不足5条，回退至 .json
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                content_list_for_refs = json.load(f)
            json_references = extract_references_from_json(content_list_for_refs)
            
            # 无论json提取结果如何，都接受它作为最终结果
            # （如果md有3条，json有50条，我们用json的。如果md有3条，json是None，我们也用json的None）
            final_references = json_references if json_references else []

        except Exception as e:
            print(f"    - ERROR: 备用方案 .json 提取过程中出错: {e}", file=sys.stderr)
            final_references = md_references if md_references else [] # 如果json读取失败，保留md的结果

    result["reference"] = final_references
    
    # --- 解析论文元数据和正文 ---
    with open(input_json_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    # ... (元数据提取逻辑保持不变)
    is_parsing_authors = False
    current_section_title = None
    current_section_text_blocks = []
    for item in content_list:
        if item.get("type") != "text": continue
        text_content = item["text"].strip()
        if not text_content: continue
        lower_text = text_content.lower().strip()
        if lower_text.startswith('references') or lower_text.startswith('acknowledgements') or lower_text.startswith('appendix'):
            break
        is_heading = item.get("text_level") == 1
        if is_heading and result["title"] is None:
            result["title"] = text_content
            is_parsing_authors = True
            continue
        if is_parsing_authors and not is_heading:
            result["authors"].append(text_content)
            continue
        if is_parsing_authors and is_heading: is_parsing_authors = False
        if is_heading:
            if current_section_title:
                full_text = "\n\n".join(current_section_text_blocks).strip()
                if full_text: result["context"][current_section_title] = full_text
            current_section_title = text_content
            current_section_text_blocks = []
        elif current_section_title:
            current_section_text_blocks.append(text_content)
    if current_section_title:
        full_text = "\n\n".join(current_section_text_blocks).strip()
        if full_text: result["context"][current_section_title] = full_text
        
    return result

# ==============================================================================
# 3. 主程序：遍历目录并处理
# ==============================================================================
def main():
    root_input_dir = "<EXPERIMENT_ROOT>/19"
    output_dir = "<EXPERIMENT_ROOT>/19/LLMxMapReduce_V2/json"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出将保存到: {output_dir}")

    paper_counter = 1
    processed_count, failed_count, low_reference_count = 0, 0, 0
    low_reference_papers = []

    all_publication_dirs = sorted([d for d in os.listdir(root_input_dir) if os.path.isdir(os.path.join(root_input_dir, d))])

    for publication_name in all_publication_dirs:
        publication_path = os.path.join(root_input_dir, publication_name)
        print(f"\n--- 正在扫描出版物: {publication_name} ---")
        all_paper_dirs = sorted([d for d in os.listdir(publication_path) if os.path.isdir(os.path.join(publication_path, d))])
        
        for paper_dir_name in all_paper_dirs:
            paper_path = os.path.join(publication_path, paper_dir_name)
            json_filename = f"{paper_dir_name}_content_list.json"
            json_filepath = os.path.join(paper_path, "auto", json_filename)
            
            if os.path.exists(json_filepath):
                print(f"[{paper_counter}] 正在处理: {paper_dir_name}")
                try:
                    paper_data = extract_paper_info(json_filepath, paper_counter, publication_name)
                    if paper_data and paper_data.get("title"):
                        ref_count = len(paper_data.get("reference", []))
                        
                        # 根据新标准（小于5）进行统计
                        if ref_count < 5:
                            low_reference_count += 1
                            low_reference_papers.append(f"{publication_name}/{paper_dir_name}")
                            print(f"    -> 警告: 最终参考文献数量为 {ref_count} (< 5)。")

                        safe_title = sanitize_filename(paper_data["title"])
                        output_filename = f"{paper_counter}_{safe_title}.json"
                        output_filepath = os.path.join(output_dir, output_filename)

                        with open(output_filepath, 'w', encoding='utf-8') as f:
                            json.dump(paper_data, f, indent=2, ensure_ascii=False)
                        
                        print(f"    -> 成功! 已保存为: {output_filename} (参考文献数: {ref_count})")
                        processed_count += 1
                    else:
                        print(f"    -> 失败: 未能从文件中提取到标题。", file=sys.stderr)
                        failed_count += 1
                    paper_counter += 1
                except Exception as e:
                    print(f"    -> 发生严重错误处理 {paper_dir_name}: {e}", file=sys.stderr)
                    failed_count += 1
                    paper_counter += 1
            
    print("\n==============================================")
    print("批处理完成!")
    print(f"成功处理: {processed_count} 篇论文")
    print(f"处理失败: {failed_count} 篇论文")
    print(f"参考文献数量小于5的论文数量: {low_reference_count}")
    print("==============================================")
    
    if low_reference_papers:
        print("\n以下论文的参考文献数量小于5:")
        for paper_name in low_reference_papers:
            print(f"- {paper_name}")

if __name__ == "__main__":
    main()
