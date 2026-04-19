import json
import re
import os

def process_text_formulas(text):
    """
    处理文本中的数学公式格式。
    - 优先将特定的行间公式块 $\\begin{array} ... \\end{array}$ 替换为 \\[ ... \\]
    - 然后将行内公式 $...$ 替换为 \(...\)
    """
    # 1. 优先替换特定的行间公式块
    # 使用 re.DOTALL 标志使 '.' 匹配包括换行在内的任何字符
    processed_text = re.sub(r'\$\\begin\{array\}(.*?)\\end\{array\}\$', r'\\\[\\begin{array}\1\\end{array}\\\]', text, flags=re.DOTALL)
    
    # 2. 再替换行内公式
    # 使用非贪婪匹配 '?' 来处理同一行有多个公式的情况
    processed_text = re.sub(r'\$([^$]+?)\$', r'\\(\1\\)', processed_text)
    
    return processed_text

# 1. 定义输入和输出文件路径
input_file_path = "<EXPERIMENT_ROOT>/19/LLMxMapReduce_V2/1_Generalized_Symmetries_in_Condensed_Matter.json"
output_file_path = "<EXPERIMENT_ROOT>/19/LLMxMapReduce_V2/content.json"

try:
    # 2. 从文件路径读取原始JSON数据
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # 3. 按照指定顺序拼接标题和上下文
    all_parts = []

    # Part 1: 添加论文标题
    title = input_data.get("title", "")
    if title:
        all_parts.append(f"# {title}")

    # 获取上下文内容，并创建一个副本以便修改
    context = input_data.get("context", {})
    context_copy = context.copy()

    # Part 2: 单独处理并添加 Abstract
    if "Abstract" in context_copy:
        abstract_content = context_copy.pop("Abstract") # 从副本中获取并移除Abstract
        processed_abstract = process_text_formulas(abstract_content)
        all_parts.append("# Abstract")
        all_parts.append(processed_abstract)

    # Part 3: 添加剩余的上下文部分，并按key排序
    # sorted() 会自然地按 "1 Introduction", "1.1...", "2.1..." 的顺序排序
    for section_title in sorted(context_copy.keys()):
        section_content = context_copy[section_title]
        processed_content = process_text_formulas(section_content)
        
        all_parts.append(f"# {section_title}")
        all_parts.append(processed_content)

    # 使用两个换行符将所有部分连接成一个长字符串
    final_text = "\n\n".join(all_parts)

    # 4. 准备输出数据
    output_data = [final_text]

    # 5. 写入新的JSON文件
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"文件已成功生成在: {output_file_path}")
    print("\n文件内容预览:")
    print(output_data[0][:600] + "...") # 稍微增加预览长度以确认顺序

except FileNotFoundError:
    print(f"错误: 输入文件未找到 at '{input_file_path}'")
except json.JSONDecodeError:
    print(f"错误: 无法解析输入文件 '{input_file_path}'。请检查其是否为有效的JSON格式。")
except Exception as e:
    print(f"处理文件时发生未知错误: {e}")
