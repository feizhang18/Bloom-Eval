# -*- coding: utf-8 -*-
"""
批量 Markdown 文件 Token 计数器

本脚本用于自动遍历指定基础路径下的所有数字子文件夹，
在每个子文件夹的 'human' 目录中找到唯一的 .md 文件，
使用 tiktoken 库计算其 token 数量，并将结果分别保存。
最后，脚本会输出一个总计和平均值的摘要。

核心功能:
1. 自动发现并遍历数字命名的子文件夹。
2. 在每个 'human' 文件夹中动态查找 .md 文件。
3. 对每个文件进行 token 计数。
4. 将每个文件的 token 数量保存在其所在的 'human' 文件夹内 (token_count.txt)。
5. 计算并显示所有文件处理后的总 token 数和平均 token 数。

使用前请确保已安装所需库:
pip install tiktoken
"""
import tiktoken
import os
import glob

# --- 1. 配置区域 ---
# 包含所有数字文件夹的基础路径 (例如: .../experiment)
# 请将其修改为您本机的实际路径
BASE_PATH = r'<EXPERIMENT_ROOT>'


# --- 2. 核心功能 ---

def count_tokens_in_file(file_path: str, tokenizer) -> int:
    """
    使用预先加载的 tiktoken tokenizer 计算指定文件中的tokens总数。

    Args:
        file_path (str): 要分析的文件的路径。
        tokenizer: 已初始化的 tiktoken tokenizer 对象。

    Returns:
        int: 文件内容的总token数。返回 -1 表示发生错误。
    """
    print(f"📖 正在处理文件: {file_path}...")
    try:
        # 使用 utf-8 编码读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用 tokenizer 对文本进行编码，结果是一个整数列表
        tokens = tokenizer.encode(content)
        
        # 列表的长度就是tokens的数量
        num_tokens = len(tokens)
        
        return num_tokens

    except Exception as e:
        print(f"❌ 错误: 读取或处理文件 '{os.path.basename(file_path)}' 时发生异常: {e}")
        return -1

def save_token_count(directory: str, count: int):
    """
    将token数量保存到指定目录下的 'token_count.txt' 文件中。

    Args:
        directory (str): 目标 'human' 文件夹的路径。
        count (int): 要保存的token数量。
    """
    output_path = os.path.join(directory, "token_count.txt")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(count))
        print(f"✅ 结果已成功保存到: {output_path}")
    except Exception as e:
        print(f"❌ 错误: 无法将结果写入文件 {output_path}: {e}")


# --- 3. 主程序执行 ---

if __name__ == "__main__":
    if not os.path.isdir(BASE_PATH):
        print(f"❌ 错误: 基础路径 '{BASE_PATH}' 不存在或不是一个目录。")
        exit()

    # 为现代GPT模型（如gpt-3.5-turbo, gpt-4）选择编码器
    # 'cl100k_base' 是目前最常用的编码方式
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print("✅ Tokenizer 'cl100k_base' 加载成功。\n")
    except Exception as e:
        print(f"❌ 错误: 初始化tokenizer失败: {e}")
        exit()
        
    all_token_counts = []

    # 遍历基础路径下的所有条目
    for folder_name in sorted(os.listdir(BASE_PATH)):
        # 确保只处理数字命名的文件夹
        if folder_name.isdigit():
            human_folder_path = os.path.join(BASE_PATH, folder_name, 'LLMxMapReduce_V2')
            
            if os.path.isdir(human_folder_path):
                # 使用 glob 查找 human 文件夹下唯一的 .md 文件
                md_files = glob.glob(os.path.join(human_folder_path, '*.md'))
                
                if len(md_files) == 1:
                    md_file_path = md_files[0]
                    
                    # 计算token
                    total_tokens = count_tokens_in_file(md_file_path, tokenizer)
                    
                    if total_tokens != -1:
                        # 保存token数量到文件
                        save_token_count(human_folder_path, total_tokens)
                        all_token_counts.append(total_tokens)
                        print(f"   - Token 数量: {total_tokens:,}\n")
                elif len(md_files) == 0:
                    print(f"⚠️ 警告: 在 '{human_folder_path}' 中未找到 .md 文件，已跳过。\n")
                else:
                    print(f"⚠️ 警告: 在 '{human_folder_path}' 中找到多个 .md 文件，已跳过。\n")
            else:
                print(f"⚠️ 警告: 目录 '{human_folder_path}' 不存在，已跳过。\n")

    # --- 4. 输出最终总结 ---
    if all_token_counts:
        total_files = len(all_token_counts)
        grand_total_tokens = sum(all_token_counts)
        average_tokens = grand_total_tokens / total_files
        
        print("\n" + "="*60)
        print("📊 **批量处理总结报告**")
        print("="*60)
        print(f"   处理文件总数: {total_files} 个")
        print(f"   Token 总计: {grand_total_tokens:,} tokens")
        print(f"   平均 Token 数: {average_tokens:,.2f} tokens/文件")
        print("="*60)
    else:
        print("\n⚠️ 未能成功处理任何文件。")
