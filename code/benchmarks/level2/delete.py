import os

# --- 配置 ---
# !!! 请确保这个路径和您主脚本中的路径完全一致 !!!
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'

# --- 安全开关 ---
# True: 只打印将要删除的文件路径，不执行任何删除操作 (推荐首先运行此模式)
# False: 真正执行删除操作
dry_run = False 

# --- 执行脚本 ---
def delete_specific_matches_file():
    """
    遍历所有数字命名的子文件夹，删除其中的 autosurvey/bench/level2/matches.json 文件。
    """
    files_found_count = 0
    
    print(f"--- 开始在 '{EXPERIMENT_ROOT}' 目录中查找目标文件 ---")
    print(f"--- 目标文件: '.../<number>/autosurvey/bench/level2/matches.json' ---")
    
    # 检查根目录是否存在
    if not os.path.isdir(EXPERIMENT_ROOT):
        print(f"错误：根目录 '{EXPERIMENT_ROOT}' 不存在，请检查路径配置。")
        return

    # 遍历根目录下的所有条目
    for item in os.listdir(EXPERIMENT_ROOT):
        # 检查条目是否为数字命名的文件夹
        if item.isdigit() and os.path.isdir(os.path.join(EXPERIMENT_ROOT, item)):
            # 构建目标 'matches.json' 文件的完整路径
            target_file = os.path.join(EXPERIMENT_ROOT, item, 'autosurvey', 'bench', 'level2', 'matches.json')
            
            # 检查目标文件是否存在
            if os.path.isfile(target_file):
                files_found_count += 1
                if dry_run:
                    print(f"[模拟删除] 设想删除文件: {target_file}")
                else:
                    try:
                        os.remove(target_file)
                        print(f"[已删除] {target_file}")
                    except OSError as e:
                        print(f"[删除失败] {target_file} - 原因: {e}")

    if files_found_count == 0:
        print("\n✅ 未找到任何需要删除的目标 'matches.json' 文件。")
        return

    print(f"\n--- 共找到 {files_found_count} 个目标文件 ---")
    
    print("\n--- 操作摘要 ---")
    if dry_run:
        print(f"✅ 模拟运行完成。共 {files_found_count} 个文件被识别。")
        print("如需真正删除，请将脚本中的 'dry_run' 设置为 False 并重新运行。")
    else:
        print(f"✅ 删除操作完成。共 {files_found_count} 个文件被处理。")

if __name__ == "__main__":
    delete_specific_matches_file()