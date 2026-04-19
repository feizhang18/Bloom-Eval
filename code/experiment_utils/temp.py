import os

def create_directory_structure():
    """
    在指定的实验目录结构中，为每个方法的文件夹创建 'bench' 子目录，
    并在其中再创建 'level1' 到 'level7' 的文件夹。
    """
    # --- 配置区 ---
    # 实验文件夹的根路径
    EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
    # 要处理的实验ID范围 (2, 3, ..., 20)
    EXPERIMENT_IDS = range(19, 20)
    # --- 结束配置 ---
    
    print("🚀 开始创建 'bench' 目录结构...")
    print("="*70)

    # 记录处理过的文件夹数量
    processed_method_folders = 0
    
    # 1. 遍历所有的实验ID文件夹 (例如 '2', '3', ...)
    for exp_id in EXPERIMENT_IDS:
        exp_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id))

        # 检查数字文件夹是否存在，不存在则跳过
        if not os.path.isdir(exp_dir):
            print(f"🟡 跳过：实验目录 {exp_dir} 不存在。")
            continue

        print(f"\n📁 正在扫描实验目录: {exp_dir}")

        # 2. 遍历该数字文件夹下的所有子文件夹 (例如 'autosurvey', 'surveyforge')
        try:
            method_names = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
        except FileNotFoundError:
            continue # 以防万一，在处理过程中目录被删除

        for method_name in method_names:
            method_dir = os.path.join(exp_dir, method_name)
            
            # 3. 在每个方法文件夹内创建 'bench' 文件夹
            bench_dir = os.path.join(method_dir, 'bench_1')
            
            # os.makedirs() 可以一次性创建多层目录， exist_ok=True 确保如果目录已存在，脚本不会报错
            os.makedirs(bench_dir, exist_ok=True)
            print(f"  - 已在 '{method_name}' 中创建/确认 'bench' 文件夹存在。")

            # 4. 在 'bench' 文件夹内创建 'level1' 到 'level7'
            for i in range(1, 8):
                level_dir = os.path.join(bench_dir, f'level{i}')
                os.makedirs(level_dir, exist_ok=True)
            
            processed_method_folders += 1

    print("\n" + "="*70)
    print("🎉 所有任务已完成！")
    print(f"📊 总结：总共为 {processed_method_folders} 个方法文件夹创建了 'bench' 目录结构。")
    print("="*70)


# --- 主程序入口 ---
if __name__ == "__main__":
    create_directory_structure()
