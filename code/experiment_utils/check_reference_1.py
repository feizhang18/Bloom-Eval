import json
import os

# --- 1. 全局配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'LLMxMapReduce_V2'        # 目标子文件夹
TARGET_FILENAME = 'reference_1.json' # 目标文件名

# --- 2. 核心检查函数 (无修改) ---
def check_json_reference_count(file_path):
    """
    检查JSON文件中的 'total_references' 字段是否与 'references' 列表的实际文章数量一致。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'total_references' not in data or 'references' not in data:
            print(f"  -> ❌ 错误: 文件中缺少 'total_references' 或 'references' 键。")
            return False

        declared_total = data['total_references']
        actual_count = len(data['references'])

        print(f"  -> 'total_references' 声明的数量: {declared_total}")
        print(f"  -> 'references' 数组中实际的文章数量: {actual_count}")

        if declared_total == actual_count:
            print("  -> ✅ 结果: 一致。")
            return True
        else:
            print(f"  -> ❌ 结果: 不一致！")
            return False

    except FileNotFoundError:
        print(f"  -> ❌ 错误: 找不到文件。")
        return False
    except json.JSONDecodeError:
        print(f"  -> ❌ 错误: 文件不是有效的JSON格式。")
        return False
    except TypeError:
        print(f"  -> ❌ 错误: 'references' 字段的值不是一个列表。")
        return False
    except Exception as e:
        print(f"  -> ❌ 发生了一个意料之外的错误: {e}")
        return False

# --- 3. 主执行流程 ---
def main():
    """
    主函数，负责遍历所有实验文件夹并调用检查函数。
    """
    print(f"🚀 开始批量检查 '{TARGET_FILENAME}' 文件...")
    consistent_count = 0
    inconsistent_count = 0
    error_count = 0

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*20} 正在处理实验ID: {exp_id} {'='*20}")
        
        # 动态构建文件路径
        json_file_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, TARGET_FILENAME)
        
        print(f"正在检查文件: {json_file_path}")

        # 调用检查函数并根据结果更新计数器
        result = check_json_reference_count(json_file_path)
        
        if result is True:
            consistent_count += 1
        else:
            # 如果文件找不到或格式错误，都算作不一致/错误
            inconsistent_count +=1


    # 打印最终总结报告
    print(f"\n\n{'='*20} 🎉 所有检查已完成! {'='*20}")
    print("📊 最终统计:")
    print(f"   - ✅ 数量一致的文件: {consistent_count}")
    print(f"   - ❌ 数量不一致或出错的文件: {inconsistent_count}")
    print("="*55)


if __name__ == '__main__':
    main()
