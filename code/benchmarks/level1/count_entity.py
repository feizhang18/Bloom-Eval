import json
from collections import defaultdict
import os

def process_and_count_entities(raw_file_path, mapping_file_path, output_file_path):
    """
    使用LLM生成的映射文件，对原始实体列表进行精确计数，并保存结果。
    这是一个核心处理函数，由主循环调用。
    """
    # --- 1. 加载输入文件，并进行前置检查 ---
    if not os.path.exists(raw_file_path) or not os.path.exists(mapping_file_path):
        print(f"❌ 错误：缺少必要的输入文件。请检查：")
        print(f"  - 原始数据文件是否存在: '{raw_file_path}'")
        print(f"  - 映射文件是否存在: '{mapping_file_path}'")
        return False # 返回失败状态

    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            
    except json.JSONDecodeError as e:
        print(f"\n❌ 错误：文件格式不正确，无法解析JSON。")
        print(f"   文件路径: {e.doc_path}") # 提示哪个文件出错了
        print(f"   详细信息: {e}")
        return False # 返回失败状态

    # --- 2. 初始化最终结果的结构 ---
    final_counts = {
        "methods_models": {},
        "datasets": {},
        "evaluation_metrics": {}
    }
    
    # --- 3. 核心计数逻辑 ---
    for category in final_counts.keys():
        raw_entities = raw_data.get(category, [])
        entity_map = mapping_data.get(f"{category}_map", {})
        category_results = {}

        for entity_alias in raw_entities:
            if entity_alias in entity_map:
                canonical_name = entity_map[entity_alias]
                if canonical_name not in category_results:
                    category_results[canonical_name] = {
                        "total_count": 0,
                        "aliases": defaultdict(int)
                    }
                category_results[canonical_name]["total_count"] += 1
                category_results[canonical_name]["aliases"][entity_alias] += 1
        
        for canonical, data in category_results.items():
            data["aliases"] = dict(sorted(data["aliases"].items(), key=lambda item: item[1], reverse=True))

        final_counts[category] = dict(sorted(category_results.items(), key=lambda item: item[1]['total_count'], reverse=True))

    # --- 4. 保存最终报告 ---
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_counts, f, indent=2, ensure_ascii=False)
        return True # 返回成功状态
    except IOError as e:
        print(f"\n❌ 错误：无法写入输出文件 '{output_file_path}'。详细信息: {e}")
        return False


def main():
    """
    主执行函数，负责遍历所有文件夹并调用处理函数。
    """
    # --- 批量处理配置 ---
    EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
    EXPERIMENT_IDS = range(1, 21)  # 遍历 2, 3, ..., 20
    METHOD_NAME = 'LLMxMapReduce_V2'
    # --- 结束配置 ---

    print(f"🚀 开始实体计数批量处理任务...")
    print(f"   目标文件夹: {EXPERIMENT_IDS.start}-{EXPERIMENT_IDS.stop - 1} 中的 '{METHOD_NAME}'")
    
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        # 1. 动态构建路径
        base_dir = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'bench', 'level1')
        raw_entities_path = os.path.join(base_dir, 'all_entity.json')
        mapping_path = os.path.join(base_dir, 'difference_entity.json')
        output_path = os.path.join(base_dir, 'final_counts.json')

        # 2. 智能跳过
        if os.path.exists(output_path):
            print(f"⏭️ 跳过：结果文件 'final_counts.json' 已存在。")
            skipped_count += 1
            continue

        # 3. 调用核心处理函数
        print(f"   - 输入 (原始): {raw_entities_path}")
        print(f"   - 输入 (映射): {mapping_path}")
        print(f"   - 输出: {output_path}")
        
        result = process_and_count_entities(
            raw_file_path=raw_entities_path,
            mapping_file_path=mapping_path,
            output_file_path=output_path
        )
        
        if result:
            print(f"✅ 成功处理实验ID: {exp_id}")
            success_count += 1
        else:
            print(f"❌ 处理实验ID: {exp_id} 时发生错误。")
            error_count += 1
            
    # 4. 打印最终总结
    print(f"\n\n{'='*30} 🎉 所有任务已完成! {'='*30}")
    print("📊 最终统计:")
    print(f"   - ✅ 成功处理: {success_count} 个文件夹")
    print(f"   - ⏭️ 跳过处理: {skipped_count} 个文件夹")
    print(f"   - ❌ 处理失败: {error_count} 个文件夹")
    print("="*72)


if __name__ == "__main__":
    main()
