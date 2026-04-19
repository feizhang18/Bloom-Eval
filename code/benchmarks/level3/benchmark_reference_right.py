import json
import re
import os
import numpy as np
from typing import Dict, List, Tuple

# --- 1. 全局配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'autosurvey'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# --- 2. 核心分析函数 ---

def analyze_single_experiment(content_path: str, references_path: str) -> Tuple[Dict, str]:
    """
    对单个实验文件夹进行引用和结构完整性审计。
    返回一个包含分数的字典和一份用于报告的字符串。
    """
    report_lines = []
    scores = {
        "citation_accuracy": 0.0,
        "structural_completeness": 0.0
    }

    # --- Part 1: 引用完整性审计 ---
    report_lines.append("\n" + "="*25 + " 引用完整性审计 " + "="*25)
    
    try:
        with open(content_path, 'r', encoding='utf-8') as f: content_data = json.load(f)
        with open(references_path, 'r', encoding='utf-8') as f: references_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        error_msg = f"❌ 错误: 无法加载输入文件。{e}"
        report_lines.append(error_msg)
        return scores, "\n".join(report_lines)

    # 从 content.json (字符串列表) 构建全文
    full_text = " ".join(content_data)
    full_text_cleaned = re.sub(r'\s+', ' ', re.sub(r'-\n\s*', '', full_text)).strip()
    
    # 提取引用标记
    markers = re.compile(r'\[([\d,\s;]+)\]').findall(full_text_cleaned)
    in_text_citations = set()
    for marker_group in markers:
        numbers = re.split(r'[,;]\s*', marker_group)
        for num in numbers:
            if num.strip().isdigit():
                in_text_citations.add(int(num.strip()))
    
    # 获取参考文献库中的编号
    total_refs = references_data.get("reference_num", 0)
    bibliography_items = set(range(1, total_refs + 1))

    # 对比分析
    unresolved_citations = in_text_citations - bibliography_items
    uncited_references = bibliography_items - in_text_citations

    report_lines.append(f"🔍 全文中找到 {len(in_text_citations)} 个独立引用编号。")
    report_lines.append(f"📚 参考文献库中应有 {len(bibliography_items)} 个条目。")
    report_lines.append(f"  - 引用了不存在的文献号 (正文 -> 文献库): {sorted(list(unresolved_citations)) if unresolved_citations else '无'}")
    report_lines.append(f"  - 未被引用的文献号 (文献库 -> 正文): {sorted(list(uncited_references)) if uncited_references else '无'}")

    # 计算引用规范性准确率
    if bibliography_items:
        # 准确率定义为：在参考文献库中，有多少比例的文献在正文中被正确引用了。
        correctly_referenced_count = len(bibliography_items - uncited_references)
        accuracy = (correctly_referenced_count / len(bibliography_items)) * 100
        scores["citation_accuracy"] = accuracy
        report_lines.append(f"📊 引用规范性准确率: {accuracy:.2f}%")
    else:
        report_lines.append("📊 引用规范性准确率: N/A (参考文献列表为空)")

    # --- Part 2: 结构完整性审计 ---
    report_lines.append("\n" + "="*25 + " 结构完整性审计 " + "="*25)
    
    target_sections = ['abstract', 'introduction', 'conclusion', 'references']
    found_sections = {section: False for section in target_sections}
    
    # 在全文中搜索关键词
    # 使用 \b 来确保匹配的是完整的单词
    if re.search(r'\bAbstract\b', full_text, re.IGNORECASE):
        found_sections['abstract'] = True
    if re.search(r'\bIntroduction\b', full_text, re.IGNORECASE):
        found_sections['introduction'] = True
    if re.search(r'\bConclusion\b', full_text, re.IGNORECASE):
        found_sections['conclusion'] = True
    
    # 通过参考文献文件是否存在且非空来判断
    if os.path.exists(references_path) and os.path.getsize(references_path) > 2: # >2 以防空json {}
        found_sections['references'] = True

    found_count = sum(found_sections.values())
    score = found_count / len(target_sections)
    scores["structural_completeness"] = score
    
    report_lines.append("📊 关键章节完整性检查:")
    for section, found in found_sections.items():
        report_lines.append(f"  - [{'✓' if found else '✗'}] {section.capitalize()}")
    report_lines.append(f"📊 结构完整性得分: {score:.2f} / 1.00")
    
    return scores, "\n".join(report_lines)

# --- 3. 主执行流程 ---
def main():
    """
    主函数，负责遍历所有实验文件夹、调用处理函数、进行统计并生成报告。
    """
    print("🚀 开始引用与结构完整性批量审计任务...")
    all_results = []
    full_report_lines = ["="*25 + " 引用与结构完整性基准测试报告 " + "="*25]

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")
        
        content_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'content.json')
        references_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, 'reference_3.json')

        if not os.path.exists(content_path) or not os.path.exists(references_path):
            print(f"❌ 错误: 缺少一个或多个输入文件，跳过此实验。")
            if not os.path.exists(content_path): print(f"   - 缺失: {content_path}")
            if not os.path.exists(references_path): print(f"   - 缺失: {references_path}")
            continue

        scores, report_str = analyze_single_experiment(content_path, references_path)
        all_results.append(scores)
        
        # 将单个实验的报告添加到总报告中
        full_report_lines.append(f"\n{'='*25} 实验ID: {exp_id} 审计报告 {'='*25}")
        full_report_lines.append(report_str)
        print(report_str) # 实时打印结果

    # --- 最终统计分析 ---
    if all_results:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_results)} 个实验结果的统计分析")
        
        metric_keys = ["citation_accuracy", "structural_completeness"]
        summary_lines.append("\n--- 各项指标的平均值与方差 ---")
        header = f"{'Metric':<28} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        for key in metric_keys:
            values = [res[key] for res in all_results]
            mean_val = np.mean(values)
            var_val = np.var(values)
            # 对准确率添加百分号
            if key == "citation_accuracy":
                summary_lines.append(f"{key:<28} | {mean_val:<11.2f}% | {var_val:<12.2f}")
            else:
                summary_lines.append(f"{key:<28} | {mean_val:<12.4f} | {var_val:<12.4f}")
        
        full_report_lines.extend(summary_lines)

    # --- 保存完整报告到文件 ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_lines))
        print(f"\n\n✅ 完整报告已成功保存至: {OUTPUT_REPORT_PATH}")
    except IOError as e:
        print(f"\n❌ 错误: 无法写入报告文件。{e}")

    print(f"\n{'='*30} 🎉 所有任务已完成! {'='*30}")

if __name__ == '__main__':
    main()
