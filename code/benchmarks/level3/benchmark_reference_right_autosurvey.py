import json
import re
import os
import numpy as np
import glob
from typing import Dict, List, Tuple, Optional

# --- 1. 全局配置 ---
EXPERIMENT_ROOT = '<EXPERIMENT_ROOT>'
EXPERIMENT_IDS = range(1, 21)  # 遍历 1 到 20 号文件夹
METHOD_NAME = 'human'
OUTPUT_REPORT_PATH = '<OUTPUT_REPORT_PATH>'

# 支持自由扩展的“参考文献”同义标题（大小写不敏感）
REFERENCE_TITLES = [
    "references", "reference", "bibliography", "works cited",
    "citations", "sources", "参考文献", "参考资料"
]

# --- 1.1 标准化与工具函数 ---

def normalize_whitespace(text: str) -> str:
    """统一换行，清理不可见空白，去除行尾空白。"""
    # 统一换行
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 替换常见不可见/特殊空白为普通空格
    invisible = {
        '\u00A0': ' ',  # NBSP
        '\u202F': ' ',  # NARROW NBSP
        '\u2009': ' ',  # THIN SPACE
        '\u200A': ' ',  # HAIR SPACE
        '\u200B': '',   # ZERO WIDTH SPACE
        '\u200C': '',   # ZWNJ
        '\u200D': '',   # ZWJ
        '\u2060': '',   # WORD JOINER
        '\ufeff': ''    # BOM
    }
    for k, v in invisible.items():
        text = text.replace(k, v)
    # 去尾随空白
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    return text

def _normalize_token(s: str) -> str:
    """与目标标题比较前的规范化：去首尾空白、去尾冒号（含全角）、压缩内部空白、转小写。"""
    s = s.strip()
    s = re.sub(r'[：:]\s*$', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def _targets_ci(targets: List[str]) -> List[str]:
    return [_normalize_token(t) for t in targets]

def _atx_title_or_none(line: str) -> Optional[str]:
    """
    是否 ATX 标题：允许任意前导空白；#*1-6；# 后空格可选（兼容 '#References'）。
    返回标题文本或 None。
    """
    m = re.match(r'^[ \t]*#{1,6}[ \t]*(.+?)\s*$', line)
    if m:
        return m.group(1)
    return None

def _is_setext_underline(line: str) -> bool:
    """是否为 Setext 下划线（=== 或 ---，允许空白）。"""
    return bool(re.match(r'^[ \t]*(=+|-+)[ \t]*$', line))

def _is_plain_title_line(line: str) -> bool:
    """
    是否为“纯文本独立行标题”：非空，不以#开头（避免与 ATX 冲突），其他留给上层匹配目标词。
    """
    if not line.strip():
        return False
    if re.match(r'^[ \t]*#{1,6}', line):
        return False  # ATX
    return True

def split_by_headings(full_text: str, target_titles: List[str]) -> Tuple[str, Optional[str], str]:
    """
    查找第一个匹配 target_titles 的“参考文献标题”，支持：
      - ATX:   '#References' / '# References' / '### References：' 等
      - 纯文本：独占一行的 'References'（末尾可带冒号）
      - Setext: 'References' + 下一行 '===' 或 '---'
    返回 (正文(剔除参考文献段), 命中的标题原文, 参考文献段内容)。
    未找到返回 (full_text, None, "")。
    """
    targets_norm = set(_targets_ci(target_titles))
    lines = full_text.split("\n")
    n = len(lines)

    hit_start_line = None         # 标题行索引
    content_start_line = None     # 参考文献内容起始行
    matched_heading_line_text = None

    i = 0
    while i < n:
        line = lines[i]

        # 1) ATX（更宽松：'#' 后可无空格）
        atx_title = _atx_title_or_none(line)
        if atx_title is not None:
            if _normalize_token(atx_title) in targets_norm:
                hit_start_line = i
                content_start_line = i + 1
                matched_heading_line_text = line.strip()
                break

        # 2) Setext 或 纯文本独立行
        if _is_plain_title_line(line):
            plain_title_norm = _normalize_token(line)
            # 2a) Setext：下一行是 === 或 ---
            if i + 1 < n and _is_setext_underline(lines[i + 1]):
                if plain_title_norm in targets_norm:
                    hit_start_line = i
                    content_start_line = i + 2
                    matched_heading_line_text = line.strip()
                    break
            # 2b) 纯文本独立行
            if plain_title_norm in targets_norm:
                hit_start_line = i
                content_start_line = i + 1
                matched_heading_line_text = line.strip()
                break

        i += 1

    if hit_start_line is None:
        return full_text, None, ""

    # 确定结束边界：下一个“任意样式标题”（ATX 或 Setext）
    end_line = n
    j = content_start_line
    while j < n:
        l = lines[j]
        # 下一个 ATX
        if _atx_title_or_none(l) is not None:
            end_line = j
            break
        # 下一个 Setext：当前行非空，且下一行是 === 或 ---
        if j + 1 < n and lines[j].strip() and _is_setext_underline(lines[j + 1]):
            end_line = j
            break
        j += 1

    # 取参考文献内容（从标题下一行或下下行开始）
    references_section = "\n".join(lines[content_start_line:end_line]).lstrip("\n")
    # 从正文中剔除参考文献（含标题和参考文献内容）
    body_lines = lines[:hit_start_line] + lines[end_line:]
    body_text = "\n".join(body_lines).strip("\n")

    return body_text, matched_heading_line_text, references_section


# --- 2. 核心分析函数 ---

def analyze_markdown_file(md_path: str) -> Tuple[Dict, str]:
    """
    对单个Markdown文件进行引用和结构完整性审计。
    返回一个包含分数的字典和一份用于报告的字符串。
    """
    report_lines = []
    scores = {
        "citation_correctness": 0.0,
        "structural_completeness": 0.0
    }

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        error_msg = f"❌ 错误: 找不到Markdown文件 at '{md_path}'"
        report_lines.append(error_msg)
        return scores, "\n".join(report_lines)

    # 标准化（关键：解决空格/换行/不可见空白）
    full_text = normalize_whitespace(full_text)

    # --- Part 1: 引用完整性审计 (基于Markdown内容) ---
    report_lines.append("\n" + "="*25 + " 引用完整性审计 " + "="*25)

    # 使用可扩展标题集合来切分正文与参考文献
    body_text, matched_heading, references_section = split_by_headings(full_text, REFERENCE_TITLES)

    if matched_heading:
        report_lines.append(f"🔖 识别到参考文献标题: {matched_heading}")
    else:
        report_lines.append("🔖 未识别到参考文献标题（将视为无参考文献部分）")

    # 提取参考文献部分定义的编号
    # 覆盖常见格式：行首空白 + 可选列表符号(- * + 或 数字.) + [数字]
    defined_refs_matches = re.findall(
        r'^[ \t]*(?:[-*+]|(?:\d+\.))?[ \t]*\[(\d+)\]',
        references_section,
        re.MULTILINE
    )
    defined_refs = {int(num) for num in defined_refs_matches}

    # 提取正文中所有引用编号，支持 [1]、[1, 2]、[1;3] 等
    #（注意：我们已把参考文献段落从正文剔除，避免 URL 中的 [ ] 干扰）
    all_bracket_content = re.findall(r'\[(.*?)\]', body_text)
    cited_refs_list = []
    for content in all_bracket_content:
        numbers_in_content = re.findall(r'\d+', content)
        cited_refs_list.extend(numbers_in_content)
    cited_refs = {int(num) for num in cited_refs_list if num}

    # 对比分析
    unresolved_citations = cited_refs - defined_refs
    uncited_references = defined_refs - cited_refs

    report_lines.append(f"🔍 正文中找到 {len(cited_refs)} 个独立引用编号。")
    report_lines.append(f"📚 参考文献部分定义了 {len(defined_refs)} 个条目。")
    report_lines.append(f"  - 引用了未定义的文献号 (正文 -> 参考文献): {sorted(list(unresolved_citations)) if unresolved_citations else '无'}")
    report_lines.append(f"  - 定义了但未被引用的文献 (参考文献 -> 正文): {sorted(list(uncited_references)) if uncited_references else '无'}")

    # 计算引用规范性得分：定义文献中被正文引用的比例
    # --- 根据FMI定义（Jaccard相似度）计算得分 ---
    
    # 1. 计算交集与并集的大小
    intersection_count = len(cited_refs.intersection(defined_refs))
    union_count = len(cited_refs.union(defined_refs))

    # 2. 计算FMI得分
    # Jaccard相似度的定义是 |A ∩ B| / |A ∪ B|
    if union_count > 0:
        fmi_score = intersection_count / union_count
    else:
        # 如果并集为空，意味着正文和参考文献部分都没有引用编号，
        # 这是一种“完美”的空状态，FMI应为1
        fmi_score = 1.0

    # 3. 更新分数和报告
    # 注意：FMI是一个0到1之间的比率，不再是百分比
    scores["citation_correctness"] = fmi_score
    report_lines.append(f"📊 格式完整性得分 (FMI): {fmi_score:.4f}")

    # --- Part 2: 结构完整性审计 (基于Markdown标题) ---
    report_lines.append("\n" + "="*25 + " 结构完整性审计 " + "="*25)

    target_sections = ['abstract', 'introduction', 'conclusion', 'references']
    found_sections = {section: False for section in target_sections}

    # 从全文中查找 Markdown ATX 标题（更宽松：'#' 后空格可选）
    headers = re.findall(r'^[ \t]*#{1,6}[ \t]*(.*)', full_text, re.MULTILINE)

    for header in headers:
        header_lower = _normalize_token(header)
        if 'abstract' in header_lower: found_sections['abstract'] = True
        if 'introduction' in header_lower: found_sections['introduction'] = True
        if 'conclusion' in header_lower: found_sections['conclusion'] = True
        if ('references' in header_lower) or ('bibliography' in header_lower) or ('参考文献' in header_lower):
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
    print("🚀 开始引用与结构完整性批量审计任务 (基于Markdown文件)...")
    all_results = []
    full_report_lines = ["="*25 + " 引用与结构完整性基准测试报告 (MD) " + "="*25]

    for exp_id in EXPERIMENT_IDS:
        print(f"\n{'='*30} 正在处理实验ID: {exp_id} {'='*30}")

        # 查找唯一的.md文件
        search_path = os.path.join(EXPERIMENT_ROOT, str(exp_id), METHOD_NAME, '*.md')
        md_files = glob.glob(search_path)

        if len(md_files) == 1:
            md_path = md_files[0]
            print(f"  - 找到Markdown文件: {os.path.basename(md_path)}")
            scores, report_str = analyze_markdown_file(md_path)
            all_results.append(scores)

            # 将单个实验的报告添加到总报告中
            full_report_lines.append(f"\n{'='*25} 实验ID: {exp_id} 审计报告 {'='*25}")
            full_report_lines.append(report_str)
            print(report_str)  # 实时打印结果
        elif len(md_files) == 0:
            print(f"❌ 错误: 在 '{os.path.dirname(search_path)}' 目录下未找到任何 .md 文件。")
        else:
            print(f"❌ 错误: 在 '{os.path.dirname(search_path)}' 目录下找到多个 .md 文件，无法确定使用哪一个。")

    # --- 最终统计分析 ---
    if all_results:
        summary_lines = [f"\n\n{'='*28} 最终统计摘要 {'='*28}"]
        summary_lines.append(f"基于 {len(all_results)} 个有效实验结果的统计分析")

        metric_keys = ["citation_correctness", "structural_completeness"]
        summary_lines.append("\n--- 各项指标的平均值与方差 ---")
        header = f"{'Metric':<28} | {'Average':<12} | {'Variance':<12}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))

        for key in metric_keys:
            values = [res[key] for res in all_results]
            mean_val = np.mean(values)
        # 注意：numpy.var 的 ddof=0（总体方差），如果要样本方差可改 ddof=1
            var_val = np.var(values)

            if key == "citation_correctness":
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
