# -*- coding: utf-8 -*-
"""
文本可读性与质量指标对比分析器 (V2.1 - 平均权重版)

本脚本用于加载两篇由人类撰写和由LLM生成的学术综述文本，
并计算一系列可读性与叙事质量指标。
核心功能是生成一个 0-1 范围内的“综合可读性相似度分数”，
此版本使用平均权重，平等看待所有核心指标。

使用前请确保已安装所需库:
pip install textstat pandas nltk numpy
"""
import json
import textstat
import pandas as pd
import re
import numpy as np
import nltk


# NLTK首次运行时需要下载分句模型
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # <--- 这是正确的异常类型
    print("首次运行：正在下载NLTK分句模型...")
    nltk.download('punkt')
    print("下载完成。")



# --- 1. 文件路径 ---
# 请将以下路径替换为您本地文件的实际路径
HUMAN_SURVEY_PATH = r'<EXPERIMENT_ROOT>/1/human/content.json'
LLM_SURVEY_PATH = r'<EXPERIMENT_ROOT>/1/LLMxMapReduce_V2/content.json'


# --- 2. 辅助函数 ---

def load_text_from_json(file_path: str) -> str:
    """
    从您提供的JSON文件加载文本内容。
    此函数会尝试将JSON中的所有文本片段连接成一个完整的字符串。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def extract_text(element):
            if isinstance(element, str):
                yield element
            elif isinstance(element, list):
                for item in element:
                    yield from extract_text(item)
            elif isinstance(element, dict):
                for value in element.values():
                    yield from extract_text(value)

        text_parts = list(extract_text(data))
        return "\n".join(part for part in text_parts if isinstance(part, str) and len(part.strip()) > 10)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return ""
    except Exception as e:
        print(f"读取或处理文件时出错 {file_path}: {e}")
        return ""

def calculate_all_metrics(text: str) -> dict:
    """
    计算给定文本的所有硬性指标，并新增句子长度的标准差。
    """
    if not text or len(text.strip()) == 0:
        return {metric: 0 for metric in [ # 返回0以便于计算
            "Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog",
            "SMOG Index", "Coleman-Liau Index", "Automated Readability Index (ARI)",
            "Word Count", "Sentence Count", "Avg Sentence Length (words)",
            "Complex Word Count (3+ syllables)", "Lexicon Count (words)",
            "Sentence Length Std Dev"
        ]}

    textstat.set_lang("en_US")

    # 新增：计算句子长度标准差
    sentences = nltk.sent_tokenize(text)
    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    sentence_length_std_dev = np.std(sentence_lengths) if sentence_lengths else 0

    metrics = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "SMOG Index": textstat.smog_index(text),
        "Coleman-Liau Index": textstat.coleman_liau_index(text),
        "Automated Readability Index (ARI)": textstat.automated_readability_index(text),
        "Word Count": textstat.lexicon_count(text, removepunct=True),
        "Sentence Count": textstat.sentence_count(text),
        "Avg Sentence Length (words)": textstat.avg_sentence_length(text),
        "Complex Word Count (3+ syllables)": textstat.difficult_words(text, 3),
        "Lexicon Count (words)": len(set(re.findall(r'\b\w+\b', text.lower()))),
        "Sentence Length Std Dev": sentence_length_std_dev
    }
    return metrics

def calculate_composite_readability_score(human_metrics: dict, llm_metrics: dict) -> (float, dict):
    """
    计算一个0-1之间的综合可读性相似度分数。
    此版本使用平均权重，平等看待所有指标。
    """
    # !!! 核心改动：所有权重均等 !!!
    weights = {
        'avg_sentence_length': 0.20,
        'lexical_diversity': 0.20,
        'sentence_length_std_dev': 0.20,
        'complex_word_ratio': 0.20,
        'avg_grade_level': 0.20
    }

    def similarity_ratio(val_human, val_llm):
        # 计算两个正数之间的相似度比率 (0-1)
        if val_human == 0 and val_llm == 0: return 1.0
        if val_human == 0 or val_llm == 0: return 0.0
        return min(val_human, val_llm) / max(val_human, val_llm)

    # 准备计算所需的比率
    ttr_human = human_metrics['Lexicon Count (words)'] / human_metrics['Word Count'] if human_metrics['Word Count'] > 0 else 0
    ttr_llm = llm_metrics['Lexicon Count (words)'] / llm_metrics['Word Count'] if llm_metrics['Word Count'] > 0 else 0
    
    cwr_human = human_metrics['Complex Word Count (3+ syllables)'] / human_metrics['Word Count'] if human_metrics['Word Count'] > 0 else 0
    cwr_llm = llm_metrics['Complex Word Count (3+ syllables)'] / llm_metrics['Word Count'] if llm_metrics['Word Count'] > 0 else 0
    
    # 计算平均年级水平
    grade_metrics = ["Flesch-Kincaid Grade", "Gunning Fog", "SMOG Index", "Coleman-Liau Index", "Automated Readability Index (ARI)"]
    avg_grade_human = np.mean([human_metrics[k] for k in grade_metrics])
    avg_grade_llm = np.mean([llm_metrics[k] for k in grade_metrics])

    # 计算各分项的相似度分数
    scores = {
        'avg_sentence_length': similarity_ratio(human_metrics['Avg Sentence Length (words)'], llm_metrics['Avg Sentence Length (words)']),
        'lexical_diversity': similarity_ratio(ttr_human, ttr_llm),
        'sentence_length_std_dev': similarity_ratio(human_metrics['Sentence Length Std Dev'], llm_metrics['Sentence Length Std Dev']),
        'complex_word_ratio': similarity_ratio(cwr_human, cwr_llm),
        'avg_grade_level': similarity_ratio(avg_grade_human, avg_grade_llm)
    }

    # 计算加权总分 (此处为算术平均值)
    final_score = sum(scores[metric] * weights[metric] for metric in weights)
    
    return final_score, scores, weights

# --- 3. 主程序 ---

if __name__ == "__main__":
    print("正在从文件中加载文本...")
    human_text = load_text_from_json(HUMAN_SURVEY_PATH)
    llm_text = load_text_from_json(LLM_SURVEY_PATH)

    if not human_text or not llm_text:
        print("未能加载一个或两个综述文本，程序将退出。")
    else:
        print("文本加载成功，正在计算各项指标...")
        
        human_metrics = calculate_all_metrics(human_text)
        llm_metrics = calculate_all_metrics(llm_text)
        
        # --- 计算综合分数 ---
        composite_score, individual_scores, weights = calculate_composite_readability_score(human_metrics, llm_metrics)

        # --- 生成并打印报告 ---
        print("\n" + "="*80)
        print("📊 文本可读性与质量指标对比分析报告 (平均权重版)")
        print("="*80)

        # 打印综合分数
        print(f"\n🏆 **综合可读性相似度指数: {composite_score:.4f}** (满分1.0)\n")
        print("--- 指数解读 ---")
        print("> 0.90:  极度相似，LLM在写作风格上与人类专家高度一致。")
        print("0.80-0.90: 高度相似，LLM的风格非常接近专家水平。")
        print("0.70-0.80: 较为相似，LLM基本抓住了核心风格，但存在一些偏差。")
        print("< 0.70:  差异明显，LLM的写作风格与人类专家有显著不同。")
        print("-" * 30)

        # 打印分数构成明细
        print("\n--- 分数构成明细 (各指标权重均等) ---\n")
        
        metric_map = {
            'avg_sentence_length': '平均句长',
            'lexical_diversity': '词汇丰富度',
            'sentence_length_std_dev': '句子节奏',
            'complex_word_ratio': '术语密度',
            'avg_grade_level': '平均年级'
        }
        
        # 动态生成报告，更具扩展性
        details_data = []
        for key, weight in weights.items():
            sim = individual_scores[key]
            contribution = sim * weight
            details_data.append({
                "指标": metric_map[key],
                "权重": f"{weight:.0%}",
                "相似度": f"{sim:.2%}",
                "对总分贡献": f"{contribution:.4f}"
            })
            
        details_df = pd.DataFrame(details_data)
        print(details_df.to_string(index=False))


        # 打印原始数据对比表
        print("\n\n--- 原始指标数据对比 ---\n")
        df = pd.DataFrame({
            'Human-Written (Gold Standard)': human_metrics,
            'LLM-Generated Survey': llm_metrics
        })
        print(df)
        
        print("\n" + "="*80)
        print("--- 分析结束 ---")