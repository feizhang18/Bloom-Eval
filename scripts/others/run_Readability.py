import os
import re
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import nltk
import numpy as np
import pandas as pd
import textstat

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, load_json_text, resolve_output_dir, to_project_relative, write_json, write_text


def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("First run: downloading NLTK sentence tokenizer...")
        nltk.download("punkt")
        print("Download complete.")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


def load_text_from_json(file_path: str) -> str:
    try:
        return load_json_text(file_path, joiner="\n", min_text_length=10)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return ""


def calculate_all_metrics(text: str) -> Dict[str, float]:
    metric_names = [
        "Flesch Reading Ease",
        "Flesch-Kincaid Grade",
        "Gunning Fog",
        "SMOG Index",
        "Coleman-Liau Index",
        "Automated Readability Index (ARI)",
        "Word Count",
        "Sentence Count",
        "Avg Sentence Length (words)",
        "Complex Word Count (3+ syllables)",
        "Lexicon Count (words)",
        "Sentence Length Std Dev",
    ]
    if not text or len(text.strip()) == 0:
        return {metric: 0.0 for metric in metric_names}

    textstat.set_lang("en_US")

    sentences = nltk.sent_tokenize(text)
    sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    sentence_length_std_dev = float(np.std(sentence_lengths)) if sentence_lengths else 0.0

    return {
        "Flesch Reading Ease": float(textstat.flesch_reading_ease(text)),
        "Flesch-Kincaid Grade": float(textstat.flesch_kincaid_grade(text)),
        "Gunning Fog": float(textstat.gunning_fog(text)),
        "SMOG Index": float(textstat.smog_index(text)),
        "Coleman-Liau Index": float(textstat.coleman_liau_index(text)),
        "Automated Readability Index (ARI)": float(textstat.automated_readability_index(text)),
        "Word Count": float(textstat.lexicon_count(text, removepunct=True)),
        "Sentence Count": float(textstat.sentence_count(text)),
        "Avg Sentence Length (words)": float(textstat.avg_sentence_length(text)),
        "Complex Word Count (3+ syllables)": float(textstat.difficult_words(text, 3)),
        "Lexicon Count (words)": float(len(set(re.findall(r"\b\w+\b", text.lower())))),
        "Sentence Length Std Dev": sentence_length_std_dev,
    }


def calculate_composite_readability_score(
    human_metrics: Dict[str, float],
    llm_metrics: Dict[str, float],
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    weights = {
        "avg_sentence_length": 0.20,
        "lexical_diversity": 0.20,
        "sentence_length_std_dev": 0.20,
        "complex_word_ratio": 0.20,
        "avg_grade_level": 0.20,
    }

    def similarity_ratio(val_human: float, val_llm: float) -> float:
        if val_human == 0 and val_llm == 0:
            return 1.0
        if val_human == 0 or val_llm == 0:
            return 0.0
        return min(val_human, val_llm) / max(val_human, val_llm)

    ttr_human = (
        human_metrics["Lexicon Count (words)"] / human_metrics["Word Count"]
        if human_metrics["Word Count"] > 0
        else 0.0
    )
    ttr_llm = (
        llm_metrics["Lexicon Count (words)"] / llm_metrics["Word Count"]
        if llm_metrics["Word Count"] > 0
        else 0.0
    )

    cwr_human = (
        human_metrics["Complex Word Count (3+ syllables)"] / human_metrics["Word Count"]
        if human_metrics["Word Count"] > 0
        else 0.0
    )
    cwr_llm = (
        llm_metrics["Complex Word Count (3+ syllables)"] / llm_metrics["Word Count"]
        if llm_metrics["Word Count"] > 0
        else 0.0
    )

    grade_metrics = [
        "Flesch-Kincaid Grade",
        "Gunning Fog",
        "SMOG Index",
        "Coleman-Liau Index",
        "Automated Readability Index (ARI)",
    ]
    avg_grade_human = float(np.mean([human_metrics[k] for k in grade_metrics]))
    avg_grade_llm = float(np.mean([llm_metrics[k] for k in grade_metrics]))

    scores = {
        "avg_sentence_length": similarity_ratio(
            human_metrics["Avg Sentence Length (words)"],
            llm_metrics["Avg Sentence Length (words)"],
        ),
        "lexical_diversity": similarity_ratio(ttr_human, ttr_llm),
        "sentence_length_std_dev": similarity_ratio(
            human_metrics["Sentence Length Std Dev"],
            llm_metrics["Sentence Length Std Dev"],
        ),
        "complex_word_ratio": similarity_ratio(cwr_human, cwr_llm),
        "avg_grade_level": similarity_ratio(avg_grade_human, avg_grade_llm),
    }

    final_score = sum(scores[metric] * weights[metric] for metric in weights)
    return final_score, scores, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Readability evaluation tool")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, required=True, help="Path to human content.json")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, required=True, help="Path to LLM content.json")
    add_common_arguments(parser, metric_name="readability", include_model=False)
    args = parser.parse_args()

    ensure_nltk_punkt()
    output_dir = resolve_output_dir(args.output_dir)

    print("Loading text from files...")
    human_text = load_text_from_json(args.content_file_human)
    llm_text = load_text_from_json(args.content_file_llm)

    if not human_text or not llm_text:
        print("Error: Could not load one or both survey texts. Exiting.")
        return

    print("Text loaded. Computing metrics...")
    human_metrics = calculate_all_metrics(human_text)
    llm_metrics = calculate_all_metrics(llm_text)
    composite_score, individual_scores, weights = calculate_composite_readability_score(
        human_metrics,
        llm_metrics,
    )

    metric_map = {
        "avg_sentence_length": "Avg Sentence Length",
        "lexical_diversity": "Lexical Diversity",
        "sentence_length_std_dev": "Sentence Rhythm",
        "complex_word_ratio": "Complex Word Ratio",
        "avg_grade_level": "Avg Grade Level",
    }

    detail_rows = []
    for key, weight in weights.items():
        similarity = individual_scores[key]
        contribution = similarity * weight
        detail_rows.append(
            {
                "metric_key": key,
                "metric_name": metric_map[key],
                "weight": weight,
                "similarity": similarity,
                "contribution": contribution,
            }
        )

    metrics_df = pd.DataFrame(
        {
            "Human-Written (Gold Standard)": human_metrics,
            "LLM-Generated Survey": llm_metrics,
        }
    )
    details_df = pd.DataFrame(detail_rows)

    intermediate_path = output_dir / "intermediate.json"
    details_csv_path = output_dir / "component_scores.csv"
    metrics_csv_path = output_dir / "raw_metrics.csv"
    result_path = output_dir / "result.json"
    report_path = output_dir / "report.txt"

    write_json(
        intermediate_path,
        {
            "human_metrics": human_metrics,
            "llm_metrics": llm_metrics,
            "individual_scores": individual_scores,
            "weights": weights,
            "composite_score": composite_score,
        },
    )

    details_df.to_csv(details_csv_path, index=False, encoding="utf-8")
    metrics_df.to_csv(metrics_csv_path, encoding="utf-8")

    write_json(
        result_path,
        build_result_payload(
            metric="Readability",
            inputs={
                "content_file_human": to_project_relative(Path(args.content_file_human)),
                "content_file_llm": to_project_relative(Path(args.content_file_llm)),
            },
            results={"composite_readability_score": composite_score},
            artifacts={
                "report_file": to_project_relative(report_path),
                "intermediate_file": to_project_relative(intermediate_path),
                "component_scores_file": to_project_relative(details_csv_path),
                "raw_metrics_file": to_project_relative(metrics_csv_path),
            },
        ),
    )

    report_lines = [
        "================================================================================",
        " Bloom-Eval Others: Readability Report",
        "================================================================================",
        f"Composite readability score: {composite_score:.4f}",
        "",
        "Score interpretation:",
        "> 0.90: Extremely similar; the LLM writing style is highly aligned with the human survey.",
        "0.80-0.90: Highly similar; the LLM style is very close to expert-level writing.",
        "0.70-0.80: Moderately similar; the LLM captures the main style but still differs.",
        "< 0.70: Noticeably different; the LLM writing style diverges from the human survey.",
        "",
        "Component scores:",
        details_df.to_string(index=False),
        "",
        "Raw metrics:",
        metrics_df.to_string(),
        "================================================================================",
    ]
    report_text = "\n".join(report_lines)

    write_text(report_path, report_text)

    print("\n" + report_text)
    print(f"Intermediate results saved to: {intermediate_path}")
    print(f"Final results saved to: {result_path}")


if __name__ == "__main__":
    main()
