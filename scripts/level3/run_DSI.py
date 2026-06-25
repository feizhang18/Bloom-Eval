import os
import re
import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import DEFAULT_HUMAN_CONTENT_FILE, DEFAULT_LLM_CONTENT_FILE, add_common_arguments, build_result_payload, format_metric_report, load_json, print_metric_summary, resolve_output_dir, to_project_relative, write_json, write_text


def normalize_whitespace(text: str) -> str:
    """Match benchmark normalization for newlines, invisible spaces, and trailing blanks."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    invisible = {
        '\u00A0': ' ',
        '\u202F': ' ',
        '\u2009': ' ',
        '\u200A': ' ',
        '\u200B': '',
        '\u200C': '',
        '\u200D': '',
        '\u2060': '',
        '\ufeff': ''
    }
    for src, dst in invisible.items():
        text = text.replace(src, dst)
    return "\n".join(line.rstrip(" \t") for line in text.split("\n"))


def _normalize_token(s: str) -> str:
    s = s.strip()
    s = re.sub(r'[：:]\s*$', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.lower()


def _iter_text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_text_values(item)
    elif isinstance(value, dict):
        if isinstance(value.get("context"), dict):
            yield from _iter_text_values(value["context"])
        else:
            for item in value.values():
                yield from _iter_text_values(item)


def load_content_text(content_path: str) -> str:
    path = Path(content_path)
    if path.suffix.lower() == ".json":
        data = load_json(path)
        return "\n".join(part for part in _iter_text_values(data) if part.strip())

    with path.open('r', encoding='utf-8') as f:
        return f.read()


def get_dsi_score(content_path: str) -> Tuple[float, str]:
    if not os.path.exists(content_path):
        return 0.0, "File not found"

    try:
        text = normalize_whitespace(load_content_text(content_path))
    except Exception as e:
        return 0.0, f"Failed to read content: {e}"

    target_sections = ['abstract', 'introduction', 'conclusion', 'references']
    found_sections = {section: False for section in target_sections}

    headers = re.findall(r'^[ \t]*#{1,6}[ \t]*(.*)', text, re.MULTILINE)
    for header in headers:
        header_lower = _normalize_token(header)
        if 'abstract' in header_lower:
            found_sections['abstract'] = True
        if 'introduction' in header_lower:
            found_sections['introduction'] = True
        if 'conclusion' in header_lower:
            found_sections['conclusion'] = True
        if ('references' in header_lower) or ('bibliography' in header_lower):
            found_sections['references'] = True

    dsi_score = sum(found_sections.values()) / len(target_sections)

    details = " | ".join(
        f"{section.capitalize()}: {'Y' if found else 'N'}"
        for section, found in found_sections.items()
    )
    report = f"Section check: {details}\nDSI score: {dsi_score:.2f}"
    return dsi_score, report

def main():
    parser = argparse.ArgumentParser(description="DSI: Document Structure Integrity Audit Tool")
    parser.add_argument("--content_file_llm", type=str, default=DEFAULT_LLM_CONTENT_FILE)
    parser.add_argument("--content_file_human", type=str, default=DEFAULT_HUMAN_CONTENT_FILE)
    add_common_arguments(parser, metric_name="dsi", include_model=False)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    l_score, l_rep = get_dsi_score(args.content_file_llm)
    h_score, h_rep = get_dsi_score(args.content_file_human)

    report_path = output_dir / "report.txt"
    result_path = output_dir / "result.json"
    inputs = {
        "content_file_human": to_project_relative(Path(args.content_file_human)),
        "content_file_llm": to_project_relative(Path(args.content_file_llm)),
    }
    metrics = {
        "human_score": h_score,
        "llm_score": l_score,
    }
    report_text = format_metric_report(
        "DSI",
        "Document Structure Integrity",
        inputs=inputs,
        results=metrics,
        sections=[
            ("Human Details", h_rep),
            ("LLM Details", l_rep),
        ],
    )
    write_text(report_path, report_text)
    write_json(
        result_path,
        build_result_payload(
            metric="DSI",
            inputs=inputs,
            results=metrics,
            artifacts={
                "report_file": to_project_relative(report_path),
                "human_report": h_rep,
                "llm_report": l_rep,
            },
        ),
    )
    print_metric_summary("DSI", report_path, result_path, results=metrics, summary_keys=("human_score", "llm_score"))

if __name__ == "__main__":
    main()
