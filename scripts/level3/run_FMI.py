import os
import re
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import add_common_arguments, build_result_payload, resolve_output_dir, to_project_relative, write_json, write_text

REFERENCE_TITLES = [
    "references", "reference", "bibliography", "works cited",
    "citations", "sources"
]


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


def _targets_ci(targets: List[str]) -> List[str]:
    return [_normalize_token(t) for t in targets]


def _atx_title_or_none(line: str) -> Optional[str]:
    m = re.match(r'^[ \t]*#{1,6}[ \t]*(.+?)\s*$', line)
    if m:
        return m.group(1)
    return None


def _is_setext_underline(line: str) -> bool:
    return bool(re.match(r'^[ \t]*(=+|-+)[ \t]*$', line))


def _is_plain_title_line(line: str) -> bool:
    if not line.strip():
        return False
    if re.match(r'^[ \t]*#{1,6}', line):
        return False
    return True


def split_by_headings(full_text: str, target_titles: List[str]) -> Tuple[str, Optional[str], str]:
    """Split body and references with the same heading rules as the benchmark script."""
    targets_norm = set(_targets_ci(target_titles))
    lines = full_text.split("\n")
    n = len(lines)

    hit_start_line = None
    content_start_line = None
    matched_heading_line_text = None

    i = 0
    while i < n:
        line = lines[i]

        atx_title = _atx_title_or_none(line)
        if atx_title is not None and _normalize_token(atx_title) in targets_norm:
            hit_start_line = i
            content_start_line = i + 1
            matched_heading_line_text = line.strip()
            break

        if _is_plain_title_line(line):
            plain_title_norm = _normalize_token(line)
            if i + 1 < n and _is_setext_underline(lines[i + 1]):
                if plain_title_norm in targets_norm:
                    hit_start_line = i
                    content_start_line = i + 2
                    matched_heading_line_text = line.strip()
                    break
            if plain_title_norm in targets_norm:
                hit_start_line = i
                content_start_line = i + 1
                matched_heading_line_text = line.strip()
                break

        i += 1

    if hit_start_line is None:
        return full_text, None, ""

    end_line = n
    j = content_start_line
    while j < n:
        line = lines[j]
        if _atx_title_or_none(line) is not None:
            end_line = j
            break
        if j + 1 < n and lines[j].strip() and _is_setext_underline(lines[j + 1]):
            end_line = j
            break
        j += 1

    references_section = "\n".join(lines[content_start_line:end_line]).lstrip("\n")
    body_lines = lines[:hit_start_line] + lines[end_line:]
    body_text = "\n".join(body_lines).strip("\n")
    return body_text, matched_heading_line_text, references_section

# --- Core logic: extract and compare ---
def get_fmi_score(md_path: str) -> Tuple[float, str]:
    if not os.path.exists(md_path):
        return 0.0, f"File not found: {md_path}"
    
    with open(md_path, 'r', encoding='utf-8') as f:
        text = normalize_whitespace(f.read())

    body_text, matched_heading, references_section = split_by_headings(text, REFERENCE_TITLES)

    defined_nums = {
        int(num)
        for num in re.findall(
            r'^[ \t]*(?:[-*+]|(?:\d+\.))?[ \t]*\[(\d+)\]',
            references_section,
            re.MULTILINE
        )
    }

    cited_refs_list = []
    for content in re.findall(r'\[(.*?)\]', body_text):
        cited_refs_list.extend(re.findall(r'\d+', content))
    cited_nums = {int(num) for num in cited_refs_list if num}

    unresolved_citations = cited_nums - defined_nums
    uncited_references = defined_nums - cited_nums

    intersection = cited_nums.intersection(defined_nums)
    union = cited_nums.union(defined_nums)
    
    fmi = len(intersection) / len(union) if union else 1.0

    heading_line = (
        f"Reference section heading found: {matched_heading}"
        if matched_heading
        else "No reference section heading found (treated as no references section)"
    )
    report = "\n".join([
        heading_line,
        f"Found {len(cited_nums)} unique citation numbers in body text.",
        f"References section defines {len(defined_nums)} entries.",
        f"  - Citations without a definition (body -> references): {sorted(list(unresolved_citations)) if unresolved_citations else 'none'}",
        f"  - Defined but uncited references (references -> body): {sorted(list(uncited_references)) if uncited_references else 'none'}",
        f"FMI score: {fmi:.4f}",
    ])
    return fmi, report

def main():
    parser = argparse.ArgumentParser(description="FMI: Citation Integrity Audit Tool")
    parser.add_argument("--content_file_llm", type=str, required=True)
    parser.add_argument("--content_file_human", type=str, required=True)
    add_common_arguments(parser, metric_name="fmi", include_model=False)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    l_score, l_rep = get_fmi_score(args.content_file_llm)
    h_score, h_rep = get_fmi_score(args.content_file_human)

    final_rep = f"=== FMI Citation Integrity Report ===\n\n[Human]\n{h_rep}\n\n[LLM]\n{l_rep}\n"
    print(final_rep)
    report_path = output_dir / "report.txt"
    result_path = output_dir / "result.json"
    write_text(report_path, final_rep)
    write_json(
        result_path,
        build_result_payload(
            metric="FMI",
            inputs={
                "content_file_human": to_project_relative(Path(args.content_file_human)),
                "content_file_llm": to_project_relative(Path(args.content_file_llm)),
            },
            results={
                "human_score": h_score,
                "llm_score": l_score,
            },
            artifacts={
                "report_file": to_project_relative(report_path),
                "human_report": h_rep,
                "llm_report": l_rep,
            },
        ),
    )

if __name__ == "__main__":
    main()
