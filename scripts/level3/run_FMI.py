import os
import re
import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import DEFAULT_HUMAN_ARTICLE_FILE, DEFAULT_HUMAN_CONTENT_FILE, DEFAULT_LLM_ARTICLE_FILE, DEFAULT_LLM_CONTENT_FILE, add_common_arguments, build_result_payload, format_metric_report, load_json, print_metric_summary, resolve_output_dir, to_project_relative, write_json, write_text

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


def infer_article_path(content_path: str) -> Optional[Path]:
    path = Path(content_path)
    if path.suffix.lower() != ".json":
        return None
    if path.exists():
        try:
            data = load_json(path)
            if isinstance(data, dict) and isinstance(data.get("context"), dict) and isinstance(data.get("reference"), list):
                return path
        except Exception:
            pass

    preferred_names = ["expert_article.json", "llm_article.json"]
    if path.parent.name == "human":
        preferred_names = ["expert_article.json", "llm_article.json"]
    elif path.parent.name == "llm":
        preferred_names = ["llm_article.json", "expert_article.json"]

    for name in preferred_names:
        candidate = path.parent / name
        if not candidate.exists():
            continue
        try:
            data = load_json(candidate)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("context"), dict) and isinstance(data.get("reference"), list):
            return candidate

    candidates = [
        candidate
        for candidate in path.parent.glob("*.json")
        if candidate.name not in {"content.json", "outline.json", "reference.json"}
        and not candidate.name.endswith((".bak", ".old_bak", ".data_json_bak"))
    ]
    for candidate in sorted(candidates):
        try:
            data = load_json(candidate)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("context"), dict) and isinstance(data.get("reference"), list):
            return candidate
    return None


def load_article_parts(content_path: str, article_path: Optional[str] = None) -> tuple[str, List[str], str]:
    selected_article = Path(article_path) if article_path else infer_article_path(content_path)
    if selected_article is not None and selected_article.exists():
        data = load_json(selected_article)
        body_text = "\n".join(str(value) for value in data.get("context", {}).values())
        references = [str(item) for item in data.get("reference", [])]
        return body_text, references, str(selected_article)
    return load_content_text(content_path), [], "content file only"


def normalize_lookup_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_year(value: str) -> str:
    cleaned = str(value).replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    match = re.search(r"(?:19|20)\d{2}", cleaned)
    return match.group(0) if match else ""


def first_author_key(author_text: str) -> str:
    text = re.sub(r"\bet\s*al\.?", "", str(author_text), flags=re.IGNORECASE)
    text = re.sub(r"\bothers\b", "", text, flags=re.IGNORECASE)
    first_author = re.split(r"\s*(?:,|;|&| and )\s*", text.strip())[0]
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ@'-]+", first_author)
    tokens = [token for token in tokens if normalize_lookup_text(token) and normalize_lookup_text(token) not in {"and", "others"}]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return normalize_lookup_text(tokens[0])
    final = tokens[-1].replace("-", "")
    if len(final) <= 5 and final.upper() == final and re.search(r"[A-Z]", final):
        return normalize_lookup_text(tokens[0])
    return normalize_lookup_text(tokens[-1])


def author_year_key(author_text: str, year_text: str) -> Optional[tuple[str, str]]:
    author = first_author_key(author_text)
    year = normalize_year(year_text)
    if not author or not year:
        return None
    return author, year


def expand_numeric_citation_group(group: str) -> List[int]:
    if not re.fullmatch(r"[\d,\s;,\-–—]+", group.strip()):
        return []
    numbers = set()
    for part in re.split(r"[,;]\s*", group):
        part = part.strip()
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= end and end - start <= 100:
                numbers.update(range(start, end + 1))
            continue
        if part.isdigit():
            numbers.add(int(part))
    return sorted(numbers)


def defined_citation_ids_from_reference_list(references: List[str]) -> set[tuple[str, object]]:
    defined: set[tuple[str, object]] = set()
    year_pattern = r"(?:19|20|2[O0])[0-9OIl]{2}[a-z]?"

    for index, reference in enumerate(references, start=1):
        text = reference.strip()
        bracket_num = re.match(r"^\s*\[(\d+)\]", text)
        dotted_num = re.match(r"^\s*(\d+)\s*[.)]", text)
        has_explicit_number = bool(bracket_num or dotted_num)
        if bracket_num:
            defined.add(("num", int(bracket_num.group(1))))
        elif dotted_num:
            defined.add(("num", int(dotted_num.group(1))))

        bracket_label = re.match(r"^\s*\[([^\]]*(?:19|20|2[O0])[0-9OIl]{2}[^\]]*)\]", text)
        if bracket_label:
            for part in re.split(r";", bracket_label.group(1)):
                match = re.search(rf"(?P<author>.+?)\s*,?\s*(?P<year>{year_pattern})", part.strip())
                if match:
                    key = author_year_key(match.group("author"), match.group("year"))
                    if key:
                        defined.add(("ay", key))

        first_year = re.search(year_pattern, text)
        if first_year:
            prefix = text[:first_year.start()]
            prefix = re.sub(r"^\s*(?:\[\d+\]|\d+\s*[.)])\s*", "", prefix)
            key = author_year_key(prefix, first_year.group(0))
            if key:
                defined.add(("ay", key))

    return defined


def filter_defined_ids_for_citation_style(cited_ids: set[tuple[str, object]], defined_ids: set[tuple[str, object]]) -> set[tuple[str, object]]:
    cited_kinds = {kind for kind, _ in cited_ids}
    if not cited_kinds:
        return defined_ids
    return {item for item in defined_ids if item[0] in cited_kinds}


def cited_ids_from_body_text(body_text: str) -> set[tuple[str, object]]:
    cited: set[tuple[str, object]] = set()
    year_pattern = r"(?:19|20|2[O0])[0-9OIl]{2}[a-z]?"

    def collect_author_year_from_group(group: str) -> None:
        for part in re.split(r";", group):
            match = re.search(rf"(?P<author>.+?)\s*,?\s*(?P<year>{year_pattern})", part.strip())
            if not match:
                continue
            key = author_year_key(match.group("author"), match.group("year"))
            if key:
                cited.add(("ay", key))

    for group in re.findall(r"\[([^\]]+)\]", body_text):
        cited.update(("num", num) for num in expand_numeric_citation_group(group))
        collect_author_year_from_group(group)

    for group in re.findall(r"\(([^)]+)\)", body_text):
        if re.fullmatch(r"[\d,\s;,\-–—]+", group.strip()):
            cited.update(("num", num) for num in expand_numeric_citation_group(group))
        collect_author_year_from_group(group)

    narrative_pattern = re.compile(
        rf"\b(?P<author>[A-Z][A-Za-zÀ-ÖØ-öø-ÿ@.'-]+(?:\s+(?:et\s+al\.?|&|and)\s*[A-Z]?[A-Za-zÀ-ÖØ-öø-ÿ@.'-]*)?)\s*"
        rf"\(\s*(?P<year>{year_pattern})\s*\)"
    )
    for match in narrative_pattern.finditer(body_text):
        key = author_year_key(match.group("author"), match.group("year"))
        if key:
            cited.add(("ay", key))

    return cited


def format_citation_id(citation_id: tuple[str, object]) -> str:
    kind, value = citation_id
    if kind == "num":
        return f"[{value}]"
    author, year = value
    return f"{author} {year}"


# --- Core logic: extract and compare ---
def get_fmi_score(content_path: str, article_path: Optional[str] = None) -> Tuple[float, str]:
    if not os.path.exists(content_path):
        return 0.0, f"File not found: {content_path}"
    
    try:
        raw_text, article_references, source_name = load_article_parts(content_path, article_path)
        text = normalize_whitespace(raw_text)
    except Exception as e:
        return 0.0, f"Failed to read content: {e}"

    body_text, matched_heading, references_section = split_by_headings(text, REFERENCE_TITLES)

    if article_references:
        defined_ids = defined_citation_ids_from_reference_list(article_references)
        cited_ids = cited_ids_from_body_text(body_text)
        definition_source = f"article JSON reference field: {source_name}"
    else:
        defined_nums = {
            ("num", int(num))
            for num in re.findall(
                r'^[ \t]*(?:[-*+]|(?:\d+\.))?[ \t]*\[(\d+)\]',
                references_section,
                re.MULTILINE
            )
        }
        defined_ids = defined_nums
        cited_ids = cited_ids_from_body_text(body_text)
        definition_source = "references section"

    unresolved_citations = cited_ids - defined_ids
    comparable_defined_ids = filter_defined_ids_for_citation_style(cited_ids, defined_ids)
    uncited_references = comparable_defined_ids - cited_ids

    intersection = cited_ids.intersection(comparable_defined_ids)
    union = cited_ids.union(comparable_defined_ids)
    
    fmi = len(intersection) / len(union) if union else 1.0

    heading_line = (
        f"Reference section heading found: {matched_heading}"
        if matched_heading
        else "No reference section heading found (treated as no references section)"
    )
    report = "\n".join([
        heading_line,
        f"Reference definitions source: {definition_source}",
        f"Found {len(cited_ids)} unique citation identifiers in body text.",
        f"Reference list defines {len(comparable_defined_ids)} comparable citation identifiers.",
        f"  - Citations without a definition (body -> references): {[format_citation_id(item) for item in sorted(unresolved_citations, key=format_citation_id)] if unresolved_citations else 'none'}",
        f"  - Defined but uncited references (references -> body): {[format_citation_id(item) for item in sorted(uncited_references, key=format_citation_id)] if uncited_references else 'none'}",
        f"FMI score: {fmi:.4f}",
    ])
    return fmi, report

def main():
    parser = argparse.ArgumentParser(description="FMI: Citation Integrity Audit Tool")
    parser.add_argument("--content_file_llm", type=str, default=DEFAULT_LLM_CONTENT_FILE)
    parser.add_argument("--content_file_human", type=str, default=DEFAULT_HUMAN_CONTENT_FILE)
    parser.add_argument("--article_file_llm", type=str, default=DEFAULT_LLM_ARTICLE_FILE, help="Optional full article JSON containing context and reference fields; inferred from content file directory when omitted.")
    parser.add_argument("--article_file_human", type=str, default=DEFAULT_HUMAN_ARTICLE_FILE, help="Optional full article JSON containing context and reference fields; inferred from content file directory when omitted.")
    add_common_arguments(parser, metric_name="fmi", include_model=False)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    l_score, l_rep = get_fmi_score(args.content_file_llm, args.article_file_llm)
    h_score, h_rep = get_fmi_score(args.content_file_human, args.article_file_human)

    report_path = output_dir / "report.txt"
    result_path = output_dir / "result.json"
    inferred_human_article = Path(args.article_file_human) if args.article_file_human else infer_article_path(args.content_file_human)
    inferred_llm_article = Path(args.article_file_llm) if args.article_file_llm else infer_article_path(args.content_file_llm)
    inputs = {
        "content_file_human": to_project_relative(Path(args.content_file_human)),
        "content_file_llm": to_project_relative(Path(args.content_file_llm)),
        "article_file_human": to_project_relative(inferred_human_article),
        "article_file_llm": to_project_relative(inferred_llm_article),
    }
    metrics = {
        "human_score": h_score,
        "llm_score": l_score,
    }
    report_text = format_metric_report(
        "FMI",
        "Citation Integrity",
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
            metric="FMI",
            inputs=inputs,
            results=metrics,
            artifacts={
                "report_file": to_project_relative(report_path),
                "human_report": h_rep,
                "llm_report": l_rep,
            },
        ),
    )
    print_metric_summary("FMI", report_path, result_path, results=metrics, summary_keys=("human_score", "llm_score"))

if __name__ == "__main__":
    main()
