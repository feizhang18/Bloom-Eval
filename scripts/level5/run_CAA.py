import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (
    DEFAULT_HUMAN_CONTENT_FILE,
    DEFAULT_LLM_CONTENT_FILE,
    add_common_arguments,
    build_result_payload,
    build_log_path,
    call_llm_with_retry,
    ensure_dir,
    format_metric_report,
    load_json_list_text,
    parse_llm_json,
    print_metric_summary,
    resolve_output_dir,
    to_project_relative,
    write_json,
    write_text,
)
from prompt_utils import load_prompt


API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"
MAX_RETRIES = 3
RETRY_DELAY = 5
DEFAULT_EXTRACT_CHUNK_TOKENS = 10000

CRITICAL_EXTRACTION_PROMPT_TEMPLATE = load_prompt("level5/CAA_critical_claim_extraction.txt")
CRITICAL_MATCHING_PROMPT_TEMPLATE = load_prompt("level5/CAA_critical_claim_matching.txt")

try:
    import tiktoken
except ImportError:
    tiktoken = None


def estimate_token_count(text: str) -> int:
    """Estimate token count without requiring tokenizer downloads."""
    if tiktoken is not None:
        try:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            pass
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def split_text_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like units while keeping sentence terminators."""
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []

    protected = {
        "e.g.": "e<prd>g<prd>",
        "i.e.": "i<prd>e<prd>",
        "et al.": "et al<prd>",
        "Fig.": "Fig<prd>",
        "Eq.": "Eq<prd>",
        "Sec.": "Sec<prd>",
        "Dr.": "Dr<prd>",
        "Mr.": "Mr<prd>",
        "Ms.": "Ms<prd>",
        "Prof.": "Prof<prd>",
        "vs.": "vs<prd>",
    }
    for source, target in protected.items():
        normalized = normalized.replace(source, target)

    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=\n)\s*(?=#{1,6}\s+)", normalized)
    sentences = []
    for part in parts:
        restored = part.replace("<prd>", ".").strip()
        if restored:
            sentences.append(restored)
    return sentences


def chunk_text_by_sentence_boundaries(text: str, max_tokens: int) -> List[str]:
    """Build chunks that start and end on sentence boundaries."""
    if max_tokens <= 0:
        return [text]

    sentences = split_text_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

        if sentence_tokens > max_tokens:
            print(
                f"  [Chunk] Warning: one sentence is estimated at {sentence_tokens} tokens, "
                f"which exceeds the {max_tokens}-token chunk target."
            )

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def dedupe_statements(statements: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for statement in statements:
        normalized = re.sub(r"\s+", " ", str(statement).strip())
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def strip_statement_prefix(statement: Any) -> str:
    text = re.sub(r"\s+", " ", str(statement).strip())
    return re.sub(r"^[HL]\d+\s*[:.]\s*", "", text)


def chunk_log_query_id(query_id: str, chunk_index: int) -> str:
    return f"{query_id}_chunk{chunk_index:03d}"


def load_text_from_json(file_path: str) -> str:
    try:
        return load_json_list_text(file_path, joiner=" ")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def extract_critical_statements(
    client: OpenAI,
    model: str,
    text: str,
    query_id: str,
    output_path: Path,
    log_dir: Optional[Path],
    *,
    chunk_tokens: int = DEFAULT_EXTRACT_CHUNK_TOKENS,
) -> List[str]:
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            return existing.get("critical_statements", [])
        except Exception:
            pass

    text_tokens = estimate_token_count(text)
    chunks = chunk_text_by_sentence_boundaries(text, chunk_tokens)
    if not chunks:
        write_json(output_path, {"critical_statements": []})
        return []

    chunk_dir = ensure_dir(output_path.parent / f"{output_path.stem}_chunks")
    all_statements: List[str] = []
    print(f"  [Chunk] Extracting critical statements from {len(chunks)} chunk(s), estimated input tokens: {text_tokens}.")

    for index, chunk in enumerate(chunks, start=1):
        chunk_output_path = chunk_dir / f"chunk_{index:03d}.json"
        if chunk_output_path.exists():
            try:
                chunk_result = json.loads(chunk_output_path.read_text(encoding="utf-8"))
                statements = chunk_result.get("critical_statements", [])
                all_statements.extend(statements)
                continue
            except Exception:
                pass

        print(f"  [Chunk] Extracting chunk {index}/{len(chunks)} (estimated tokens: {estimate_token_count(chunk)}).")
        prompt = CRITICAL_EXTRACTION_PROMPT_TEMPLATE.format(text=chunk)
        raw_response = call_llm_with_retry(
            client,
            model,
            prompt,
            build_log_path(log_dir, chunk_log_query_id(query_id, index)),
            temperature=0.0,
            max_tokens=8192,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
            failure_log_message=f"Failed after {MAX_RETRIES} retries.",
        )

        try:
            result = parse_llm_json(raw_response, kind="auto")
            statements = result.get("critical_statements", [])
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Could not parse critical statement extraction result for chunk {index}: {e}")
            statements = []

        write_json(
            chunk_output_path,
            {
                "chunk_index": index,
                "total_chunks": len(chunks),
                "estimated_tokens": estimate_token_count(chunk),
                "critical_statements": statements,
            },
        )
        all_statements.extend(statements)

    statements = dedupe_statements(all_statements)
    write_json(
        output_path,
        {
            "critical_statements": statements,
            "chunk_tokens": chunk_tokens,
            "estimated_input_tokens": text_tokens,
            "total_chunks": len(chunks),
        },
    )
    return statements


def find_semantic_matches(
    client: OpenAI,
    model: str,
    human_statements: List[str],
    llm_statements: List[str],
    query_id: str,
    output_path: Path,
    log_dir: Optional[Path],
) -> Dict[str, Any]:
    human_str = json.dumps(human_statements, ensure_ascii=False, indent=2)
    llm_str = json.dumps(llm_statements, ensure_ascii=False, indent=2)
    prompt = CRITICAL_MATCHING_PROMPT_TEMPLATE.format(
        expert_statements_str=human_str,
        llm_statements_str=llm_str,
    )

    llm_response = call_llm_with_retry(
        client,
        model,
        prompt,
        build_log_path(log_dir, query_id),
        temperature=0.0,
        max_tokens=8192,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY,
        failure_log_message=f"Failed after {MAX_RETRIES} retries.",
    )
    try:
        result = parse_llm_json(llm_response, kind="object")
        if "matched_critical_pairs" not in result or not isinstance(result["matched_critical_pairs"], list):
            result = {"matched_critical_pairs": []}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Could not parse model response as JSON. {e}")
        result = {"matched_critical_pairs": []}

    normalized_pairs = []
    for pair in result.get("matched_critical_pairs", []):
        if not isinstance(pair, dict):
            continue
        expert_statement = strip_statement_prefix(pair.get("expert_critical_statement", ""))
        llm_statement = strip_statement_prefix(pair.get("llm_critical_statement", ""))
        if not expert_statement or not llm_statement:
            continue
        normalized_pairs.append(
            {
                "expert_critical_statement": expert_statement,
                "llm_critical_statement": llm_statement,
            }
        )
    result = {"matched_critical_pairs": normalized_pairs}

    write_json(output_path, result)
    return result


def calculate_metrics(human_count: int, llm_count: int, matched_count: int) -> Dict[str, float]:
    tp = matched_count
    fp = llm_count - tp
    fn = human_count - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "human_critical_statements": human_count,
        "llm_critical_statements": llm_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Critical Argument Alignment (CAA) evaluation tool")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, default=DEFAULT_HUMAN_CONTENT_FILE, help="Path to human content.json")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, default=DEFAULT_LLM_CONTENT_FILE, help="Path to LLM content.json")
    parser.add_argument("--extract_chunk_tokens", type=int, default=DEFAULT_EXTRACT_CHUNK_TOKENS, help="Target token count for sentence-boundary chunks during critical-statement extraction.")
    add_common_arguments(parser, metric_name="caa", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        sys.exit(1)

    output_dir = resolve_output_dir(args.output_dir)
    raw_log_dir = ensure_dir(output_dir / "logs") if args.save_raw_response else None
    client = OpenAI(api_key=API_KEY, base_url=args.base_url)

    human_text = load_text_from_json(args.content_file_human)
    llm_text = load_text_from_json(args.content_file_llm)
    if not human_text or not llm_text:
        print("Error: Missing or invalid input files.")
        sys.exit(1)

    human_extract_path = output_dir / "human_critical_claims.json"
    llm_extract_path = output_dir / "llm_critical_claims.json"
    matching_path = output_dir / "critical_matching.json"

    human_statements = extract_critical_statements(
        client,
        args.model,
        human_text,
        "human_critical_extract",
        human_extract_path,
        raw_log_dir,
        chunk_tokens=args.extract_chunk_tokens,
    )
    llm_statements = extract_critical_statements(
        client,
        args.model,
        llm_text,
        "llm_critical_extract",
        llm_extract_path,
        raw_log_dir,
        chunk_tokens=args.extract_chunk_tokens,
    )

    matched_data = find_semantic_matches(
        client,
        args.model,
        human_statements,
        llm_statements,
        "critical_matching",
        matching_path,
        raw_log_dir,
    )

    num_matched = len(matched_data.get("matched_critical_pairs", []))
    metrics = calculate_metrics(len(human_statements), len(llm_statements), num_matched)

    intermediate = {
        "human_critical_statements": human_statements,
        "llm_critical_statements": llm_statements,
        "matched_critical_pairs": matched_data.get("matched_critical_pairs", []),
    }
    intermediate_path = output_dir / "intermediate.json"
    write_json(intermediate_path, intermediate)

    report_path = output_dir / "report.txt"
    result_path = output_dir / "result.json"
    inputs = {
        "content_file_human": to_project_relative(Path(args.content_file_human)),
        "content_file_llm": to_project_relative(Path(args.content_file_llm)),
    }
    config = {
        "model": args.model,
        "base_url": args.base_url,
        "extract_chunk_tokens": args.extract_chunk_tokens,
    }
    report_text = format_metric_report(
        "CAA",
        "Critical Argument Alignment",
        inputs=inputs,
        results=metrics,
        config=config,
    )
    write_text(report_path, report_text)

    write_json(
        result_path,
        build_result_payload(
            metric="CAA",
            inputs=inputs,
            results=metrics,
            config=config,
            artifacts={
                "report_file": to_project_relative(report_path),
                "intermediate_file": to_project_relative(intermediate_path),
                "human_claims_file": to_project_relative(human_extract_path),
                "llm_claims_file": to_project_relative(llm_extract_path),
                "matching_file": to_project_relative(matching_path),
            },
        ),
    )
    print_metric_summary(
        "CAA",
        report_path,
        result_path,
        results=metrics,
        summary_keys=("precision", "recall", "f1_score"),
        artifacts={"intermediate": intermediate_path},
    )


if __name__ == "__main__":
    main()
