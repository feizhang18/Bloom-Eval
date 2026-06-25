import os
import time
import argparse
import re
import sys
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Any, Optional

# sys.path setup to import prompt_utils
# Assumes this script is located under Bloom-Eval/scripts/level1/
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (
    DEFAULT_HUMAN_CONTENT_FILE,
    DEFAULT_LLM_CONTENT_FILE,
    add_common_arguments,
    build_result_payload,
    call_llm_for_json,
    ensure_dir,
    format_metric_report,
    load_json,
    print_metric_summary,
    resolve_output_dir,
    save_json,
    to_project_relative,
    write_json,
    write_text,
)
from prompt_utils import load_prompt

API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EXTRACT_CHUNK_TOKENS = 10000

PROMPT_EXTRACT = load_prompt("level1/FCons_claim_extraction.txt")
PROMPT_MATCH = load_prompt("level1/FCons_claim_matching.txt")

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


def dedupe_claims(claims: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for claim in claims:
        normalized = re.sub(r"\s+", " ", str(claim).strip())
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def chunk_log_path(log_path: Optional[Path], chunk_index: int) -> Optional[Path]:
    if log_path is None:
        return None
    return log_path.with_name(f"{log_path.stem}_chunk{chunk_index:03d}{log_path.suffix}")


def step1_extract_claims(
    client: OpenAI,
    model: str,
    text_content: str,
    output_path: Path,
    log_path: Optional[Path],
    *,
    chunk_tokens: int = DEFAULT_EXTRACT_CHUNK_TOKENS,
) -> List[str]:
    """Stage 1: Extract factual statements from text."""
    if os.path.exists(output_path):
        data = load_json(output_path)
        return data.get("factual_statements", data.get("actual_claims", []))

    text_tokens = estimate_token_count(text_content)
    chunks = chunk_text_by_sentence_boundaries(text_content, chunk_tokens)
    if not chunks:
        save_json({"factual_statements": []}, output_path)
        return []

    if len(chunks) == 1:
        prompt = PROMPT_EXTRACT.replace("{text}", chunks[0])
        result = call_llm_for_json(client, model, prompt, log_path)
        statements = result.get("factual_statements", result.get("actual_claims", []))
        save_json({"factual_statements": statements}, output_path)
        return statements

    print(f"  [Chunk] Splitting extraction input into {len(chunks)} chunks (estimated tokens: {text_tokens}).")
    chunk_dir = ensure_dir(output_path.parent / f"{output_path.stem}_chunks")
    all_statements: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_output_path = chunk_dir / f"chunk_{index:03d}.json"
        if chunk_output_path.exists():
            chunk_data = load_json(chunk_output_path)
            statements = chunk_data.get("factual_statements", chunk_data.get("actual_claims", []))
        else:
            print(f"  [Chunk] Extracting chunk {index}/{len(chunks)} (estimated tokens: {estimate_token_count(chunk)}).")
            prompt = PROMPT_EXTRACT.replace("{text}", chunk)
            result = call_llm_for_json(client, model, prompt, chunk_log_path(log_path, index))
            statements = result.get("factual_statements", result.get("actual_claims", []))
            save_json(
                {
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "estimated_tokens": estimate_token_count(chunk),
                    "factual_statements": statements,
                },
                chunk_output_path,
            )
        all_statements.extend(statements)

    statements = dedupe_claims(all_statements)
    save_json({"factual_statements": statements}, output_path)
    return statements

def step2_match_claims(client: OpenAI, model: str, expert_list: List[str], llm_list: List[str], output_path: Path, log_path: Optional[Path]) -> List[Dict]:
    """Stage 2: Use LLM to determine semantic equivalence between two sets of factual statements."""
    if os.path.exists(output_path):
        return load_json(output_path).get("matched_pairs", [])

    expert_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(expert_list)])
    llm_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(llm_list)])

    prompt = PROMPT_MATCH.replace("{expert_statements_str}", expert_str)
    prompt = prompt.replace("{llm_statements_str}", llm_str)

    result = call_llm_for_json(client, model, prompt, log_path)
    matched_pairs = result.get("matched_pairs", [])

    save_json({"matched_pairs": matched_pairs}, output_path)
    return matched_pairs

def step3_calculate_metrics(expert_list: List[str], llm_list: List[str], matched_pairs: List[Dict]) -> Dict[str, float]:
    """Stage 3: Compute precision, recall, and F1."""
    total_expert = len(expert_list)
    total_llm = len(llm_list)
    tp = len(matched_pairs)
    fp = total_llm - tp
    fn = total_expert - tp

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
        "total_expert": total_expert,
        "total_llm": total_llm,
    }

def main():
    parser = argparse.ArgumentParser(description="Factual Claims End-to-End Evaluation Pipeline")
    parser.add_argument("--content_file_llm", "--llm_file", dest="content_file_llm", type=str, default=DEFAULT_LLM_CONTENT_FILE, help="Path to the LLM-generated content.json")
    parser.add_argument("--content_file_human", "--human_file", dest="content_file_human", type=str, default=DEFAULT_HUMAN_CONTENT_FILE, help="Path to the human-expert content.json")
    parser.add_argument("--extract_chunk_tokens", type=int, default=DEFAULT_EXTRACT_CHUNK_TOKENS, help="Target token count for sentence-boundary chunks during factual-claim extraction.")
    add_common_arguments(parser, metric_name="fcons", default_model=DEFAULT_MODEL)
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Please set OPENAI_API_KEY in the environment.")
        sys.exit(1)

    output_dir = resolve_output_dir(args.output_dir)
    client = OpenAI(api_key=API_KEY, base_url=args.base_url)
    ts = time.strftime("%Y%m%d_%H%M%S")

    dir_human = ensure_dir(output_dir / "human")
    dir_llm = ensure_dir(output_dir / "llm")
    dir_logs = ensure_dir(output_dir / "logs") if args.save_raw_response else None

    print("Running FCons...")

    human_data = load_json(args.content_file_human)
    llm_data = load_json(args.content_file_llm)
    human_text = human_data[0] if isinstance(human_data, list) else human_data
    llm_text = llm_data[0] if isinstance(llm_data, list) else llm_data

    # Stage 1: Extract factual claims
    human_claims = step1_extract_claims(client, args.model, human_text, dir_human / "factual_claims.json", (dir_logs / f"ext_human_{ts}.txt") if dir_logs else None, chunk_tokens=args.extract_chunk_tokens)
    llm_claims = step1_extract_claims(client, args.model, llm_text, dir_llm / "factual_claims.json", (dir_logs / f"ext_llm_{ts}.txt") if dir_logs else None, chunk_tokens=args.extract_chunk_tokens)

    if not human_claims or not llm_claims:
        print("\nError: One side produced no factual statements; cannot proceed with matching.")
        sys.exit(1)

    matched_pairs = step2_match_claims(client, args.model, human_claims, llm_claims, output_dir / "matched_pairs.json", (dir_logs / f"match_{ts}.txt") if dir_logs else None)

    report_path = output_dir / "report.txt"
    result_path = output_dir / "result.json"
    metrics = step3_calculate_metrics(human_claims, llm_claims, matched_pairs)
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
        "FCons",
        "Factual Claim Consistency",
        inputs=inputs,
        results=metrics,
        config=config,
    )
    write_text(report_path, report_text)
    write_json(
        result_path,
        build_result_payload(
            metric="FCons",
            inputs=inputs,
            results=metrics,
            config=config,
            artifacts={
                "report_file": to_project_relative(report_path),
                "human_dir": to_project_relative(dir_human),
                "llm_dir": to_project_relative(dir_llm),
            },
        ),
    )
    print_metric_summary("FCons", report_path, result_path, results=metrics, summary_keys=("precision", "recall", "f1_score"))

if __name__ == "__main__":
    main()
