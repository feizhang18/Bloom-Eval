import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5-mini"


def add_common_arguments(
    parser: argparse.ArgumentParser,
    metric_name: str,
    default_model: str = DEFAULT_MODEL,
    include_model: bool = True,
) -> None:
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results") / metric_name.lower(),
        help="Output directory for results, relative to project root.",
    )
    parser.add_argument(
        "--save_raw_response",
        action="store_true",
        help="Save raw LLM responses to log files.",
    )
    if include_model:
        parser.add_argument(
            "--model",
            type=str,
            default=os.getenv("BLOOM_EVAL_MODEL", default_model),
            help="LLM model name.",
        )
        parser.add_argument(
            "--base_url",
            type=str,
            default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            help="LLM API base URL.",
        )


def resolve_output_dir(output_dir: Path) -> Path:
    output_path = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_project_relative(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def load_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    write_json(path, data)


def build_log_path(
    log_dir: Optional[Path],
    purpose: str,
    *,
    suffix: str = "_raw_response.txt",
    timestamp_format: str = "%Y%m%d-%H%M%S",
) -> Optional[Path]:
    if log_dir is None:
        return None
    timestamp = time.strftime(timestamp_format)
    return log_dir / f"{timestamp}_{purpose}{suffix}"


def _strip_code_fences(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        return cleaned[7:-3].strip()
    if cleaned.startswith("```"):
        return cleaned[3:-3].strip()
    return cleaned


def call_llm(
    client: Any,
    model: str,
    prompt: str,
    log_file: Optional[Path] = None,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> str:
    log_name = log_file.name if log_file is not None else "none"
    if verbose:
        print(f"  [LLM] Requesting model, log: {log_name}")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    answer = (response.choices[0].message.content or "").strip()
    if log_file is not None:
        write_text(log_file, answer)
    return answer


def call_llm_with_retry(
    client: Any,
    model: str,
    prompt: str,
    log_file: Optional[Path] = None,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    failure_log_message: Optional[str] = None,
    verbose: bool = True,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return call_llm(
                client,
                model,
                prompt,
                log_file,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                verbose=verbose,
            )
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    if log_file is not None and failure_log_message is not None and last_error is not None:
        write_text(log_file, f"{failure_log_message}\nLast error: {last_error}")
    if last_error is None:
        raise RuntimeError("LLM request failed without an explicit exception.")
    raise last_error


def extract_json_snippet(content: str, kind: Literal["object", "array"]) -> str:
    cleaned = _strip_code_fences(content)
    if kind == "object":
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    else:
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a JSON {kind} in the response.")
    return match.group(0)


def parse_llm_json(
    content: str,
    *,
    kind: Literal["object", "array", "auto"] = "auto",
    replace_nbsp: bool = False,
) -> Any:
    cleaned = _strip_code_fences(content)
    if replace_nbsp:
        cleaned = cleaned.replace("\xa0", " ")
    if kind == "auto":
        return json.loads(cleaned)
    return json.loads(extract_json_snippet(cleaned, kind))


def call_llm_for_json(
    client: Any,
    model: str,
    prompt: str,
    log_file: Optional[Path] = None,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    try:
        answer = call_llm(
            client,
            model,
            prompt,
            log_file,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            verbose=verbose,
        )
        return parse_llm_json(answer, kind="auto")
    except Exception as e:
        print(f"  [Error] LLM call or JSON parse failed: {e}")
        return {}


def load_json_text(
    file_path: os.PathLike[str] | str,
    *,
    joiner: str = "\n",
    min_text_length: int = 0,
) -> str:
    data = load_json(file_path)

    def extract_text(element: Any) -> Iterable[str]:
        if isinstance(element, str):
            yield element
        elif isinstance(element, list):
            for item in element:
                yield from extract_text(item)
        elif isinstance(element, dict):
            for value in element.values():
                yield from extract_text(value)

    text_parts = [
        part.strip()
        for part in extract_text(data)
        if isinstance(part, str) and len(part.strip()) > min_text_length
    ]
    return joiner.join(text_parts)


def load_json_list_text(
    file_path: os.PathLike[str] | str,
    *,
    joiner: str = "\n",
) -> str:
    data = load_json(file_path)
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return joiner.join(data)
    return ""


def load_json_field_text(
    file_path: os.PathLike[str] | str,
    field: str,
) -> str:
    data = load_json(file_path)
    if isinstance(data, dict):
        value = data.get(field, "")
        return value if isinstance(value, str) else ""
    return ""


def build_result_payload(
    metric: str,
    inputs: Dict[str, Any],
    results: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "metric": metric,
        "inputs": inputs,
        "results": results,
    }
    if config:
        payload["config"] = config
    if artifacts:
        payload["artifacts"] = artifacts
    return payload
