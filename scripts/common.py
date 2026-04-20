import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


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
