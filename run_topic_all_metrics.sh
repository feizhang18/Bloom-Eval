#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

TOPIC_DIR=""
HUMAN_DIR="${PROJECT_ROOT}/data/cs_01/human"
LLM_DIR="${PROJECT_ROOT}/data/cs_01/llm"
TASK_FILE=""
OUTPUT_DIR=""
HUMAN_ARTICLE_FILE=""
LLM_ARTICLE_FILE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL=""
BASE_URL=""
SAVE_RAW_RESPONSE=0
HUMAN_DIR_SET=0
LLM_DIR_SET=0

usage() {
  cat <<'EOF'
Usage:
  ./run_topic_all_metrics.sh [options]
  ./run_topic_all_metrics.sh --human-dir DIR --llm-dir DIR [options]
  ./run_topic_all_metrics.sh --topic-dir DIR [options]

Default input:
  data/cs_01/human and data/cs_01/llm

Input layout:
  human dir:
    content.json
    outline.json
    reference.json
    expert_article.json

  llm dir:
    content.json
    outline.json
    reference.json
    llm_article.json  # required by FMI when available

Options:
  --topic-dir DIR          Topic directory containing human/ and llm/
  --human-dir DIR          Human input directory
  --llm-dir DIR            LLM input directory
  --task-file FILE         Explicit task JSON for FAP/FNov/ROQ, default: human/expert_article.json
  --human-article-file FILE
                           Explicit full human article JSON for FMI, default: human/expert_article.json
  --llm-article-file FILE  Explicit full LLM article JSON for FMI, default: llm/llm_article.json if present
  --output-dir DIR         Root output directory for all metrics, default: results/<topic>/all_metrics
  --python BIN             Python executable, default: python3
  --model NAME             Override --model for API-based metrics
  --base-url URL           Override --base_url for API-based metrics
  --save-raw-response      Pass --save_raw_response to all metrics
  -h, --help               Show this help

Examples:
  ./run_topic_all_metrics.sh \
    --human-dir data/cs_01/human \
    --llm-dir /path/to/cs_01/llm \
    --task-file data/cs_01/human/expert_article.json \
    --output-dir results/cs_01_all

  ./run_topic_all_metrics.sh \
    --topic-dir /path/to/cs_01 \
    --model gpt-5-mini \
    --save-raw-response
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Missing file: $path"
}

discover_task_file() {
  local dir="$1"
  local task_file

  task_file="${dir}/expert_article.json"
  if [[ -f "$task_file" ]]; then
    echo "$task_file"
    return
  fi

  task_file="$(find "$dir" -maxdepth 1 -type f -name '[0-9]*_*.json' | sort | head -n 1)"
  if [[ -n "$task_file" ]]; then
    echo "$task_file"
    return
  fi

  find "$dir" -maxdepth 1 -type f -name '*.json' \
    ! -name 'content.json' \
    ! -name 'outline.json' \
    ! -name 'reference.json' \
    ! -name '*.bak' \
    ! -name '*.old_bak' \
    ! -name '*.data_json_bak' \
    | sort | head -n 2
}

discover_human_article_file() {
  local dir="$1"
  local article_file="${dir}/expert_article.json"
  if [[ -f "$article_file" ]]; then
    echo "$article_file"
    return
  fi
  discover_task_file "$dir" | sed '/^$/d' | head -n 1
}

discover_llm_article_file() {
  local dir="$1"
  local article_file="${dir}/llm_article.json"
  if [[ -f "$article_file" ]]; then
    echo "$article_file"
    return
  fi
  find "$dir" -maxdepth 1 -type f -name '*.json' \
    ! -name 'content.json' \
    ! -name 'outline.json' \
    ! -name 'reference.json' \
    ! -name '*.bak' \
    ! -name '*.old_bak' \
    ! -name '*.data_json_bak' \
    | sort | head -n 1
}

run_metric() {
  local name="$1"
  shift

  local metric_dir="${OUTPUT_DIR}/${name}"
  mkdir -p "$metric_dir"

  echo
  echo "Running ${name}"
  echo "Output: ${metric_dir}"

  if "$PYTHON_BIN" "$@" --output_dir "$metric_dir"; then
    SUCCEEDED+=("$name")
    echo "[OK] ${name}"
  else
    FAILED+=("$name")
    echo "[FAIL] ${name}" >&2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic-dir)
      TOPIC_DIR="$2"
      shift 2
      ;;
    --human-dir)
      HUMAN_DIR="$2"
      HUMAN_DIR_SET=1
      shift 2
      ;;
    --llm-dir)
      LLM_DIR="$2"
      LLM_DIR_SET=1
      shift 2
      ;;
    --task-file)
      TASK_FILE="$2"
      shift 2
      ;;
    --human-article-file)
      HUMAN_ARTICLE_FILE="$2"
      shift 2
      ;;
    --llm-article-file)
      LLM_ARTICLE_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --save-raw-response)
      SAVE_RAW_RESPONSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ -n "$TOPIC_DIR" ]]; then
  [[ -d "$TOPIC_DIR" ]] || die "Topic directory not found: $TOPIC_DIR"
  [[ "$HUMAN_DIR_SET" -eq 1 ]] || HUMAN_DIR="${TOPIC_DIR}/human"
  [[ "$LLM_DIR_SET" -eq 1 ]] || LLM_DIR="${TOPIC_DIR}/llm"
fi

[[ -d "$HUMAN_DIR" ]] || die "Human directory not found: $HUMAN_DIR"
[[ -d "$LLM_DIR" ]] || die "LLM directory not found: $LLM_DIR"

HUMAN_DIR="$(cd "$HUMAN_DIR" && pwd)"
LLM_DIR="$(cd "$LLM_DIR" && pwd)"

HUMAN_CONTENT_JSON="${HUMAN_DIR}/content.json"
HUMAN_OUTLINE_JSON="${HUMAN_DIR}/outline.json"
HUMAN_REFERENCE_JSON="${HUMAN_DIR}/reference.json"
LLM_CONTENT_JSON="${LLM_DIR}/content.json"
LLM_OUTLINE_JSON="${LLM_DIR}/outline.json"
LLM_REFERENCE_JSON="${LLM_DIR}/reference.json"

require_file "$HUMAN_CONTENT_JSON"
require_file "$HUMAN_OUTLINE_JSON"
require_file "$HUMAN_REFERENCE_JSON"
require_file "$LLM_CONTENT_JSON"
require_file "$LLM_OUTLINE_JSON"
require_file "$LLM_REFERENCE_JSON"

if [[ -z "$TASK_FILE" ]]; then
  TASK_FILE_CANDIDATES="$(discover_task_file "$HUMAN_DIR")"
  TASK_FILE_COUNT="$(printf '%s\n' "$TASK_FILE_CANDIDATES" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "$TASK_FILE_COUNT" -eq 1 ]]; then
    TASK_FILE="$TASK_FILE_CANDIDATES"
  elif [[ "$TASK_FILE_COUNT" -gt 1 ]]; then
    die "Multiple task JSON candidates found under ${HUMAN_DIR}; use --task-file explicitly"
  fi
fi
[[ -n "$TASK_FILE" ]] || die "Task file not found under ${HUMAN_DIR}; use --task-file explicitly"
require_file "$TASK_FILE"
TASK_FILE="$(cd "$(dirname "$TASK_FILE")" && pwd)/$(basename "$TASK_FILE")"

if [[ -z "$HUMAN_ARTICLE_FILE" ]]; then
  HUMAN_ARTICLE_FILE="$(discover_human_article_file "$HUMAN_DIR")"
fi
[[ -n "$HUMAN_ARTICLE_FILE" ]] || die "Human article JSON not found under ${HUMAN_DIR}; use --human-article-file explicitly"
require_file "$HUMAN_ARTICLE_FILE"
HUMAN_ARTICLE_FILE="$(cd "$(dirname "$HUMAN_ARTICLE_FILE")" && pwd)/$(basename "$HUMAN_ARTICLE_FILE")"

if [[ -z "$LLM_ARTICLE_FILE" ]]; then
  LLM_ARTICLE_FILE="$(discover_llm_article_file "$LLM_DIR")"
fi
if [[ -n "$LLM_ARTICLE_FILE" ]]; then
  require_file "$LLM_ARTICLE_FILE"
  LLM_ARTICLE_FILE="$(cd "$(dirname "$LLM_ARTICLE_FILE")" && pwd)/$(basename "$LLM_ARTICLE_FILE")"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  die "OPENAI_API_KEY is not set. The full 16-metric run includes API-based metrics."
fi

TOPIC_NAME="$(basename "$(dirname "$HUMAN_DIR")")"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/results/${TOPIC_NAME}/all_metrics"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

RAW_ARGS=()
MODEL_ARGS=()
FMI_ARTICLE_ARGS=()

if [[ "$SAVE_RAW_RESPONSE" -eq 1 ]]; then
  RAW_ARGS+=(--save_raw_response)
fi
if [[ -n "$MODEL" ]]; then
  MODEL_ARGS+=(--model "$MODEL")
fi
if [[ -n "$BASE_URL" ]]; then
  MODEL_ARGS+=(--base_url "$BASE_URL")
fi
FMI_ARTICLE_ARGS+=(--article_file_human "$HUMAN_ARTICLE_FILE")
if [[ -n "$LLM_ARTICLE_FILE" ]]; then
  FMI_ARTICLE_ARGS+=(--article_file_llm "$LLM_ARTICLE_FILE")
fi

SUCCEEDED=()
FAILED=()

echo "Topic: ${TOPIC_NAME}"
echo "Human dir: ${HUMAN_DIR}"
echo "LLM dir:   ${LLM_DIR}"
echo "Task file: ${TASK_FILE}"
echo "Human article: ${HUMAN_ARTICLE_FILE}"
echo "LLM article: ${LLM_ARTICLE_FILE:-none}"
echo "Output:    ${OUTPUT_DIR}"

run_metric "EFid" \
  "${PROJECT_ROOT}/scripts/level1/run_EFid.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "FCons" \
  "${PROJECT_ROOT}/scripts/level1/run_FCons.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "HIRC" \
  "${PROJECT_ROOT}/scripts/level1/run_HIRC.py" \
  --reference_file_human "$HUMAN_REFERENCE_JSON" \
  --reference_file_llm "$LLM_REFERENCE_JSON" \
  "${RAW_ARGS[@]}"

run_metric "CF" \
  "${PROJECT_ROOT}/scripts/level2/run_CF.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --reference_file_human "$HUMAN_REFERENCE_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  --reference_file_llm "$LLM_REFERENCE_JSON" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "OTC" \
  "${PROJECT_ROOT}/scripts/level2/run_OTC.py" \
  --outline_file_human "$HUMAN_OUTLINE_JSON" \
  --outline_file_llm "$LLM_OUTLINE_JSON" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "TBal" \
  "${PROJECT_ROOT}/scripts/level2/run_TBal.py" \
  --reference_file_human "$HUMAN_REFERENCE_JSON" \
  --reference_file_llm "$LLM_REFERENCE_JSON" \
  "${RAW_ARGS[@]}"

run_metric "TFSim" \
  "${PROJECT_ROOT}/scripts/level2/run_TFSim.py" \
  --reference_file_human "$HUMAN_REFERENCE_JSON" \
  --reference_file_llm "$LLM_REFERENCE_JSON" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "DSI" \
  "${PROJECT_ROOT}/scripts/level3/run_DSI.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  "${RAW_ARGS[@]}"

run_metric "FAP" \
  "${PROJECT_ROOT}/scripts/level3/run_FAP.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  --task_file "$TASK_FILE" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "FMI" \
  "${PROJECT_ROOT}/scripts/level3/run_FMI.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  "${FMI_ARTICLE_ARGS[@]}" \
  "${RAW_ARGS[@]}"

run_metric "SCS" \
  "${PROJECT_ROOT}/scripts/level4/run_SCS.py" \
  --outline_file_llm "$LLM_OUTLINE_JSON" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "SCons" \
  "${PROJECT_ROOT}/scripts/level4/run_SCons.py" \
  --outline_file_human "$HUMAN_OUTLINE_JSON" \
  --outline_file_llm "$LLM_OUTLINE_JSON" \
  "${RAW_ARGS[@]}"

run_metric "STS" \
  "${PROJECT_ROOT}/scripts/level4/run_STS.py" \
  --outline_file_human "$HUMAN_OUTLINE_JSON" \
  --outline_file_llm "$LLM_OUTLINE_JSON" \
  "${RAW_ARGS[@]}"

run_metric "CAA" \
  "${PROJECT_ROOT}/scripts/level5/run_CAA.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "FNov" \
  "${PROJECT_ROOT}/scripts/level6/run_FNov.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  --task_file "$TASK_FILE" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "ROQ" \
  "${PROJECT_ROOT}/scripts/level6/run_ROQ.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  --task_file "$TASK_FILE" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

SCORE_SUMMARY_FILE="${OUTPUT_DIR}/score_summary.md"
"$PYTHON_BIN" - "$OUTPUT_DIR" "$SCORE_SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any

output_dir = Path(sys.argv[1])
score_summary_file = Path(sys.argv[2])


def load_results(metric: str) -> dict[str, Any]:
    path = output_dir / metric / "result.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("results", {})


def get_value(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def f1(metric: str) -> str:
    return fmt(get_value(load_results(metric), "f1_score"))


def f1_ds(metric: str, ds_key: str) -> str:
    results = load_results(metric)
    return f"{fmt(get_value(results, 'f1_score'))} / {fmt(get_value(results, ds_key))}"


def efid_f1_ds() -> str:
    results = load_results("EFid")
    ds = get_value(results, "ds_score")
    if ds is not None:
        return f"{fmt(get_value(results, 'f1_score'))} / {fmt(ds)}"
    sims = [
        get_value(results, "jensen_shannon_sim"),
        get_value(results, "hellinger_sim"),
        get_value(results, "total_variation_sim"),
    ]
    ds = None if any(value is None for value in sims) else sum(sims) / len(sims)
    return f"{fmt(get_value(results, 'f1_score'))} / {fmt(ds)}"


rows = [
    ("Entity Fidelity EFid", "F1 / DS", efid_f1_ds()),
    ("High-Impact Reference Coverage HIRC", "F1", f1("HIRC")),
    ("Factual Consistency FCons", "F1", f1("FCons")),
    ("Outline Topic Coverage OTC", "F1", f1("OTC")),
    ("Citation Faithfulness CF", "F1", fmt(get_value(load_results("CF"), "llm", "f1_score"))),
    ("Thematic Focus Similarity TFSim", "F1 / DS", f1_ds("TFSim", "ds_score")),
    ("Thematic Balance TBal", "1 - Gini", fmt(get_value(load_results("TBal"), "llm", "breadth_score"))),
    ("Formatting Integrity FMI", "Jaccard similarity", fmt(get_value(load_results("FMI"), "llm_score"))),
    ("Document Structure Integrity DSI", "Checklist completion ratio", fmt(get_value(load_results("DSI"), "llm_score"))),
    ("Framework Application FAP", "GRADE relative score", fmt(get_value(load_results("FAP"), "fap_score"))),
    ("Semantic Tree Similarity STS", "Normalized tree-edit similarity", fmt(get_value(load_results("STS"), "semantic_tree_similarity"))),
    ("Shape Consistency SCons", "Depth/breadth consistency score", fmt(get_value(load_results("SCons"), "shape_consistency"))),
    ("Structural Clarity Score SCS", "1 - redundant sections / total sections", fmt(get_value(load_results("SCS"), "scs_llm"))),
    ("Critical Analysis Alignment CAA", "F1", f1("CAA")),
    ("Framework Novelty FNov", "GRADE relative score", fmt(get_value(load_results("FNov"), "framework_novelty_score"))),
    ("Research Outlook Quality ROQ", "GRADE relative score", fmt(get_value(load_results("ROQ"), "research_heuristics_score"))),
]

lines = [
    "| Metric | Reported metric | Score |",
    "|---|---|---|",
]
lines.extend(f"| {metric} | {reported_metric} | {score} |" for metric, reported_metric, score in rows)
score_summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
{
  echo "Topic: ${TOPIC_NAME}"
  echo "Human dir: ${HUMAN_DIR}"
  echo "LLM dir: ${LLM_DIR}"
  echo "Task file: ${TASK_FILE}"
  echo "Human article: ${HUMAN_ARTICLE_FILE}"
  echo "LLM article: ${LLM_ARTICLE_FILE:-none}"
  echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
  echo "Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
  echo
  echo "Score summary: ${SCORE_SUMMARY_FILE}"
  echo
  cat "${SCORE_SUMMARY_FILE}"
} > "${SUMMARY_FILE}"

echo "Completed"
echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "Score summary: ${SCORE_SUMMARY_FILE}"
cat "${SCORE_SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
