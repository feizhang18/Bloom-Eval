#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

TOPIC_DIR=""
HUMAN_DIR=""
LLM_DIR=""
TASK_FILE=""
OUTPUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL=""
BASE_URL=""
SAVE_RAW_RESPONSE=0
INCLUDE_READABILITY=0

usage() {
  cat <<'EOF'
Usage:
  ./run_topic_all_metrics.sh --human-dir DIR --llm-dir DIR [options]
  ./run_topic_all_metrics.sh --topic-dir DIR [options]

Required input layout:
  human dir:
    content.json
    outline.json
    reference.json
    <digits>_*.json   # task file, unless --task-file is provided

  llm dir:
    content.json
    outline.json
    reference.json

Options:
  --topic-dir DIR          Topic directory containing human/ and llm/
  --human-dir DIR          Human input directory
  --llm-dir DIR            LLM input directory
  --task-file FILE         Explicit task JSON for FAP/FNov/ROQ
  --output-dir DIR         Root output directory for all metrics
  --python BIN             Python executable, default: python3
  --model NAME             Override --model for API-based metrics
  --base-url URL           Override --base_url for API-based metrics
  --save-raw-response      Pass --save_raw_response to all metrics
  --include-readability    Also run the extra readability script (outside the core 16)
  -h, --help               Show this help

Examples:
  ./run_topic_all_metrics.sh \
    --human-dir data/experimental_data/cs_01/human \
    --llm-dir /path/to/cs_01/llm \
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
  find "$dir" -maxdepth 1 -type f -name '[0-9]*_*.json' | sort | head -n 1
}

prepare_markdown_from_content_json() {
  local src_json="$1"
  local dst_md="$2"
  "$PYTHON_BIN" - "$src_json" "$dst_md" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    parts = [item for item in data if isinstance(item, str)]
    text = "\n".join(parts)
elif isinstance(data, str):
    text = data
elif isinstance(data, dict):
    parts = [value for value in data.values() if isinstance(value, str)]
    text = "\n".join(parts)
else:
    text = str(data)

dst.write_text(text, encoding="utf-8")
PY
}

run_metric() {
  local name="$1"
  shift

  local metric_dir="${OUTPUT_DIR}/${name}"
  mkdir -p "$metric_dir"

  echo
  echo "=================================================="
  echo "Running ${name}"
  echo "Output: ${metric_dir}"
  echo "=================================================="

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
      shift 2
      ;;
    --llm-dir)
      LLM_DIR="$2"
      shift 2
      ;;
    --task-file)
      TASK_FILE="$2"
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
    --include-readability)
      INCLUDE_READABILITY=1
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
  [[ -n "$HUMAN_DIR" ]] || HUMAN_DIR="${TOPIC_DIR}/human"
  [[ -n "$LLM_DIR" ]] || LLM_DIR="${TOPIC_DIR}/llm"
fi

[[ -n "$HUMAN_DIR" ]] || die "--human-dir is required unless --topic-dir is provided"
[[ -n "$LLM_DIR" ]] || die "--llm-dir is required unless --topic-dir is provided"
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
  TASK_FILE="$(discover_task_file "$HUMAN_DIR")"
fi
[[ -n "$TASK_FILE" ]] || die "Task file not found under ${HUMAN_DIR}; use --task-file explicitly"
require_file "$TASK_FILE"
TASK_FILE="$(cd "$(dirname "$TASK_FILE")" && pwd)/$(basename "$TASK_FILE")"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  die "OPENAI_API_KEY is not set. The full 16-metric run includes API-based metrics."
fi

TOPIC_NAME="$(basename "$(dirname "$HUMAN_DIR")")"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/results/all_metrics/${TOPIC_NAME}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

HUMAN_CONTENT_MD="${TMP_DIR}/human_content.md"
LLM_CONTENT_MD="${TMP_DIR}/llm_content.md"
prepare_markdown_from_content_json "$HUMAN_CONTENT_JSON" "$HUMAN_CONTENT_MD"
prepare_markdown_from_content_json "$LLM_CONTENT_JSON" "$LLM_CONTENT_MD"

RAW_ARGS=()
MODEL_ARGS=()

if [[ "$SAVE_RAW_RESPONSE" -eq 1 ]]; then
  RAW_ARGS+=(--save_raw_response)
fi
if [[ -n "$MODEL" ]]; then
  MODEL_ARGS+=(--model "$MODEL")
fi
if [[ -n "$BASE_URL" ]]; then
  MODEL_ARGS+=(--base_url "$BASE_URL")
fi

SUCCEEDED=()
FAILED=()

echo "Topic: ${TOPIC_NAME}"
echo "Human dir: ${HUMAN_DIR}"
echo "LLM dir:   ${LLM_DIR}"
echo "Task file: ${TASK_FILE}"
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
  --content_file_human "$HUMAN_CONTENT_MD" \
  --content_file_llm "$LLM_CONTENT_MD" \
  "${RAW_ARGS[@]}"

run_metric "FAP" \
  "${PROJECT_ROOT}/scripts/level3/run_FAP.py" \
  --content_file_human "$HUMAN_CONTENT_JSON" \
  --content_file_llm "$LLM_CONTENT_JSON" \
  --task_file "$TASK_FILE" \
  "${RAW_ARGS[@]}" "${MODEL_ARGS[@]}"

run_metric "FMI" \
  "${PROJECT_ROOT}/scripts/level3/run_FMI.py" \
  --content_file_human "$HUMAN_CONTENT_MD" \
  --content_file_llm "$LLM_CONTENT_MD" \
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

if [[ "$INCLUDE_READABILITY" -eq 1 ]]; then
  run_metric "Readability" \
    "${PROJECT_ROOT}/scripts/others/run_Readability.py" \
    --content_file_human "$HUMAN_CONTENT_JSON" \
    --content_file_llm "$LLM_CONTENT_JSON" \
    "${RAW_ARGS[@]}"
fi

SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
{
  echo "Topic: ${TOPIC_NAME}"
  echo "Human dir: ${HUMAN_DIR}"
  echo "LLM dir: ${LLM_DIR}"
  echo "Task file: ${TASK_FILE}"
  echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
  echo "Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
} > "${SUMMARY_FILE}"

echo
echo "=================================================="
echo "Completed"
echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "Summary: ${SUMMARY_FILE}"
echo "=================================================="

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
