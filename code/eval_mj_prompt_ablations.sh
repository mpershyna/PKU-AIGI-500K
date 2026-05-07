#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash code/eval_mj_prompt_ablations_lambda_0_0083.sh [LAMBDA_VALUE]

Examples:
  bash code/eval_mj_prompt_ablations_lambda_0_0083.sh 0.0083
  bash code/eval_mj_prompt_ablations_lambda_0_0083.sh 0.015
  bash code/eval_mj_prompt_ablations_lambda_0_0083.sh 0.0275
  bash code/eval_mj_prompt_ablations_lambda_0_0083.sh 0.05

Environment overrides:
  LAMBDA_VALUE=0.05 bash code/eval_mj_prompt_ablations_lambda_0_0083.sh
  CHECKPOINT=/path/to/checkpoint_latest.pth.tar bash code/eval_mj_prompt_ablations_lambda_0_0083.sh 0.05
  USE_CUDA=0 bash code/eval_mj_prompt_ablations_lambda_0_0083.sh 0.05
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 1
fi

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
LAMBDA_VALUE="${1:-${LAMBDA_VALUE:-0.0083}}"
LAMBDA_DECIMAL="${LAMBDA_VALUE//_/.}"
LAMBDA_SLUG="${LAMBDA_DECIMAL//./_}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mj_only_${LAMBDA_SLUG}/128_${LAMBDA_DECIMAL}/checkpoint_latest.pth.tar}"
TEST_DIR="${TEST_DIR:-data/MJ/test}"
RESULTS_DIR="${RESULTS_DIR:-results}"
USE_CUDA="${USE_CUDA:-1}"

if [[ ! -f "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${TEST_DIR}" ]]; then
  echo "MJ test directory not found: ${TEST_DIR}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

CUDA_ARGS=()
if [[ "${USE_CUDA}" != "0" ]]; then
  CUDA_ARGS+=(--cuda)
fi

CASES=(
  "empty"
  "word_shuffled"
  "swapped"
)

PROMPT_FILES=(
  "data/MJ/MJ_test_empty_prompts.txt"
  "data/MJ/MJ_test_word_shuffled_prompts.txt"
  "data/MJ/MJ_test_swapped_prompts.txt"
)

RESULT_FILES=(
  "${RESULTS_DIR}/mj_${LAMBDA_SLUG}_empty_prompts_results.txt"
  "${RESULTS_DIR}/mj_${LAMBDA_SLUG}_word_shuffled_prompts_results.txt"
  "${RESULTS_DIR}/mj_${LAMBDA_SLUG}_swapped_prompts_results.txt"
)

for i in "${!CASES[@]}"; do
  case_name="${CASES[$i]}"
  prompt_file="${PROMPT_FILES[$i]}"
  result_file="${RESULT_FILES[$i]}"

  if [[ ! -f "${prompt_file}" ]]; then
    echo "Prompt file not found for ${case_name}: ${prompt_file}" >&2
    exit 1
  fi

  echo "[$(date -Is)] Evaluating ${case_name} prompts -> ${result_file}"

  {
    echo "Started: $(date -Is)"
    echo "Case: ${case_name}"
    echo "Lambda: ${LAMBDA_DECIMAL}"
    echo "Checkpoint: ${CHECKPOINT}"
    echo "Test directory: ${TEST_DIR}"
    echo "Prompt file: ${prompt_file}"
    echo

    "${PYTHON}" code/eval.py \
      --checkpoint "${CHECKPOINT}" \
      --data-i "${TEST_DIR}" \
      --data-t "${prompt_file}" \
      "${CUDA_ARGS[@]}"

    echo
    echo "Finished: $(date -Is)"
  } >"${result_file}" 2>&1
done

echo "[$(date -Is)] Done. Results written to:"
printf '  %s\n' "${RESULT_FILES[@]}"
