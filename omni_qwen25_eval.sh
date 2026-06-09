#!/usr/bin/env bash
set -euo pipefail

# Minimal Qwen2.5-Omni random-pruning eval script.
# Override these variables on the command line if needed, e.g.
#   TASK_NAMES="videomme" RETAIN_RATIOS="0.35 0.45" bash examples/omni_qwen25_random_eval.sh

TASK_NAMES="${TASK_NAMES:-worldsense dailyomni av_speakerbench_audiovisual avut omnivideobench lvomnibench}"
RETAIN_RATIOS="${RETAIN_RATIOS:-0.25}"
METHOD_NAMES="${METHOD_NAMES:-vidcom2}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
MODEL_TAG="${MODEL_TAG:-omni7b}"
MODEL_NAME="qwen2_5_omni"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-768}"
LOG_ROOT="${LOG_ROOT:-/mnt/workspace/lijiasheng-20260417/VidCom2/logs}"

RANDOM_SEED="${RANDOM_SEED:-1234}"
RANDOM_PRINT="${RANDOM_PRINT:-1}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi


for method_name in ${METHOD_NAMES}; do
  echo ""
  echo "###############################################################################"
  echo "# METHOD: ${method_name}"
  echo "###############################################################################"

  for task_name in ${TASK_NAMES}; do
    # 支持每个任务单独指定 ratio，优先级：RETAIN_RATIOS_<task_name> > RETAIN_RATIOS
    task_ratio_var="RETAIN_RATIOS_${task_name}"
    ratios="${RETAIN_RATIOS}"
    if [[ -n "${!task_ratio_var-}" ]]; then
      ratios="${!task_ratio_var}"
      echo "Using per-task ratios for '${task_name}': ${ratios}"
    fi
    for retain_ratio in ${ratios}; do
      echo "========================================"
      echo "Running random visual-only evaluation"
      echo "  Model: ${MODEL_PATH}"
      echo "  Method: ${method_name}"
      echo "  Task: ${task_name}"
      echo "  Video retain ratio: ${retain_ratio}"
      echo "  Audio retain ratio: 1.0"
      echo "  Max frames: ${MAX_NUM_FRAMES}"
      echo "========================================"

      ts="$(date +"%m-%d-%H-%M")"
      job_name="${MODEL_NAME}_${MODEL_TAG}_${method_name}_visualonly_${task_name}_v${retain_ratio}_f${MAX_NUM_FRAMES}"
      output_path="${LOG_ROOT}/${MODEL_NAME}/${method_name}_visualonly/${task_name}/${ts}_${job_name}"
      log_path="${LOG_ROOT}/${MODEL_NAME}/${method_name}_visualonly/${task_name}/${ts}_${job_name}.log"
      port=$((30000 + RANDOM % 10000))
      mkdir -p "$(dirname "${output_path}")"

      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      COMPRESSOR="${method_name}" \
      R_RATIO="${retain_ratio}" \
      RANDOM_R_RATIO="${retain_ratio}" \
      RANDOM_SEED="${RANDOM_SEED}" \
      RANDOM_PRINT="${RANDOM_PRINT}" \
      RANDOM_ASSERT_AUDIO_PRESERVED=1 \
      TOKEN_STATS=1 \
      TOKEN_STATS_CASE=1 \
      VIDCOM_TOKEN_STATS=1 \
      VIDCOM_TOKEN_STATS_CASE=1 \
      accelerate launch \
        --main_process_port "${port}" \
        --num_processes "${NUM_PROCESSES}" \
        -m lmms_eval \
        --model "${MODEL_NAME}" \
        --model_args "pretrained=${MODEL_PATH},attn_implementation=${ATTN_IMPLEMENTATION},max_num_frames=${MAX_NUM_FRAMES}" \
        --tasks "${task_name}" \
        --batch_size 1 \
        "${LIMIT_ARGS[@]}" \
        --log_samples \
        --output_path "${output_path}" 2>&1 | tee "${log_path}"

      echo "Completed: method=${method_name} task=${task_name} retain_ratio=${retain_ratio}"
      echo "Log: ${log_path}"
    done
  done
done
