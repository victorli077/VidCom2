set -euo pipefail

# Qwen2.5-Omni OmniZip eval script — runs both audio-preserving and non-preserving modes.
# Override TASK_NAMES on the command line if needed, e.g.
#   TASK_NAMES="worldsense omnivideobench avut dailyomni" bash omni_qwen25_omnizip_eval.sh

TASK_NAMES="${TASK_NAMES:-worldsense omnivideobench avut dailyomni}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-3B}"
MODEL_TAG="${MODEL_TAG:-omni3b}"
MODEL_NAME="qwen2_5_omni"
METHOD="omnizip"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-768}"
LOG_ROOT="${LOG_ROOT:-/mnt/workspace/lijiasheng-20260417/VidCom2/logs}"

OMNIZIP_G="${OMNIZIP_G:-3}"
OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO:-0.05}"
OMNIZIP_RHO_MIN="${OMNIZIP_RHO_MIN:-0.35}"
OMNIZIP_RHO_MAX="${OMNIZIP_RHO_MAX:-0.85}"

# Per-task retain ratios for audio-preserving mode (OMNIZIP_AUDIO_PRESERVE=1)
export RETAIN_RATIOS_worldsense="0.3492 0.1865"
export RETAIN_RATIOS_omnivideobench="0.3667 0.2084"
export RETAIN_RATIOS_avut="0.3757 0.2199"
export RETAIN_RATIOS_dailyomni="0.3391 0.1739"

# Per-task (r_ratio, rho_audio) pairs for non-preserving mode (OMNIZIP_AUDIO_PRESERVE=0)
#   total40 original -> rho_audio=0.3
#   total25 original -> rho_audio=0.4
declare -A OMNIZIP_RHO_AUDIO_MAP=(
    [worldsense]="0.3764 0.3 0.2083 0.4"
    [omnivideobench]="0.3840 0.3 0.2158 0.4"
    [avut]="0.3862 0.3 0.2210 0.4"
    [dailyomni]="0.3623 0.3 0.1933 0.4"
)

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

# ── Mode 1: audio-preserving (OMNIZIP_AUDIO_PRESERVE=1) ──────────────────────
#   Each task runs two r_ratio values; rho_audio is fixed at 0.3.
OMNIZIP_AUDIO_PRESERVE=1
OMNIZIP_RHO_AUDIO_MODE1=0.3

echo ""
echo "########################################################################"
echo "### MODE 1 — Audio Preserving (OMNIZIP_AUDIO_PRESERVE=1)               ###"
echo "########################################################################"
echo ""

for task_name in ${TASK_NAMES}; do
    task_ratio_var="RETAIN_RATIOS_${task_name}"
    ratios="${RETAIN_RATIOS:-0.35}"
    if [[ -n "${!task_ratio_var-}" ]]; then
        ratios="${!task_ratio_var}"
    fi
    for retain_ratio in ${ratios}; do
        rho_video="$(python -c "r=float('${retain_ratio}'); print(max(0.0, min(1.0, 1.0-r)))")"

        echo "========================================"
        echo "Running OmniZip audio-preserving evaluation"
        echo "  Model: ${MODEL_PATH}"
        echo "  Task: ${task_name}"
        echo "  Video retain ratio: ${retain_ratio}"
        echo "  OmniZip rho video: ${rho_video}"
        echo "  OmniZip rho audio: ${OMNIZIP_RHO_AUDIO_MODE1}"
        echo "  Audio preserve: ${OMNIZIP_AUDIO_PRESERVE}"
        echo "  Max frames: ${MAX_NUM_FRAMES}"
        echo "========================================"

        ts="$(date +"%m-%d-%H-%M")"
        variant="audio_preserve"
        job_name="${MODEL_NAME}_${MODEL_TAG}_${METHOD}_${variant}_${task_name}_v${retain_ratio}_f${MAX_NUM_FRAMES}"
        output_path="${LOG_ROOT}/${MODEL_NAME}/${METHOD}_${variant}/${task_name}/${ts}_${job_name}"
        log_path="${LOG_ROOT}/${MODEL_NAME}/${METHOD}_${variant}/${task_name}/${ts}_${job_name}.log"
        port=$((30000 + RANDOM % 10000))
        mkdir -p "$(dirname "${output_path}")"

        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        COMPRESSOR="${METHOD}" \
        R_RATIO="${retain_ratio}" \
        OMNIZIP_RHO_VIDEO="${rho_video}" \
        OMNIZIP_RHO_AUDIO="${OMNIZIP_RHO_AUDIO_MODE1}" \
        OMNIZIP_G="${OMNIZIP_G}" \
        OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO}" \
        OMNIZIP_RHO_MIN="${OMNIZIP_RHO_MIN}" \
        OMNIZIP_RHO_MAX="${OMNIZIP_RHO_MAX}" \
        OMNIZIP_AUDIO_PRESERVE="${OMNIZIP_AUDIO_PRESERVE}" \
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
            --log_samples_suffix "${METHOD}_${variant}" \
            --output_path "${output_path}" 2>&1 | tee "${log_path}"

        echo "Completed: task=${task_name} retain_ratio=${retain_ratio}"
        echo "Log: ${log_path}"
    done
done

# ── Mode 0: audio-pruned (OMNIZIP_AUDIO_PRESERVE=0) ──────────────────────────
#   Each task runs two (r_ratio, rho_audio) pairs: (total40_rho_audio, total25_rho_audio).
OMNIZIP_AUDIO_PRESERVE=0

echo ""
echo "########################################################################"
echo "### MODE 0 — Audio Pruned (OMNIZIP_AUDIO_PRESERVE=0)                   ###"
echo "########################################################################"
echo ""

for task_name in ${TASK_NAMES}; do
    if [[ ! -v "OMNIZIP_RHO_AUDIO_MAP[${task_name}]" ]]; then
        echo "WARNING: no config found for task '${task_name}', skipping."
        continue
    fi
    read -r r_ratio_40 rho_audio_40 r_ratio_25 rho_audio_25 <<< "${OMNIZIP_RHO_AUDIO_MAP[${task_name}]}"

    for r_ratio in "${r_ratio_40}" "${r_ratio_25}"; do
        if [[ "${r_ratio}" == "${r_ratio_40}" ]]; then
            rho_audio="${rho_audio_40}"
        else
            rho_audio="${rho_audio_25}"
        fi
        rho_video="$(python -c "r=float('${r_ratio}'); print(max(0.0, min(1.0, 1.0-r)))")"

        echo "========================================"
        echo "Running OmniZip audio-pruned evaluation"
        echo "  Model: ${MODEL_PATH}"
        echo "  Task: ${task_name}"
        echo "  Video retain ratio: ${r_ratio}"
        echo "  OmniZip rho video: ${rho_video}"
        echo "  OmniZip rho audio: ${rho_audio}"
        echo "  Audio preserve: ${OMNIZIP_AUDIO_PRESERVE}"
        echo "  Max frames: ${MAX_NUM_FRAMES}"
        echo "========================================"

        ts="$(date +"%m-%d-%H-%M")"
        variant="audio_pruned"
        job_name="${MODEL_NAME}_${MODEL_TAG}_${METHOD}_${variant}_${task_name}_v${r_ratio}_a${rho_audio}_f${MAX_NUM_FRAMES}"
        output_path="${LOG_ROOT}/${MODEL_NAME}/${METHOD}_${variant}/${task_name}/${ts}_${job_name}"
        log_path="${LOG_ROOT}/${MODEL_NAME}/${METHOD}_${variant}/${task_name}/${ts}_${job_name}.log"
        port=$((30000 + RANDOM % 10000))
        mkdir -p "$(dirname "${output_path}")"

        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        COMPRESSOR="${METHOD}" \
        R_RATIO="${r_ratio}" \
        OMNIZIP_RHO_VIDEO="${rho_video}" \
        OMNIZIP_RHO_AUDIO="${rho_audio}" \
        OMNIZIP_G="${OMNIZIP_G}" \
        OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO}" \
        OMNIZIP_RHO_MIN="${OMNIZIP_RHO_MIN}" \
        OMNIZIP_RHO_MAX="${OMNIZIP_RHO_MAX}" \
        OMNIZIP_AUDIO_PRESERVE="${OMNIZIP_AUDIO_PRESERVE}" \
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
            --log_samples_suffix "${METHOD}_${variant}" \
            --output_path "${output_path}" 2>&1 | tee "${log_path}"

        echo "Completed: task=${task_name} r_ratio=${r_ratio} rho_audio=${rho_audio}"
        echo "Log: ${log_path}"
    done
done

echo ""
echo "All evaluations finished."
