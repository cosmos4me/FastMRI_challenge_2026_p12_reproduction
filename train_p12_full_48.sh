#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT="${DATA_ROOT:-/home/interns/data/fmri}"
LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${DATA_ROOT}/leaderboard}"
GPU_NUM="${GPU_NUM:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-48}"
LR="${LR:-0.0002}"
MODEL_TYPE=p12_stable_unified_promptmr_plus
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
CHECKPOINT_CASCADES="${CHECKPOINT_CASCADES:-12}"
CPU_OFFLOAD_CASCADES="${CPU_OFFLOAD_CASCADES:-4}"
BLOCK_CHECKPOINTING="${BLOCK_CHECKPOINTING:-1}"
VALIDATION_FIRST_EPOCHS="${VALIDATION_FIRST_EPOCHS:-10}"
VALIDATION_LAST_EPOCHS="${VALIDATION_LAST_EPOCHS:-10}"
DISABLE_VALIDATION="${DISABLE_VALIDATION:-1}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-48}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

DEFAULT_RUN_NAME=P12_stable_unified_g5dc_1080_e48
DESCRIPTION="P12-stable: P11 cascades 1-8 + zero-init bounded G5 dynamic DC/prior routing in cascades 9-12"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

if [[ "${DISABLE_VALIDATION}" != "0" && "${DISABLE_VALIDATION}" != "1" ]]; then
  echo "ERROR: DISABLE_VALIDATION must be 0 or 1" >&2
  exit 1
fi

REQUIRED_DATA=(
  "${DATA_ROOT}/train/image" "${DATA_ROOT}/train/kspace"
  "${DATA_ROOT}/val/image" "${DATA_ROOT}/val/kspace"
)
if [[ "${DISABLE_VALIDATION}" == "0" ]]; then
  REQUIRED_DATA+=(
    "${LEADERBOARD_ROOT}/acc4/image" "${LEADERBOARD_ROOT}/acc4/kspace"
    "${LEADERBOARD_ROOT}/acc8/image" "${LEADERBOARD_ROOT}/acc8/kspace"
  )
fi

for required in "${REQUIRED_DATA[@]}"; do
  if [[ ! -d "${required}" ]] || \
     ! find "${required}" -maxdepth 1 -type f -print -quit | grep -q .; then
    echo "ERROR: missing or empty data directory: ${required}" >&2
    exit 1
  fi
done

if [[ -x /home/interns/.conda/envs/gflow/bin/python3.10 ]]; then
  DEFAULT_PYTHON=/home/interns/.conda/envs/gflow/bin/python3.10
elif command -v python3 >/dev/null 2>&1; then
  DEFAULT_PYTHON=python3
else
  DEFAULT_PYTHON=python
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
  echo "P12 initialization: resume ${RESUME_CHECKPOINT}"
else
  echo "P12 initialization: random scratch (seed 430)"
fi

VALIDATION_ARGS=()
if [[ "${DISABLE_VALIDATION}" == "1" ]]; then
  VALIDATION_ARGS+=(--disable-validation)
fi

export CUDA_VISIBLE_DEVICES="${GPU_NUM}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/utils/model:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.6,max_split_size_mb:32}"
export P11_CHECKPOINT_CASCADES="${CHECKPOINT_CASCADES}"
export P11_CPU_OFFLOAD_CASCADES="${CPU_OFFLOAD_CASCADES}"
export P11_CHECKPOINT_UNET_BLOCKS="${BLOCK_CHECKPOINTING}"
export ADAMW_FOREACH="${ADAMW_FOREACH:-0}"
export MAX_TRAIN_LOSS="${MAX_TRAIN_LOSS:-5.0}"
export TRAIN_RECOVERY_INTERVAL="${TRAIN_RECOVERY_INTERVAL:-100}"

echo "P12 model: ${MODEL_TYPE}"
echo "P12 run: ${RUN_NAME}; physical GPU=${GPU_NUM}; epochs=${NUM_EPOCHS}"
echo "P12 architecture: ${DESCRIPTION}"
echo "P12 exclusions: no all-cascade FiLM, whitening, sensitivity smoothing, SENSE/PCG, SPIRiT, or morphology routing"
echo "P12 data: train+val fully sampled anatomy; 16-epoch 50/50 acc4/acc8 all-offset cycle"
if [[ "${DISABLE_VALIDATION}" == "1" ]]; then
  echo "P12 validation: disabled; training and checkpoint saving remain enabled"
else
  echo "P12 validation: public leaderboard acc4 + acc8; first=${VALIDATION_FIRST_EPOCHS}, last=${VALIDATION_LAST_EPOCHS}"
fi
echo "P12 1080 memory: foreach=${ADAMW_FOREACH}, cpu_offload_cascades=${CPU_OFFLOAD_CASCADES}, block_checkpointing=${BLOCK_CHECKPOINTING}, max_train_loss=${MAX_TRAIN_LOSS}, recovery_interval=${TRAIN_RECOVERY_INTERVAL}"

"${PYTHON_BIN}" -u train.py \
  -g 0 \
  -b 1 \
  -e "${NUM_EPOCHS}" \
  -l "${LR}" \
  -r 100 \
  -n "${RUN_NAME}" \
  -t "${DATA_ROOT}/train" \
  --extra-data-path-train "${DATA_ROOT}/val" \
  -v "${LEADERBOARD_ROOT}/acc4" \
  --extra-data-path-val "${LEADERBOARD_ROOT}/acc8" \
  --model-type "${MODEL_TYPE}" \
  --cascade 12 \
  --chans 16 \
  --sens_chans 8 \
  --loss-mode challenge \
  --foreground-l1-weight 0.005 \
  --optimizer adamw \
  --weight-decay 0.01 \
  --conditioning-lr-scale 0.25 \
  --scheduler warmup_step \
  --warmup-epochs 2 \
  --scheduler-step-epochs 24 40 \
  --scheduler-gamma 0.3 \
  --grad-accum-steps 1 \
  --grad-clip-norm 0.1 \
  --balanced-acc-offset-cycle \
  --bbox-loss-weight 0 \
  --foreground-loss-weight 0 \
  --bbox-sample-weight 1 \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --allow-tf32 \
  --save-every-epoch \
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}" \
  --validation-first-epochs "${VALIDATION_FIRST_EPOCHS}" \
  --validation-last-epochs "${VALIDATION_LAST_EPOCHS}" \
  "${VALIDATION_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --seed 430
