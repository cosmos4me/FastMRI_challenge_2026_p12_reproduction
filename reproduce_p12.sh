#!/usr/bin/env bash
# One-command, deterministic P12 train -> epoch 40--48 soup -> official evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/Data}"
LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${DATA_ROOT}/leaderboard}"
GPU_NUM="${GPU_NUM:-0}"
RUN_NAME="${RUN_NAME:-P12_reproduction_e48}"
if [[ -x /home/interns/.conda/envs/gflow/bin/python3.10 ]]; then
  DEFAULT_PYTHON=/home/interns/.conda/envs/gflow/bin/python3.10
elif command -v python3 >/dev/null 2>&1; then
  DEFAULT_PYTHON=python3
else
  DEFAULT_PYTHON=python
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
RESULT_DIR="${SCRIPT_DIR}/../result/${RUN_NAME}"
CHECKPOINT_DIR="${RESULT_DIR}/checkpoints"

# Training never reads leaderboard labels: train + val only.
DATA_ROOT="${DATA_ROOT}" GPU_NUM="${GPU_NUM}" NUM_EPOCHS=48 RUN_NAME="${RUN_NAME}" \
DISABLE_VALIDATION=1 KEEP_LAST_CHECKPOINTS=48 PYTHON_BIN="${PYTHON_BIN}" \
  bash train_p12_full_48.sh

CHECKPOINTS=()
for epoch in {40..48}; do
  printf -v checkpoint "%s/epoch_%03d.pt" "${CHECKPOINT_DIR}" "${epoch}"
  [[ -s "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }
  CHECKPOINTS+=("${checkpoint}")
done
"${PYTHON_BIN}" average_checkpoints.py "${CHECKPOINTS[@]}" \
  --output "${CHECKPOINT_DIR}/best_model.pt"

# The organiser's recon_eval.py is unchanged.  P12 W-TTA lives only in
# utils/learning/test_part.py, the allowed model I/O contract.
CUDA_VISIBLE_DEVICES="${GPU_NUM}" \
P11_CHECKPOINT_CASCADES=0 \
P11_CPU_OFFLOAD_CASCADES=0 \
P11_CHECKPOINT_UNET_BLOCKS=0 \
P12_TTA_BATCHED=1 \
P12_TTA_IDENTITY_WEIGHT=0.65 \
P12_OUTPUT_SCALE=1.0025 \
"${PYTHON_BIN}" recon_eval.py \
  -g 0 -n "${RUN_NAME}" -p "${LEADERBOARD_ROOT}" \
  --cascade 12 --chans 16 --sens_chans 8
