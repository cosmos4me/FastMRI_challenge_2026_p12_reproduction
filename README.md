# P12 FastMRI 2026 Reproduction

This repository is a minimal, reproducible package for the final unified P12
knee-MRI reconstruction model used in the 2026 SNU FastMRI Challenge Phase 2.
It intentionally excludes unrelated baselines, ablations, checkpoint assembly
scripts, and training artifacts.

## What is included

- **Architecture:** 12-cascade PromptMR+ reconstruction model with a shared
  acc4/acc8 backbone, actual-mask acquisition descriptor, and bounded learned
  soft data-consistency/prior routing in cascades 9--12.
- **Training:** exact 48-epoch schedule, train+val fully-sampled anatomy,
  balanced acc4/acc8 and offset remasking, challenge-aligned SSIM loss plus
  foreground L1.
- **Selection:** equal parameter average of epochs 40--48.
- **Inference:** one model only; physically correct W-axis reflection TTA:
  `0.65 * original + 0.35 * reflected`, then output scale `1.0025`.
- **Harness:** `recon_eval.py` is included unchanged.  TTA is implemented only
  in the permitted `utils/learning/test_part.py` model-I/O contract.

## Data layout

Set `DATA_ROOT` to a directory containing:

```text
DATA_ROOT/
  train/{image,kspace}/
  val/{image,kspace}/
  leaderboard/
    acc4/{image,kspace}/
    acc8/{image,kspace}/
```

`leaderboard` is used only in the final evaluation step.  Training reads only
`train` and `val`.

## Environment

Tested inference environment: Python 3.10, PyTorch 2.3.1 + CUDA 12.1, GTX1080
8GB.  On Vessl, install dependencies once:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
```

For a preconfigured Conda environment, point `PYTHON_BIN` to its Python binary.

## One-command reproduction

```bash
cd FastMRI_challenge_2026_p12_reproduction
DATA_ROOT=/root/Data GPU_NUM=0 PYTHON_BIN=python3 bash reproduce_p12.sh
```

The command performs, in order:

1. random-seed-fixed 48-epoch training on `train + val`;
2. preserves all 48 numbered checkpoints;
3. averages `epoch_040.pt` through `epoch_048.pt` into `best_model.pt`;
4. runs the unmodified official `recon_eval.py` on leaderboard data.

Outputs are written to `../result/P12_reproduction_e48/` by default.  Override
`RUN_NAME` if several runs share the same parent directory.

## Vessl GTX1080: one command in tmux

The default launcher enables cascade/block checkpointing and limited CPU boundary
offload so that the 12-cascade P12 model fits on a GTX1080 8GB card.  From a
Vessl terminal, launch the full reproduction in a detachable tmux session:

```bash
cd ~/FastMRI_challenge_2026_p12_reproduction

RUN_NAME=P12_reproduction_e48
mkdir -p /root/result/${RUN_NAME}

tmux new-session -d -s p12_repro \
"cd ~/FastMRI_challenge_2026_p12_reproduction && \
set -o pipefail && \
DATA_ROOT=/root/Data GPU_NUM=0 RUN_NAME=${RUN_NAME} \
PYTHON_BIN=/usr/local/bin/python3 \
bash reproduce_p12.sh 2>&1 | tee /root/result/${RUN_NAME}/train_and_eval.log"
```

If `/usr/local/bin/python3` is not the environment where PyTorch is installed,
replace `PYTHON_BIN` with the correct executable.  Monitor or attach with:

```bash
tmux attach -t p12_repro
# or, without attaching:
tail -f /root/result/P12_reproduction_e48/train_and_eval.log
```

The session survives terminal disconnection.  `reproduce_p12.sh` runs the
leaderboard evaluation only after all 48 training epochs and the epoch 40--48
checkpoint average finish.

## Resume after interruption

```bash
DATA_ROOT=/root/Data GPU_NUM=0 RUN_NAME=P12_reproduction_e48 \
RESUME_CHECKPOINT=/root/result/P12_reproduction_e48/checkpoints/epoch_024.pt \
NUM_EPOCHS=48 PYTHON_BIN=python3 bash train_p12_full_48.sh
```

After training, run the averaging and evaluation commands from
`reproduce_p12.sh`, or simply run that script only when starting from scratch.

## Submission checkpoint

The supplied final checkpoint should be placed at:

```text
../result/P12_reproduction_e48/checkpoints/best_model.pt
```

The package uses only measured multi-coil k-space and its actual sampling mask
at inference.  It does not use supplied GRAPPA, bbox annotations, or image H5
fields for reconstruction.

## Third-party attribution

`utils/model/promptmr_plus/` includes the PromptMR+ implementation and its
license notice at `utils/model/promptmr_plus/LICENSE.md`.
