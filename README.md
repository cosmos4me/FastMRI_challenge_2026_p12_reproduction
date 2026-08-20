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
- **Inference:** one model only; the original and physically correct W-axis
  reflection are batched into one forward pass, blended as
  `0.65 * original + 0.35 * reflected`, then scaled by `1.0025`.
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
PYTHON_BIN=python3 \
bash reproduce_p12.sh 2>&1 | tee /root/result/${RUN_NAME}/train_and_eval.log"
```

If `python3` is not the environment where PyTorch is installed, replace
`PYTHON_BIN` with the correct executable.  Monitor or attach with:

```bash
tmux attach -t p12_repro
# or, without attaching:
tail -f /root/result/P12_reproduction_e48/train_and_eval.log
```

The session survives terminal disconnection.  `reproduce_p12.sh` runs the
leaderboard evaluation only after all 48 training epochs and the epoch 40--48
checkpoint average finish.

## Evaluate the supplied final checkpoint on Vessl

The model parameter file is distributed separately from this source repository
(for example as the required `.pt` submission attachment).  It is the equal
parameter average of epochs 40--48 and has SHA-256:

```text
ca8d6f95c984447a208e45411872b3c40ee37dcf8b59c8ce95fff586661c505f
```

After placing it at `/root/p12_avg_epoch040_048.pt`, run the exact final
inference configuration below.  The official `recon_eval.py` file is unchanged;
the TTA is implemented only in `utils/learning/test_part.py`.

```bash
cd ~/FastMRI_challenge_2026_p12_reproduction
git pull --ff-only origin main

RUN_NAME=P12_avg40_48_wtta
mkdir -p /root/result/${RUN_NAME}/checkpoints
cp /root/p12_avg_epoch040_048.pt \
  /root/result/${RUN_NAME}/checkpoints/best_model.pt

export PYTHONPATH="$PWD/utils/model:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0 \
P11_CHECKPOINT_CASCADES=0 \
P11_CPU_OFFLOAD_CASCADES=0 \
P11_CHECKPOINT_UNET_BLOCKS=0 \
P12_TTA_BATCHED=1 \
P12_TTA_IDENTITY_WEIGHT=0.65 \
P12_OUTPUT_SCALE=1.0025 \
python3 recon_eval.py \
  -g 0 -n "${RUN_NAME}" -p /root/Data/leaderboard \
  --cascade 12 --chans 16 --sens_chans 8
```

### Exact final-result pipeline

The evaluation command above is the complete pipeline for the provided weight:

```text
p12_avg_epoch040_048.pt
  -> copied as checkpoints/best_model.pt
  -> utils.learning.test_part.load_model() strict-loads P12
  -> for every slice: measured k-space + actual mask
  -> original and PE/W-axis reflected inputs are stacked as batch size 2
  -> both reconstructions are produced by one batched model forward
  -> 0.65 * original + 0.35 * inverse-reflected reconstruction
  -> output * 1.0025
  -> unchanged recon_eval.py scores and writes final H5 outputs
```

The W reflection is not an image-file augmentation: it is applied to the
measured raw k-space using the centered-FFT phase ramp, and the Cartesian mask
is reflected with it.  The inverse reflection is applied before averaging.
This is the exact final TTA implementation in `utils/learning/test_part.py`.

`recon_eval.py` prints `SSIM_full`, `SSIM_bbox`, and `ms/slice`; it also writes
the final submission-format reconstructions to:

```text
/root/result/P12_avg40_48_wtta/reconstructions_leaderboard/acc4/
/root/result/P12_avg40_48_wtta/reconstructions_leaderboard/acc8/
```

The command uses one P12 checkpoint only and applies the selected physical
PE-axis reflection TTA `0.65 * original + 0.35 * reflection`, followed by
multiplicative output scale `1.0025`.  No bbox annotation, image H5 field,
supplied GRAPPA, or filename-derived acceleration is used by reconstruction.

## Measured final result

The exact command above was measured on the organiser's GTX1080 environment:

| Metric | Result |
|---|---:|
| SSIM_full | 0.9333 |
| SSIM_bbox | 0.9317 |
| SSIM_full (acc4 / acc8) | 0.9504 / 0.9162 |
| SSIM_bbox (acc4 / acc8) | 0.9515 / 0.9120 |
| Reconstruction time | 2322.10 s total |
| Time per slice | 1049.3 ms |

Using the displayed rounded SSIM values, `Score50 = 0.9325`. With the challenge
time bonus formula, the displayed values correspond to approximately
`0.932995`; the official value may differ slightly because scoring uses
unrounded metrics internally.

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
