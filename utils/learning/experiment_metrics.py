"""Research-only validation reporting with the fixed challenge metrics."""

import csv
import json
from pathlib import Path

import h5py
import torch

from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full


def evaluate_challenge_metrics(args, reconstructions, device):
    totals = {
        acc: {"full": 0.0, "bbox": 0.0, "slices": 0, "boxes": 0}
        for acc in ("acc4", "acc8")
    }
    per_volume = {}
    ssim = SSIM().to(device)
    validation_roots = [Path(args.data_path_val)] + [
        Path(root)
        for root in (getattr(args, "extra_data_path_val", []) or [])
    ]
    image_directories = [root / "image" for root in validation_roots]
    image_paths = {}
    for image_directory in image_directories:
        if not image_directory.is_dir():
            raise FileNotFoundError(
                f"validation image directory not found: {image_directory}"
            )
        for path in image_directory.glob("*.h5"):
            previous = image_paths.get(path.name)
            if previous is not None and previous != path:
                raise RuntimeError(
                    "duplicate validation image filename across roots: "
                    f"{path.name}: {previous}, {path}"
                )
            image_paths[path.name] = path

    with torch.inference_mode():
        for fname, reconstruction in reconstructions.items():
            acc = "acc4" if "acc4" in fname else "acc8"
            volume = {"acceleration": acc, "full_sum": 0.0, "full_count": 0,
                      "bbox_sum": 0.0, "bbox_count": 0}
            image_path = image_paths.get(fname)
            if image_path is None:
                searched = ", ".join(str(path) for path in image_directories)
                raise FileNotFoundError(
                    f"validation target {fname} not found in: {searched}"
                )
            with h5py.File(image_path, "r") as hf:
                target = hf[args.target_key][:]
                maximum = hf.attrs[args.max_key]
                annotations = json.loads(hf.attrs.get("annotations", "{}"))

            for slice_index in range(target.shape[0]):
                recon_t = torch.from_numpy(reconstruction[slice_index]).to(device)
                target_t = torch.from_numpy(target[slice_index]).to(device)
                foreground_t = torch.from_numpy(
                    foreground_mask(target[slice_index])
                ).to(device=device, dtype=torch.float32)
                value = ssim_full(ssim, recon_t, target_t, foreground_t, maximum)
                if value is not None:
                    totals[acc]["full"] += value
                    volume["full_sum"] += value
                    volume["full_count"] += 1
                    totals[acc]["slices"] += 1
                for box in annotations.get(str(slice_index), []):
                    value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                    if value is not None:
                        totals[acc]["bbox"] += value
                        volume["bbox_sum"] += value
                        volume["bbox_count"] += 1
                        totals[acc]["boxes"] += 1

            volume["ssim_full"] = volume["full_sum"] / max(volume["full_count"], 1)
            volume["ssim_bbox"] = volume["bbox_sum"] / max(volume["bbox_count"], 1)
            per_volume[fname] = volume
    details = {}
    for acc, total in totals.items():
        details[acc] = {
            "ssim_full": total["full"] / max(total["slices"], 1),
            "ssim_bbox": total["bbox"] / max(total["boxes"], 1),
            "slices": total["slices"],
            "boxes": total["boxes"],
        }
    specialist_acceleration = {
        "p01m_acc4_promptmr_plus": "acc4",
        "p02m_acc4_promptmr_plus": "acc4",
        "p07m_acc8_multimask_promptmr_plus": "acc8",
        "p11m_acc8_sampling_aware_promptmr_plus": "acc8",
    }.get(getattr(args, "model_type", ""))
    active_accelerations = (
        (specialist_acceleration,)
        if specialist_acceleration is not None
        else ("acc4", "acc8")
    )
    full = sum(
        details[acc]["ssim_full"] for acc in active_accelerations
    ) / len(active_accelerations)
    bbox = sum(
        details[acc]["ssim_bbox"] for acc in active_accelerations
    ) / len(active_accelerations)
    return {
        "ssim_full": full,
        "ssim_bbox": bbox,
        "score_50_50": 0.5 * (full + bbox),
        "active_accelerations": list(active_accelerations),
        "details": details,
        "per_volume": per_volume,
    }


def append_metric_record(result_dir, epoch, metrics, learning_rate):
    path = Path(result_dir) / "val_challenge_metrics.csv"
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "ssim_full": metrics["ssim_full"],
        "ssim_bbox": metrics["ssim_bbox"],
        "score_50_50": metrics["score_50_50"],
        "acc4_full": metrics["details"]["acc4"]["ssim_full"],
        "acc4_bbox": metrics["details"]["acc4"]["ssim_bbox"],
        "acc8_full": metrics["details"]["acc8"]["ssim_full"],
        "acc8_bbox": metrics["details"]["acc8"]["ssim_bbox"],
    }
    write_header = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    (Path(result_dir) / "latest_val_challenge_metrics.json").write_text(
        json.dumps({"epoch": epoch, **metrics}, indent=2) + "\n"
    )


def append_training_record(
    result_dir, epoch, train_loss, components, learning_rate
):
    """Append one row per epoch for plotting and experiment comparison."""
    path = Path(result_dir) / "train_loss_metrics.csv"
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_loss,
        "weighted_full_loss": components.get("full", 0.0),
        "weighted_bbox_loss": components.get("bbox", 0.0),
        "weighted_foreground_l1_loss": components.get(
            "foreground_l1", 0.0
        ),
        "missing_kspace_raw_loss": components.get(
            "missing_kspace_raw", 0.0
        ),
        "weighted_missing_kspace_loss": components.get(
            "missing_kspace_weighted", 0.0
        ),
        "aux_fi_raw_loss": components.get("aux_fi_raw", 0.0),
        "aux_prompt_raw_loss": components.get("aux_prompt_raw", 0.0),
        "weighted_aux_fi_loss": components.get("aux_fi_weighted", 0.0),
        "weighted_aux_prompt_loss": components.get("aux_prompt_weighted", 0.0),
        "aux_low_raw_loss": components.get("aux_low_raw", 0.0),
        "aux_mid_raw_loss": components.get("aux_mid_raw", 0.0),
        "weighted_aux_low_loss": components.get(
            "aux_low_weighted", 0.0
        ),
        "weighted_aux_mid_loss": components.get(
            "aux_mid_weighted", 0.0
        ),
        "aux_factor": components.get("aux_factor", 0.0),
    }
    write_header = not path.exists()
    fieldnames = list(row)
    if not write_header:
        # Older H10--H14 runs predate foreground L1 logging. Preserve their
        # existing CSV schema when training is resumed, while new H15 runs
        # record the additional component from their first epoch onward.
        with path.open(newline="") as file:
            existing_header = next(csv.reader(file), None)
        if existing_header:
            fieldnames = existing_header
            row = {key: row.get(key, 0.0) for key in fieldnames}
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
