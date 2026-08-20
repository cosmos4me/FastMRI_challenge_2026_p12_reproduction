"""M02 training loop with AdamW, warmup-cosine, and challenge-metric selection."""

import copy
import gc
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from utils.common.loss_function import (
    BBoxAwareSSIMLoss,
    ChallengeAlignedSSIMLoss,
    normalized_missing_kspace_l1,
)
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.learning.experiment_metrics import (
    append_metric_record,
    append_training_record,
    evaluate_challenge_metrics,
)
from utils.learning.optim import build_optimizer, build_scheduler
from utils.learning.train_part import validate
from utils.model.model_factory import build_model


def _cpu_snapshot(value):
    """Clone nested training state to CPU without retaining CUDA storage."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _capture_recovery_state(model, optimizer, scheduler, updates):
    return {
        "model": _cpu_snapshot(model.state_dict()),
        "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "scheduler": (
            copy.deepcopy(scheduler.state_dict())
            if scheduler is not None
            else None
        ),
        "updates": int(updates),
    }


def _restore_recovery_state(snapshot, model, optimizer, scheduler):
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    if scheduler is not None and snapshot["scheduler"] is not None:
        scheduler.load_state_dict(snapshot["scheduler"])
    optimizer.zero_grad(set_to_none=True)


def auxiliary_loss_factor(args, epoch):
    aux_enabled = (
        getattr(args, "aux_fi_weight", 0.0) > 0
        or getattr(args, "aux_prompt_weight", 0.0) > 0
        or getattr(args, "p11_low_aux_weight", 0.0) > 0
        or getattr(args, "p11_mid_aux_weight", 0.0) > 0
    )
    if not aux_enabled:
        return 0.0
    decay_start = getattr(args, "aux_decay_start_epoch", 0)
    if decay_start <= 0:
        return 1.0
    current_epoch = epoch + 1
    if current_epoch <= decay_start:
        return 1.0
    if current_epoch >= args.num_epochs:
        return 0.0
    return (args.num_epochs - current_epoch) / (
        args.num_epochs - decay_start
    )


def pe_frequency_band_l1(
    prediction,
    target,
    maximum,
    foreground_mask,
    cutoff,
):
    """Max-normalized PE-spectrum L1 inside a centered frequency band."""
    if not 0.0 < cutoff <= 1.0:
        raise ValueError("cutoff must be in (0, 1]")
    scale = maximum[:, None, None].clamp_min(1e-8)
    foreground = foreground_mask.to(dtype=prediction.dtype)
    prediction = prediction / scale * foreground
    target = target / scale * foreground
    prediction_spectrum = torch.fft.fftshift(
        torch.fft.fft(
            torch.fft.ifftshift(prediction, dim=-1),
            dim=-1,
            norm="ortho",
        ),
        dim=-1,
    )
    target_spectrum = torch.fft.fftshift(
        torch.fft.fft(
            torch.fft.ifftshift(target, dim=-1),
            dim=-1,
            norm="ortho",
        ),
        dim=-1,
    )
    normalized_frequency = torch.linspace(
        -1.0,
        1.0,
        prediction.shape[-1],
        device=prediction.device,
        dtype=prediction.dtype,
    ).abs()
    band = normalized_frequency <= cutoff
    return (
        prediction_spectrum[..., band] - target_spectrum[..., band]
    ).abs().mean()


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type):
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_epoch = start_iter = time.perf_counter()
    total_loss = 0.0
    component_totals = {
        "full": 0.0,
        "bbox": 0.0,
        "foreground_l1": 0.0,
        "missing_kspace_raw": 0.0,
        "missing_kspace_weighted": 0.0,
        "aux_fi_raw": 0.0,
        "aux_prompt_raw": 0.0,
        "aux_fi_weighted": 0.0,
        "aux_prompt_weighted": 0.0,
        "aux_factor": 0.0,
        "aux_low_raw": 0.0,
        "aux_mid_raw": 0.0,
        "aux_low_weighted": 0.0,
        "aux_mid_weighted": 0.0,
    }
    component_steps = 0
    aux_factor = auxiliary_loss_factor(args, epoch)
    max_train_loss = float(os.environ.get("MAX_TRAIN_LOSS", "0"))
    if max_train_loss < 0:
        raise ValueError("MAX_TRAIN_LOSS must be non-negative")
    recovery_interval = int(
        os.environ.get("TRAIN_RECOVERY_INTERVAL", "0")
    )
    if recovery_interval < 0:
        raise ValueError("TRAIN_RECOVERY_INTERVAL must be non-negative")
    use_h11_aux = aux_factor > 0.0 and (
        getattr(args, "aux_fi_weight", 0.0) > 0
        or getattr(args, "aux_prompt_weight", 0.0) > 0
    )
    use_p11_aux = aux_factor > 0.0 and (
        getattr(args, "p11_low_aux_weight", 0.0) > 0
        or getattr(args, "p11_mid_aux_weight", 0.0) > 0
    )
    use_aux = use_h11_aux or use_p11_aux

    optimizer.zero_grad(set_to_none=True)
    successful_updates = 0
    recovery_state = None
    if recovery_interval:
        recovery_state = _capture_recovery_state(
            model, optimizer, scheduler, successful_updates
        )
        print(
            "Training rollback enabled: CPU snapshot every "
            f"{recovery_interval} optimizer updates"
        )
    for iteration, data in enumerate(data_loader):
        mask, kspace, target, maximum, bbox_data, foreground_mask, fname, slice_num = (
            data[:8]
        )
        h16_payload = data[8] if len(data) > 8 else None
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        bbox_data = {
            key: value.cuda(non_blocking=True)
            for key, value in bbox_data.items()
        }
        foreground_mask = foreground_mask.cuda(non_blocking=True)

        aux_fi_value = 0.0
        aux_prompt_value = 0.0
        aux_fi_contribution = 0.0
        aux_prompt_contribution = 0.0
        aux_low_value = 0.0
        aux_mid_value = 0.0
        aux_low_contribution = 0.0
        aux_mid_contribution = 0.0
        missing_kspace_value = 0.0
        missing_kspace_contribution = 0.0
        # Keep graph-bearing locals explicit so a skipped step can release
        # them before the next forward evaluates its right-hand side.
        outputs = output = loss = None
        missing_kspace_loss = missing_kspace_term = None
        aux_fi_loss = aux_prompt_loss = None
        aux_fi_term = aux_prompt_term = None
        aux_low_loss = aux_mid_loss = None
        aux_low_term = aux_mid_term = None
        missing_weight = getattr(
            args, "missing_kspace_loss_weight", 0.0
        )
        use_missing_kspace = h16_payload is not None and missing_weight > 0

        if h16_payload is not None:
            adjacent_kspace = h16_payload["adjacent_kspace"].cuda(
                non_blocking=True
            )
            if use_missing_kspace:
                full_kspace = h16_payload["full_kspace"].cuda(
                    non_blocking=True
                )
                outputs = model(
                    kspace,
                    mask,
                    adjacent_kspace=adjacent_kspace,
                    return_kspace=True,
                )
                output = outputs.reconstruction
            else:
                output = model(
                    kspace,
                    mask,
                    adjacent_kspace=adjacent_kspace,
                )
        elif use_aux:
            outputs = model(
                kspace, mask, return_intermediates=True
            )
            output = outputs.final
        else:
            output = model(kspace, mask)

        loss = loss_type(
            output, target, maximum, bbox_data, foreground_mask
        )
        if use_missing_kspace:
            missing_kspace_loss = normalized_missing_kspace_l1(
                outputs.predicted_kspace,
                full_kspace,
                mask,
            )
            missing_kspace_term = missing_weight * missing_kspace_loss
            loss = loss + missing_kspace_term
            missing_kspace_value = float(
                missing_kspace_loss.detach().cpu()
            )
            missing_kspace_contribution = float(
                missing_kspace_term.detach().cpu()
            )
        if use_h11_aux:
            acceleration = bbox_data["acceleration"]
            aux_fi_loss = loss_type.auxiliary_full_loss(
                outputs.fi_decoder,
                target,
                maximum,
                foreground_mask,
                acceleration,
            )
            aux_prompt_loss = loss_type.auxiliary_full_loss(
                outputs.prompt2,
                target,
                maximum,
                foreground_mask,
                acceleration,
            )
            aux_fi_term = (
                aux_factor
                * args.aux_fi_weight
                * aux_fi_loss
            )
            aux_prompt_term = (
                aux_factor
                * args.aux_prompt_weight
                * aux_prompt_loss
            )
            loss = loss + aux_fi_term + aux_prompt_term
            aux_fi_value = float(aux_fi_loss.detach().cpu())
            aux_prompt_value = float(aux_prompt_loss.detach().cpu())
            aux_fi_contribution = float(aux_fi_term.detach().cpu())
            aux_prompt_contribution = float(aux_prompt_term.detach().cpu())
        elif use_p11_aux:
            aux_low_loss = pe_frequency_band_l1(
                outputs.cascade4,
                target,
                maximum,
                foreground_mask,
                cutoff=0.34,
            )
            aux_mid_loss = pe_frequency_band_l1(
                outputs.cascade8,
                target,
                maximum,
                foreground_mask,
                cutoff=0.68,
            )
            aux_low_term = (
                aux_factor * args.p11_low_aux_weight * aux_low_loss
            )
            aux_mid_term = (
                aux_factor * args.p11_mid_aux_weight * aux_mid_loss
            )
            loss = loss + aux_low_term + aux_mid_term
            aux_low_value = float(aux_low_loss.detach().cpu())
            aux_mid_value = float(aux_mid_loss.detach().cpu())
            aux_low_contribution = float(aux_low_term.detach().cpu())
            aux_mid_contribution = float(aux_mid_term.detach().cpu())
        guarded_loss_value = float(loss.detach())
        loss_is_finite = math.isfinite(guarded_loss_value)
        loss_is_excessive = (
            loss_is_finite
            and max_train_loss > 0
            and guarded_loss_value > max_train_loss
        )
        if not loss_is_finite or loss_is_excessive:
            reason = (
                f"excessive loss {guarded_loss_value:.6g}"
                if loss_is_excessive
                else "non-finite loss"
            )
            print(
                f"WARNING: skipping {reason} "
                f"at epoch={epoch + 1} iteration={iteration} "
                f"fname={fname} slice={slice_num}"
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = output = loss = None
            missing_kspace_loss = missing_kspace_term = None
            aux_fi_loss = aux_prompt_loss = None
            aux_fi_term = aux_prompt_term = None
            aux_low_loss = aux_mid_loss = None
            aux_low_term = aux_mid_term = None
            # Non-reentrant checkpointing can leave Python reference cycles
            # after an aborted graph. Collect them before returning memory to CUDA.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if recovery_state is not None:
                _restore_recovery_state(
                    recovery_state, model, optimizer, scheduler
                )
                successful_updates = recovery_state["updates"]
                print(
                    "RECOVERY: restored last known-good CPU snapshot "
                    f"from optimizer update {successful_updates}"
                )
            continue
        # Refresh recovery only after the current forward has proved that
        # the post-update model is still numerically healthy. Snapshotting
        # immediately after optimizer.step could accidentally preserve the
        # very update whose damage becomes visible on the next slice.
        if (
            recovery_state is not None
            and successful_updates - recovery_state["updates"]
            >= recovery_interval
        ):
            recovery_state = _capture_recovery_state(
                model, optimizer, scheduler, successful_updates
            )
        group_start = (iteration // args.grad_accum_steps) * args.grad_accum_steps
        group_size = min(
            args.grad_accum_steps,
            len(data_loader) - group_start,
        )
        (loss / group_size).backward()
        is_update = (
            (iteration + 1) % args.grad_accum_steps == 0
            or iteration + 1 == len(data_loader)
        )
        if is_update:
            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm
                )
                if not torch.isfinite(grad_norm):
                    print(
                        "WARNING: skipping non-finite gradients "
                        f"at epoch={epoch + 1} iteration={iteration} "
                        f"fname={fname} slice={slice_num}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    outputs = output = loss = None
                    missing_kspace_loss = missing_kspace_term = None
                    aux_fi_loss = aux_prompt_loss = None
                    aux_fi_term = aux_prompt_term = None
                    aux_low_loss = aux_mid_loss = None
                    aux_low_term = aux_mid_term = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if recovery_state is not None:
                        _restore_recovery_state(
                            recovery_state, model, optimizer, scheduler
                        )
                        successful_updates = recovery_state["updates"]
                        print(
                            "RECOVERY: restored last known-good CPU snapshot "
                            f"from optimizer update {successful_updates}"
                        )
                    continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            successful_updates += 1
        total_loss += loss.item()

        components = getattr(loss_type, "last_components", None)
        if components:
            component_totals["full"] += components["full"]
            component_totals["bbox"] += components["bbox"]
            component_totals["foreground_l1"] += components.get(
                "foreground_l1", 0.0
            )
            component_totals["missing_kspace_raw"] += missing_kspace_value
            component_totals["missing_kspace_weighted"] += (
                missing_kspace_contribution
            )
            component_totals["aux_fi_raw"] += aux_fi_value
            component_totals["aux_prompt_raw"] += aux_prompt_value
            component_totals["aux_fi_weighted"] += aux_fi_contribution
            component_totals["aux_prompt_weighted"] += (
                aux_prompt_contribution
            )
            component_totals["aux_low_raw"] += aux_low_value
            component_totals["aux_mid_raw"] += aux_mid_value
            component_totals["aux_low_weighted"] += aux_low_contribution
            component_totals["aux_mid_weighted"] += aux_mid_contribution
            component_totals["aux_factor"] += aux_factor
            component_steps += 1

        aux_total_contribution = (
            aux_fi_contribution
            + aux_prompt_contribution
            + aux_low_contribution
            + aux_mid_contribution
        )
        if iteration % args.report_interval == 0:
            print(
                f"Epoch = [{epoch + 1:3d}/{args.num_epochs:3d}] "
                f"Iter = [{iteration:4d}/{len(data_loader):4d}] "
                f"Loss = {loss.item():.4g} "
                f"KLoss = {missing_kspace_contribution:.3g} "
                f"LR = {optimizer.param_groups[0]['lr']:.3g} "
                f"Aux = {aux_total_contribution:.3g} "
                f"Time = {time.perf_counter() - start_iter:.4f}s"
            )
            start_iter = time.perf_counter()

        if iteration == 0 and torch.cuda.is_available():
            print(
                "CUDA peak after first train step: "
                f"allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                f"reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
            )

    averages = {
        key: value / max(component_steps, 1)
        for key, value in component_totals.items()
    }
    if torch.cuda.is_available():
        print(
            "CUDA peak for training epoch: "
            f"allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
            f"reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
        )
    return (
        total_loss / len(data_loader),
        averages,
        time.perf_counter() - start_epoch,
    )


def save_model(
    args,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    best_challenge_score,
    challenge_metrics,
    is_new_val_best,
    is_new_challenge_best,
):
    checkpoint = {
        "epoch": epoch,
        "args": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "best_challenge_score": best_challenge_score,
        "challenge_metrics": challenge_metrics,
        "exp_dir": args.exp_dir,
    }
    latest = args.exp_dir / "model.pt"
    torch.save(checkpoint, latest)
    if args.save_every_epoch:
        shutil.copyfile(latest, args.exp_dir / f"epoch_{epoch:03d}.pt")
        keep_last = int(getattr(args, "keep_last_checkpoints", 0))
        if keep_last > 0:
            epoch_checkpoints = sorted(
                args.exp_dir.glob("epoch_[0-9][0-9][0-9].pt")
            )
            for old_checkpoint in epoch_checkpoints[:-keep_last]:
                old_checkpoint.unlink()
    if is_new_val_best:
        shutil.copyfile(latest, args.exp_dir / "best_val_loss_model.pt")
    if is_new_challenge_best:
        shutil.copyfile(latest, args.exp_dir / "best_50_50_model.pt")
        shutil.copyfile(latest, args.exp_dir / "best_model.pt")


def should_run_validation(epoch, num_epochs, first_epochs=0, last_epochs=0):
    """Return whether a one-based epoch belongs to either validation window."""
    first_epochs = max(0, int(first_epochs))
    last_epochs = max(0, int(last_epochs))
    if first_epochs == 0 and last_epochs == 0:
        return True
    return epoch <= first_epochs or epoch > num_epochs - last_epochs


def train(args):
    device = torch.device(
        f"cuda:{args.GPU_NUM}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        allow_tf32 = bool(getattr(args, "allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(
            getattr(args, "cudnn_benchmark", False)
        )
        if allow_tf32:
            torch.set_float32_matmul_precision("high")
    print("Current device:", device)

    model = build_model(
        model_type=args.model_type,
        num_cascades=args.cascade,
        chans=args.chans,
        sens_chans=args.sens_chans,
    ).to(device)
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    resume = None
    if args.resume_checkpoint is not None:
        resume = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(resume["model"])
        print(f"Resumed model weights from {args.resume_checkpoint}")
    elif args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if args.allow_partial_init:
            incompatible = model.load_state_dict(
                checkpoint["model"], strict=False
            )
            allowed_new_parameters = (
                ".film.",
                ".high_resolution.",
                "context_encoder.",
                "context_fusions.",
            )
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not any(
                    token in key for token in allowed_new_parameters
                )
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Partial initialization mismatch: "
                    f"invalid missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            print(
                "Partially initialized; new architecture parameters:",
                incompatible.missing_keys,
            )
        else:
            model.load_state_dict(checkpoint["model"])
        print(f"Initialized model weights from {args.init_checkpoint}")

    train_loader = create_data_loaders(
        data_path=args.data_path_train, args=args, shuffle=True
    )
    val_loader = None
    if not getattr(args, "disable_validation", False):
        val_loader = create_data_loaders(data_path=args.data_path_val, args=args)
    optimizer = build_optimizer(args, model)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = build_scheduler(args, optimizer, updates_per_epoch)

    if args.loss_mode == "challenge":
        loss_type = ChallengeAlignedSSIMLoss(
            train_loader.dataset.get_challenge_counts(),
            cross_acc_remask=args.cross_acc_remask,
            foreground_l1_weight=getattr(
                args, "foreground_l1_weight", 0.0
            ),
            active_accelerations=(
                (4,) if getattr(args, "all_anatomy_acc4", False)
                else (8,) if getattr(args, "all_anatomy_acc8", False)
                else (4, 8)
            ),
        ).to(device)
        print("Challenge loss counts:", train_loader.dataset.challenge_counts)
    else:
        loss_type = BBoxAwareSSIMLoss(
            args.bbox_loss_weight, args.foreground_loss_weight
        ).to(device)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None and resume.get("scheduler") is not None:
            scheduler.load_state_dict(resume["scheduler"])
        best_val_loss = float(resume["best_val_loss"])
        best_challenge_score = float(resume.get("best_challenge_score", -1.0))
        start_epoch = int(resume["epoch"])
        print(f"Resuming at epoch {start_epoch + 1}")
    else:
        best_val_loss = float("inf")
        best_challenge_score = -1.0
        start_epoch = 0
    last_challenge_metrics = (
        resume.get("challenge_metrics") if resume is not None else None
    )

    existing_log = Path(args.val_loss_dir) / "val_loss_log.npy"
    if resume is not None and existing_log.is_file():
        val_loss_log = np.load(existing_log)
    else:
        val_loss_log = np.empty((0, 2))

    for epoch in range(start_epoch, args.num_epochs):
        train_loader.dataset.set_epoch(epoch)
        if getattr(args, "acc8_mask_offset_augmentation", False):
            print(
                "P07 acc8 mask-offset coverage:",
                train_loader.dataset.get_acc8_mask_offset_counts(),
            )
        if getattr(args, "balanced_acc_offset_cycle", False):
            print(
                "P12 balanced acc/offset coverage:",
                train_loader.dataset.get_balanced_acc_offset_counts(),
            )
        if getattr(args, "mri_augment", False):
            augment_probability = (
                train_loader.dataset.augmentation_probability()
            )
            print(
                "MRI augmentation probability: "
                f"{augment_probability:.2f}"
            )
        if hasattr(model, "set_adaptation_scale"):
            ramp_epochs = getattr(
                args, "h12_adaptation_ramp_epochs", 0
            )
            adaptation_scale = (
                1.0 if ramp_epochs <= 0
                else min(1.0, (epoch + 1) / ramp_epochs)
            )
            model.set_adaptation_scale(adaptation_scale)
            print(f"H12 train adaptation scale: {adaptation_scale:.2f}")
        if args.loss_mode == "challenge":
            loss_type.set_dataset_counts(
                train_loader.dataset.get_challenge_counts()
            )
        print(f"Epoch #{epoch + 1:2d} ............... {args.net_name} ...............")
        train_loss, train_components, train_time = train_epoch(
            args, epoch, model, train_loader, optimizer, scheduler, loss_type
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        append_training_record(
            args.val_loss_dir,
            epoch + 1,
            train_loss,
            train_components,
            learning_rate,
        )
        if hasattr(model, "set_adaptation_scale"):
            model.set_adaptation_scale(1.0)
        run_validation = (
            not getattr(args, "disable_validation", False)
            and should_run_validation(
                epoch + 1,
                args.num_epochs,
                getattr(args, "validation_first_epochs", 0),
                getattr(args, "validation_last_epochs", 0),
            )
        )
        is_new_val_best = False
        is_new_challenge_best = False
        reconstructions = targets = inputs = None
        if run_validation:
            # Persist the completed epoch before any validation I/O. A missing
            # or malformed validation file must never discard hours of training.
            save_model(
                args=args,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                best_challenge_score=best_challenge_score,
                challenge_metrics=last_challenge_metrics,
                is_new_val_best=False,
                is_new_challenge_best=False,
            )
            print("Saved pre-validation recovery checkpoint: model.pt")
            (
                val_total,
                num_subjects,
                reconstructions,
                targets,
                inputs,
                val_time,
            ) = validate(args, model, val_loader)
            val_loss = float(val_total / num_subjects)
            challenge = evaluate_challenge_metrics(
                args, reconstructions, device
            )
            last_challenge_metrics = challenge
            append_metric_record(
                args.val_loss_dir, epoch + 1, challenge, learning_rate
            )

            val_loss_log = np.append(
                val_loss_log, np.array([[epoch, val_total]]), axis=0
            )
            np.save(
                os.path.join(args.val_loss_dir, "val_loss_log"),
                val_loss_log,
            )

            is_new_val_best = val_loss < best_val_loss
            is_new_challenge_best = (
                challenge["score_50_50"] > best_challenge_score
            )
            best_val_loss = min(best_val_loss, val_loss)
            best_challenge_score = max(
                best_challenge_score, challenge["score_50_50"]
            )
        save_model(
            args=args,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            best_challenge_score=best_challenge_score,
            challenge_metrics=last_challenge_metrics,
            is_new_val_best=is_new_val_best,
            is_new_challenge_best=is_new_challenge_best,
        )

        if run_validation:
            print(
                f"Epoch = [{epoch + 1:4d}/{args.num_epochs:4d}] "
                f"TrainLoss = {train_loss:.4g} ValLoss = {val_loss:.4g} "
                f"SSIM_full = {challenge['ssim_full']:.6f} "
                f"SSIM_bbox = {challenge['ssim_bbox']:.6f} "
                f"Score50/50 = {challenge['score_50_50']:.6f} "
                f"TrainTime = {train_time:.1f}s ValTime = {val_time:.1f}s"
            )
            print(
                "  acc4: "
                f"full={challenge['details']['acc4']['ssim_full']:.6f} "
                f"bbox={challenge['details']['acc4']['ssim_bbox']:.6f}; "
                "acc8: "
                f"full={challenge['details']['acc8']['ssim_full']:.6f} "
                f"bbox={challenge['details']['acc8']['ssim_bbox']:.6f}"
            )
            if is_new_challenge_best:
                print("New best 50/50 checkpoint: best_50_50_model.pt")
                save_reconstructions(
                    reconstructions,
                    args.val_dir,
                    targets=targets,
                    inputs=inputs,
                )
        else:
            print(
                f"Epoch = [{epoch + 1:4d}/{args.num_epochs:4d}] "
                f"TrainLoss = {train_loss:.4g} Validation = skipped "
                f"TrainTime = {train_time:.1f}s"
            )
