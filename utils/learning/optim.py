"""Optimizer and per-update learning-rate schedules for controlled ablations."""

import math
import os

import torch

from utils.learning.cpu_offload_adamw import CPUOffloadAdamW


def build_optimizer(args, model):
    conditioning_scale = getattr(args, "conditioning_lr_scale", 1.0)
    conditioning_prefixes = (
        "acquisition_encoder.",
        "late_routers.",
        "final_soft_dc.",
    )
    if conditioning_scale < 1.0:
        base_parameters = []
        conditioning_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = (
                conditioning_parameters
                if name.startswith(conditioning_prefixes)
                else base_parameters
            )
            destination.append(parameter)
        if not conditioning_parameters:
            raise ValueError(
                "--conditioning-lr-scale requires P12-stable parameters"
            )
        parameters = [
            {"params": base_parameters, "lr": args.lr},
            {
                "params": conditioning_parameters,
                "lr": args.lr * conditioning_scale,
            },
        ]
        print(
            "P12 conditioning LR: "
            f"base={args.lr:g}, routing={args.lr * conditioning_scale:g}"
        )
    else:
        parameters = model.parameters()
    if args.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=args.lr)
    if args.optimizer == "adamw":
        if os.environ.get("CPU_OFFLOAD_ADAMW", "0") == "1":
            print("Optimizer: CPU-offloaded AdamW")
            return CPUOffloadAdamW(
                parameters, lr=args.lr, weight_decay=args.weight_decay
            )
        foreach_value = os.environ.get("ADAMW_FOREACH", "1")
        if foreach_value not in {"0", "1"}:
            raise ValueError("ADAMW_FOREACH must be 0 or 1")
        foreach = foreach_value == "1"
        print(f"Optimizer: AdamW foreach={foreach}")
        return torch.optim.AdamW(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            foreach=foreach,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def build_scheduler(args, optimizer, updates_per_epoch):
    if args.scheduler == "none":
        return None
    if args.scheduler in {"smartify_step", "warmup_step"}:
        milestones = [
            int(epoch * updates_per_epoch)
            for epoch in args.scheduler_step_epochs
        ]
        if args.scheduler == "smartify_step":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=args.scheduler_gamma,
            )
        warmup_updates = max(
            1, int(round(args.warmup_epochs * updates_per_epoch))
        )

        def warmup_step_multiplier(update):
            if update < warmup_updates:
                return max(1, update + 1) / warmup_updates
            decays = sum(update >= milestone for milestone in milestones)
            return args.scheduler_gamma ** decays

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_step_multiplier
        )
    if args.scheduler != "warmup_cosine":
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    total_updates = max(1, int(args.num_epochs * updates_per_epoch))
    warmup_updates = min(
        total_updates - 1, int(round(args.warmup_epochs * updates_per_epoch))
    )
    min_ratio = args.min_lr / args.lr

    def lr_multiplier(update):
        if warmup_updates > 0 and update < warmup_updates:
            return max(1, update + 1) / warmup_updates
        decay_updates = max(1, total_updates - warmup_updates - 1)
        progress = min(1.0, max(0.0, (update - warmup_updates) / decay_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
