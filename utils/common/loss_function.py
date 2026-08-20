"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class BBoxAwareSSIMLoss(SSIMLoss):
    """Blend whole-image SSIM loss with SSIM inside annotation boxes."""

    def __init__(self, bbox_weight=0.0, foreground_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        if bbox_weight < 0 or foreground_weight < 0 or bbox_weight + foreground_weight > 1:
            raise ValueError("loss weights must be non-negative and sum to at most 1")
        self.bbox_weight = bbox_weight
        self.foreground_weight = foreground_weight

    def forward(self, X, Y, data_range, bbox_mask, foreground_mask=None):
        # Newer data loaders return both the legacy union mask and the exact
        # annotation boxes. Keep old M01/M02 commands reproducible.
        if isinstance(bbox_mask, dict):
            bbox_mask = bbox_mask["mask"]
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        ssim_map = ((2 * ux * uy + C1) * (2 * vxy + C2)) / (
            (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
        )

        full_loss = 1 - ssim_map.mean(dim=(1, 2, 3))
        if self.bbox_weight == 0.0 and self.foreground_weight == 0.0:
            return full_loss.mean()

        pad = self.win_size // 2
        valid_mask = bbox_mask[:, None, pad:-pad, pad:-pad].to(ssim_map.dtype)
        mask_sum = valid_mask.sum(dim=(1, 2, 3))
        bbox_loss = 1 - (ssim_map * valid_mask).sum(dim=(1, 2, 3)) / mask_sum.clamp_min(1)
        bbox_loss = torch.where(mask_sum > 0, bbox_loss, full_loss)
        if self.foreground_weight > 0:
            foreground_valid = foreground_mask[:, None, pad:-pad, pad:-pad].to(ssim_map.dtype)
            foreground_sum = foreground_valid.sum(dim=(1, 2, 3))
            foreground_loss = 1 - (
                (ssim_map * foreground_valid).sum(dim=(1, 2, 3))
                / foreground_sum.clamp_min(1)
            )
            foreground_loss = torch.where(foreground_sum > 0, foreground_loss, full_loss)
        else:
            foreground_loss = full_loss
        full_weight = 1 - self.bbox_weight - self.foreground_weight
        return (full_weight * full_loss + self.bbox_weight * bbox_loss
                + self.foreground_weight * foreground_loss).mean()


class ChallengeAlignedSSIMLoss(SSIMLoss):
    """Differentiable surrogate matching the fixed challenge aggregation.

    The official metric averages foreground SSIM over slices and bbox SSIM over
    individual boxes, separately for acc4/acc8, before the final 50/50 mean.
    """

    def __init__(
        self,
        dataset_counts,
        cross_acc_remask=False,
        foreground_l1_weight=0.0,
        active_accelerations=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_acc_remask = bool(cross_acc_remask)
        self.foreground_l1_weight = float(foreground_l1_weight)
        if self.foreground_l1_weight < 0:
            raise ValueError("foreground_l1_weight must be non-negative")
        self.total_slices = int(dataset_counts["total_slices"])
        self.total_boxes = int(dataset_counts["total_boxes"])
        self.slice_counts = {
            4: int(dataset_counts["slice_counts"][4]),
            8: int(dataset_counts["slice_counts"][8]),
        }
        self.box_counts = {
            4: int(dataset_counts["box_counts"][4]),
            8: int(dataset_counts["box_counts"][8]),
        }
        self.active_accelerations = tuple(
            active_accelerations
            if active_accelerations is not None
            else (4, 8)
        )
        if not self.active_accelerations or any(
            acc not in (4, 8) for acc in self.active_accelerations
        ):
            raise ValueError("active accelerations must be a subset of (4, 8)")
        if self.total_slices <= 0 or self.total_boxes <= 0:
            raise ValueError("challenge loss requires non-empty slices and boxes")
        if any(self.slice_counts[acc] <= 0 for acc in self.active_accelerations):
            raise ValueError("active accelerations require training slices")
        if any(self.box_counts[acc] <= 0 for acc in self.active_accelerations):
            raise ValueError("active accelerations require training boxes")
        self.last_components = {}

    def set_dataset_counts(self, dataset_counts):
        """Update inverse-frequency weights after an epoch remask flip."""
        self.slice_counts = {
            4: int(dataset_counts["slice_counts"][4]),
            8: int(dataset_counts["slice_counts"][8]),
        }
        self.box_counts = {
            4: int(dataset_counts["box_counts"][4]),
            8: int(dataset_counts["box_counts"][8]),
        }
        if any(self.slice_counts[acc] <= 0 for acc in self.active_accelerations):
            raise ValueError("active accelerations require training slices")
        if any(self.box_counts[acc] <= 0 for acc in self.active_accelerations):
            raise ValueError("active accelerations require training boxes")

    def _foreground_loss(self, X, Y, data_range, foreground):
        """Match ssim_full: mask images first, then average valid pixels."""
        X = X * foreground
        Y = Y * foreground
        X4 = X.unsqueeze(1)
        Y4 = Y.unsqueeze(1)
        ranges = data_range[:, None, None, None]
        C1 = (self.k1 * ranges) ** 2
        C2 = (self.k2 * ranges) ** 2
        ux = F.conv2d(X4, self.w)
        uy = F.conv2d(Y4, self.w)
        uxx = F.conv2d(X4 * X4, self.w)
        uyy = F.conv2d(Y4 * Y4, self.w)
        uxy = F.conv2d(X4 * Y4, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        ssim_map = ((2 * ux * uy + C1) * (2 * vxy + C2)) / (
            (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
        )
        pad = self.win_size // 2
        valid = foreground[:, None, pad:-pad, pad:-pad].to(ssim_map.dtype)
        denom = valid.sum(dim=(1, 2, 3))
        score = (ssim_map * valid).sum(dim=(1, 2, 3)) / denom.clamp_min(1)
        return torch.where(denom > 0, 1 - score, torch.zeros_like(score))

    def auxiliary_full_loss(
        self, X, Y, data_range, foreground_mask, acceleration
    ):
        """Return acc4/acc8-balanced foreground full-SSIM loss.

        Unlike ``forward``, this is a full-score loss rather than the 0.5
        full component of the final 50/50 objective. H11 applies its explicit
        0.05 and 0.10 coefficients outside this method.
        """
        acceleration = acceleration.to(device=X.device)
        if not torch.all((acceleration == 4) | (acceleration == 8)):
            raise ValueError("acceleration metadata must be 4 or 8")
        foreground = foreground_mask.to(device=X.device, dtype=X.dtype)
        full_loss = self._foreground_loss(X, Y, data_range, foreground)
        full_weight = torch.empty_like(full_loss)
        for acc in (4, 8):
            selected = acceleration == acc
            full_weight[selected] = (
                self.total_slices / (2.0 * self.slice_counts[acc])
            )
        return (full_weight * full_loss).mean()

    def _box_loss_sums(self, X, Y, data_range, boxes, box_count):
        sums = []
        for batch_index in range(X.shape[0]):
            sample_sum = X.new_zeros(())
            count = int(box_count[batch_index].item())
            for box_index in range(count):
                x0, y0, x1, y1 = (
                    int(value.item()) for value in boxes[batch_index, box_index]
                )
                if x1 - x0 < self.win_size or y1 - y0 < self.win_size:
                    continue
                sample_sum = sample_sum + super().forward(
                    X[batch_index:batch_index + 1, y0:y1, x0:x1],
                    Y[batch_index:batch_index + 1, y0:y1, x0:x1],
                    data_range[batch_index:batch_index + 1],
                )
            sums.append(sample_sum)
        return torch.stack(sums)

    @staticmethod
    def _foreground_l1_loss(X, Y, data_range, foreground):
        """Return per-slice foreground MAE normalized by the target range."""
        foreground_sum = foreground.sum(dim=(1, 2))
        absolute_error = (
            (X - Y).abs() * foreground
        ).sum(dim=(1, 2)) / foreground_sum.clamp_min(1)
        normalized = absolute_error / data_range.to(X.dtype).clamp_min(1e-12)
        return torch.where(
            foreground_sum > 0,
            normalized,
            torch.zeros_like(normalized),
        )

    def forward(self, X, Y, data_range, bbox_data, foreground_mask):
        if not isinstance(bbox_data, dict):
            raise TypeError("challenge loss requires exact bbox metadata")
        foreground = foreground_mask.to(device=X.device, dtype=X.dtype)
        boxes = bbox_data["boxes"].to(device=X.device)
        box_count = bbox_data["count"].to(device=X.device)
        acceleration = bbox_data["acceleration"].to(device=X.device)

        full_loss = self._foreground_loss(X, Y, data_range, foreground)
        box_loss_sum = self._box_loss_sums(X, Y, data_range, boxes, box_count)
        foreground_l1 = self._foreground_l1_loss(
            X, Y, data_range, foreground
        )

        full_weight = torch.empty_like(full_loss)
        box_weight = torch.empty_like(full_loss)
        l1_weight = torch.empty_like(full_loss)
        group_count = len(self.active_accelerations)
        valid_acceleration = torch.zeros_like(acceleration, dtype=torch.bool)
        for acc in self.active_accelerations:
            selected = acceleration == acc
            valid_acceleration |= selected
            full_weight[selected] = (
                self.total_slices
                / (2.0 * group_count * self.slice_counts[acc])
            )
            box_weight[selected] = (
                self.total_slices
                / (2.0 * group_count * self.box_counts[acc])
            )
            l1_weight[selected] = (
                self.total_slices
                / (group_count * self.slice_counts[acc])
            )
        if not torch.all(valid_acceleration):
            raise ValueError(
                f"acceleration must be in {self.active_accelerations}"
            )

        weighted_full = full_weight * full_loss
        weighted_bbox = box_weight * box_loss_sum
        weighted_foreground_l1 = (
            self.foreground_l1_weight * l1_weight * foreground_l1
        )
        total = weighted_full + weighted_bbox + weighted_foreground_l1
        self.last_components = {
            "full": float(weighted_full.detach().mean().cpu()),
            "bbox": float(weighted_bbox.detach().mean().cpu()),
            "foreground_l1": float(
                weighted_foreground_l1.detach().mean().cpu()
            ),
            "total": float(total.detach().mean().cpu()),
            "boxes": int(box_count.detach().sum().cpu()),
        }
        return total.mean()


def normalized_missing_kspace_l1(
    predicted_kspace: torch.Tensor,
    target_kspace: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Complex L1 on unmeasured lines, normalized per slice and coil."""
    if predicted_kspace.shape != target_kspace.shape:
        raise ValueError("predicted and target k-space shapes must match")
    if predicted_kspace.ndim != 5 or predicted_kspace.shape[-1] != 2:
        raise ValueError("k-space must have shape [B, coils, H, W, 2]")

    missing = ~mask.bool()[..., 0]
    missing = missing.expand_as(predicted_kspace[..., 0])
    missing_count = missing.sum(dim=(-2, -1)).clamp_min(1)
    complex_error = torch.linalg.vector_norm(
        predicted_kspace - target_kspace,
        dim=-1,
    )
    mean_missing_error = (
        complex_error * missing.to(complex_error.dtype)
    ).sum(dim=(-2, -1)) / missing_count
    target_rms = target_kspace.square().sum(dim=-1).mean(
        dim=(-2, -1)
    ).sqrt().clamp_min(1e-12)
    return (mean_missing_error / target_rms).mean()
