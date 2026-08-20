"""Stable unified acc4/acc8 PromptMR+ reconstruction.

This model keeps the proven P07/P11 backbone unchanged in cascades 1--8.  It
uses the actual sampling mask only to route bounded data-consistency and prior
strengths in cascades 9--12.  The routing design follows the useful component
of G0/G3/G4/G5 while deliberately excluding the failed forced whitening,
sensitivity smoothing, direct SENSE inversion, SPIRiT proposal, dual-domain
tail, and morphology router experiments.

Every new output layer is zero initialized, so at initialization this class is
exactly the underlying P11 inference path.  Unlike the previous P12-full, no
multiplicative FiLM is applied inside the PromptUNet encoder.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from utils.model.p00s_acc8_promptmr_plus import P00SAcc8PromptMRPlus
from utils.model.p11m_acc8_sampling_aware_promptmr_plus import (
    P11MAcc8SamplingAwarePromptMRPlus,
    P11SoftReconstructionOutput,
)
from utils.model.p12_unified_acquisition_promptmr_plus import (
    ExactAcquisitionDescriptor,
)
from utils.model.promptmr_plus.math import complex_abs, complex_mul
from utils.model.promptmr_plus.mri_ops import rss, sens_expand, sens_reduce


class StableAcquisitionEncoder(nn.Module):
    """Encode exact mask metadata once without changing the measured data."""

    output_channels = 8

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(ExactAcquisitionDescriptor.output_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.output_channels),
            nn.LayerNorm(self.output_channels),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.network(descriptor)[:, :, None, None]


class StableDynamicRouter(nn.Module):
    """Small zero-initialized router for late DC and learned-prior gains."""

    def __init__(self, static_channels: int = 8, hidden: int = 12):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(static_channels + 2, hidden, 3, padding=1),
            nn.PReLU(hidden),
        )
        self.output = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        routed = self.output(self.trunk(condition))
        # G5 showed that DC routing is the useful path.  The narrower bounds
        # prevent a correlated update from destabilizing twelve recurrences.
        dc_gain = 1.0 + 0.10 * torch.tanh(routed[:, 0:1])
        prior_gain = 1.0 + 0.05 * torch.tanh(routed[:, 1:2])
        return dc_gain, prior_gain


class StableFinalSoftDC(nn.Module):
    """Zero-start, bounded final measured-residual correction."""

    def __init__(self, condition_channels: int = 10, hidden: int = 8):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv2d(condition_channels, hidden, 3, padding=1),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.map[-1].weight)
        nn.init.zeros_(self.map[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        image: torch.Tensor,
        measured_residual: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        dc_map = 0.75 + 0.50 * torch.sigmoid(self.map(condition))
        dc_map = F.interpolate(
            dc_map,
            size=image.shape[-3:-1],
            mode="bilinear",
            align_corners=False,
        )
        strength = 0.10 * torch.tanh(self.alpha)
        return image - strength * dc_map.unsqueeze(-1) * measured_residual


class P12StableUnifiedPromptMRPlus(P11MAcc8SamplingAwarePromptMRPlus):
    """One stable P11-width model shared by balanced acc4 and acc8 views."""

    specialist_acceleration = None
    requires_scan_calibration = False

    def __init__(self, num_cascades: int = 12, **kwargs):
        super().__init__(num_cascades=num_cascades, **kwargs)
        if num_cascades != 12:
            raise ValueError("P12-stable requires exactly 12 cascades")
        self.acquisition_descriptor = ExactAcquisitionDescriptor()
        self.acquisition_encoder = StableAcquisitionEncoder()
        self.late_routers = nn.ModuleList(
            StableDynamicRouter(
                static_channels=StableAcquisitionEncoder.output_channels
            )
            for _ in range(4)
        )
        self.final_soft_dc = StableFinalSoftDC(
            StableAcquisitionEncoder.output_channels + 2
        )

    def load_warm_start_state_dict(
        self, source_state: Dict[str, torch.Tensor]
    ) -> Dict[str, int]:
        incompatible = self.load_state_dict(source_state, strict=False)
        allowed = (
            "acquisition_encoder.",
            "late_routers.",
            "final_soft_dc.",
        )
        invalid = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed)
        ]
        if invalid or incompatible.unexpected_keys:
            raise RuntimeError(
                "P12-stable warm-start mismatch: "
                f"missing={invalid}, unexpected={incompatible.unexpected_keys}"
            )
        return {
            "transferred": len(source_state),
            "new": len(incompatible.missing_keys),
        }

    @staticmethod
    def _measured_projection(
        image: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        predicted_kspace = sens_expand(image, sens_maps, 1)
        measured = torch.where(
            mask,
            predicted_kspace,
            torch.zeros(1, device=image.device, dtype=image.dtype),
        )
        return sens_reduce(measured, sens_maps, 1)

    @staticmethod
    def _relative_magnitude(
        value: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        magnitude = complex_abs(value)
        reference_magnitude = complex_abs(reference)
        scale = reference_magnitude.square().mean(
            dim=(-2, -1), keepdim=True
        ).sqrt().clamp_min(1e-8)
        return torch.tanh(magnitude / scale)

    def _dynamic_condition(
        self,
        static_condition: torch.Tensor,
        measured_residual: torch.Tensor,
        previous_update: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        # Diagnostics are observations, not a trainable way for the backbone
        # to make its own routing problem artificially easy.
        with torch.no_grad():
            residual = self._relative_magnitude(
                measured_residual.detach(), reference.detach()
            )
            disagreement = self._relative_magnitude(
                previous_update.detach(), reference.detach()
            )
            dynamic = torch.cat((residual, disagreement), dim=1)
            dynamic = F.avg_pool2d(dynamic, kernel_size=4, stride=4)
        static = static_condition.expand(
            -1, -1, dynamic.shape[-2], dynamic.shape[-1]
        )
        return torch.cat((static, dynamic), dim=1)

    def _late_cascade(
        self,
        index: int,
        image: torch.Tensor,
        image_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history,
        previous_update: torch.Tensor,
        static_condition: torch.Tensor,
    ):
        cascade = self.backbone.cascades[index]
        measured_projection = self._measured_projection(
            image, mask, sens_maps
        )
        measured_residual = measured_projection - image_zf
        condition = self._dynamic_condition(
            static_condition,
            measured_residual,
            previous_update,
            image_zf,
        )
        dc_gain, prior_gain = self.late_routers[index - 8](condition)
        target_size = image.shape[-3:-1]
        dc_gain = F.interpolate(
            dc_gain, size=target_size, mode="bilinear", align_corners=False
        ).unsqueeze(-1)
        prior_gain = F.interpolate(
            prior_gain, size=target_size, mode="bilinear", align_corners=False
        ).unsqueeze(-1)

        if cascade.model.n_buffer > 0:
            buffer = torch.cat(
                [measured_projection]
                + [latent] * (cascade.model.n_buffer - 3)
                + [image_zf, measured_residual],
                dim=1,
            )
        else:
            buffer = None

        # The original scalar is trainable but cannot cross into a recurrently
        # explosive negative or very large DC step in P12-stable.
        effective_dc_weight = cascade.dc_weight.clamp(0.0, 1.5)
        soft_dc = measured_residual * effective_dc_weight * dc_gain
        model_term, latent, history = cascade.model(image, history, buffer)
        updated = image - soft_dc - prior_gain * model_term
        return updated, latent, history, updated - image, condition

    def _run_cascade(
        self,
        index: int,
        image: torch.Tensor,
        image_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history,
        previous_update: torch.Tensor,
        static_condition: torch.Tensor,
    ):
        cascade = self.backbone.cascades[index]

        def step(
            current_image,
            zero_filled,
            current_latent,
            current_mask,
            sensitivities,
            current_history,
            last_update,
            static,
        ):
            if index < 8:
                updated, new_latent, new_history = cascade(
                    current_image,
                    zero_filled,
                    current_latent,
                    current_mask,
                    sensitivities,
                    current_history,
                )
                empty_condition = static.new_zeros(
                    (static.shape[0], 10, 1, 1)
                )
                return (
                    updated,
                    new_latent,
                    new_history,
                    updated - current_image,
                    empty_condition,
                )
            return self._late_cascade(
                index,
                current_image,
                zero_filled,
                current_latent,
                current_mask,
                sensitivities,
                current_history,
                last_update,
                static,
            )

        checkpoint_indices = getattr(
            self.backbone, "checkpoint_cascade_indices", ()
        )
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and index in checkpoint_indices
        )
        arguments = (
            image,
            image_zf,
            latent,
            mask,
            sens_maps,
            history,
            previous_update,
            static_condition,
        )
        if not use_checkpoint:
            return step(*arguments)
        offload_indices = getattr(
            self.backbone, "checkpoint_cpu_offload_indices", ()
        )
        context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if index in offload_indices
            else nullcontext()
        )
        with context:
            return torch.utils.checkpoint.checkpoint(
                step, *arguments, use_reentrant=False
            )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        return_intermediates: bool = False,
    ):
        if masked_kspace.ndim != 5 or masked_kspace.shape[-1] != 2:
            raise ValueError(
                "masked_kspace must have shape [B, coils, H, W, 2]"
            )
        mask = mask.bool()
        num_low_frequencies = torch.full(
            (masked_kspace.shape[0],),
            round(0.08 * masked_kspace.shape[-2]),
            dtype=torch.long,
            device=masked_kspace.device,
        )
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpoint:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.backbone.sens_net,
                masked_kspace,
                mask,
                num_low_frequencies,
                ("cartesian",),
                self.compute_sens_per_coil,
                use_reentrant=False,
            )
        else:
            sens_maps = self.backbone.sens_net(
                masked_kspace,
                mask,
                num_low_frequencies,
                ("cartesian",),
                self.compute_sens_per_coil if self.training else True,
            )

        image_zf = sens_reduce(masked_kspace, sens_maps, 1)
        image = image_zf.clone()
        latent = image_zf.clone()
        history = None
        previous_update = torch.zeros_like(image_zf)
        descriptor = self.acquisition_descriptor(mask).detach()
        static_condition = self.acquisition_encoder(descriptor)
        requested = {4, 8} if return_intermediates else set()
        intermediates = {}
        last_condition = None

        for index in range(self.num_cascades):
            image, latent, history, previous_update, condition = (
                self._run_cascade(
                    index,
                    image,
                    image_zf,
                    latent,
                    mask,
                    sens_maps,
                    history,
                    previous_update,
                    static_condition,
                )
            )
            if index >= 8:
                last_condition = condition
            one_based = index + 1
            if one_based in requested:
                intermediates[one_based] = rss(
                    complex_abs(complex_mul(image, sens_maps)), dim=1
                )

        measured_projection = self._measured_projection(
            image, mask, sens_maps
        )
        measured_residual = measured_projection - image_zf
        if last_condition is None:
            last_condition = self._dynamic_condition(
                static_condition,
                measured_residual,
                previous_update,
                image_zf,
            )
        image = self.final_soft_dc(
            image, measured_residual, last_condition
        )

        final = rss(complex_abs(complex_mul(image, sens_maps)), dim=1)
        final = P00SAcc8PromptMRPlus._center_crop_or_pad(final)
        if not return_intermediates:
            return final
        return P11SoftReconstructionOutput(
            final=final,
            cascade4=P00SAcc8PromptMRPlus._center_crop_or_pad(
                intermediates[4]
            ),
            cascade8=P00SAcc8PromptMRPlus._center_crop_or_pad(
                intermediates[8]
            ),
        )
