"""P12: one acquisition-conditioned PromptMR+ for acc4 and acc8.

The reconstruction backbone and sensitivity estimator retain the proven P11
shape.  P12 differs in two controlled ways:

1. one set of weights reconstructs both accelerations; and
2. a descriptor computed only from the actual mask conditions the learned
   data consistency and, in the full variant, every PromptUNet cascade.

No acceleration expert, lesion label, bbox, image-file field, or supplied
GRAPPA reconstruction is used at inference. P12-full conditions learned DC and
every PromptUNet cascade with bounded, zero-initialized modulation.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from utils.model.p00s_acc8_promptmr_plus import P00SAcc8PromptMRPlus
from utils.model.p11m_acc8_sampling_aware_promptmr_plus import (
    P11MAcc8SamplingAwarePromptMRPlus,
    P11SoftReconstructionOutput,
)
from utils.model.promptmr_plus.math import complex_abs, complex_mul
from utils.model.promptmr_plus.mri_ops import rss, sens_expand, sens_reduce


class ExactAcquisitionDescriptor(nn.Module):
    """Encode the measured mask lattice and its complex point-spread peaks.

    The descriptor is deterministic.  Candidate lattice mismatch distinguishes
    acc4 from acc8 even inside the contiguous ACS block.  Complex PSF samples
    are taken at fractional alias displacements, so widths such as 322/372 do
    not get forced into the inaccurate ``round(width / acceleration)`` model.
    """

    output_dim = 37

    @staticmethod
    def _line_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim < 2:
            raise ValueError("mask must include batch and PE dimensions")
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        batch, width = mask.shape[0], mask.shape[-1]
        return mask.reshape(batch, -1, width).bool().any(dim=1)

    @staticmethod
    def _lattice(line_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, width = line_mask.shape
        index = torch.arange(width, device=line_mask.device)
        # The challenge ACS occupies about 8%; exclude a conservative 12%
        # while fitting the regular lattice.
        center = (width - 1) / 2.0
        outside_acs = (index.to(torch.float32) - center).abs() > 0.06 * width
        observed = line_mask.to(torch.float32)
        errors = []
        candidates = []
        for acceleration in (4, 8):
            for offset in range(acceleration):
                predicted = (index % acceleration == offset).to(observed.dtype)
                mismatch = (observed - predicted).abs()
                mismatch = mismatch[:, outside_acs].mean(dim=1)
                errors.append(mismatch)
                candidates.append((acceleration, offset))
        best = torch.stack(errors, dim=1).argmin(dim=1)
        accelerations = torch.tensor(
            [item[0] for item in candidates], device=line_mask.device
        )[best]
        offsets = torch.tensor(
            [item[1] for item in candidates], device=line_mask.device
        )[best]
        return accelerations, offsets

    @staticmethod
    def _acs_fraction(line_mask: torch.Tensor) -> torch.Tensor:
        batch, width = line_mask.shape
        index = torch.arange(width, device=line_mask.device).expand(batch, -1)
        center = width // 2
        zeros = ~line_mask
        left_zero = torch.where(
            zeros & (index < center), index, torch.full_like(index, -1)
        ).amax(dim=1)
        right_zero = torch.where(
            zeros & (index > center), index, torch.full_like(index, width)
        ).amin(dim=1)
        length = (right_zero - left_zero - 1).clamp(min=1, max=width)
        return length.to(torch.float32) / float(width)

    @staticmethod
    def _fractional_psf_peaks(
        line_mask: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> torch.Tensor:
        batch, width = line_mask.shape
        centered = line_mask.to(torch.float32)
        psf = torch.fft.ifft(
            torch.fft.ifftshift(centered, dim=-1).to(torch.complex64),
            dim=-1,
            norm="ortho",
        )
        psf = psf / psf[:, :1].abs().clamp_min(1e-6)
        peaks = []
        for alias_index in range(1, 8):
            position = alias_index * width / acceleration.to(torch.float32)
            lower_float = torch.floor(position)
            lower = lower_float.to(torch.long).remainder(width)
            upper = (lower + 1).remainder(width)
            fraction = (position - lower_float).to(psf.real.dtype)
            lower_value = psf.gather(1, lower[:, None])[:, 0]
            upper_value = psf.gather(1, upper[:, None])[:, 0]
            value = lower_value * (1.0 - fraction) + upper_value * fraction
            valid = (alias_index < acceleration).to(psf.real.dtype)
            value = value * valid
            peaks.extend((value.real, value.imag, value.abs(), valid))
        return torch.stack(peaks, dim=1)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        line_mask = self._line_mask(mask)
        acceleration, offset = self._lattice(line_mask)
        acceleration_f = acceleration.to(torch.float32)
        offset_f = offset.to(torch.float32)
        phase = 2.0 * math.pi * offset_f / acceleration_f
        width = float(line_mask.shape[-1])
        scalars = torch.stack(
            (
                (acceleration == 4).to(torch.float32),
                (acceleration == 8).to(torch.float32),
                acceleration_f / 8.0,
                offset_f / (acceleration_f - 1.0).clamp_min(1.0),
                torch.sin(phase),
                torch.cos(phase),
                torch.full_like(acceleration_f, width / 512.0),
                line_mask.to(torch.float32).mean(dim=1),
                self._acs_fraction(line_mask),
            ),
            dim=1,
        )
        peaks = self._fractional_psf_peaks(line_mask, acceleration)
        descriptor = torch.cat((scalars, peaks), dim=1)
        if descriptor.shape[1] != self.output_dim:
            raise RuntimeError(
                f"P12 acquisition descriptor has {descriptor.shape[1]} fields"
            )
        return descriptor


class AcquisitionConditioner(nn.Module):
    """Create cascade-specific bounded feature modulation and DC scaling."""

    def __init__(self, num_cascades: int, feature_conditioning: bool):
        super().__init__()
        hidden = 96
        self.feature_conditioning = bool(feature_conditioning)
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(ExactAcquisitionDescriptor.output_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.cascade_embedding = nn.Embedding(num_cascades, hidden)
        self.dc_head = nn.Linear(hidden, 1)
        if self.feature_conditioning:
            # FiLM pairs: 16, 32, 48, 64. Prompt additions: 24, 16, 8.
            self.feature_head = nn.Linear(
                hidden, 2 * (16 + 32 + 48 + 64) + 24 + 16 + 8
            )
            nn.init.zeros_(self.feature_head.weight)
            nn.init.zeros_(self.feature_head.bias)
        else:
            self.feature_head = None
        nn.init.zeros_(self.dc_head.weight)
        nn.init.zeros_(self.dc_head.bias)

    def encode_descriptor(self, descriptor: torch.Tensor) -> torch.Tensor:
        """Encode one observed mask once, then reuse it in all cascades."""
        return self.descriptor_encoder(descriptor)

    def condition_encoded(
        self,
        encoded_descriptor: torch.Tensor,
        cascade_index: int,
    ) -> Tuple[Optional[Tuple[torch.Tensor, ...]], torch.Tensor]:
        index = torch.full(
            (encoded_descriptor.shape[0],),
            cascade_index,
            device=encoded_descriptor.device,
            dtype=torch.long,
        )
        encoded = torch.nn.functional.gelu(
            encoded_descriptor + self.cascade_embedding(index)
        )
        dc_scale = 1.0 + 0.25 * torch.tanh(self.dc_head(encoded))
        dc_scale = dc_scale[:, :, None, None, None]
        if self.feature_head is None:
            return None, dc_scale

        values = self.feature_head(encoded)
        sizes = (16, 16, 32, 32, 48, 48, 64, 64, 24, 16, 8)
        pieces = values.split(sizes, dim=1)
        condition = tuple(
            0.10 * torch.tanh(piece[:, :, None, None])
            for piece in pieces
        )
        return condition, dc_scale

    def forward(
        self,
        descriptor: torch.Tensor,
        cascade_index: int,
    ) -> Tuple[Optional[Tuple[torch.Tensor, ...]], torch.Tensor]:
        return self.condition_encoded(
            self.encode_descriptor(descriptor), cascade_index
        )


class _P12UnifiedBase(P11MAcc8SamplingAwarePromptMRPlus):
    """Shared P11 execution with optional acquisition conditioning."""

    specialist_acceleration = None
    conditioning_mode = "none"

    def __init__(
        self,
        num_cascades: int = 12,
        compute_sens_per_coil: bool = True,
        **kwargs,
    ):
        super().__init__(
            num_cascades=num_cascades,
            compute_sens_per_coil=compute_sens_per_coil,
            **kwargs,
        )
        self.acquisition_descriptor = ExactAcquisitionDescriptor()
        self.acquisition_conditioner = None
        if self.conditioning_mode in {"dc", "full"}:
            self.acquisition_conditioner = AcquisitionConditioner(
                num_cascades,
                feature_conditioning=self.conditioning_mode == "full",
            )

    def load_warm_start_state_dict(
        self, source_state: Dict[str, torch.Tensor]
    ) -> Dict[str, int]:
        incompatible = self.load_state_dict(source_state, strict=False)
        allowed_prefixes = (
            "acquisition_conditioner.",
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "P12 warm-start mismatch: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        return {
            "transferred": len(source_state),
            "cloned": 0,
            "new_conditioner_parameters": len(incompatible.missing_keys),
        }

    def _condition(
        self,
        encoded_descriptor: torch.Tensor,
        cascade_index: int,
    ) -> Tuple[Optional[Tuple[torch.Tensor, ...]], torch.Tensor]:
        if self.acquisition_conditioner is None:
            scale = encoded_descriptor.new_ones(
                (encoded_descriptor.shape[0], 1, 1, 1, 1)
            )
            return None, scale
        return self.acquisition_conditioner.condition_encoded(
            encoded_descriptor, cascade_index
        )

    @staticmethod
    def _cascade_equation(
        cascade: nn.Module,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history_feat,
        physics_condition: Optional[Tuple[torch.Tensor, ...]],
        dc_scale: torch.Tensor,
    ):
        current_kspace = sens_expand(current_img, sens_maps, 1)
        measured_projection = sens_reduce(
            torch.where(mask, current_kspace, torch.zeros_like(current_kspace)),
            sens_maps,
            1,
        )
        if cascade.model.n_buffer > 0:
            buffer = torch.cat(
                [measured_projection]
                + [latent] * (cascade.model.n_buffer - 3)
                + [img_zf, measured_projection - img_zf],
                dim=1,
            )
        else:
            buffer = None
        soft_dc = (
            (measured_projection - img_zf)
            * cascade.dc_weight
            * dc_scale
        )
        model_term, latent, history_feat = cascade.model(
            current_img,
            history_feat,
            buffer,
            physics_condition=physics_condition,
        )
        return current_img - soft_dc - model_term, latent, history_feat

    def _run_cascade(
        self,
        cascade_index: int,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history_feat,
        encoded_descriptor: torch.Tensor,
    ):
        cascade = self.backbone.cascades[cascade_index]

        def step(image, zero_filled, state, current_mask, sensitivities, history, encoded):
            condition, dc_scale = self._condition(encoded, cascade_index)
            return self._cascade_equation(
                cascade,
                image,
                zero_filled,
                state,
                current_mask,
                sensitivities,
                history,
                condition,
                dc_scale,
            )

        checkpoint_indices = getattr(
            self.backbone, "checkpoint_cascade_indices", ()
        )
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and cascade_index in checkpoint_indices
        )
        if not use_checkpoint:
            return step(
                current_img,
                img_zf,
                latent,
                mask,
                sens_maps,
                history_feat,
                encoded_descriptor,
            )
        offload_indices = getattr(
            self.backbone, "checkpoint_cpu_offload_indices", ()
        )
        context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if cascade_index in offload_indices
            else nullcontext()
        )
        with context:
            return torch.utils.checkpoint.checkpoint(
                step,
                current_img,
                img_zf,
                latent,
                mask,
                sens_maps,
                history_feat,
                encoded_descriptor,
                use_reentrant=False,
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

        img_zf = sens_reduce(masked_kspace, sens_maps, 1)
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        history_feat = None
        # The mask is observed, discrete acquisition metadata.  It never needs
        # a gradient path and is encoded once per slice, not once per cascade.
        descriptor = (
            self.acquisition_descriptor(mask).detach()
            if self.acquisition_conditioner is not None
            else masked_kspace.new_zeros(
                (masked_kspace.shape[0], ExactAcquisitionDescriptor.output_dim)
            )
        )
        encoded_descriptor = (
            self.acquisition_conditioner.encode_descriptor(descriptor)
            if self.acquisition_conditioner is not None
            else descriptor
        )
        requested = {4, 8} if return_intermediates else set()
        intermediates = {}

        for cascade_index in range(self.num_cascades):
            img_pred, latent, history_feat = self._run_cascade(
                cascade_index,
                img_pred,
                img_zf,
                latent,
                mask,
                sens_maps,
                history_feat,
                encoded_descriptor,
            )
            one_based = cascade_index + 1
            if one_based in requested:
                intermediates[one_based] = rss(
                    complex_abs(complex_mul(img_pred, sens_maps)), dim=1
                )

        final = rss(complex_abs(complex_mul(img_pred, sens_maps)), dim=1)
        final = P00SAcc8PromptMRPlus._center_crop_or_pad(final)
        if not return_intermediates:
            return final
        return P11SoftReconstructionOutput(
            final=final,
            cascade4=P00SAcc8PromptMRPlus._center_crop_or_pad(intermediates[4]),
            cascade8=P00SAcc8PromptMRPlus._center_crop_or_pad(intermediates[8]),
        )


class P12UnifiedAcquisitionPromptMRPlus(_P12UnifiedBase):
    """Full P12: descriptor-conditioned DC and all-cascade feature modulation."""

    conditioning_mode = "full"
