"""P00S: sMaRtIfy-style lean PromptMR+ acc8 specialist.

This controlled baseline keeps the 2025 team's memory-efficient reconstruction
settings while using this repository's challenge-aligned acc8 data and loss.
Unlike P00, P00S reconstructs only the measured central slice; adjacent-slice
fusion is deliberately reserved for a later ablation.
"""

import torch
from torch import nn
from torch.nn import functional as F

from utils.model.promptmr_plus import PromptMR


class P00SAcc8PromptMRPlus(nn.Module):
    """Eight-cascade, single-slice PromptMR+ trained only on acc8 views."""

    requires_adjacent_slices = False
    specialist_acceleration = 8
    output_shape = (384, 384)

    def __init__(
        self,
        num_cascades: int = 8,
        sens_chans: int = 8,
        chans: int = 8,
        gradient_checkpointing: bool = True,
        compute_sens_per_coil: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if num_cascades < 2:
            raise ValueError("P00S requires at least two PromptMR+ cascades")
        if chans != 8 or sens_chans != 8:
            raise ValueError(
                "P00S reproduces the controlled sMaRtIfy widths; "
                "use --chans 8 --sens_chans 8"
            )

        self.num_cascades = int(num_cascades)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.compute_sens_per_coil = bool(compute_sens_per_coil)

        self.backbone = PromptMR(
            num_cascades=num_cascades,
            num_adj_slices=1,
            n_feat0=8,
            feature_dim=[24, 32, 40],
            prompt_dim=[8, 16, 24],
            sens_n_feat0=8,
            sens_feature_dim=[12, 16, 20],
            sens_prompt_dim=[4, 8, 12],
            len_prompt=[3, 3, 3],
            prompt_size=[16, 8, 4],
            n_enc_cab=[2, 3, 3],
            n_dec_cab=[2, 2, 3],
            n_skip_cab=[1, 1, 1],
            n_bottleneck_cab=3,
            no_use_ca=False,
            mask_center=True,
            learnable_prompt=False,
            adaptive_input=True,
            n_buffer=4,
            n_history=3,
            use_sens_adj=False,
            reduction=2,
            sens_reduction=2,
            checkpoint_cascades=False,
        )

    @classmethod
    def _center_crop_or_pad(cls, image: torch.Tensor) -> torch.Tensor:
        target_height, target_width = cls.output_shape
        height, width = image.shape[-2:]
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)
        if pad_height or pad_width:
            image = F.pad(
                image,
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                ),
            )
            height, width = image.shape[-2:]
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return image[
            ...,
            top:top + target_height,
            left:left + target_width,
        ]

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if masked_kspace.ndim != 5 or masked_kspace.shape[-1] != 2:
            raise ValueError(
                "masked_kspace must have shape [B, coils, H, W, 2]"
            )
        num_low_frequencies = torch.full(
            (masked_kspace.shape[0],),
            round(0.08 * masked_kspace.shape[-2]),
            dtype=torch.long,
            device=masked_kspace.device,
        )
        outputs = self.backbone(
            masked_kspace,
            mask.bool(),
            num_low_frequencies=num_low_frequencies,
            mask_type=("cartesian",),
            use_checkpoint=(
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ),
            compute_sens_per_coil=self.compute_sens_per_coil,
        )
        return self._center_crop_or_pad(outputs["img_pred"])
