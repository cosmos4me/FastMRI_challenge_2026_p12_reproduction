"""P00M: 3090-trained medium PromptMR+ acc8 baseline.

Capacity is spent on central-slice reconstruction depth and multiscale feature
width. Sensitivity, prompt size, and short history remain lean so the trained
model can still be evaluated on the challenge GTX 1080.
"""

import torch
from torch import nn

from utils.model.p00s_acc8_promptmr_plus import P00SAcc8PromptMRPlus
from utils.model.promptmr_plus import PromptMR


class P00MAcc8PromptMRPlus(nn.Module):
    """Ten-cascade medium PromptMR+ trained exclusively on acc8 views."""

    requires_adjacent_slices = False
    specialist_acceleration = 8
    output_shape = (384, 384)

    def __init__(
        self,
        num_cascades: int = 10,
        sens_chans: int = 8,
        chans: int = 16,
        gradient_checkpointing: bool = True,
        compute_sens_per_coil: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if num_cascades < 2:
            raise ValueError("P00M requires at least two PromptMR+ cascades")
        if chans != 16 or sens_chans != 8:
            raise ValueError(
                "P00M baseline requires --chans 16 --sens_chans 8"
            )

        self.num_cascades = int(num_cascades)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.compute_sens_per_coil = bool(compute_sens_per_coil)

        self.backbone = PromptMR(
            num_cascades=num_cascades,
            num_adj_slices=1,
            n_feat0=16,
            feature_dim=[32, 48, 64],
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
            # Training batches all coils to use the 3090 efficiently. During
            # evaluation, automatically switch to the sMa-style per-coil path
            # so the same checkpoint remains safe on an 8 GB GTX 1080.
            compute_sens_per_coil=(
                self.compute_sens_per_coil if self.training else True
            ),
        )
        return P00SAcc8PromptMRPlus._center_crop_or_pad(
            outputs["img_pred"]
        )
