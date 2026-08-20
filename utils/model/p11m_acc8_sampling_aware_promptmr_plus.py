"""P11-soft: P07M inference architecture with training-only band supervision.

No reconstruction state is filtered or projected. During training only,
cascade 4 and cascade 8 RSS reconstructions can be returned for low-frequency
and low+mid-frequency auxiliary losses. Normal inference is exactly P07M.
"""

from dataclasses import dataclass
import os
from typing import Dict

import torch

from utils.model.p00s_acc8_promptmr_plus import P00SAcc8PromptMRPlus
from utils.model.p07m_acc8_multimask_promptmr_plus import (
    P07MAcc8MultiMaskPromptMRPlus,
)


@dataclass
class P11SoftReconstructionOutput:
    final: torch.Tensor
    cascade4: torch.Tensor
    cascade8: torch.Tensor


class P11MAcc8SamplingAwarePromptMRPlus(P07MAcc8MultiMaskPromptMRPlus):
    """Exact P07M model with optional training-only intermediate outputs."""

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
        checkpoint_count = int(
            os.environ.get("P11_CHECKPOINT_CASCADES", "4")
        )
        if checkpoint_count < 0 or checkpoint_count > num_cascades:
            raise ValueError(
                "P11_CHECKPOINT_CASCADES must be between 0 and "
                f"{num_cascades}"
            )
        # Checkpoint only the last N cascades. Internal activations are similar
        # in size across cascades; this makes memory/time scale predictably and
        # keeps the cascade-4/8 auxiliary taps outside the default N=4 region.
        self.backbone.checkpoint_cascades = checkpoint_count > 0
        self.backbone.checkpoint_cascade_indices = frozenset(
            range(num_cascades - checkpoint_count, num_cascades)
        )
        self.checkpoint_cascade_count = checkpoint_count
        cpu_offload_count = int(
            os.environ.get("P11_CPU_OFFLOAD_CASCADES", "4")
        )
        if cpu_offload_count < 0 or cpu_offload_count > checkpoint_count:
            raise ValueError(
                "P11_CPU_OFFLOAD_CASCADES must be between 0 and the "
                f"checkpoint count ({checkpoint_count})"
            )
        checkpoint_indices = sorted(
            self.backbone.checkpoint_cascade_indices
        )
        # Offload the earliest checkpoint boundaries. They otherwise remain
        # resident throughout the late-cascade backward recomputations where
        # the 8 GB card reaches its peak.
        self.backbone.checkpoint_cpu_offload_indices = frozenset(
            checkpoint_indices[:cpu_offload_count]
        )
        self.cpu_offload_cascade_count = cpu_offload_count
        block_checkpoint_value = os.environ.get(
            "P11_CHECKPOINT_UNET_BLOCKS", "0"
        )
        if block_checkpoint_value not in {"0", "1"}:
            raise ValueError(
                "P11_CHECKPOINT_UNET_BLOCKS must be 0 or 1"
            )
        self.block_checkpointing = block_checkpoint_value == "1"
        self.backbone.sens_net.norm_unet.unet.checkpoint_blocks = (
            self.block_checkpointing
        )
        for cascade in self.backbone.cascades:
            cascade.model.unet.checkpoint_blocks = self.block_checkpointing

    def load_warm_start_state_dict(
        self, source_state: Dict[str, torch.Tensor]
    ) -> Dict[str, int]:
        """Optionally load an architecture-identical P07M checkpoint."""
        self.load_state_dict(source_state, strict=True)
        return {
            "transferred": len(source_state),
            "cloned": 0,
            "new_conditioner_parameters": 0,
        }

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
        num_low_frequencies = torch.full(
            (masked_kspace.shape[0],),
            round(0.08 * masked_kspace.shape[-2]),
            dtype=torch.long,
            device=masked_kspace.device,
        )
        requested = (4, 8) if return_intermediates else ()
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
            compute_sens_per_coil=(
                self.compute_sens_per_coil if self.training else True
            ),
            intermediate_cascades=requested,
        )
        final = P00SAcc8PromptMRPlus._center_crop_or_pad(
            outputs["img_pred"]
        )
        if not return_intermediates:
            return final
        intermediates = outputs["intermediate_preds"]
        if 4 not in intermediates or 8 not in intermediates:
            raise RuntimeError("P11-soft cascade 4/8 outputs were not produced")
        return P11SoftReconstructionOutput(
            final=final,
            cascade4=P00SAcc8PromptMRPlus._center_crop_or_pad(
                intermediates[4]
            ),
            cascade8=P00SAcc8PromptMRPlus._center_crop_or_pad(
                intermediates[8]
            ),
        )
