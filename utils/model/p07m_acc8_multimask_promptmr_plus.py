"""P07M: scratch-trained, twelve-cascade, multi-mask acc8 PromptMR+.

The architecture deliberately stays P00M-compatible. P07M changes capacity
from ten to twelve cascades while the data pipeline cycles through all eight
official equispaced acc8 offsets for every fully sampled training volume.
"""

from utils.model.p00m_acc8_promptmr_plus import P00MAcc8PromptMRPlus


class P07MAcc8MultiMaskPromptMRPlus(P00MAcc8PromptMRPlus):
    """P00M-width acc8 specialist with exactly twelve PromptMR+ cascades."""

    specialist_acceleration = 8

    def __init__(self, num_cascades=12, **kwargs):
        if num_cascades != 12:
            raise ValueError("P07M requires --cascade 12")
        super().__init__(num_cascades=num_cascades, **kwargs)
