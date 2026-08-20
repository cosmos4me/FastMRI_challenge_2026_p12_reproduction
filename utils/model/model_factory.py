"""P12-only model factory for the reproducibility package."""

from utils.model.p12_stable_unified_promptmr_plus import P12StableUnifiedPromptMRPlus

MODEL_TYPE = "p12_stable_unified_promptmr_plus"

def build_model(model_type: str, num_cascades: int, chans: int, sens_chans: int):
    if model_type != MODEL_TYPE:
        raise ValueError(f"This reproduction package supports only {MODEL_TYPE}, got {model_type}")
    return P12StableUnifiedPromptMRPlus(
        num_cascades=num_cascades, chans=chans, sens_chans=sens_chans
    )

def checkpoint_model_type(checkpoint) -> str:
    checkpoint_args = checkpoint.get("args")
    model_type = getattr(checkpoint_args, "model_type", MODEL_TYPE)
    if model_type != MODEL_TYPE:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")
    return model_type
