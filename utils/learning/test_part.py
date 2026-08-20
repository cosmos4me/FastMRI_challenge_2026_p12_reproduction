import h5py
import math
import os
import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.transforms import to_tensor
from utils.model.model_factory import build_model, checkpoint_model_type

# ---------------------------------------------------------------------------
# Team-editable reconstruction contract.
# recon_eval.py (the fixed timing harness) only calls the three functions
# below. This branch feeds `kspace` + `mask` (k-space domain) to a VarNet; a
# U-Net branch reimplements the same three functions for the image domain.
# ---------------------------------------------------------------------------
INPUT_KIND = "kspace"      # harness delivers the kspace H5 to prep_volume


def load_model(args, device):
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    saved_args = checkpoint.get('args')
    model = build_model(
        model_type=checkpoint_model_type(checkpoint),
        num_cascades=getattr(saved_args, 'cascade', args.cascade),
        chans=getattr(saved_args, 'chans', args.chans),
        sens_chans=getattr(saved_args, 'sens_chans', args.sens_chans),
    ).to(device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def prep_volume(image_path, kspace_path, device):
    """Load one volume's k-space and mask onto the host. Untimed: no model compute here."""
    with h5py.File(kspace_path, 'r') as hf:
        kspace = hf['kspace'][:]
        mask = np.array(hf['mask'])
    return {"kspace": kspace, "mask": mask, "device": device, "num_slices": kspace.shape[0]}


def _neg_axis_with_phase(kspace, dim):
    """Exact centered-grid reflection in raw k-space, with its phase ramp."""
    size = kspace.shape[dim]
    reflected = torch.roll(torch.flip(kspace, dims=(dim,)), 1, dims=(dim,))
    coordinate = torch.arange(size, device=kspace.device, dtype=kspace.dtype)
    coordinate = coordinate - size // 2
    phase = 2.0 * math.pi * coordinate / size
    shape = [1] * (kspace.ndim - 1)
    shape[dim] = size
    cosine = torch.cos(phase).reshape(shape)
    sine = torch.sin(phase).reshape(shape)
    real, imag = reflected[..., 0], reflected[..., 1]
    return torch.stack((real * cosine - imag * sine,
                        real * sine + imag * cosine), dim=-1)


def _reflect_pe_mask(mask):
    return torch.roll(torch.flip(mask, dims=(3,)), 1, dims=(3,))


def recon_slice(model, ctx, s):
    """Timed P12 reconstruction with the selected physical W-axis TTA."""
    if getattr(model, "requires_adjacent_slices", False):
        raise RuntimeError("This P12 reproduction package is single-slice only")
    device = ctx["device"]
    sampled_mask = ctx["mask"]
    raw_kspace = to_tensor(ctx["kspace"][s] * sampled_mask)
    raw_kspace = torch.stack((raw_kspace.real, raw_kspace.imag), dim=-1)
    raw_kspace = raw_kspace.unsqueeze(0).to(device=device)
    mask_t = torch.from_numpy(
        sampled_mask.reshape(1, 1, raw_kspace.shape[-2], 1).astype(np.float32)
    ).byte().unsqueeze(0).to(device=device)

    # Submission-selected values, determined before final harness execution.
    identity_weight = float(os.environ.get("P12_TTA_IDENTITY_WEIGHT", "0.65"))
    reflected_weight = 1.0 - identity_weight
    output_scale = float(os.environ.get("P12_OUTPUT_SCALE", "1.0025"))
    if not 0.0 <= identity_weight <= 1.0 or output_scale <= 0.0:
        raise ValueError("Invalid P12 TTA weight or output scale")

    reflected_kspace = _neg_axis_with_phase(raw_kspace, dim=3)
    reflected_mask = _reflect_pe_mask(mask_t)
    batched_tta = os.environ.get("P12_TTA_BATCHED", "1") != "0"
    if batched_tta:
        paired_kspace = torch.cat((raw_kspace, reflected_kspace), dim=0)
        paired_mask = torch.cat((mask_t, reflected_mask), dim=0)
        paired_output = model(paired_kspace, paired_mask)
        original, reflected = paired_output[0], paired_output[1]
    else:
        original = model(raw_kspace, mask_t)[0]
        reflected = model(reflected_kspace, reflected_mask)[0]
    reflected = torch.flip(reflected, dims=(1,))
    return output_scale * (identity_weight * original + reflected_weight * reflected)


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for data in data_loader:
            mask, kspace, _, _, _, _, fnames, slices = data[:8]
            h16_payload = data[8] if len(data) > 8 else None
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            if h16_payload is None:
                output = model(kspace, mask)
            else:
                adjacent_kspace = h16_payload["adjacent_kspace"].cuda(
                    non_blocking=True
                )
                output = model(
                    kspace,
                    mask,
                    adjacent_kspace=adjacent_kspace,
                )

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    model = load_model(args, device)

    forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
