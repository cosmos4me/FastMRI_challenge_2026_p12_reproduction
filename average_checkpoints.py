"""Average compatible VarNet checkpoints for a zero-latency ensemble."""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError("At least one state dict is required")
    keys = list(state_dicts[0])
    if any(list(state_dict) != keys for state_dict in state_dicts[1:]):
        raise ValueError("Checkpoint model state dicts have different keys")

    averaged = OrderedDict()
    for key in keys:
        tensors = [state_dict[key] for state_dict in state_dicts]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"Shape mismatch for parameter {key}")
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            accumulator = tensors[0].to(dtype=torch.float64)
            for tensor in tensors[1:]:
                accumulator = accumulator + tensor.to(dtype=torch.float64)
            averaged[key] = (accumulator / len(tensors)).to(dtype=tensors[0].dtype)
        else:
            averaged[key] = tensors[-1].clone()
    return averaged


def main():
    args = parse_args()
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoints
    ]
    output = dict(checkpoints[-1])
    output["model"] = average_state_dicts(
        [checkpoint["model"] for checkpoint in checkpoints]
    )
    output["averaged_from"] = [str(path) for path in args.checkpoints]
    output.pop("optimizer", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved {len(checkpoints)}-checkpoint average to {args.output}")


if __name__ == "__main__":
    main()
