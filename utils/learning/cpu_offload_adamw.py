"""Single-GPU AdamW with CPU-resident master parameters and moments."""

import math

import torch


class CPUOffloadAdamW(torch.optim.Optimizer):
    """Keep AdamW persistent state off GPU without a DeepSpeed dependency.

    The CUDA model and gradients remain unchanged. At each optimizer update,
    gradients are copied to CPU, the FP32 master parameters and Adam moments
    are updated on CPU, and the new parameters are copied back to CUDA.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    ):
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if eps < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        super().__init__(
            params,
            dict(
                lr=float(lr),
                betas=tuple(betas),
                eps=float(eps),
                weight_decay=float(weight_decay),
            ),
        )

    @staticmethod
    def _cpu_fp32(tensor):
        return tensor.detach().to(
            device="cpu", dtype=torch.float32, copy=True
        )

    def _initialize_state(self, parameter):
        state = self.state[parameter]
        if state:
            return state
        master = self._cpu_fp32(parameter)
        state["step"] = 0
        state["master_param"] = master
        state["exp_avg"] = torch.zeros_like(master)
        state["exp_avg_sq"] = torch.zeros_like(master)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError(
                        "CPUOffloadAdamW does not support sparse gradients"
                    )

                state = self._initialize_state(parameter)
                state["step"] += 1
                step = state["step"]
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_cpu = self._cpu_fp32(gradient)

                if weight_decay:
                    master.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad_cpu, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_cpu, grad_cpu, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denominator = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(eps)
                master.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-(lr / bias_correction1),
                )
                parameter.copy_(master, non_blocking=False)

        return loss

    def load_state_dict(self, state_dict):
        """Load optimizer tensors directly to CPU, avoiding a CUDA peak."""
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("optimizer parameter-group count mismatch")

        self.state.clear()
        for current_group, saved_group in zip(
            self.param_groups, saved_groups
        ):
            saved_parameters = saved_group["params"]
            current_parameters = current_group["params"]
            if len(saved_parameters) != len(current_parameters):
                raise ValueError("optimizer parameter count mismatch")
            for key, value in saved_group.items():
                if key != "params":
                    current_group[key] = value
            for saved_id, parameter in zip(
                saved_parameters, current_parameters
            ):
                saved_state = state_dict["state"].get(saved_id, {})
                if not saved_state:
                    continue
                restored = {}
                for key, value in saved_state.items():
                    if key == "step":
                        restored[key] = int(
                            value.item()
                            if torch.is_tensor(value)
                            else value
                        )
                    elif torch.is_tensor(value):
                        restored[key] = self._cpu_fp32(value)
                    else:
                        restored[key] = value
                restored.setdefault(
                    "master_param", self._cpu_fp32(parameter)
                )
                restored.setdefault(
                    "exp_avg", torch.zeros_like(restored["master_param"])
                )
                restored.setdefault(
                    "exp_avg_sq",
                    torch.zeros_like(restored["master_param"]),
                )
                self.state[parameter] = restored
