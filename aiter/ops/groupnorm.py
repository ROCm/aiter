from ..jit.core import compile_ops
import torch
from typing import Optional

from torch import Tensor


# JIT-compiled binding to the C++ kernel. Output `y` and scratch `workspace`
# are allocated by the Python wrapper below and passed in; the kernel writes
# into them in-place and returns None.
@compile_ops("module_groupnorm", fc_name="_groupnorm_run", develop=True)
def _groupnorm_run(
    y: Tensor,
    workspace: Tensor,
    input: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> None: ...


# Grow-only scratch workspace cache, keyed by device. The C++ kernel must not
# allocate; it reuses whatever buffer we hand it (empty is fine — the kernel
# overwrites every slot it reads). Reusing across calls avoids a per-call
# allocation that otherwise dominates small-shape latency.
#
# Resident memory is bounded, not unbounded: ws_slots = 2 * gridx * outer, and
# gridx is capped by the 4096 grid limit in groupnorm.cu, so the cached buffer
# never exceeds ~8192 float32 (~32 KB) per device regardless of input size.
# This matches main's C++ mean_accumulator_, which was likewise grow-only and
# resident; no extra shrink/free logic is warranted.
_workspace_cache: dict[torch.device, Tensor] = {}


def _groupnorm_ws_slots(input: Tensor, num_groups: int) -> int:
    # Exact float32-slot count the kernel needs: 2 * gridx * outer_size.
    # gridx derivation mirrors launchGroupNormKernel in groupnorm.cu; keep in sync.
    THREADS_PER_BLOCK = 1024
    STEPS_PER_THREAD = 8
    outer = input.shape[0] * num_groups
    inner = input.numel() // outer
    gridx = (inner + STEPS_PER_THREAD * THREADS_PER_BLOCK - 1) // (
        STEPS_PER_THREAD * THREADS_PER_BLOCK
    )
    if inner % 4 == 0 and gridx >= 16:
        gridx = max(1, gridx // 4)
    gridx = min((4096 + outer - 1) // outer, gridx)
    return 2 * gridx * outer


def _groupnorm_workspace(ws_slots: int, device: torch.device) -> Tensor:
    ws = _workspace_cache.get(device)
    if ws is None or ws.numel() < ws_slots:
        ws = torch.empty(ws_slots, dtype=torch.float32, device=device)
        _workspace_cache[device] = ws
    return ws


def groupnorm_run(
    input: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> Tensor:
    """Group Normalization. Allocates output and reuses a cached scratch
    workspace, then calls the HIP kernel. Returns the normalized output."""
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    y = torch.empty_like(input)

    ws_slots = _groupnorm_ws_slots(input, num_groups)
    workspace = _groupnorm_workspace(ws_slots, input.device)

    _groupnorm_run(y, workspace, input, num_groups, weight, bias, eps)
    return y


class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = torch.nn.Parameter(
                torch.ones(num_channels, device=device, dtype=dtype)
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(num_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, use_torch: bool = False) -> torch.Tensor:
        if use_torch or not self.affine:
            # fallback to PyTorch for non-affine or debug mode
            return torch.nn.functional.group_norm(
                x,
                self.num_groups,
                weight=self.weight if self.affine else None,
                bias=self.bias if self.affine else None,
                eps=self.eps,
            )
        else:
            return groupnorm_run(x, self.num_groups, self.weight, self.bias, self.eps)
