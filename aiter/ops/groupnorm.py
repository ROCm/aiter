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


def groupnorm_run(
    input: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> Tensor:
    """Group Normalization. Allocates output and scratch workspace, then calls
    the HIP kernel. Returns the normalized output tensor."""
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    y = torch.empty_like(input)

    # Workspace upper bound that matches the C-side grid cap (see groupnorm.cu):
    # num_acc_slots = gridx * outer, with gridx <= ceil(4096 / outer);
    # the kernel needs 2 * num_acc_slots float32 slots.
    outer = input.shape[0] * num_groups
    ws_slots = 2 * ((4096 + outer - 1) // outer * outer)
    workspace = torch.empty(ws_slots, dtype=torch.float32, device=input.device)

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
