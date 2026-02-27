from ..jit.core import compile_ops
import torch
from typing import Optional

from torch import Tensor


def gen_groupnorm_fake_tensors(
    input: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> Tensor:
    return torch.empty_like(input)


@compile_ops(
    "module_groupnorm", fc_name="_groupnorm_run", gen_fake=gen_groupnorm_fake_tensors
)
def _groupnorm_run(
    input: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> Tensor: ...


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
            return _groupnorm_run(x, self.num_groups, self.weight, self.bias, self.eps)
