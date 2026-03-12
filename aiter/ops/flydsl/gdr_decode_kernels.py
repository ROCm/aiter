import functools
from typing import Dict, Optional
import torch


def flydsl_gdr_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    from .kernels.gdr_decode import Args, get_func

    b, sq, num_k_heads, head_k_dim = q.shape
    num_v_heads = v.shape[-2]
    head_v_dim = v.shape[-1]
    args_ = Args(
        dtype=q.dtype,
        b=b,
        sq=sq,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
    )
    out = torch.zeros(
        (b, sq, num_v_heads, head_v_dim), dtype=args_.dtype, device=q.device
    )
    EXE = gdr_decode(args_)
    EXE(
        q,
        k,
        v,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        initial_state,
        out,
        args_.b,
        float(1.0 / (args_.head_k_dim**0.5)),
    )
    return out
