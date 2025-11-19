import torch
from aiter.ops.triton._triton_kernels.split_qkv import _split_qkv_kernel


def split_qkv(
    qkv,
    q_size,
    kv_size,
):

    q = torch.empty(qkv.shape[0], q_size, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(qkv.shape[0], kv_size, dtype=qkv.dtype, device=qkv.device)
    v = torch.empty(qkv.shape[0], kv_size, dtype=qkv.dtype, device=qkv.device)

    grid = qkv.shape[0]

    # TODO: Add support for dim
    _split_qkv_kernel[(grid,)](
        qkv,
        q,
        k,
        v,
        qkv.stride(0),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        q_size,
        kv_size,
    )

    return q, k, v
