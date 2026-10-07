# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import re

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

# The parser is imported by AOT enumeration and the tuner; torch and the
# compiled-kernel modules stay lazy inside the functions that need them.
_FLYDSL_DECODE_RE = re.compile(
    r"^flydsl_decode_t(\d+)x(\d+)x(\d+)_kw(\d+)_nb(\d+)_sk(\d+)$"
)

# Rows of A the kernel computes per workgroup.
DECODE_TILE_M = 16
# K extent of one WG main-loop iteration: each k-wave consumes one K256 tile.
DECODE_K_TILE = 256


def decode_tile_k(k_waves):
    return DECODE_K_TILE * k_waves


def flydsl_decode_name(tile_n, k_waves, num_buffers, split_k):
    return (
        f"flydsl_decode_t{DECODE_TILE_M}x{tile_n}x{decode_tile_k(k_waves)}"
        f"_kw{k_waves}_nb{num_buffers}_sk{split_k}"
    )


def parse_flydsl_decode_name(name):
    """Parse flydsl_decode_t{TM}x{tile_n}x{TK}_kw{k_waves}_nb{num_buffers}_sk{split_k}.

    Returns None if not that form or if TM/TK disagree with k_waves.
    """
    mt = _FLYDSL_DECODE_RE.match(name) if isinstance(name, str) else None
    if mt is None:
        return None
    tm, tile_n, tk, k_waves, num_buffers, split_k = (int(g) for g in mt.groups())
    if tm != DECODE_TILE_M or tk != decode_tile_k(k_waves):
        return None
    return {
        "tile_n": tile_n,
        "k_waves": k_waves,
        "num_buffers": num_buffers,
        "split_k": split_k,
    }


def decode_gemm_mxfp4_lds_bytes(M, K, tile_n, k_waves, split_k=1):
    """Maximum per-WG LDS, including padded K and the intra-WG reduction."""
    tiles = -(-K // DECODE_K_TILE)
    if not isinstance(split_k, int) or not 1 <= split_k <= tiles:
        raise ValueError("requires integer split_k in [1,ceil(K/256)]")
    padded_slice_k = -(-tiles // split_k) * DECODE_K_TILE
    # The single-group specialization never uses partial LDS; it is eliminated.
    partial_bytes = 16 * tile_n * 4 if k_waves == 2 else 0
    return M * padded_slice_k // 2 + padded_slice_k + partial_bytes


# gfx950 per-CU LDS capacity; A residency beyond it cannot launch.
DECODE_GEMM_MXFP4_LDS_LIMIT = get_lds_capacity_bytes("gfx950")


def flydsl_decode_gemm_mxfp4(
    A,
    B,
    A_scale,
    B_scale,
    out,
    *,
    tile_n=32,
    k_waves=2,
    num_buffers=4,
    split_k=1,
    workspace=None,
):
    """Decode A4W4 with asm-compatible packed operands and shuffled e8m0 scales.

    A is row-major [M,K/2]; B is [N,K/2] preshuffled with layout=(16,16),
    which requires K%64==0. Scales use the per_1x32 shuffle=True layout
    with K rounded to 256 and at least 32 A rows. split_k partitions K256
    tiles across WGs and reduces compact fp32 [split_k,M,N] partials in a
    separate launch. workspace may supply that buffer; otherwise it is
    allocated per call, including from the graph pool during capture.
    """
    import flydsl.expr as fx
    import torch

    from aiter.ops.flydsl.kernels.decode_gemm_mxfp4 import compile_decode_gemm_mxfp4
    from aiter.ops.flydsl.kernels.decode_gemm_mxfp4_reduce import (
        compile_decode_gemm_mxfp4_reduce,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    M, packed_k = A.shape
    N = B.shape[0]
    K = packed_k * 2
    if not (1 <= M <= 16 and (tile_n, k_waves) in ((32, 2), (64, 1))):
        raise ValueError("requires M in [1,16] and (tile_n,k_waves)=(32,2) or (64,1)")
    if N % tile_n or K % 64 or num_buffers < 1:
        raise ValueError(
            "requires N%tile_n==0, K%64==0 for the (16,16) B layout, and num_buffers>=1"
        )
    if B.shape != (N, packed_k) or out.shape != (M, N) or out.dtype != torch.bfloat16:
        raise ValueError("requires B [N,K/2] and bf16 out [M,N]")
    lds_bytes = decode_gemm_mxfp4_lds_bytes(M, K, tile_n, k_waves, split_k)
    if lds_bytes > DECODE_GEMM_MXFP4_LDS_LIMIT:
        raise ValueError(
            f"decode_gemm_mxfp4: A residency needs {lds_bytes} B LDS > {DECODE_GEMM_MXFP4_LDS_LIMIT}; "
            "use asm/backend fallback"
        )
    if workspace is not None and (
        workspace.shape != (split_k, M, N)
        or workspace.dtype != torch.float32
        or workspace.device != A.device
        or not workspace.is_contiguous()
    ):
        raise ValueError(
            "requires contiguous fp32 workspace [split_k,M,N] on operand device"
        )
    tensors = (A, B, A_scale, B_scale, out)
    if any(t.device != A.device or not t.is_contiguous() for t in tensors):
        raise ValueError("operands must be contiguous on the same device")
    if any(t.element_size() != 1 for t in tensors[:4]):
        raise ValueError("packed fp4 and e8m0 operands must have one-byte storage")
    padded_k = -(-K // DECODE_K_TILE) * 256
    if A_scale.numel() < padded_k or B_scale.numel() < N * padded_k // 32:
        raise ValueError("requires padded shuffled A scales and shuffled B scales")
    arch = torch.cuda.get_device_properties(A.device).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise ValueError(f"decode MXFP4 supports gfx950, got {arch}")
    target = out
    if split_k > 1:
        if workspace is None:
            workspace = torch.empty(
                (split_k, M, N), device=A.device, dtype=torch.float32
            )
        target = workspace
    exe = compile_decode_gemm_mxfp4(
        M=M,
        N=N,
        K=K,
        tile_n=tile_n,
        k_waves=k_waves,
        num_buffers=num_buffers,
        split_k=split_k,
    )
    _run_compiled(
        exe,
        A.view(torch.int32).view(-1),
        B.view(torch.int32).view(-1),
        A_scale.view(torch.int32).view(-1),
        B_scale.view(torch.int32).view(-1),
        target.view(-1),
        fx.Stream(torch.cuda.current_stream(A.device)),
    )
    if split_k > 1:
        _run_compiled(
            compile_decode_gemm_mxfp4_reduce(M=M, N=N, split_k=split_k),
            workspace.view(-1),
            out.view(-1),
            fx.Stream(torch.cuda.current_stream(A.device)),
        )
    return out
