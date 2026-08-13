# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Kimi Delta Attention (KDA) chunkwise prefill.

Two layers live here:

``kda_chunk_fwd``
    A thin PyTorch wrapper over the FlyDSL kernels in ``kernels.kda_kernel`` /
    ``kernels.kda_split``.  It takes the dense ``[B, H, T, D]`` layout the
    kernel is built around and expects the gate/norm preprocessing to have
    already been applied.

``flydsl_chunk_kda``
    A drop-in replacement for ``fla.ops.kda.chunk_kda`` as Kimi-K3 calls it:
    packed varlen ``[1, total_tokens, H, D]`` inputs, raw (pre-activation)
    gate and beta, and a V-first recurrent state.  It returns ``None`` when
    the shape or configuration is outside what the FlyDSL kernel supports, so
    callers can fall back to the Triton path without a try/except.

    "Supported" here is as much about speed as correctness: this path wins on
    batches of short sequences and loses on long ones, so it also declines
    shapes it would run correctly but slowly (see ``_MAX_SEQLEN``).

Only the sequence-parallel ("split") kernel is used.  The fused single-kernel
variant factors its C x C tiles against one per-chunk reference row, and with
Kimi's ``gate_lower_bound = -5`` a 32-token chunk can accumulate up to 160
nats of decay -- far enough into the exponent range that the tiles overflow
and the output goes NaN (the final state stays correct, so the failure is
silent for a whole prefill before it shows up).  The split path builds the
same tiles per chunk with a bounded construction and is exact over that whole
range; it is also the faster of the two at prefill chunk counts.
"""

from __future__ import annotations

import functools
import math
import os

import torch

from .kernels.kda_kernel import (
    LDS_GRANULE,
    LDS_PER_CU,
    fwd_lds_bytes,
)
from .kernels.kda_split import build_kda_prep_module, build_kda_scan_module
from .kda_varlen import kda_pack_prepare, kda_unpack_output

__all__ = [
    "kda_chunk_fwd",
    "flydsl_chunk_kda",
    "flydsl_kda_supported",
]

# C=32 with a 4-way value split keeps LDS ~71 KB (two workgroups/CU); with the
# O(C^2) triangular-solve shrink that beats C=64 by ~2.9x despite more chunks.
CHUNK_SIZE = 32
DV_SPLIT = 4

# The prep kernel materializes six O(T) tiles per chunk; past this share of free
# VRAM, report the shape unsupported rather than risk an OOM mid-request.
SPLIT_MEM_FRACTION = 0.25

# Right-padding to the batch max is only cheap on a roughly rectangular batch;
# a lopsided one inflates the token count, so fall back past this ratio.
_MAX_PAD_RATIO = float(os.environ.get("AITER_KDA_FLYDSL_MAX_PAD_RATIO", "1.5"))

# The scan walks T/C chunks serially at fixed width (num_seqs*heads*dv_split),
# so it loses to fla past ~2K tokens/seq (1K for narrow batches); both env-tunable.
_MAX_SEQLEN = int(os.environ.get("AITER_KDA_FLYDSL_MAX_SEQLEN", "2048"))
_MAX_SEQLEN_NARROW = int(os.environ.get("AITER_KDA_FLYDSL_MAX_SEQLEN_NARROW", "1024"))
_NARROW_BATCH = 4

_SUPPORTED_ARCHS = ("gfx950",)


def _split_workspace_bytes(BH, NC, C, DK):
    n = BH * NC
    return 2 * n * (2 * C * C + 2 * C * DK + DK * C) + 4 * n * DK


def _free_bytes(device):
    """Free memory the split workspace can claim: driver-free plus the caching
    allocator's own unallocated pool (which a KV-cache-heavy server has reserved)."""
    free, _ = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free + cached


@functools.lru_cache(maxsize=8)
def _num_cus(device_index):
    """CUs on the target device; read rather than hard-code so the dispatch
    rule below does not silently mistune on a different part."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=8)
def _arch(device_index):
    name = torch.cuda.get_device_properties(device_index).gcnArchName
    return name.split(":")[0]


def _wgs_per_cu(DK, DV, C, dv_split):
    """Resident workgroups per CU for a split: 160 KB divided by the
    granule-rounded LDS footprint."""
    lds = fwd_lds_bytes(DK=DK, DV=DV, C=C, DV_SPLIT=dv_split)
    return LDS_PER_CU // (-(-lds // LDS_GRANULE) * LDS_GRANULE)


@functools.lru_cache(maxsize=256)
def _auto_dv_split(BH, DK, DV, C, num_cus, ev_max):
    """Value-channel split chosen from the launch shape: the finest split that
    still fills idle CUs (d_fill) and keeps two workgroups per CU resident (d_occ)."""
    legal = [
        d
        for d in (1, 2, 4)
        if DV % d == 0
        and DV // d >= 32  # EV is an MMA M/N extent the 2x2 wave grid halves
        and DV // d <= ev_max
        and fwd_lds_bytes(DK=DK, DV=DV, C=C, DV_SPLIT=d) <= LDS_PER_CU
    ]
    if not legal:
        return DV_SPLIT
    fits = [d for d in legal if BH * d <= num_cus]
    d_fill = max(fits) if fits else min(legal)
    occ2 = [d for d in legal if _wgs_per_cu(DK, DV, C, d) >= 2]
    d_occ = min(occ2) if occ2 else d_fill
    return min(max(legal), max(d_fill, d_occ))


# Each new (BH, T) is a fresh ~35 ms module build shared by the step's 69 KDA
# layers; keep the cache wide enough that ordinary shape churn does not evict.
@functools.lru_cache(maxsize=256)
def _get_split_modules(BH, T, DK, DV, C, dv_split, BLOCK, has_h0, out_dtype_str):
    import flydsl.expr as fx

    fx_out = {"bf16": fx.BFloat16, "fp32": fx.Float32}[out_dtype_str]
    prep = build_kda_prep_module(BH=BH, T=T, DK=DK, DV=DV, C=C, BLOCK=BLOCK)
    scan = build_kda_scan_module(
        BH=BH,
        T=T,
        DK=DK,
        DV=DV,
        C=C,
        DV_SPLIT=dv_split,
        BLOCK=BLOCK,
        has_initial_state=has_h0,
        store_final_state=True,
        out_dtype=fx_out,
    )
    torch_out_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[out_dtype_str]
    return prep, scan, torch_out_dtype


def _split_workspace(BH, NC, C, DK, device):
    """The six per-chunk tiles for the scan, from one pool rather than six
    torch.empty calls (their host cost shows up at short sequences)."""
    n = BH * NC
    shapes = [
        ("a", (n, C, C)),
        ("gk", (n, C, DK)),
        ("gq", (n, C, DK)),
        ("aqk", (n, C, C)),
        ("kt", (n, DK, C)),
    ]
    counts = [math.prod(shape) for _, shape in shapes]
    pool = torch.empty(sum(counts), dtype=torch.bfloat16, device=device)
    ws, off = {}, 0
    for (name, shape), cnt in zip(shapes, counts):
        ws[name] = pool[off : off + cnt].view(*shape)
        off += cnt
    ws["dec"] = torch.empty(n, DK, dtype=torch.float32, device=device)
    ws["_pool"] = pool  # keep the backing storage alive
    return ws


def kda_chunk_fwd(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    scale=None,
    chunk_size=CHUNK_SIZE,
    dv_split=None,
    block=256,
    out_dtype="bf16",
    output_final_state=True,
    stream=None,
):
    """Chunkwise KDA forward over the split FlyDSL kernels: dense [B,H,T,D] bf16
    q/k/v, fp32 g (per-channel log-decay, not pre-summed) and beta; returns (o, ht)."""
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    C = chunk_size
    assert T % C == 0, f"T={T} must be a multiple of {C}"
    assert q.is_contiguous() and k.is_contiguous()
    assert v.is_contiguous() and g.is_contiguous()
    scale = DK**-0.5 if scale is None else float(scale)

    BH, NC = B * H, T // C
    if dv_split is None:
        # ev_max: the kernel stages an initial state through the (C x DK) tile,
        # so EV has to stay <= C on that path.
        dv_split = _auto_dv_split(
            BH,
            DK,
            DV,
            C,
            _num_cus(q.device.index),
            C if initial_state is not None else DV,
        )
    EV = DV // dv_split

    n2 = BH * NC
    q2 = q.view(n2, C * DK)
    k2 = k.view(n2, C * DK)
    g2 = g.view(n2, C * DK)
    b2 = beta.view(n2, C)
    v4 = v.view(n2, C, dv_split, EV)

    has_h0 = initial_state is not None
    prep, scan, torch_out_dtype = _get_split_modules(
        BH, T, DK, DV, C, dv_split, block, has_h0, out_dtype
    )

    dev = q.device
    o = torch.empty(n2, C, dv_split, EV, dtype=torch_out_dtype, device=dev)
    # the kernel keeps the state transposed (S^T, dv x dk)
    htt = torch.empty(BH, dv_split, EV, DK, dtype=torch.float32, device=dev)
    if has_h0:
        h0t = initial_state.transpose(-1, -2).contiguous().reshape(BH, dv_split, EV, DK)
    else:
        h0t = htt  # unread when the module was built with has_initial_state=False

    if stream is None:
        stream = torch.cuda.current_stream()
    ws = _split_workspace(BH, NC, C, DK, dev)
    prep(
        q2,
        k2,
        g2,
        b2,
        ws["a"],
        ws["gk"],
        ws["gq"],
        ws["aqk"],
        ws["kt"],
        ws["dec"],
        float(scale),
        stream=stream,
    )
    # the scan stages these with 128-bit reads, so hand it flat views
    def flat(x):
        return x.reshape(BH * NC, -1)

    scan(
        flat(ws["a"]),
        flat(ws["gk"]),
        flat(ws["gq"]),
        flat(ws["aqk"]),
        flat(ws["kt"]),
        ws["dec"],
        v4,
        o,
        h0t,
        htt,
        NC,
        stream=stream,
    )

    o = o.view(B, H, T, DV)
    # htt is [BH, dv_split, EV, DK] contiguous, so one view + one transpose
    # lands on exactly the [B, H, DK, DV] tensor the caller expects.
    ht = htt.view(B, H, DV, DK).transpose(-1, -2) if output_final_state else None
    return o, ht


def flydsl_kda_supported(head_k: int, head_v: int, dtype, device) -> bool:
    """Can this device / geometry run the FlyDSL KDA kernel at all?"""
    if _arch(device.index) not in _SUPPORTED_ARCHS:
        return False
    if dtype != torch.bfloat16:
        return False
    # BLOCK is fixed at 2*DK by the in-place cumulative sum, and the 256-thread
    # workgroup the kernel is built around pins DK to 128.
    if head_k != 128 or head_v != 128:
        return False
    return True


def flydsl_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    out: torch.Tensor | None = None,
):
    """fla.ops.kda.chunk_kda on the FlyDSL kernel for packed varlen [1,T,H,D]
    input; returns (o, final_state), or None when the shape/config is unsupported."""
    if q.dim() != 4 or q.shape[0] != 1:
        return None
    num_tokens, num_heads, head_k = q.shape[1], q.shape[2], q.shape[3]
    head_v = v.shape[-1]
    if v.shape[2] != num_heads:  # GVA (HV > H) is not implemented here
        return None
    if not flydsl_kda_supported(head_k, head_v, q.dtype, q.device):
        return None
    if use_gate_in_kernel and A_log is None:
        return None
    if initial_state is not None and initial_state.dtype != torch.float32:
        return None

    num_seqs = int(cu_seqlens.numel()) - 1
    if num_seqs < 1 or num_tokens == 0:
        return None
    if initial_state is not None and initial_state.shape[0] != num_seqs:
        return None

    t_pad = -(-int(max_seqlen) // CHUNK_SIZE) * CHUNK_SIZE
    if t_pad <= 0:
        return None
    if t_pad > (_MAX_SEQLEN if num_seqs >= _NARROW_BATCH else _MAX_SEQLEN_NARROW):
        return None
    padded_tokens = num_seqs * t_pad
    # A short prefill pads up to a whole chunk per sequence; only bail out when
    # the batch is lopsided enough that padding dominates.
    if padded_tokens > _MAX_PAD_RATIO * num_tokens + num_seqs * CHUNK_SIZE:
        return None

    BH = num_seqs * num_heads
    NC = t_pad // CHUNK_SIZE
    if _split_workspace_bytes(BH, NC, CHUNK_SIZE, head_k) >= SPLIT_MEM_FRACTION * _free_bytes(
        q.device
    ):
        return None

    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    beta_scale = 2.0 if allow_neg_eigval else 1.0
    q_pad, k_pad, v_pad, g_pad, beta_pad = kda_pack_prepare(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        g.squeeze(0),
        beta.squeeze(0),
        cu_seqlens,
        num_seqs,
        t_pad,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound if use_gate_in_kernel else None,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_gate=use_gate_in_kernel,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        beta_scale=beta_scale,
    )

    # kda_chunk_fwd wants [B,H,DK,DV] and transposes internally, so a V-first
    # state is handed over as a transposed view (the transposes cancel, no copy).
    h0 = initial_state
    if h0 is not None and state_v_first:
        h0 = h0.transpose(-1, -2)

    o_pad, ht = kda_chunk_fwd(
        q_pad,
        k_pad,
        v_pad,
        g_pad,
        beta_pad,
        initial_state=h0,
        scale=scale,
        output_final_state=output_final_state,
    )

    if ht is not None and state_v_first:
        ht = ht.transpose(-1, -2)

    if out is not None:
        kda_unpack_output(o_pad, out, cu_seqlens, num_seqs)
        return out, ht

    o = q.new_empty((num_tokens, num_heads, head_v))
    kda_unpack_output(o_pad, o, cu_seqlens, num_seqs)
    return o.unsqueeze(0), ht
