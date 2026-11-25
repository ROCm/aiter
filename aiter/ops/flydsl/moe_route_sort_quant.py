# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The fused MoE routing kernel in front of fused_moe (DeepSeek-V4.1-Flash TP4, gfx950).

`MoeRouteSortQuant` keeps the kernel's persistent buffers (workspace, the sorted scale buffer zeroed once) and runs the
routing kernel for 257-2048 rows in two launches (`build_route_p1` / `build_route_p2`).
`fused_moe_with_routing(x, logits, bias, w1, w2, **fused_moe_kwargs)` runs it, then aiter.fused_moe with its decode
sort+quant hook (`_is_fused_decode_sort_quant_enabled` / `_fused_decode_sort_quant`) pointed at the kernel's
outputs, so stage 1 takes the prequantized MXFP8 a1 and the sorted e8m0 scales and no sort or quant runs inside
fused_moe. topk_ids / topk_weights passed to fused_moe are the kernel's own.
"""

import contextlib
import threading

import torch

from aiter.ops.flydsl.kernels import moe_route_sort_quant as rsq
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

_TLS = threading.local()


def _max_sorted(m, bm):
    routes = m * rsq.TOPK
    active = min(rsq.E_ALL, routes)
    return (routes + active * (bm - 1) + bm - 1) // bm * bm


def block_m_for(m):
    """The sort block of the tuned DeepSeek-V4.1 MoE rows: 64 for the 2048 bucket, else 32."""
    return 64 if m > 1024 else 32


class MoeRouteSortQuant:
    def __init__(self, device="cuda", max_tokens=rsq.MAX_TOKENS):
        self.dev = torch.device(device)
        self.exe1 = rsq.build_route_p1()
        self.exe = {bm: rsq.build_route_p2(block_m=bm) for bm in (32, 64)}
        # launch 1 -> launch 2 hand-off, shared by every layer (calls run one after another on one stream)
        self.slots = torch.empty(
            (max_tokens, rsq.TOPK), dtype=torch.int32, device=self.dev
        )
        self.tsc = torch.empty(
            (max_tokens, rsq.GROUPS), dtype=torch.uint8, device=self.dev
        )
        self.ws = torch.zeros(rsq.WS_INTS, dtype=torch.int32, device=self.dev)
        self.max_sorted = max(_max_sorted(max_tokens, 32), _max_sorted(max_tokens, 64))
        self.a1_scale = torch.zeros(
            (self.max_sorted, rsq.GROUPS), dtype=torch.uint8, device=self.dev
        )

    def __call__(
        self, logits, bias_f32, x, rsf=1.0, shared_w=1.0, num_token_non_padded=None
    ):
        m = x.shape[0]
        nv = (
            num_token_non_padded.to(torch.int32).reshape(1)
            if num_token_non_padded is not None
            else torch.full((1,), m, dtype=torch.int32, device=x.device)
        )
        assert (
            1 <= m <= rsq.MAX_TOKENS
            and x.shape[1] == rsq.DIM
            and logits.shape == (m, rsq.E_ROUTED)
        )
        bm = block_m_for(m)
        s = _max_sorted(m, bm)
        b = (s + bm - 1) // bm
        dev = x.device
        out = {
            "topk_ids": torch.empty((m, rsq.TOPK), dtype=torch.int32, device=dev),
            "topk_w": torch.empty((m, rsq.TOPK), dtype=torch.float32, device=dev),
            "sorted_ids": torch.empty((s,), dtype=torch.int32, device=dev),
            "sorted_w": torch.empty((s,), dtype=torch.float32, device=dev),
            "sorted_eids": torch.empty((b,), dtype=torch.int32, device=dev),
            "num_valid": torch.empty((2,), dtype=torch.int32, device=dev),
            "a1": torch.empty((m, rsq.DIM), dtype=torch.float8_e4m3fn, device=dev),
            "a1_scale": self.a1_scale[:s].view(torch.float8_e8m0fnu),
            "block_m": bm,
        }
        stream = torch.cuda.current_stream()
        _run_compiled(
            self.exe1,
            ptr_arg(logits),
            ptr_arg(bias_f32),
            ptr_arg(x),
            ptr_arg(out["topk_ids"]),
            ptr_arg(out["topk_w"]),
            ptr_arg(out["a1"].view(torch.uint8)),
            ptr_arg(self.slots),
            ptr_arg(self.tsc),
            ptr_arg(self.ws),
            ptr_arg(nv),
            m,
            float(rsf),
            float(shared_w),
            rsq.workers(m),
            stream,
        )
        _run_compiled(
            self.exe[bm],
            ptr_arg(out["topk_ids"]),
            ptr_arg(out["topk_w"]),
            ptr_arg(self.slots),
            ptr_arg(self.tsc),
            ptr_arg(out["sorted_ids"]),
            ptr_arg(out["sorted_w"]),
            ptr_arg(out["sorted_eids"]),
            ptr_arg(out["num_valid"]),
            ptr_arg(self.a1_scale),
            ptr_arg(self.ws),
            ptr_arg(nv),
            m,
            rsq.workers(m),
            stream,
        )
        return out


def _enabled(*a, **k):
    return getattr(_TLS, "route", None) is not None


def _sort_quant(
    hidden_states,
    topk_ids,
    topk_weights,
    *,
    model_dim,
    global_experts,
    block_m,
    moebuf_dtype,
    accumulate,
    compact_scale=False,
    num_local_tokens=None,
):
    r = _TLS.route
    if (
        block_m != r["block_m"]
        or global_experts != rsq.E_ALL
        or compact_scale
        or num_local_tokens is not None
    ):
        raise ValueError(
            f"moe_route_sort_quant: block_m {block_m}, experts {global_experts}, compact_scale {compact_scale}"
        )
    m = hidden_states.shape[0]
    # accumulate (atomic stage 2) needs a zeroed output; the FP8 route-out stage 2 overwrites it
    moe_buf = (
        torch.zeros((m, model_dim), dtype=moebuf_dtype, device=hidden_states.device)
        if accumulate
        else torch.empty((0, 0), dtype=moebuf_dtype, device=hidden_states.device)
    )
    return (
        r["sorted_ids"],
        r["sorted_w"],
        r["sorted_eids"],
        r["num_valid"],
        moe_buf,
        r["a1"],
        r["a1_scale"],
    )


@contextlib.contextmanager
def _hook(route):
    import aiter.fused_moe as fm

    old = fm._is_fused_decode_sort_quant_enabled, fm._fused_decode_sort_quant
    fm._is_fused_decode_sort_quant_enabled, fm._fused_decode_sort_quant = (
        _enabled,
        _sort_quant,
    )
    _TLS.route = route
    try:
        yield
    finally:
        _TLS.route = None
        fm._is_fused_decode_sort_quant_enabled, fm._fused_decode_sort_quant = old


def fused_moe_with_routing(
    router: MoeRouteSortQuant, x, logits, bias_f32, w1, w2, rsf=1.0, shared_w=1.0, **kw
):
    import aiter.fused_moe as fm

    r = router(logits, bias_f32, x, rsf, shared_w)
    with _hook(r):
        return fm.fused_moe(x, w1, w2, r["topk_w"], r["topk_ids"], **kw)
