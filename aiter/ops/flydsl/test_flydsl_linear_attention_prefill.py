# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention Prefill (chunk_gated_delta_h) regressions.

Usage:
    pytest -sv aiter/ops/flydsl/test_flydsl_linear_attention_prefill.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import triton
from torch.profiler import ProfilerActivity, profile

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL Linear Attention Prefill tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.kernels.chunk_gated_delta_h import (
        chunk_gated_delta_rule_fwd_h_flydsl,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h_opt,
        chunk_gated_delta_rule_fwd_h_opt_vk,
    )
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL Linear Attention Prefill kernels: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")


# -- Global test configuration ------------------------------------------


@dataclass
class PrefillArgs:
    K: int
    V: int
    Hk: int
    Hv: int
    tp: int
    full_prompt_len: int
    model_name: str = ""
    BT: int = 64
    max_num_batched_tokens: int = 32768
    dtype: torch.dtype = torch.bfloat16
    is_varlen: bool = True
    output_final_state: bool = True

    @property
    def Hg(self):
        return self.Hk // self.tp

    @property
    def H(self):
        return self.Hv // self.tp

    def __repr__(self):
        tag = self.model_name + "_" if self.model_name else ""
        tag += f"K{self.K}_V{self.V}_Hk{self.Hk}_Hv{self.Hv}"
        tag += f"_TP{self.tp}_T{self.full_prompt_len}"
        if not self.is_varlen:
            tag += "_novarlen"
        if not self.output_final_state:
            tag += "_nofs"
        return tag


NUM_WARMUP = 5
NUM_ITERS = 50

PREFILL_PARAMS = [
    # non-varlen + no final state
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=1,
        full_prompt_len=2500,
        model_name="Qwen3.5-35B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=1,
        full_prompt_len=60000,
        model_name="Qwen3.5-35B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=2,
        full_prompt_len=2500,
        model_name="Qwen3.5-35B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=2,
        full_prompt_len=60000,
        model_name="Qwen3.5-35B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=1,
        full_prompt_len=2500,
        model_name="Qwen3.5-397B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=1,
        full_prompt_len=60000,
        model_name="Qwen3.5-397B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=2,
        full_prompt_len=2500,
        model_name="Qwen3.5-397B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=2500,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=2,
        full_prompt_len=60000,
        model_name="Qwen3.5-397B",
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens=60000,
    ),
    # varlen + final_state (default path)
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=4,
        full_prompt_len=1024,
        model_name="meta-Qwen3.5",
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=4,
        full_prompt_len=8192,
        model_name="meta-Qwen3.5",
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=8,
        full_prompt_len=1024,
        model_name="meta-Qwen3.5",
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=8,
        full_prompt_len=8192,
        model_name="meta-Qwen3.5",
    ),
]


# -- Helper functions ---------------------------------------------------


def _build_context_lens(full_prompt_len, max_tokens=32768):
    context_lens = []
    remaining = max_tokens
    while remaining > 0:
        cur = min(full_prompt_len, remaining)
        context_lens.append(cur)
        remaining -= cur
    return context_lens


def _build_cu_seqlens(context_lens, device="cuda"):
    scheduled_q_lens = context_lens
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(scheduled_q_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    return scheduled_q_lens, cu_seqlens


def _make_inputs(
    context_lens,
    args: PrefillArgs = None,
    *,
    tp=1,
    K_dim=128,
    V_dim=128,
    Hk_dim=16,
    Hv_dim=64,
    dtype=torch.bfloat16,
    device="cuda",
    with_initial_state=True,
    is_varlen=True,
):
    if args is not None:
        tp = args.tp
        K_dim = args.K
        V_dim = args.V
        Hk_dim = args.Hk
        Hv_dim = args.Hv
        dtype = args.dtype
        is_varlen = args.is_varlen

    Hg = Hk_dim // tp
    H = Hv_dim // tp

    if is_varlen:
        scheduled_q_lens, cu_seqlens = _build_cu_seqlens(context_lens, device=device)
        T_total = int(cu_seqlens[-1].item())
        N = len(scheduled_q_lens)
        B = 1
    else:
        T_total = sum(context_lens)
        B = 1
        N = B
        cu_seqlens = None
        scheduled_q_lens = context_lens

    k = torch.randn(B, T_total, Hg, K_dim, dtype=dtype, device=device) * 0.1
    w_orig = torch.randn(B, T_total, H, K_dim, dtype=dtype, device=device) * 0.1
    u_orig = torch.randn(B, T_total, H, V_dim, dtype=dtype, device=device) * 0.1
    g = torch.randn(T_total, H, dtype=torch.float32, device=device).abs() * -0.5
    g = g.cumsum(dim=0)

    w_c = w_orig.permute(0, 2, 1, 3).contiguous()
    u_c = u_orig.permute(0, 2, 1, 3).contiguous()

    initial_state = None
    if with_initial_state:
        initial_state = (
            torch.randn(N, H, V_dim, K_dim, dtype=torch.float32, device=device) * 0.01
        )

    return k, w_orig, u_orig, w_c, u_c, g, initial_state, cu_seqlens, scheduled_q_lens


# -- Pure-PyTorch reference ----------------------------------------------


def ref_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
):
    """Reference in FP32 for correctness checking."""
    B, T, Hg_dim, K_dim = k.shape
    H_dim, V_dim = u.shape[-2], u.shape[-1]
    BT_dim = chunk_size
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT_dim)
    else:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        NT = sum(triton.cdiv(int(seq_len), BT_dim) for seq_len in seq_lens)
    gqa_ratio = H_dim // Hg_dim

    h_out = k.new_zeros(B, NT, H_dim, V_dim, K_dim, dtype=torch.float32)
    v_new_out = torch.zeros_like(u, dtype=torch.float32)

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    final_state = (
        torch.zeros(N, H_dim, V_dim, K_dim, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    for b_idx in range(B):
        if cu_seqlens is not None:
            seqs = [
                (s, cu_seqlens[s].item(), cu_seqlens[s + 1].item()) for s in range(N)
            ]
        else:
            seqs = [(b_idx, 0, T)]

        chunk_offset = 0
        for seq_idx, bos, eos in seqs:
            seq_len = eos - bos
            seq_nt = triton.cdiv(seq_len, BT_dim)

            for i_h in range(H_dim):
                i_hg = i_h // gqa_ratio
                h_state = torch.zeros(
                    V_dim, K_dim, dtype=torch.float32, device=k.device
                )
                if initial_state is not None:
                    h_state = initial_state[seq_idx, i_h].float().clone()

                for i_t in range(seq_nt):
                    t_start = i_t * BT_dim
                    t_end = min(t_start + BT_dim, seq_len)
                    actual_bt = t_end - t_start

                    h_out[b_idx, chunk_offset + i_t, i_h] = h_state.clone()

                    w_chunk = w[b_idx, bos + t_start : bos + t_end, i_h].float()
                    u_chunk = u[b_idx, bos + t_start : bos + t_end, i_h].float()
                    b_v = u_chunk - w_chunk @ h_state.T
                    v_new_out[b_idx, bos + t_start : bos + t_end, i_h] = b_v

                    last_idx = bos + t_end - 1
                    g_last = g[last_idx, i_h].float()
                    g_chunk = g[bos + t_start : bos + t_end, i_h].float()

                    mask = torch.zeros(BT_dim, device=k.device)
                    mask[:actual_bt] = 1.0
                    gate = torch.where(
                        mask[:actual_bt].bool(),
                        torch.exp(g_last - g_chunk),
                        torch.zeros_like(g_chunk),
                    )
                    b_v_gated = b_v * gate.unsqueeze(-1)

                    h_state = h_state * torch.exp(g_last)
                    k_chunk = k[b_idx, bos + t_start : bos + t_end, i_hg].float()
                    b_v_gated_cast = b_v_gated.to(k.dtype).float()
                    h_state = h_state + b_v_gated_cast.T @ k_chunk

                if output_final_state:
                    final_state[seq_idx, i_h] = h_state

            chunk_offset += seq_nt

    return h_out, v_new_out.to(u.dtype), final_state


def _normalize_opt_v_new(vn_opt):
    """Convert opt v_new layout [B, H, T, V] back to [B, T, H, V]."""
    return vn_opt.permute(0, 2, 1, 3).contiguous()


# -- Performance benchmark ----------------------------------------------


def _bench_fn(fn, *args, **kwargs):
    """Average per-iter device kernel time (us) via torch.profiler.

    Mirrors the methodology used by ``0422_gdr_prefill_kernel_bench.py``:
    capture all CUDA events under ProfilerActivity.CUDA during a fixed-iter
    measurement window, then sum each unique kernel's
    ``self_device_time_total`` and divide by ``niters``. Because each
    ``_launch()`` closure dispatches a single K5 device kernel (with all host
    preparation lifted out), the resulting number is the per-iter K5 device
    time, free of Python launcher overhead.
    """
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(NUM_ITERS):
            fn(*args, **kwargs)
    torch.cuda.synchronize()

    total_us = 0.0
    for evt in prof.key_averages():
        if evt.device_type is None or "cuda" not in str(evt.device_type).lower():
            continue
        total_us += evt.self_device_time_total / NUM_ITERS
    return total_us


# -- Correctness tests ---------------------------------------------------

PREFILL_TEST_IDS = [repr(p) for p in PREFILL_PARAMS]


def _assert_k5_outputs_match_ref(
    h_out, vn_out, fs_out, h_ref, vn_ref, fs_ref, *, output_final_state, label
):
    """Compare a K5 backend's outputs against the PyTorch FP32 reference.

    All backends in this file return VK-ordered ``h`` / ``final_state`` and
    ``v_new`` in head-major ``[B, H, T, V]`` layout (which we permute back to
    ``[B, T, H, V]`` for comparison via ``_normalize_opt_v_new``).
    """
    torch.testing.assert_close(
        h_out.float(),
        h_ref.float(),
        atol=1e-1,
        rtol=1e-1,
        msg=f"{label}: h mismatch",
    )
    torch.testing.assert_close(
        _normalize_opt_v_new(vn_out).float(),
        vn_ref.float(),
        atol=1e-1,
        rtol=1e-1,
        msg=f"{label}: v_new mismatch",
    )
    if output_final_state:
        torch.testing.assert_close(
            fs_out.float(),
            fs_ref.float(),
            atol=1e-1,
            rtol=1e-1,
            msg=f"{label}: final_state mismatch",
        )
    else:
        assert fs_out is None, f"{label}: expected None final_state"
        assert fs_ref is None


class TestCorrectness:
    """Correctness against PyTorch FP32 reference for all three K5 backends."""

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl(self, args: PrefillArgs):
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w_orig,
            u_orig,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

        _assert_k5_outputs_match_ref(
            h_fly,
            vn_fly,
            fs_fly,
            h_ref,
            vn_ref,
            fs_ref,
            output_final_state=args.output_final_state,
            label="flydsl",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_triton_vk(self, args: PrefillArgs):
        """Triton VK K5 (h: [V, K]) -- same input/output layout as FlyDSL."""
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_vk, vn_vk, fs_vk = chunk_gated_delta_rule_fwd_h_opt_vk(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w_orig,
            u_orig,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

        _assert_k5_outputs_match_ref(
            h_vk,
            vn_vk,
            fs_vk,
            h_ref,
            vn_ref,
            fs_ref,
            output_final_state=args.output_final_state,
            label="triton_vk",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_triton_kv(self, args: PrefillArgs):
        """Triton KV K5 (h: [K, V]) -- h0/h/final_state are transposed.

        We feed the wrapper a KV-layout ``initial_state`` (transposed from the
        VK-layout ``h0`` produced by ``_make_inputs``), and transpose the
        returned ``h`` / ``final_state`` back to VK so they compare to the
        common FP32 reference.
        """
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h0_kv = h0.transpose(-2, -1).contiguous() if h0 is not None else None

        h_kv, vn_kv, fs_kv = chunk_gated_delta_rule_fwd_h_opt(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0_kv,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w_orig,
            u_orig,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

        # KV-layout outputs need to be transposed back to VK for comparison.
        h_kv_vk = h_kv.transpose(-2, -1).contiguous()
        fs_kv_vk = fs_kv.transpose(-2, -1).contiguous() if fs_kv is not None else None

        _assert_k5_outputs_match_ref(
            h_kv_vk,
            vn_kv,
            fs_kv_vk,
            h_ref,
            vn_ref,
            fs_ref,
            output_final_state=args.output_final_state,
            label="triton_kv",
        )


_perf_results: list[dict] = []


class TestPerformance:
    """Kernel-only performance comparison: FlyDSL vs Triton opt_vk vs Triton opt3_kv."""

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_perf_comparison(self, args: PrefillArgs):
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, _, _, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)
        total_tokens = sum(context_lens)

        # Triton KV wrapper expects initial_state in [N, H, K, V]; tests carry VK.
        h0_kv = h0.transpose(-2, -1).contiguous() if h0 is not None else None

        # K5 launch closures: each invokes the K5 host wrapper of its backend.
        def flydsl_launch():
            chunk_gated_delta_rule_fwd_h_flydsl(
                k=k,
                w=w_c,
                u=u_c,
                g=g,
                initial_state=h0,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )

        def triton_vk_launch():
            chunk_gated_delta_rule_fwd_h_opt_vk(
                k=k,
                w=w_c,
                u=u_c,
                g=g,
                initial_state=h0,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )

        def triton_opt3_launch():
            chunk_gated_delta_rule_fwd_h_opt(
                k=k,
                w=w_c,
                u=u_c,
                g=g,
                initial_state=h0_kv,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )

        # Warmup FlyDSL once so its internal BV-autotune sweep does not
        # leak into the timed window. Triton's own ``triton.autotune`` is
        # already absorbed by ``_bench_fn``'s NUM_WARMUP=5 prelude.
        flydsl_launch()
        torch.cuda.synchronize()

        us_triton_opt3 = _bench_fn(triton_opt3_launch)
        us_fly = _bench_fn(flydsl_launch)
        us_triton_vk = _bench_fn(triton_vk_launch)

        speedup_vk = us_triton_vk / us_fly if us_fly > 0 else float("inf")
        speedup_opt3 = us_triton_opt3 / us_fly if us_fly > 0 else float("inf")

        _perf_results.append(
            {
                "Model": args.model_name or "-",
                "TP": args.tp,
                "K": args.K,
                "V": args.V,
                "Hg": args.Hg,
                "H": args.H,
                "SeqLen": args.full_prompt_len,
                "T": total_tokens,
                "varlen": args.is_varlen,
                "final_st": args.output_final_state,
                "FlyDSL_vk(us)": us_fly,
                "Triton_vk(us)": us_triton_vk,
                "Triton_kv(us)": us_triton_opt3,
                "vs_triton_vk": speedup_vk,
                "vs_triton_kv": speedup_opt3,
            }
        )


def _print_perf_table():
    if not _perf_results:
        return

    cols = [
        ("Model", 16),
        ("TP", 4),
        ("K", 5),
        ("V", 5),
        ("Hg", 4),
        ("H", 4),
        ("SeqLen", 7),
        ("T", 7),
        ("varlen", 7),
        ("final_st", 9),
        ("FlyDSL_vk(us)", 15),
        ("Triton_vk(us)", 15),
        ("Triton_kv(us)", 15),
        ("vs_triton_vk", 13),
        ("vs_triton_kv", 13),
    ]

    header = " | ".join(name.rjust(width) for name, width in cols)
    sep = "-+-".join("-" * width for _, width in cols)

    lines = [
        "",
        "=" * len(header),
        "K5 Prefill Performance Summary (K5 device kernel time only, via torch.profiler)",
        "=" * len(header),
        header,
        sep,
    ]

    for row in _perf_results:
        cells = []
        for name, width in cols:
            val = row[name]
            if isinstance(val, bool):
                cells.append(("Y" if val else "N").rjust(width))
            elif isinstance(val, float):
                if name in ("vs_triton_vk", "vs_triton_kv"):
                    cells.append(f"{val:.3f}x".rjust(width))
                else:
                    cells.append(f"{val:.2f}".rjust(width))
            else:
                cells.append(str(val).rjust(width))
        lines.append(" | ".join(cells))

    lines.append(sep)
    lines.append("")
    print("\n".join(lines))


@pytest.fixture(scope="session", autouse=True)
def _print_summary_table(request):
    """Print the summary performance table after all tests finish."""
    yield
    _print_perf_table()
