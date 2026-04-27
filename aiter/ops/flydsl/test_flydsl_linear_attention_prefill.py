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
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        flydsl_gdr_prefill,
    )
    from aiter.ops.flydsl.kernels import (
        chunk_gated_delta_h as flydsl_chunk_gated_delta_h_mod,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_opt,
        chunk_gated_delta_rule_fwd_kernel_h_opt_vk,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
        prepare_chunk_offsets as _prepare_chunk_offsets,
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


def _build_chunk_offsets(cu_seqlens, chunk_size):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(lens, chunk_size),
        ]
    ).cumsum(-1)


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


# -- Triton opt_vk reference ---------------------------------------------

_FLA_CHUNK_SIZE_OPT_VK = 64


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


def _prepare_flydsl_kernel_launch(
    k,
    w,
    u,
    g=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    wu_contiguous=True,
    bv=0,
):
    mod = flydsl_chunk_gated_delta_h_mod
    batch_size, total_tokens, hg, head_k_dim = k.shape
    bt = chunk_size

    if wu_contiguous:
        num_heads = w.shape[1]
        head_v_dim = u.shape[-1]
        t_flat = w.shape[2]
    else:
        num_heads = u.shape[-2]
        head_v_dim = u.shape[-1]
        t_flat = w.shape[1]

    if cu_seqlens is None:
        num_seqs, num_chunks, chunk_offsets = (
            batch_size,
            triton.cdiv(total_tokens, bt),
            None,
        )
    else:
        num_seqs = len(cu_seqlens) - 1
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        num_chunks = sum(triton.cdiv(int(seq_len), bt) for seq_len in lens.tolist())
        chunk_offsets = _build_chunk_offsets(cu_seqlens, bt).to(torch.int32)

    use_g = g is not None
    use_h0 = initial_state is not None
    is_varlen = cu_seqlens is not None

    if bv <= 0:
        shape_key = (
            head_k_dim,
            head_v_dim,
            bt,
            num_heads,
            hg,
            t_flat,
            num_seqs,
            use_g,
            use_h0,
            output_final_state,
            save_new_value,
            is_varlen,
            wu_contiguous,
        )
        if shape_key not in mod._autotune_cache:
            mod.chunk_gated_delta_rule_fwd_h_flydsl(
                k,
                w,
                u,
                g=g,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
                save_new_value=save_new_value,
                cu_seqlens=cu_seqlens,
                wu_contiguous=wu_contiguous,
            )
            torch.cuda.synchronize()
        bv = mod._autotune_cache[shape_key]

    launch_fn = mod._get_or_compile(
        head_k_dim,
        head_v_dim,
        bt,
        bv,
        num_heads,
        hg,
        use_g,
        use_h0,
        output_final_state,
        save_new_value,
        is_varlen,
        wu_contiguous,
    )

    h = k.new_empty(batch_size, num_chunks, num_heads, head_v_dim, head_k_dim)
    final_state = (
        k.new_empty(num_seqs, num_heads, head_v_dim, head_k_dim, dtype=torch.float32)
        if output_final_state
        else None
    )
    v_new_buf = k.new_empty(batch_size, num_heads, t_flat, head_v_dim, dtype=u.dtype)

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    g_arg = g if g is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    vn_arg = v_new_buf
    cu_arg = (
        cu_seqlens.to(torch.int32) if cu_seqlens is not None else dummy.to(torch.int32)
    )
    co_arg = chunk_offsets if chunk_offsets is not None else dummy.to(torch.int32)
    stream = torch.cuda.current_stream()

    def _launch():
        mod._launch_kernel(
            launch_fn,
            bv,
            head_v_dim,
            num_seqs,
            num_heads,
            k,
            u,
            w,
            vn_arg,
            g_arg,
            h,
            h0_arg,
            ht_arg,
            cu_arg,
            co_arg,
            total_tokens,
            t_flat,
            stream,
        )

    _launch.best_bv = bv
    return _launch


# -- Triton K5 launch closures (host prep done once, only device kernel timed) --
# These mirror the host wrappers in
# ``aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_delta_h.py``
# (`chunk_gated_delta_rule_fwd_h_opt` / `_opt_vk`), but lift all per-call host
# work (allocations + chunk_offsets + grid lambda + initial_state transpose) out
# of the closure so ``_bench_fn`` only times the underlying ``triton.jit``
# device kernel launch -- the same convention FlyDSL's autotune harness uses.


def _prepare_triton_opt3_kv_kernel_launch(
    k,
    w,
    u,
    g=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
):
    """Triton K5 (KV layout) launch closure: device-only launch in the body."""
    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
        NT = int(chunk_offsets[-1].item())

    # The KV wrapper expects initial_state in [N, H, K, V]; tests carry VK.
    h0_kv = (
        initial_state.transpose(-2, -1).contiguous()
        if initial_state is not None
        else None
    )
    # h is in KV layout: [B, NT, H, K, V].
    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    def _launch():
        chunk_gated_delta_rule_fwd_kernel_h_opt[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=None,
            h=h,
            h0=h0_kv,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            T_flat=T_flat,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
        )

    return _launch


def _prepare_triton_opt_vk_kernel_launch(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=_FLA_CHUNK_SIZE_OPT_VK,
    save_new_value=True,
    cu_seqlens=None,
):
    """Triton K5 (VK layout) launch closure: device-only launch in the body."""
    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
        NT = int(chunk_offsets[-1].item())

    # h is in VK layout: [B, NT, H, V, K].
    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    def _launch():
        chunk_gated_delta_rule_fwd_kernel_h_opt_vk[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=gk,
            h=h,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            T_flat=T_flat,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
        )

    return _launch


# -- Correctness tests ---------------------------------------------------

PREFILL_TEST_IDS = [repr(p) for p in PREFILL_PARAMS]


class TestCorrectness:
    """Correctness against PyTorch reference."""

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl(self, args: PrefillArgs):
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = flydsl_gdr_prefill(
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

        torch.testing.assert_close(h_fly.float(), h_ref.float(), atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            _normalize_opt_v_new(vn_fly).float(), vn_ref.float(), atol=1e-1, rtol=1e-1
        )
        if args.output_final_state:
            torch.testing.assert_close(
                fs_fly.float(), fs_ref.float(), atol=1e-1, rtol=1e-1
            )
        else:
            assert fs_fly is None
            assert fs_ref is None


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

        flydsl_launch = _prepare_flydsl_kernel_launch(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )
        triton_vk_launch = _prepare_triton_opt_vk_kernel_launch(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )
        triton_opt3_launch = _prepare_triton_opt3_kv_kernel_launch(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

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
