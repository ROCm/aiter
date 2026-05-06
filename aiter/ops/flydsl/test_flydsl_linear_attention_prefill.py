# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention Prefill (chunk_gated_delta_h) regressions.

Usage:
    pytest -sv aiter/ops/flydsl/test_flydsl_linear_attention_prefill.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dataclass_replace

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
    # SSM-state dtype for h0 / final_state. The kernel keeps the f32
    # accumulator unchanged for both choices; bf16 only affects HBM
    # bandwidth/footprint of the SSM state.
    ssm_state_dtype: torch.dtype = torch.float32

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
        if self.ssm_state_dtype == torch.bfloat16:
            tag += "_stateBF16"
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
        model_name="Qwen3.5-tp4-1k",
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=4,
        full_prompt_len=8192,
        model_name="Qwen3.5-tp4-8k",
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=8,
        full_prompt_len=1024,
        model_name="Qwen3.5-tp8-1k",
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=8,
        full_prompt_len=8192,
        model_name="Qwen3.5-tp8-8k",
    ),
]


# Mirror every base shape with a bf16-SSM-state variant. The bf16 vs f32
# kernel paths only differ in two ``if const_expr`` branches:
#   - h0 load (gated by USE_INITIAL_STATE)
#   - ht store (gated by STORE_FINAL_STATE)
# The bf16 mirror keeps ``output_final_state`` from the base shape, so:
#   - ``_nofs`` shapes (use_h0=True, store_fs=False) cover the h0 load path
#   - default shapes (use_h0=True, store_fs=True) cover both paths
# Only ``(use_h0=False, store_fs=False)`` would generate IR identical to
# the f32 path; none of the current PREFILL_PARAMS hits that combo, so we
# do not filter here. If you add such a case later, gate the mirror with
# ``if _base.output_final_state or _make_inputs(...) provides h0``.
PREFILL_PARAMS.extend(
    [
        _dataclass_replace(_base, ssm_state_dtype=torch.bfloat16)
        for _base in list(PREFILL_PARAMS)
    ]
)


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
    ssm_state_dtype=torch.float32,
):
    if args is not None:
        tp = args.tp
        K_dim = args.K
        V_dim = args.V
        Hk_dim = args.Hk
        Hv_dim = args.Hv
        dtype = args.dtype
        is_varlen = args.is_varlen
        ssm_state_dtype = args.ssm_state_dtype

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
        # Always allocate in f32 first to keep numerical noise small for
        # references built off this tensor, then cast to the requested
        # state dtype when it differs (e.g. bf16-state path).
        initial_state = (
            torch.randn(N, H, V_dim, K_dim, dtype=torch.float32, device=device) * 0.01
        )
        if ssm_state_dtype != torch.float32:
            initial_state = initial_state.to(ssm_state_dtype)

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
    h_out,
    vn_out,
    fs_out,
    h_ref,
    vn_ref,
    fs_ref,
    *,
    output_final_state,
    label,
    atol=5e-2,
    rtol=5e-2,
):
    """Compare a K5 backend's outputs against the PyTorch FP32 reference.

    All backends in this file return VK-ordered ``h`` / ``final_state`` and
    ``v_new`` in head-major ``[B, H, T, V]`` layout (which we permute back to
    ``[B, T, H, V]`` for comparison via ``_normalize_opt_v_new``).

    The same tolerance applies to all dtypes (f32-state and bf16-state) and
    all three outputs. The bf16-state path's only extra noise relative to
    f32-state is one ``truncf`` on the final_state, which stays well within
    bf16 ULP for sane inputs and never exceeds the historical f32-state
    margins.
    """
    torch.testing.assert_close(
        h_out.float(),
        h_ref.float(),
        atol=atol,
        rtol=rtol,
        msg=f"{label}: h mismatch",
    )
    torch.testing.assert_close(
        _normalize_opt_v_new(vn_out).float(),
        vn_ref.float(),
        atol=atol,
        rtol=rtol,
        msg=f"{label}: v_new mismatch",
    )
    if output_final_state:
        torch.testing.assert_close(
            fs_out.float(),
            fs_ref.float(),
            atol=atol,
            rtol=rtol,
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
        if args.ssm_state_dtype != torch.float32:
            pytest.skip("Triton VK reference only supports f32 SSM state.")
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
        if args.ssm_state_dtype != torch.float32:
            pytest.skip("Triton KV reference only supports f32 SSM state.")
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


# -- bf16 SSM-state correctness ------------------------------------------


# A small, fast subset of shapes used to validate the bf16-state code path
# (h0 / final_state in bf16). Picked to cover both the non-varlen and varlen
# launch routes while keeping kernel JIT compile time low.
STATE_BF16_PARAMS = [
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=2,
        full_prompt_len=2500,
        model_name="Qwen3.5-35B-bf16state",
        is_varlen=False,
        output_final_state=True,
        max_num_batched_tokens=2500,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=4,
        full_prompt_len=1024,
        model_name="Qwen3.5-tp4-1k-bf16state",
        is_varlen=True,
        output_final_state=True,
        max_num_batched_tokens=8192,
    ),
]
STATE_BF16_TEST_IDS = [repr(p) for p in STATE_BF16_PARAMS]


class TestStateDtypeBF16:
    """Validate that ``state_dtype=bfloat16`` matches the ``float32`` path.

    The bf16-state kernel keeps the f32 accumulator unchanged and only
    rounds h0 (extf) and final_state (truncf) at the HBM boundary, so its
    output should agree with the f32-state kernel up to one bf16 trunc
    error on the SSM state plus accumulated round-off through the chunk
    loop. We compare against the *flydsl f32-state* path on the exact same
    shape rather than the PyTorch reference, which gives the tightest
    regression signal for this specific feature.
    """

    @pytest.mark.parametrize("args", STATE_BF16_PARAMS, ids=STATE_BF16_TEST_IDS)
    def test_state_bf16_matches_state_f32(self, args: PrefillArgs):
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, _, _, w_c, u_c, g, h0_f32, cu, _ = _make_inputs(context_lens, args=args)
        h0_bf16 = h0_f32.to(torch.bfloat16)

        h_f32, vn_f32, fs_f32 = chunk_gated_delta_rule_fwd_h_flydsl(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0_f32,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )
        h_bf16, vn_bf16, fs_bf16 = chunk_gated_delta_rule_fwd_h_flydsl(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0_bf16,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

        # final_state dtype must follow the input dtype.
        if args.output_final_state:
            assert (
                fs_f32 is not None and fs_f32.dtype == torch.float32
            ), f"f32 path produced {fs_f32.dtype} final_state"
            assert (
                fs_bf16 is not None and fs_bf16.dtype == torch.bfloat16
            ), f"bf16 path produced {fs_bf16.dtype} final_state"
        else:
            assert fs_f32 is None and fs_bf16 is None

        # h and v_new are bf16 in both paths (decoupled from state dtype).
        assert h_f32.dtype == h_bf16.dtype == k.dtype
        if vn_f32 is not None:
            assert vn_f32.dtype == vn_bf16.dtype == u_c.dtype

        # The two paths diverge only by the rounding applied to h0/ht. With
        # f32 accumulation this stays well within bf16 ULP * (1 + chunk
        # length) for sane inputs.
        atol = 5e-2
        rtol = 5e-2
        torch.testing.assert_close(
            h_bf16.float(),
            h_f32.float(),
            atol=atol,
            rtol=rtol,
            msg="bf16-state vs f32-state: h mismatch",
        )
        if vn_f32 is not None:
            torch.testing.assert_close(
                vn_bf16.float(),
                vn_f32.float(),
                atol=atol,
                rtol=rtol,
                msg="bf16-state vs f32-state: v_new mismatch",
            )
        if args.output_final_state:
            torch.testing.assert_close(
                fs_bf16.float(),
                fs_f32.float(),
                atol=atol,
                rtol=rtol,
                msg="bf16-state vs f32-state: final_state mismatch",
            )

    @pytest.mark.parametrize("args", STATE_BF16_PARAMS, ids=STATE_BF16_TEST_IDS)
    def test_state_dtype_kwarg_no_initial_state(self, args: PrefillArgs):
        """``state_dtype`` kwarg controls final_state dtype when h0 is None."""
        if not args.output_final_state:
            pytest.skip("kwarg only meaningful when final_state is requested")
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, _, _, w_c, u_c, g, _, cu, _ = _make_inputs(
            context_lens, args=args, with_initial_state=False
        )

        _, _, fs_f32 = chunk_gated_delta_rule_fwd_h_flydsl(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu,
            # default -> f32
        )
        assert fs_f32 is not None and fs_f32.dtype == torch.float32

        _, _, fs_bf16 = chunk_gated_delta_rule_fwd_h_flydsl(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu,
            state_dtype=torch.bfloat16,
        )
        assert fs_bf16 is not None and fs_bf16.dtype == torch.bfloat16

    def test_state_dtype_conflict_raises(self):
        """Mismatched ``state_dtype`` and ``initial_state.dtype`` must raise."""
        args = STATE_BF16_PARAMS[0]
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, _, _, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)
        with pytest.raises(ValueError):
            chunk_gated_delta_rule_fwd_h_flydsl(
                k,
                w_c,
                u_c,
                g=g,
                initial_state=h0,  # f32
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
                state_dtype=torch.bfloat16,  # conflict
            )

    def test_state_dtype_unsupported_raises(self):
        """Unsupported state dtypes must raise (e.g. fp16)."""
        args = STATE_BF16_PARAMS[0]
        context_lens = _build_context_lens(
            args.full_prompt_len, args.max_num_batched_tokens
        )
        k, _, _, w_c, u_c, g, _, cu, _ = _make_inputs(
            context_lens, args=args, with_initial_state=False
        )
        with pytest.raises(ValueError):
            chunk_gated_delta_rule_fwd_h_flydsl(
                k,
                w_c,
                u_c,
                g=g,
                initial_state=None,
                output_final_state=True,
                cu_seqlens=cu,
                state_dtype=torch.float16,
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

        # Triton K5 host wrappers only accept f32 ``initial_state`` and always
        # produce an f32 ``final_state``. When FlyDSL is benched with a bf16
        # SSM state, we still want a Triton baseline for comparison, so we
        # promote h0 to f32 once (outside the timed window) and feed it to
        # the Triton closures. The resulting "Triton(f32) vs FlyDSL(bf16)"
        # row answers the practical question "how much does enabling
        # bf16-state win against the existing Triton baseline?".
        h0_triton_vk = (
            h0.float() if (h0 is not None and h0.dtype != torch.float32) else h0
        )
        h0_kv = (
            h0_triton_vk.transpose(-2, -1).contiguous()
            if h0_triton_vk is not None
            else None
        )

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
                initial_state=h0_triton_vk,
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
                "state": "bf16" if args.ssm_state_dtype == torch.bfloat16 else "fp32",
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

    # Columns shared by both per-state tables. ``state`` is omitted because
    # each subtable describes a single SSM-state dtype in its title.
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
    border = "=" * len(header)

    def _fmt_row(row):
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
        return " | ".join(cells)

    # Bucket rows by SSM-state dtype, keeping each bucket's ordering
    # consistent with the original ``_perf_results.append`` order so that
    # rows line up with the parametrize id order.
    rows_fp32 = [r for r in _perf_results if r["state"] == "fp32"]
    rows_bf16 = [r for r in _perf_results if r["state"] == "bf16"]

    lines = ["", border]
    lines.append(
        "K5 Prefill Performance Summary "
        "(K5 device kernel time only, via torch.profiler)"
    )
    lines.append(
        "  Triton K5 references always use fp32 SSM state; only FlyDSL's "
        "SSM-state dtype changes between the two tables below."
    )
    lines.append(border)

    def _emit_subtable(title, rows):
        if not rows:
            return
        lines.append("")
        lines.append(title)
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for row in rows:
            lines.append(_fmt_row(row))
        lines.append(sep)

    _emit_subtable("[FlyDSL SSM state = fp32]", rows_fp32)
    _emit_subtable("[FlyDSL SSM state = bf16]", rows_bf16)
    lines.append("")
    print("\n".join(lines))


@pytest.fixture(scope="session", autouse=True)
def _print_summary_table(request):
    """Print the summary performance table after all tests finish."""
    yield
    _print_perf_table()
