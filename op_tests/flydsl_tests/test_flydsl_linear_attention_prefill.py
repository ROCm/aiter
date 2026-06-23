# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention Prefill (chunk_gated_delta_h) regressions.

Usage:
    rm -rf ~/.triton/cache
    export GATED_DELTA_RULE_TRITON_AUTOTUNE=1
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=7 pytest -sv op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -s
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=7 python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "varlen-32k-aws" -v -s
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import triton
import triton.language as tl
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
        chunk_gated_delta_rule_fwd_h_flydsl_kv,
        chunk_gated_delta_rule_fwd_h_flydsl_vk,
        chunk_gated_delta_rule_fwd_h_flydsl_vk_naive,
        chunk_gated_delta_rule_fwd_h_flydsl_kv_naive,
        chunk_gated_delta_rule_fwd_h_flydsl_naive,
        chunk_gated_delta_rule_fwd_h_flydsl_naive_opt,
        chunk_gated_delta_rule_fwd_h_flydsl_hipport,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h_opt_vk,
    )
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL Linear Attention Prefill kernels: {exc}",
        allow_module_level=True,
    )

try:
    from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h as chunk_gated_delta_rule_fwd_h_vllm,
    )

    _HAS_VLLM_K5 = True
except Exception:
    chunk_gated_delta_rule_fwd_h_vllm = None
    _HAS_VLLM_K5 = False

# HIP/C++ K5 (chunk_gated_delta_rule_fwd_h.cu). JIT-compiled on first call.
# Same public VK outputs as the FlyDSL / Triton opt_vk backends, but it
# requires K=V=128 + bf16 inputs, so cases that violate that are skipped
# in the correctness test and excluded from the perf launch.
try:
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_hip_fn,
    )

    _HAS_HIP_K5 = True
except Exception:
    chunk_gated_delta_rule_fwd_h_hip_fn = None
    _HAS_HIP_K5 = False

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
    # If set, override ``_build_context_lens(full_prompt_len,
    # max_num_batched_tokens)`` and use these segment lengths verbatim.
    # Used by trace-derived ragged-batch cases (e.g. the prefill_gdr.log
    # 407-shape set imported below) that cannot be expressed as the
    # "k equal segments + remainder" recipe ``_build_context_lens``
    # produces. ``None`` (the default) preserves the existing behavior
    # for every hand-written ``PrefillGroup`` row.
    context_lens: object = None  # list[int] | None
    # Free-form tag used in __repr__ when ``context_lens`` is set, so
    # parametrized-test IDs stay short and unique even when many trace
    # shapes share the same ``(T, num_seqs)``. Typical values are a log
    # count or a hex digest of cu_seqlens.
    trace_tag: str = ""

    @property
    def Hg(self):
        return self.Hk // self.tp

    @property
    def H(self):
        return self.Hv // self.tp

    def resolve_context_lens(self):
        """Return the per-segment token counts this case wants.

        For trace-derived cases this is the ``cu_seqlens`` diff list
        captured from the source workload; for hand-written cases it is
        the equal-length recipe ``_build_context_lens`` emits.
        """
        if self.context_lens is not None:
            return list(self.context_lens)
        return _build_context_lens(self.full_prompt_len, self.max_num_batched_tokens)

    def __repr__(self):
        # Trace-derived cases have a bespoke cu_seqlens; surface enough
        # to identify the shape but elide the cu_seqlens themselves
        # (they can be 64+ entries long).
        if self.context_lens is not None:
            n = len(self.context_lens)
            T = sum(self.context_lens)
            tag = self.model_name or "trace"
            tag += f"_T{T}_n{n}"
            if self.trace_tag:
                tag += f"_{self.trace_tag}"
            return tag
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


@dataclass
class PrefillGroup:
    """A compact spec for a family of ``PrefillArgs`` cases that share every
    field except ``tp`` and ``full_prompt_len``.

    ``expand_groups`` takes a list of these and returns the flat
    ``PrefillArgs`` list that ``pytest.parametrize`` consumes. For each
    group, the (tps x full_prompt_lens) Cartesian product is materialised,
    and ``max_num_batched_tokens`` defaults to ``full_prompt_len`` when not
    explicitly set (matches the existing per-case behavior of the
    non-varlen rows). varlen/fs cases that previously left
    ``max_num_batched_tokens`` at its dataclass default (32768) can omit
    it here too.

    The display tag still encodes (tp, full_prompt_len) via
    ``PrefillArgs.__repr__``, so pytest IDs stay unique even when several
    expanded cases share the same ``model_name``.
    """

    model_name: str
    Hv: int
    tps: list
    full_prompt_lens: list
    Hk: int = 16
    K: int = 128
    V: int = 128
    BT: int = 64
    dtype: torch.dtype = torch.bfloat16
    is_varlen: bool = True
    output_final_state: bool = True
    ssm_state_dtype: torch.dtype = torch.float32
    # Three semantics for ``max_num_batched_tokens``:
    #   - int : use this exact value for every expanded case (e.g. you want
    #           a fixed scheduler budget across a sweep of full_prompt_len).
    #   - "full_prompt_len" : tie it to each case's full_prompt_len. The
    #           original non-varlen Qwen3.5-35B / 397B rows wrote
    #           ``max_num_batched_tokens=full_prompt_len`` explicitly, which
    #           makes ``_build_context_lens`` return exactly one segment.
    #   - None (default) : fall back to the ``PrefillArgs`` dataclass
    #           default (32768). The original varlen rows omitted this
    #           field, so they implicitly used 32768 -- which makes
    #           ``_build_context_lens(1024, 32768)`` produce 32 segments of
    #           length 1024. Preserving that behavior is what keeps the
    #           varlen path's per-case shape unchanged across this refactor.
    max_num_batched_tokens: object = None
    # Optional "trace-derived 3-segment" expansion knob. When set, each
    # expanded case overrides ``_build_context_lens`` with the explicit
    # 3-segment layout ``[head, mid_seqlen, full_prompt_len - head - mid_seqlen]``,
    # i.e. cu_seqlens = [0, head, head + mid_seqlen, full_prompt_len].
    # This reproduces the worst K5 regression family found in bench
    # results 20260603 (n=3, T ~= 16384, middle segment == 10000): the
    # K5 kernel exhibits a near-constant ~543us cost across this whole
    # cluster regardless of head_seqlen, while triton K5 varies with the
    # head split between ~460-495us. Sweeping head_seqlens lets us probe
    # the kernel's sensitivity (or lack thereof) to the head boundary.
    # Group is materialised as the (tps x full_prompt_lens x head_seqlens)
    # Cartesian product when this is not None.
    head_seqlens: object = None  # list[int] | None
    mid_seqlen: int = 10000
    # Number of segments per expanded case when ``head_seqlens`` is set:
    #   num_segments=3 (default): context_lens = [head, mid_seqlen, full_len-head-mid_seqlen]
    #     -> cu_seqlens = [0, head, head+mid_seqlen, full_len]   (n=3)
    #   num_segments=2          : context_lens = [head, full_len-head]
    #     -> cu_seqlens = [0, head, full_len]                    (n=2)
    #     ``mid_seqlen`` is ignored in this mode; the tail length is whatever
    #     remains after ``head``. Used to cover the n=2 T=16384 regression
    #     clusters (head near 6400 / 8192 / 9912 / 10000) found in the
    #     bench_gdr 20260604 trace.
    num_segments: int = 3


def expand_groups(groups):
    out = []
    for g in groups:
        for tp in g.tps:
            for full_len in g.full_prompt_lens:
                if g.max_num_batched_tokens == "full_prompt_len":
                    mnbt = full_len
                elif g.max_num_batched_tokens is None:
                    mnbt = 32768  # PrefillArgs dataclass default
                else:
                    mnbt = g.max_num_batched_tokens

                # head_seqlens=None : preserve the original "equal split via
                # _build_context_lens" behavior. Otherwise materialise one
                # PrefillArgs per (tp, full_len, head) triple with an
                # explicit 3-segment cu_seqlens layout
                # [head, mid_seqlen, full_len - head - mid_seqlen].
                if g.head_seqlens is None:
                    out.append(
                        PrefillArgs(
                            K=g.K,
                            V=g.V,
                            Hk=g.Hk,
                            Hv=g.Hv,
                            tp=tp,
                            full_prompt_len=full_len,
                            model_name=g.model_name,
                            BT=g.BT,
                            max_num_batched_tokens=mnbt,
                            dtype=g.dtype,
                            is_varlen=g.is_varlen,
                            output_final_state=g.output_final_state,
                            ssm_state_dtype=g.ssm_state_dtype,
                        )
                    )
                else:
                    for head in g.head_seqlens:
                        if g.num_segments == 2:
                            tail = full_len - head
                            if tail <= 0:
                                raise ValueError(
                                    f"head_seqlens (num_segments=2) produced "
                                    f"non-positive tail ({tail}) for "
                                    f"group={g.model_name!r} "
                                    f"full_prompt_len={full_len} head={head}."
                                )
                            context_lens = [head, tail]
                            tag = f"head{head}_tail{tail}"
                        elif g.num_segments == 3:
                            tail = full_len - head - g.mid_seqlen
                            if tail <= 0:
                                raise ValueError(
                                    f"head_seqlens (num_segments=3) produced "
                                    f"non-positive tail ({tail}) for "
                                    f"group={g.model_name!r} "
                                    f"full_prompt_len={full_len} head={head} "
                                    f"mid_seqlen={g.mid_seqlen}. Drop this "
                                    f"(full_len, head) combo or raise "
                                    f"full_prompt_len."
                                )
                            context_lens = [head, g.mid_seqlen, tail]
                            tag = f"head{head}_mid{g.mid_seqlen}"
                        else:
                            raise ValueError(
                                f"num_segments={g.num_segments} unsupported; "
                                f"only 2 or 3 are implemented."
                            )
                        out.append(
                            PrefillArgs(
                                K=g.K,
                                V=g.V,
                                Hk=g.Hk,
                                Hv=g.Hv,
                                tp=tp,
                                full_prompt_len=full_len,
                                model_name=g.model_name,
                                BT=g.BT,
                                max_num_batched_tokens=mnbt,
                                dtype=g.dtype,
                                is_varlen=g.is_varlen,
                                output_final_state=g.output_final_state,
                                ssm_state_dtype=g.ssm_state_dtype,
                                context_lens=context_lens,
                                trace_tag=tag,
                            )
                        )
    return out


_PREFILL_GROUPS = [
    # non-varlen + no final state (Qwen3.5-35B family, Hv=32).
    # Original rows set max_num_batched_tokens == full_prompt_len so that
    # _build_context_lens emits exactly one segment of length full_prompt_len.
    PrefillGroup(
        model_name="Qwen3.5-35B",
        Hv=32,
        tps=[1, 2],
        full_prompt_lens=[2500, 60000, 128000],
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens="full_prompt_len",
    ),
    # non-varlen + no final state (Qwen3.5-397B family, Hv=64).
    PrefillGroup(
        model_name="Qwen3.5-397B",
        Hv=64,
        tps=[1, 2],
        full_prompt_lens=[2500, 60000, 128000],
        is_varlen=False,
        output_final_state=False,
        max_num_batched_tokens="full_prompt_len",
    ),
    # varlen + final_state (default path), TP=4 / TP=8 share everything
    # else, so they collapse into a single group. Original rows left
    # max_num_batched_tokens at the PrefillArgs default of 32768, which
    # makes _build_context_lens slice 32768 into ceil(32768/full_len)
    # equal-length segments (e.g. 32 segments of length 1024 for the
    # 1k row). Keeping ``max_num_batched_tokens=None`` here preserves that.
    PrefillGroup(
        model_name="varlen-32k-qwen",
        Hv=64,
        tps=[4, 8],
        full_prompt_lens=[1024, 2048, 4096, 8192],
        max_num_batched_tokens=32768,
    ),
    PrefillGroup(
        model_name="varlen-16k-aws",
        Hv=32,
        tps=[1],
        full_prompt_lens=[1000, 5000, 10000],
        max_num_batched_tokens=16384,
    ),
    PrefillGroup(
        model_name="varlen-32k-aws",
        Hv=32,
        tps=[1],
        # full_prompt_lens=[1000, 5000, 10000],
        full_prompt_lens=[1000],
        max_num_batched_tokens=32768,
    ),
    PrefillGroup(
        model_name="flydsl-k5-n1",
        Hv=32,
        tps=[1],
        full_prompt_lens=[5000, 10000],
        max_num_batched_tokens="full_prompt_len",
    ),
    PrefillGroup(
        model_name="flydsl-k5-n3-mid10k",
        Hv=32,
        tps=[1],
        full_prompt_lens=[16384],
        max_num_batched_tokens=16384,
        head_seqlens=[5, 10, 65, 704, 936, 1820, 4467, 5508],
        mid_seqlen=10000,
    ),
    PrefillGroup(
        model_name="flydsl-k5-n2-16k",
        Hv=32,
        tps=[1],
        full_prompt_lens=[16384],
        max_num_batched_tokens=16384,
        head_seqlens=[4000, 6396, 8192, 9912, 10000],
        num_segments=2,
    ),
]

PREFILL_PARAMS = expand_groups(_PREFILL_GROUPS)


PREFILL_TEST_IDS = [repr(p) for p in PREFILL_PARAMS]


# -- bf16 SSM-state params (paired with TestStateDtypeBF16 below) ------

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
    # g is head-major [H, T_total] (matches Triton VK / HIP / FlyDSL K5).
    # cumsum is along the T dim.
    g = torch.randn(H, T_total, dtype=torch.float32, device=device).abs() * -0.5
    g = g.cumsum(dim=1)

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
                    g_last = g[i_h, last_idx].float()
                    g_chunk = g[i_h, bos + t_start : bos + t_end].float()

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


def _hip_k5_supported(args: PrefillArgs) -> bool:
    """The HIP K5 kernel only handles K=V=128, bf16 inputs, chunk_size=64."""
    return (
        _HAS_HIP_K5
        and args.K == 128
        and args.V == 128
        and args.dtype == torch.bfloat16
        and args.BT == 64
    )


def chunk_gated_delta_rule_fwd_h_hip_k5(
    k,
    w,
    u,
    g=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """HIP/C++ K5 host wrapper, adapted to this file's K5 calling convention.

    Mirrors the FlyDSL / Triton ``opt_vk`` backends: takes the GQA-layout
    ``k`` ([B, T, Hg, K]), head-major ``w`` / ``u`` ([B, H, T, K/V]), and a
    head-major cumulative-gate ``g`` ([H, T_total] or [B, H, T_total]) in
    natural-log space, and returns VK-ordered ``h`` ([B, NT, H, V, K]),
    head-major ``v_new`` ([B, H, T, V]), and VK ``final_state``
    ([N, H, V, K]) -- identical public outputs to the other backends, so
    the shared ``_assert_k5_outputs_match_ref`` comparator applies directly.

    The underlying kernel's ``USE_EXP2`` path expects log2-space gates, so we
    pass ``use_exp2=False`` here to keep the natural-log-space ``g`` contract
    shared with the PyTorch reference (the kernel then applies the LOG2E
    scale internally).
    """
    B = w.shape[0]
    H = w.shape[1]
    T_flat = w.shape[2]

    # The HIP wrapper wants a 3-D head-major g [B, H, T_flat]. This file
    # produces a 2-D [H, T_total] gate for the B=1 varlen / dense cases.
    if g is not None:
        if g.dim() == 2:
            g_hip = g.reshape(1, H, T_flat).contiguous()
        else:
            g_hip = g.contiguous()
    else:
        g_hip = None

    return chunk_gated_delta_rule_fwd_h_hip_fn(
        k,
        w,
        u,
        g=g_hip,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        use_exp2=False,
        g_head_major=True,
    )


# -- Performance benchmark ----------------------------------------------


_K5_KERNEL_PREFIXES = [
    "chunk_gdn_fwd_h_flydsl_vk",
    "chunk_gdn_fwd_h_flydsl_kv",
    "chunk_gdn_fwd_h_flydsl_naive",
    "chunk_gated_delta_rule_fwd_kernel_h",
]

# The HIP/C++ K5 kernel is a templated __global__ whose profiler symbol is
# either the demangled ``...chunk_gated_delta_rule_fwd_h_hip_kernel<...>`` or
# a mangled ``_ZN...`` form. Match it as a substring (the templated name never
# appears at offset 0 after demangling because of the leading return type).
_K5_KERNEL_SUBSTRINGS = [
    "chunk_gated_delta_rule_fwd_h_hip_kernel",
]


def _is_k5_kernel(name: str) -> bool:
    """Return True if *name* is a K5 hidden-state recurrence kernel."""
    if any(name.startswith(p) for p in _K5_KERNEL_PREFIXES):
        return True
    return any(s in name for s in _K5_KERNEL_SUBSTRINGS)


def _bench_fn(fn, *args, **kwargs):
    """Average per-iter K5 kernel time (us) via torch.profiler.

    Only counts kernels whose name matches ``_K5_KERNEL_PREFIXES``
    (chunk_gdn_fwd_h_flydsl_vk, chunk_gated_delta_rule_fwd_kernel_h*).
    This excludes memset, dtype-cast, and any other non-K5 GPU work.
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
        if _is_k5_kernel(evt.key):
            total_us += evt.self_device_time_total / NUM_ITERS
    return total_us


# -- Correctness tests ---------------------------------------------------


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
        context_lens = args.resolve_context_lens()
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

    @pytest.mark.skipif(not _HAS_HIP_K5, reason="HIP K5 kernel not importable")
    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_hip(self, args: PrefillArgs):
        """HIP/C++ K5 impl. Same public VK outputs as the baseline flydsl
        path. The kernel only supports K=V=128 + bf16 + chunk_size=64, so
        cases outside that envelope are skipped."""
        if not _hip_k5_supported(args):
            pytest.skip(
                reason="HIP K5 kernel requires K=V=128, bfloat16, chunk_size=64"
            )
        # Pin the seed so the bf16 v_new comparison is reproducible: a few
        # ``u - w @ h`` entries cancel to near-zero, where the rtol band is
        # tiny and an unlucky random draw can push bf16 round-off past it.
        torch.manual_seed(42)
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_hip, vn_hip, fs_hip = chunk_gated_delta_rule_fwd_h_hip_k5(
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
            h_hip,
            vn_hip,
            fs_hip,
            h_ref,
            vn_ref,
            fs_ref,
            output_final_state=args.output_final_state,
            label="hip",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl_kv(self, args: PrefillArgs):
        """Separate KV-layout FlyDSL K5 impl (VWARP 16x16x16 + 3-wave +
        coalesced KV h-store). Same VK public outputs as the baseline flydsl
        path; only the BV==64 configs exercise the KV kernel, others fall back."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_kv(
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
            label="flydsl_kv",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl_vk(self, args: PrefillArgs):
        """Separate VK-layout FlyDSL K5 impl (forked from the KV variant;
        VWARP 16x16x16 + 3-wave). Same VK public outputs as the baseline
        flydsl path; only the BV==64 configs exercise the VK kernel, others
        fall back."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_vk(
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
            label="flydsl_vk",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl_vk_naive(self, args: PrefillArgs):
        """Naive (un-pipelined) VK-fork FlyDSL K5 impl (same VWARP layout +
        coalesced VK h-store as flydsl_vk, all prefetch/pipeline scheduling
        removed). Same VK public outputs as the baseline flydsl path; only the
        BV==64 configs exercise the naive VK kernel, others fall back."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_vk_naive(
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
            label="flydsl_vk_naive",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_hipport(self, args: PrefillArgs):
        """HIP-PORT FlyDSL K5 impl: detail-for-detail replica of the hand-tuned
        HIP/C++ kernel's split-M scheme (K=16 mfma, register h-state, coalesced
        VK transpose store, software-pipeline panel prefetch). Same public VK
        outputs as the baseline flydsl path; only the BV==64 configs exercise
        the hipport kernel, others fall back to the baseline."""
        # Pin the seed so the bf16 v_new comparison is reproducible: a few
        # ``u - w @ h`` entries cancel to near-zero, where the rtol band is
        # tiny and an unlucky random draw can push bf16 round-off past it
        # (mirrors test_correctness_hip).
        torch.manual_seed(42)
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_hipport(
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
            label="hipport",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl_naive(self, args: PrefillArgs):
        """Naive (un-pipelined) fork of the BASELINE FlyDSL K5 kernel: same
        baseline layout / VK public outputs, with all prefetch / software-
        pipeline scheduling removed (cross-chunk w prefetch, OPT-VC g/gk/u
        emitter queue, OPT-K / OPT-W interleaves). Works at any BV (not gated
        on BV==64). Used to baseline the raw bottleneck structure in a trace."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_naive(
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
            label="flydsl_naive",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl_naive_opt(self, args: PrefillArgs):
        """Naive-OPT fork: the naive kernel with the OPT-DGL w-load (direct
        HBM->LDS buffer_load_lds + two-sided XOR swizzle) forced on. Same
        public VK outputs / numerics as the naive fork; only the w staging
        path differs (no ds_write / no VGPR staging for w)."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_naive_opt(
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
            label="flydsl_naive_opt",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl_kv_naive(self, args: PrefillArgs):
        """Naive (un-pipelined) KV-fork FlyDSL K5 impl (same VWARP layout +
        coalesced KV h-store as flydsl_kv, all prefetch/pipeline scheduling
        removed). Same VK public outputs as the baseline flydsl path; only the
        BV==64 configs exercise the naive KV kernel, others fall back."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_kv_naive(
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
            label="flydsl_kv_naive",
        )

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_triton_vk(self, args: PrefillArgs):
        """Triton VK K5 (h: [V, K]) -- same input/output layout as FlyDSL."""
        if args.ssm_state_dtype != torch.float32:
            pytest.skip("Triton VK reference only supports f32 SSM state.")
        context_lens = args.resolve_context_lens()
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

    @pytest.mark.skipif(
        not _HAS_VLLM_K5,
        reason="vllm.model_executor.layers.fla.ops.chunk_delta_h not importable",
    )
    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_vllm(self, args: PrefillArgs):
        """vLLM upstream K5 (h: [V, K]) -- same input/output layout as FlyDSL.

        vLLM's ``chunk_gated_delta_rule_fwd_h`` is the FLA upstream port that
        powers ``vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule``,
        and shares the ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``
        kernel source with aiter's ``opt_vk``. We still cover it explicitly
        to catch upstream version drift (e.g. signature, scaling factors,
        or default chunk_size changes that would not show up in the aiter
        ``opt_vk`` test above).
        """
        if args.ssm_state_dtype != torch.float32:
            pytest.skip("vLLM K5 reference only supports f32 SSM state.")
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        # vLLM's host wrapper infers ``H = u.shape[-2]`` and ``T = k.shape[1]``
        # (T-major), so it needs the un-permuted ``w_orig`` / ``u_orig`` of
        # shape ``[B, T, H, *]``, NOT the H-major ``w_c`` / ``u_c`` that
        # aiter's ``opt_vk`` consumes. Feeding ``u_c`` would make vLLM think
        # ``H = T`` and try to allocate ``(B, NT, T, V, K)`` for ``h``
        # (terabytes for long contexts). vLLM supports GQA natively, so we
        # also do NOT repeat_interleave ``k``.
        # vLLM's upstream K5 host wrapper indexes ``g`` as token-major
        # ``[T_total, H]`` (FLA's original layout). Our test fixture now
        # produces ``g`` in head-major ``[H, T_total]`` (matching the
        # aiter / FlyDSL K5 convention), so transpose+contiguous it back
        # to token-major before launching vLLM.
        g_token_major = g.transpose(0, 1).contiguous()
        h_vllm, vn_vllm, fs_vllm = chunk_gated_delta_rule_fwd_h_vllm(
            k,
            w_orig,
            u_orig,
            g=g_token_major,
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

        # vLLM's ``v_new = empty_like(u_orig)`` is already T-major [B, T, H, V],
        # matching ``vn_ref`` directly -- we must NOT route through
        # ``_assert_k5_outputs_match_ref`` because that helper calls
        # ``_normalize_opt_v_new`` (permute 0,2,1,3) on the assumption that the
        # K5 returned ``v_new`` in head-major [B, H, T, V] layout (aiter's
        # convention). ``h`` and ``final_state`` follow the same vk layout as
        # the aiter ``opt_vk`` path, so we compare them directly.
        atol, rtol = 5e-2, 5e-2
        torch.testing.assert_close(
            h_vllm.float(),
            h_ref.float(),
            atol=atol,
            rtol=rtol,
            msg="vllm: h mismatch",
        )
        torch.testing.assert_close(
            vn_vllm.float(),
            vn_ref.float(),
            atol=atol,
            rtol=rtol,
            msg="vllm: v_new mismatch",
        )
        if args.output_final_state:
            torch.testing.assert_close(
                fs_vllm.float(),
                fs_ref.float(),
                atol=atol,
                rtol=rtol,
                msg="vllm: final_state mismatch",
            )
        else:
            assert fs_vllm is None, "vllm: expected None final_state"

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_triton_origin_opt(self, args: PrefillArgs):
        """triton_origin_opt K5: standalone fwd_h (BV=16 + exp2) variant.

        Same kernel as triton_origin but with BV=16 + USE_EXP2 (the
        ``new pipeline`` config from ``0423_gdr_prefill_bench_standalone.py``).
        Unlike triton_origin this K5 supports GQA natively (the kernel
        takes ``Hg`` as a constexpr), so the GQA expand step is skipped.
        """
        if args.ssm_state_dtype != torch.float32:
            pytest.skip("triton_origin_opt reference only supports f32 SSM state.")
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        # GQA-aware K5: no repeat_interleave needed. Hidden state in [K,V]
        # layout (same as triton_origin), so use h0 transposed from the
        # VK reference layout.
        h0_kv = h0.transpose(-2, -1).contiguous() if h0 is not None else None

        h_origin_opt, vn_origin_opt, fs_origin_opt = (
            chunk_gated_delta_rule_fwd_h_origin_opt(
                k,
                w_orig,
                u_orig,
                g=g,
                initial_state=h0_kv,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )
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

        h_origin_opt_vk = h_origin_opt.transpose(-2, -1).contiguous()
        fs_origin_opt_vk = (
            fs_origin_opt.transpose(-2, -1).contiguous()
            if fs_origin_opt is not None
            else None
        )

        atol, rtol = 5e-2, 5e-2
        torch.testing.assert_close(
            h_origin_opt_vk.float(),
            h_ref.float(),
            atol=atol,
            rtol=rtol,
            msg="triton_origin_opt: h mismatch",
        )
        torch.testing.assert_close(
            vn_origin_opt.float(),
            vn_ref.float(),
            atol=atol,
            rtol=rtol,
            msg="triton_origin_opt: v_new mismatch",
        )
        if args.output_final_state:
            torch.testing.assert_close(
                fs_origin_opt_vk.float(),
                fs_ref.float(),
                atol=atol,
                rtol=rtol,
                msg="triton_origin_opt: final_state mismatch",
            )


_perf_results: list[dict] = []


def _run_perf_comparison(args: PrefillArgs):
    """Per-shape K5 perf body used by ``TestPerformance``.

    Bench 4 backends (FlyDSL, Triton opt_vk, Triton origin_opt, vLLM)
    under identical inputs and append a row to ``_perf_results``. The
    summary table is printed once at session teardown by
    ``_print_summary_table``.
    """
    context_lens = args.resolve_context_lens()
    k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)
    total_tokens = sum(context_lens)

    # Triton K5 host wrappers only accept f32 ``initial_state`` and always
    # produce an f32 ``final_state``. When FlyDSL is benched with a bf16
    # SSM state, we still want a Triton baseline for comparison, so we
    # promote h0 to f32 once (outside the timed window) and feed it to
    # the Triton closures. The resulting "Triton(f32) vs FlyDSL(bf16)"
    # row answers the practical question "how much does enabling
    # bf16-state win against the existing Triton baseline?".
    h0_triton_vk = h0.float() if (h0 is not None and h0.dtype != torch.float32) else h0

    # triton_origin_opt uses a [K, V] hidden-state layout, so its h0
    # is the VK reference h0 transposed on the last two dims.
    h0_origin_kv = (
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

    def flydsl_kv_launch():
        chunk_gated_delta_rule_fwd_h_flydsl_kv(
            k=k,
            w=w_c,
            u=u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    def flydsl_vk_launch():
        chunk_gated_delta_rule_fwd_h_flydsl_vk(
            k=k,
            w=w_c,
            u=u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    def flydsl_vk_naive_launch():
        chunk_gated_delta_rule_fwd_h_flydsl_vk_naive(
            k=k,
            w=w_c,
            u=u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    def flydsl_kv_naive_launch():
        chunk_gated_delta_rule_fwd_h_flydsl_kv_naive(
            k=k,
            w=w_c,
            u=u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    def flydsl_naive_launch():
        chunk_gated_delta_rule_fwd_h_flydsl_naive(
            k=k,
            w=w_c,
            u=u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    def flydsl_naive_opt_launch():
        chunk_gated_delta_rule_fwd_h_flydsl_naive_opt(
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

    # vLLM upstream K5 (chunk_delta_h.chunk_gated_delta_rule_fwd_h). Uses
    # the same vk hidden-state layout and same final_state dtype as the
    # aiter ``opt_vk`` wrapper, but unlike ``opt_vk`` its host wrapper
    # infers ``H = u.shape[-2]`` (T-major), so it requires the un-permuted
    # ``w_orig`` / ``u_orig`` -- feeding the H-major ``w_c`` / ``u_c``
    # would make vLLM compute ``H = T`` and try to allocate a
    # terabyte-sized ``h``. vLLM supports GQA natively, so ``k`` is also
    # passed un-expanded. We still feed the same h0_triton_vk (fp32) as
    # ``triton_vk_launch`` because vk hidden-state layout is identical.
    def vllm_launch():
        chunk_gated_delta_rule_fwd_h_vllm(
            k=k,
            w=w_orig,
            u=u_orig,
            g=g,
            initial_state=h0_triton_vk,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    def triton_origin_opt_launch():
        # GQA-aware (uses unexpanded k) BV=16 + exp2 variant of fwd_h
        # from the standalone bench's new pipeline. Hidden state in
        # [K,V] layout, so it needs h0_origin_kv (h0 transposed from
        # the VK reference layout).
        chunk_gated_delta_rule_fwd_h_origin_opt(
            k=k,
            w=w_orig,
            u=u_orig,
            g=g,
            initial_state=h0_origin_kv,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    # HIP/C++ K5. Same VK hidden-state layout as the flydsl backends, so it
    # consumes the same h0 (fp32) and head-major w_c / u_c. Only valid for
    # K=V=128 + bf16; gated by ``_hip_k5_supported`` below.
    hip_supported = _hip_k5_supported(args)

    def hip_launch():
        chunk_gated_delta_rule_fwd_h_hip_k5(
            k=k,
            w=w_c,
            u=u_c,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
        )

    # Warmup FlyDSL once so its internal BV-autotune sweep does not
    # leak into the timed window. Triton's own ``triton.autotune`` is
    # already absorbed by ``_bench_fn``'s NUM_WARMUP=5 prelude, except
    # for vLLM upstream's K5 -- its first call also runs a BV/warps/
    # stages sweep and the 5-iter prelude is not always enough to
    # converge the autotuner on long-T shapes. Pre-warm it once here
    # for parity with FlyDSL so that ``us_vllm`` reflects steady-state
    # kernel time, not the autotune sweep.
    flydsl_launch()
    flydsl_kv_launch()
    flydsl_vk_launch()
    flydsl_vk_naive_launch()
    flydsl_kv_naive_launch()
    flydsl_naive_launch()
    flydsl_naive_opt_launch()
    if _HAS_VLLM_K5:
        vllm_launch()
    if hip_supported:
        hip_launch()
    torch.cuda.synchronize()

    us_fly = _bench_fn(flydsl_launch)
    us_fly_kv = _bench_fn(flydsl_kv_launch)
    us_fly_vk = _bench_fn(flydsl_vk_launch)
    us_fly_vk_naive = _bench_fn(flydsl_vk_naive_launch)
    us_fly_kv_naive = _bench_fn(flydsl_kv_naive_launch)
    us_fly_naive = _bench_fn(flydsl_naive_launch)
    us_fly_naive_opt = _bench_fn(flydsl_naive_opt_launch)
    us_triton_vk = _bench_fn(triton_vk_launch)
    us_triton_origin_opt = _bench_fn(triton_origin_opt_launch)
    us_vllm = _bench_fn(vllm_launch) if _HAS_VLLM_K5 else float("nan")
    us_hip = _bench_fn(hip_launch) if hip_supported else float("nan")

    fly_vs_vk = us_triton_vk / us_fly if us_fly > 0 else float("inf")
    fly_vs_origin_opt = us_triton_origin_opt / us_fly if us_fly > 0 else float("inf")
    fly_vs_vllm = (
        us_vllm / us_fly if (us_fly > 0 and us_vllm == us_vllm) else float("nan")
    )
    # HIP vs FlyDSL ratio (>1 means HIP is slower than the FlyDSL baseline).
    hip_vs_fly = (
        us_hip / us_fly if (us_fly > 0 and us_hip == us_hip) else float("nan")
    )

    _perf_results.append(
        {
            "Model": args.model_name or "-",
            "TP": args.tp,
            "Hg": args.Hg,
            "H": args.H,
            "SeqLen": args.full_prompt_len,
            "T": total_tokens,
            "varlen": args.is_varlen,
            "final_st": args.output_final_state,
            "FlyDSL_vk(us)": us_fly,
            "FlyDSL_kv(us)": us_fly_kv,
            "FlyDSL_kvnaive(us)": us_fly_kv_naive,
            "FlyDSL_vkfork(us)": us_fly_vk,
            "FlyDSL_vknaive(us)": us_fly_vk_naive,
            "FlyDSL_naive(us)": us_fly_naive,
            "FlyDSL_naive_opt(us)": us_fly_naive_opt,
            "Triton_vk(us)": us_triton_vk,
            "Triton_origin_opt(us)": us_triton_origin_opt,
            "vLLM_vk(us)": us_vllm,
            "HIP_vk(us)": us_hip,
            "flydsl_vs_vk": fly_vs_vk,
            "flydsl_vs_origin_opt": fly_vs_origin_opt,
            "flydsl_vs_vllm": fly_vs_vllm,
            "hip_vs_flydsl": hip_vs_fly,
        }
    )


class TestPerformance:
    """Kernel-only performance comparison: FlyDSL vs Triton opt_vk vs Triton opt3_kv."""

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_perf_comparison(self, args: PrefillArgs):
        _run_perf_comparison(args)


_naive_opt_perf_results: list[dict] = []


class TestNaiveOptPerformance:
    """Focused A/B: naive vs naive_opt (DGL w-load) device kernel time.

    Isolates the OPT-DGL w-load win by benching the two forks on identical
    inputs. The DGL fork should not be slower than the naive fork; we assert
    a small regression guard band and record the measured speedup for the
    summary table printed at session teardown.
    """

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_perf_naive_vs_naive_opt(self, args: PrefillArgs):
        context_lens = args.resolve_context_lens()
        k, _, _, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)

        def naive_launch():
            chunk_gated_delta_rule_fwd_h_flydsl_naive(
                k=k,
                w=w_c,
                u=u_c,
                g=g,
                initial_state=h0,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )

        def naive_opt_launch():
            chunk_gated_delta_rule_fwd_h_flydsl_naive_opt(
                k=k,
                w=w_c,
                u=u_c,
                g=g,
                initial_state=h0,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )

        naive_us = _bench_fn(naive_launch)
        naive_opt_us = _bench_fn(naive_opt_launch)
        speedup = naive_us / naive_opt_us if naive_opt_us > 0 else float("nan")

        _naive_opt_perf_results.append(
            {
                "id": repr(args),
                "naive_us": naive_us,
                "naive_opt_us": naive_opt_us,
                "speedup": speedup,
            }
        )

        # Regression guard: DGL must not be materially slower than naive.
        # Allow a 5% band for measurement noise on tiny shapes.
        assert naive_opt_us <= naive_us * 1.05, (
            f"naive_opt ({naive_opt_us:.1f} us) is slower than naive "
            f"({naive_us:.1f} us) by more than 5% on {args!r}"
        )


def _print_naive_opt_perf_table():
    if not _naive_opt_perf_results:
        return
    lines = [
        "",
        "=" * 78,
        "naive vs naive_opt (DGL w-load) -- K5 device kernel time (us)",
        "=" * 78,
        f"{'shape':<44} {'naive':>9} {'naive_opt':>10} {'speedup':>8}",
        "-" * 78,
    ]
    for r in _naive_opt_perf_results:
        lines.append(
            f"{r['id']:<44} {r['naive_us']:>9.1f} "
            f"{r['naive_opt_us']:>10.1f} {r['speedup']:>7.2f}x"
        )
    lines.append("-" * 78)
    print("\n".join(lines))


def _print_perf_table():
    if not _perf_results:
        return

    cols = [
        ("Model", "Model", 16),
        ("TP", "TP", 3),
        ("Hg", "Hg", 3),
        ("H", "H", 3),
        ("SeqLen", "SeqLen", 7),
        ("T", "T", 7),
        ("var", "varlen", 3),
        ("fs", "final_st", 3),
        ("FlyDSL", "FlyDSL_vk(us)", 8),
        ("FlyDSL_n", "FlyDSL_naive(us)", 12),
        ("FlyDSL_nopt", "FlyDSL_naive_opt(us)", 12),
        ("FlyDSL_kv", "FlyDSL_kv(us)", 9),
        ("FlyDSL_kvn", "FlyDSL_kvnaive(us)", 10),
        ("Tri_vk", "Triton_vk(us)", 8),
        ("vLLM", "vLLM_vk(us)", 8),
        ("HIP", "HIP_vk(us)", 8),
        ("fly/vk", "flydsl_vs_vk", 7),
        ("fly/vllm", "flydsl_vs_vllm", 8),
        ("hip/fly", "hip_vs_flydsl", 8),
    ]

    def _fmt_cell(val, key, width):
        if isinstance(val, bool):
            return ("Y" if val else "N").rjust(width)
        if isinstance(val, float):
            if val != val:  # NaN (vLLM column when vllm not installed)
                return "-".rjust(width)
            return (f"{val:.2f}x" if "_vs_" in key else f"{val:.1f}").rjust(width)
        return str(val).rjust(width)

    header = " | ".join(display.rjust(w) for display, _, w in cols)
    sep = "-+-".join("-" * w for _, _, w in cols)
    border = "=" * len(header)

    lines = [
        "",
        border,
        "K5 Prefill Performance Summary (K5 device kernel time only, via torch.profiler)",
        border,
        "",
        sep,
        header,
        sep,
    ]
    for row in _perf_results:
        lines.append(" | ".join(_fmt_cell(row[k], k, w) for _, k, w in cols))
    lines.append(sep)
    lines.append("")
    print("\n".join(lines))


@pytest.fixture(scope="session", autouse=True)
def _print_summary_table(request):
    """Print the summary performance table after all tests finish."""
    yield
    _print_perf_table()
    _print_naive_opt_perf_table()


# -- bf16 SSM-state correctness ----------------------------------------


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
        context_lens = args.resolve_context_lens()
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
        context_lens = args.resolve_context_lens()
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
        context_lens = args.resolve_context_lens()
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
        context_lens = args.resolve_context_lens()
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


# -- triton_origin_opt: BV=16 + exp2 variant of fwd_h --------------------
#
# Inlined from the (now-deleted) standalone benchmark script
# ``0423_gdr_prefill_bench_standalone.py``. Launches the same recurrence
# kernel as the existing ``chunk_gated_delta_rule_fwd_h`` (``triton_origin``
# in this file), but with two changes that the standalone bench's new
# pipeline applies on top of the original RTP config:
#
#   - BV = 16 (was 32): smaller V-tile -> more (V/BV) blocks per (B*H)
#     program-id pair -> better occupancy on MI355X.
#   - USE_EXP2 = True (was False): emits a single ``v_exp_f32`` per
#     gate evaluation instead of the ``v_log + v_mul + v_exp`` chain that
#     ``tl.exp`` lowers to.
#
# Because USE_EXP2 expects gates pre-scaled by ``1/ln(2)``, the wrapper
# multiplies the supplied ``g`` (already a per-chunk cumsum, as produced
# by ``_make_inputs``) by RCP_LN2 before launching. That scale step is
# excluded from the K5 kernel time -- we time only the kernel itself, in
# line with how the other K5 wrappers in this file are benchmarked.
#
# The kernel itself is wrapped in a ``triton.autotune`` sweep over
# (BV, num_warps, num_stages); the standalone version pinned BV=16
# only, but exposing the sweep here matches what aiter's own
# ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` does internally
# and lets each shape pick its own best config on first run.

_RCP_LN2 = 1.0 / 0.6931471805599453

_exp = tl.exp
_exp2 = tl.math.exp2


# Decorator stack mirrors FLA's K5 kernels (Heuristics outer, Autotune
# inner) so that Triton 3.x writes the sweep result to its persistent
# autotune cache (`~/.triton/autotune`) via ``cache_results=True``. After
# the first run each (H, K, V, BT, IS_VARLEN) key is served from disk and
# subsequent runs no longer launch the full BV/warps/stages sweep -- the
# rocprof kernel-stats CSV then reports the same ~56 calls as the other
# K5 kernels, instead of the 9000+ that an un-cached sweep produces.
#
# ``Hg`` is intentionally excluded from ``key``: it only affects host-side
# address arithmetic (``H // Hg`` divisor for the K block-ptr), not the
# compiled binary or tile shape, so adding it would just multiply the
# number of unique keys and force redundant sweeps.
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in (16, 32, 64)
        for num_warps in (2, 4)
        for num_stages in (2, 3)
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_origin_opt(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v2 = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v2, b_v.to(p_v2.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, _exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = _exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, _exp(b_g_last - b_g), 0)[:, None]
                b_g_last = _exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


_TRITON_ORIGIN_OPT_KERNEL = chunk_gated_delta_rule_fwd_kernel_h_origin_opt


def chunk_gated_delta_rule_fwd_h_origin_opt(
    k,
    w,
    u,
    g=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """``triton_origin_opt`` K5: USE_EXP2 + autotuned BV/warps/stages variant.

    Mirrors the standalone bench's ``fwd_h`` host wrapper but adds a
    Triton autotune sweep over ``BV ? {16, 32, 64}``, ``num_warps ?
    {2, 4}``, ``num_stages ? {1, 2, 3}``. Keyed on ``(H, Hg, K, V, BT,
    IS_VARLEN)`` so each shape picks its own best config on first run.

    Inputs use the GQA layout from PREFILL_PARAMS unchanged -- since this
    K5 kernel accepts ``Hg`` directly, no ``repeat_interleave`` is needed
    (unlike the original ``chunk_gated_delta_rule_fwd_h``, which is
    MHA-only).

    NOTE: The RCP_LN2 scale required by USE_EXP2=True is applied here so
    that callers can pass the same per-chunk-cumsum ``g`` as the other
    K5 wrappers. This scale is a cheap elementwise multiply and is
    excluded from the kernel-time measurement when ``_bench_fn`` profiles
    only the kernel launch.
    """
    import triton as _triton

    B, T, Hg, K = k.shape
    V = u.shape[-1]
    H = u.shape[-2]
    BT = 64
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, _triton.cdiv(T, BT), None
    else:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    # USE_EXP2=True expects gates pre-scaled by 1/ln(2). Cheap elementwise
    # op; excluded from kernel time when profiled via torch.profiler.
    # ``triton_origin_opt`` reads g with token-major offsets
    # ``(bos*H + row*H + i_h)`` so transpose head-major [H, T_total] back
    # to token-major [T_total, H] before launch.
    if g is not None:
        if g.dim() == 2:
            g_for_kernel = g.transpose(0, 1).contiguous()  # [T_total, H]
        else:
            g_for_kernel = g.transpose(-2, -1).contiguous()
        g_scaled = g_for_kernel * _RCP_LN2
    else:
        g_scaled = None

    def grid(meta):
        return (_triton.cdiv(V, meta["BV"]), N * H)

    _TRITON_ORIGIN_OPT_KERNEL[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g_scaled,
        gk=None,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=True,
    )
    return h, v_new, final_state
