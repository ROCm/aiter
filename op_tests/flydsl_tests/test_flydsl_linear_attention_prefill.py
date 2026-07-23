# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention Prefill (chunk_gated_delta_h) regressions.

Usage:
    rm -rf ~/.triton/cache
    export GATED_DELTA_RULE_TRITON_AUTOTUNE=1
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=7 pytest -sv op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -s
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=7 python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "varlen-64k-qwen-ptpc-ali" -v -s
    bash op_tests/flydsl_tests/run_test_flydsl_gdr_k5_prefill.sh
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
        chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip,
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

# When True, ``test_consistency_flydsl_mfma16_hip_vs_hip`` requires the
# flydsl-hip fork to match HIP/C++ BIT-FOR-BIT (torch.equal). Until the LDS /
# layout / (optionally) numeric alignment work fully lands it stays False and
# the test only records the gap + asserts a loose same-algorithm band.
_MFMA16_HIP_VS_HIP_BITEXACT = False

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
    # Appended to the display id when a group sweeps multiple
    # ``max_num_batched_tokens`` values, so a fixed (tp, full_prompt_len) stays
    # unique across the batched-token sweep. Empty for single-value groups, so
    # their ids are unchanged.
    bt_tag: str = ""

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
        if self.bt_tag:
            tag += f"_{self.bt_tag}"
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
    # Semantics for ``max_num_batched_tokens``:
    #   - list/tuple : sweep -- materialise one case per element (Cartesian with
    #           tps x full_prompt_lens). Each element is itself one of the specs
    #           below (int / "full_prompt_len" / None). For the varlen path this
    #           sweeps the batch size N = mnbt // full_prompt_len. ids get an
    #           ``mnbt{value}`` suffix so a fixed (tp, full_prompt_len) stays
    #           unique. Example: ``max_num_batched_tokens=[16384, 32768, 65536]``.
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
        # ``max_num_batched_tokens`` may be a single spec (int / "full_prompt_len"
        # / None) OR a list/tuple of such specs. A list materialises one case per
        # value (Cartesian with tps x full_prompt_lens) -- e.g. to sweep the
        # scheduler token budget, which for the varlen path sweeps the batch size
        # (N = mnbt // full_prompt_len). When more than one value is present, ids
        # gain an ``mnbt{value}`` suffix so a fixed (tp, full_prompt_len) stays
        # unique; a single value keeps the original ids unchanged.
        mnbt_specs = g.max_num_batched_tokens
        if not isinstance(mnbt_specs, (list, tuple)):
            mnbt_specs = [mnbt_specs]
        _sweep_mnbt = len(mnbt_specs) > 1
        for tp in g.tps:
            for full_len in g.full_prompt_lens:
                for mnbt_spec in mnbt_specs:
                    if mnbt_spec == "full_prompt_len":
                        mnbt = full_len
                    elif mnbt_spec is None:
                        mnbt = 32768  # PrefillArgs dataclass default
                    else:
                        mnbt = mnbt_spec
                    bt_tag = f"mnbt{mnbt}" if _sweep_mnbt else ""

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
                                bt_tag=bt_tag,
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
                            if _sweep_mnbt:
                                tag = f"{tag}_mnbt{mnbt}"
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
    PrefillGroup(
        model_name="Qwen3.5-397B-ptpc-ali",
        Hv=64,
        tps=[8],
        full_prompt_lens=[1024, 2048, 4096, 8192],
        is_varlen=False,
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
        model_name="varlen-64k-qwen-ptpc-ali",
        Hv=64,
        tps=[8],
        full_prompt_lens=[8192],
        # max_num_batched_tokens=[8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536],
        max_num_batched_tokens=[65536],
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
        full_prompt_lens=[1000, 5000, 10000],
        # full_prompt_lens=[1000],
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


def _is_gfx950() -> bool:
    """Whether the current GPU is CDNA4 / gfx950 (MI350).

    The baseline / ``naive`` / ``naive_opt`` FlyDSL K5 forks emit the
    ``mfma_f32_16x16x32_bf16`` (K=32 bf16) MFMA and ``mfma32_vk`` emits
    ``mfma_f32_32x32x16_bf16`` -- both are gfx950-only instructions. On gfx942
    (CDNA3 / MI300) they fail to compile with an LLVM ``Cannot select``
    abort, so the perf harness skips them there. The remaining forks
    (``kv`` / ``mfma16_hip`` / ``mfma16_2wave_opt1`` / ``mfma16_3wave_opt2``)
    use the K=16 ``mfma_f32_16x16x16bf16_1k`` and run on both.
    """
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return "gfx950" in arch


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
    "chunk_gdn_fwd_h_flydsl_mfma16",
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


def _assert_mean_abs_within(out, ref, *, mean_atol, label):
    """Guard the *mean* absolute error, not just the per-element worst case.

    ``torch.testing.assert_close``'s ``atol`` only bounds the single worst
    element, which for bf16 K5 over a long chunked recurrence sits close to
    the 5e-2 element tolerance (a few outlier tokens at the tail of the
    33-segment chain). The mean abs error, by contrast, is ~2-3e-3 in
    practice and is what actually moves when an implementation regresses the
    *whole* distribution (e.g. a gating / accumulation bug) without yet
    tripping any single element past 5e-2. Bound it here.
    """
    mean_abs = (out.float() - ref.float()).abs().mean().item()
    assert mean_abs <= mean_atol, (
        f"{label}: mean abs error {mean_abs:.3e} exceeds mean_atol "
        f"{mean_atol:.3e} (per-element atol may still pass; this guards "
        f"whole-distribution drift)"
    )


def _assert_close_lowmem(a, b, *, atol, rtol, msg, chunk_rows=1 << 22):
    """Memory-frugal elementwise ``|a-b| <= atol + rtol*|b|`` check.

    Equivalent in semantics to ``torch.testing.assert_close(a, b, atol, rtol)``
    but streams over a flattened view in row chunks so it never materialises
    more than one chunk-sized fp32 temporary. Used for the mfma16_2wave_opt1/triton_vk
    consistency check, where the h / v_new tensors at long context are
    multi-GiB and the stock ``assert_close`` (which up-casts both whole tensors
    and builds a full mismatch report) OOMs on a 256 GiB card. On mismatch it
    reports the worst element's abs error / allowed tol rather than dumping the
    entire tensor.
    """
    assert a.shape == b.shape, f"{msg}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    af = a.reshape(-1)
    bf = b.reshape(-1)
    n = af.numel()
    worst_abs = 0.0
    worst_allowed = 0.0
    worst_idx = -1
    n_bad = 0
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        ac = af[s:e].float()
        bc = bf[s:e].float()
        abs_e = (ac - bc).abs()
        allowed = atol + rtol * bc.abs()
        bad = abs_e > allowed
        nb = int(bad.sum().item())
        if nb:
            n_bad += nb
            # track the single worst (abs - allowed) margin in this chunk
            margin = abs_e - allowed
            mi = int(margin.argmax().item())
            if abs_e[mi].item() - allowed[mi].item() > worst_abs - worst_allowed:
                worst_abs = abs_e[mi].item()
                worst_allowed = allowed[mi].item()
                worst_idx = s + mi
        del ac, bc, abs_e, allowed, bad
    assert n_bad == 0, (
        f"{msg}: {n_bad}/{n} elements exceed atol={atol:g}+rtol={rtol:g}*|b|. "
        f"Worst @ flat idx {worst_idx}: abs_err={worst_abs:.3e} > "
        f"allowed={worst_allowed:.3e}."
    )


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
    mean_atol=5e-3,
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

    Two complementary bounds are enforced per output:
      * ``atol`` / ``rtol`` (5e-2): the per-element worst case.
      * ``mean_atol`` (5e-3): the mean abs error, which catches a regression
        that shifts the whole distribution before any single element trips
        the looser element tolerance. Measured mean abs is ~2-3e-3 on the
        varlen-32k-aws shape, so 5e-3 leaves ~2x headroom over bf16 noise.
    """
    h_out_f = h_out.float()
    vn_out_f = _normalize_opt_v_new(vn_out).float()
    torch.testing.assert_close(
        h_out_f,
        h_ref.float(),
        atol=atol,
        rtol=rtol,
        msg=f"{label}: h mismatch",
    )
    _assert_mean_abs_within(h_out_f, h_ref, mean_atol=mean_atol, label=f"{label} h")
    torch.testing.assert_close(
        vn_out_f,
        vn_ref.float(),
        atol=atol,
        rtol=rtol,
        msg=f"{label}: v_new mismatch",
    )
    _assert_mean_abs_within(
        vn_out_f, vn_ref, mean_atol=mean_atol, label=f"{label} v_new"
    )
    if output_final_state:
        fs_out_f = fs_out.float()
        torch.testing.assert_close(
            fs_out_f,
            fs_ref.float(),
            atol=atol,
            rtol=rtol,
            msg=f"{label}: final_state mismatch",
        )
        _assert_mean_abs_within(
            fs_out_f, fs_ref, mean_atol=mean_atol, label=f"{label} final_state"
        )
    else:
        assert fs_out is None, f"{label}: expected None final_state"
        assert fs_ref is None


class TestCorrectness:
    """Correctness against PyTorch FP32 reference for all three K5 backends."""

    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_correctness_flydsl_mfma16_hip(self, args: PrefillArgs):
        """mfma16 / HIP-aligned FlyDSL K5 impl (formerly the "vk" fork): 16x16x16
        MFMA + HIP warp partition. Same VK public outputs as the baseline flydsl
        path; only the BV==64 configs exercise the kernel, others fall back."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
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
            label="flydsl_mfma16_hip",
        )


# -- Performance benchmark (flydsl-hip vs hip vs triton) -----------------

_perf_results: list[dict] = []


def _run_perf_comparison(args: PrefillArgs):
    """Bench the same shape on flydsl-hip / hip(C++) / triton(opt_vk) and record
    a row into ``_perf_results``; the session-scoped ``_print_summary_table``
    fixture prints an aligned table after all tests finish. hip/triton are
    mainline backends used only as references; hip is skipped for shapes it does
    not support (needs K=V=128, bf16, chunk_size=64)."""
    context_lens = args.resolve_context_lens()
    k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)
    ofs = args.output_final_state
    total_tokens = int(cu[-1].item())

    us_fly = _bench_fn(
        chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip,
        k,
        w_c,
        u_c,
        g=g,
        initial_state=h0,
        output_final_state=ofs,
        cu_seqlens=cu,
    )
    us_tri = _bench_fn(
        chunk_gated_delta_rule_fwd_h_opt_vk,
        k,
        w_c,
        u_c,
        g=g,
        initial_state=h0,
        output_final_state=ofs,
        cu_seqlens=cu,
    )
    if _HAS_HIP_K5 and _hip_k5_supported(args):
        us_hip = _bench_fn(
            chunk_gated_delta_rule_fwd_h_hip_k5,
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=ofs,
            cu_seqlens=cu,
        )
    else:
        us_hip = float("nan")

    has_hip = us_hip == us_hip  # not NaN
    _perf_results.append(
        {
            "Model": args.model_name or "-",
            "TP": args.tp,
            "Hg": args.Hg,
            "H": args.H,
            "SeqLen": args.full_prompt_len,
            "T": total_tokens,
            "varlen": args.is_varlen,
            "final_st": ofs,
            "fly_hip": us_fly,
            "HIP": us_hip,
            "Triton": us_tri,
            # speedup vs hip (hip is the baseline): >1 faster than hip, <1 slower.
            "fly/hip": (us_hip / us_fly) if has_hip else float("nan"),
            "tri/hip": (us_hip / us_tri) if has_hip else float("nan"),
        }
    )


def _print_perf_table():
    if not _perf_results:
        return
    _model_w = max([len("Model")] + [len(str(r["Model"])) for r in _perf_results])
    # (header_display, row_key, width): header uses the 1st, cell lookup the 2nd.
    cols = [
        ("Model", "Model", _model_w),
        ("TP", "TP", 2),
        ("Hg", "Hg", 2),
        ("H", "H", 2),
        ("SeqLen", "SeqLen", 6),
        ("T", "T", 6),
        ("varlen", "varlen", 6),
        ("final_st", "final_st", 8),
        ("FlyDSL_hip(us)", "fly_hip", 14),
        ("HIP(us)", "HIP", 8),
        ("Triton(us)", "Triton", 10),
        ("fly/hip", "fly/hip", 7),
        ("tri/hip", "tri/hip", 7),
    ]

    def _fmt_cell(val, key, width):
        if isinstance(val, bool):
            return ("Y" if val else "N").rjust(width)
        if isinstance(val, float):
            if val != val:  # NaN (hip skipped for unsupported shapes)
                return "-".rjust(width)
            return (f"{val:.2f}x" if "/" in key else f"{val:.1f}").rjust(width)
        return str(val).rjust(width)

    header = "|".join(disp.rjust(w) for disp, _, w in cols)
    sep = "+".join("-" * w for _, _, w in cols)
    border = "=" * len(header)
    lines = [
        "",
        border,
        "K5 Prefill Perf Summary (mfma16_hip vs hip vs triton; K5 device kernel us via "
        "torch.profiler; fly/hip & tri/hip = speedup vs hip, >1 faster / <1 slower)",
        border,
        "",
        sep,
        header,
        sep,
    ]
    for row in _perf_results:
        lines.append("|".join(_fmt_cell(row[k], k, w) for _, k, w in cols))
    lines.append(sep)
    lines.append("")
    print("\n".join(lines))


@pytest.fixture(scope="session", autouse=True)
def _print_summary_table(request):
    """Print the perf summary table after all tests in the session finish."""
    yield
    _print_perf_table()


class TestPerformance:
    @pytest.mark.parametrize("args", PREFILL_PARAMS, ids=PREFILL_TEST_IDS)
    def test_perf_comparison(self, args: PrefillArgs):
        _run_perf_comparison(args)
