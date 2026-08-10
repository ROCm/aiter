# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention Prefill (chunk_gated_delta_h) regressions.

Usage:
    rm -rf ~/.triton/cache
    export GATED_DELTA_RULE_TRITON_AUTOTUNE=1
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=7 pytest -sv op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -s
    FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=7 python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "varlen-64k-qwen-ptpc-ali" -v -s
"""

from __future__ import annotations

import math
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
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
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
except ImportError:
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
except ImportError:
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
    # Batch size B for the dense (non-varlen) path. When >1, ``_make_inputs``
    # builds ``g`` as a 3D ``[B, H, T_flat]`` layout, exercising the dense B>1
    # batch-head gate-offset path (the kernel's ``g_head_base`` must include the
    # ``i_n*H*T_flat`` batch stride). The varlen path ignores this field (always
    # B=1, N segments). Defaults to 1 so existing dense cases are unchanged.
    dense_batch: int = 1
    # Whether to provide ``g``. False takes the ``g=None`` (USE_G=False) path,
    # covering the masking of the last chunk's padding rows when there is no g
    # (otherwise invalid tokens' v_new would flow through gated_v and corrupt
    # the state update).
    use_g: bool = True
    # g layout, matching the wrapper/HIP contract. False (default) -> token-major
    # 3D [B, T_flat, H] (== HIP default); True -> head-major 3D [B, H, T_flat].
    g_head_major: bool = False

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
            if not self.use_g:
                tag += "_nog"
            if self.g_head_major:
                tag += "_ghm"
            return tag
        tag = self.model_name + "_" if self.model_name else ""
        tag += f"K{self.K}_V{self.V}_Hk{self.Hk}_Hv{self.Hv}"
        tag += f"_TP{self.tp}_T{self.full_prompt_len}"
        if self.bt_tag:
            tag += f"_{self.bt_tag}"
        if not self.is_varlen:
            tag += "_novarlen"
        if self.dense_batch != 1:
            tag += f"_B{self.dense_batch}"
        if not self.use_g:
            tag += "_nog"
        if self.g_head_major:
            tag += "_ghm"
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
    # dense (non-varlen) batch size; when >1, g becomes 3D [B,H,T_flat] (see PrefillArgs).
    dense_batch: int = 1
    # whether to provide g; False takes the g=None (USE_G=False) path (see PrefillArgs).
    use_g: bool = True
    # g layout: False (default) token-major [B,T,H]; True head-major [B,H,T] (see PrefillArgs).
    g_head_major: bool = False


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
                                dense_batch=g.dense_batch,
                                use_g=g.use_g,
                                g_head_major=g.g_head_major,
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
                                    dense_batch=g.dense_batch,
                                    use_g=g.use_g,
                                    g_head_major=g.g_head_major,
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
        # dense B>1: g becomes 3D [B,H,T], validating the kernel's batch-head
        # gate offset (g_sh_base includes i_n*H*T_flat). Bumping B to 2 covers
        # the i_n>0 batch-stride branch. Also set g_head_major=True so this group
        # exercises the head-major dense layout (the other dense/varlen groups
        # cover the default token-major layout).
        dense_batch=2,
        g_head_major=True,
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
        max_num_batched_tokens=[8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536],
        # max_num_batched_tokens=[65536],
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
        model_name="aws-n1",
        Hv=32,
        tps=[1],
        full_prompt_lens=[5000, 10000],
        max_num_batched_tokens="full_prompt_len",
    ),
    PrefillGroup(
        model_name="aws-n3-mid10k",
        Hv=32,
        tps=[1],
        full_prompt_lens=[16384],
        max_num_batched_tokens=16384,
        # head=0 creates an empty first segment (cu_seqlens=[0,0,10000,16384]),
        # validating that empty varlen sequences skip the W/K prologue, do not
        # read out of bounds, and pass state through h0->ht correctly.
        head_seqlens=[0, 10, 65, 704, 936, 1820, 4467, 5508],
        mid_seqlen=10000,
    ),
    PrefillGroup(
        model_name="aws-n2-16k",
        Hv=32,
        tps=[1],
        full_prompt_lens=[16384],
        max_num_batched_tokens=16384,
        head_seqlens=[4000, 6396, 8192, 9912, 10000],
        num_segments=2,
        # head-major varlen coverage (other varlen groups use token-major).
        g_head_major=True,
    ),
]

PREFILL_PARAMS = expand_groups(_PREFILL_GROUPS)

# Explicit empty-TAIL varlen case (cu_seqlens=[0, 6384, 16384, 16384]; last
# segment length 0). The existing empty-segment group only covers an empty
# FIRST segment (bos=0), which can only read token 0 and never reaches the
# buffer tail; the original OOB was a tail prologue over-read that requires
# bos==eos==T_total. ``context_lens`` is used verbatim (bypasses the
# ``expand_groups`` tail>0 guard), and the 0-length tail also validates the
# reference passes ``initial_state`` straight through to ``final_state``.
PREFILL_PARAMS = list(PREFILL_PARAMS) + [
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=32,
        tp=1,
        full_prompt_len=16384,
        model_name="flydsl-k5-empty-tail",
        is_varlen=True,
        output_final_state=True,
        max_num_batched_tokens=16384,
        context_lens=[6384, 10000, 0],
    ),
]


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
        ssm_state_dtype=torch.bfloat16,
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
        ssm_state_dtype=torch.bfloat16,
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
    dense_batch=1,
    use_g=True,
    g_head_major=False,
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
        dense_batch = args.dense_batch
        use_g = args.use_g
        g_head_major = args.g_head_major

    Hg = Hk_dim // tp
    H = Hv_dim // tp

    if is_varlen:
        scheduled_q_lens, cu_seqlens = _build_cu_seqlens(context_lens, device=device)
        T_total = int(cu_seqlens[-1].item())
        N = len(scheduled_q_lens)
        B = 1
    else:
        T_total = sum(context_lens)
        B = dense_batch
        N = B
        cu_seqlens = None
        scheduled_q_lens = context_lens

    k = torch.randn(B, T_total, Hg, K_dim, dtype=dtype, device=device) * 0.1
    w_orig = torch.randn(B, T_total, H, K_dim, dtype=dtype, device=device) * 0.1
    u_orig = torch.randn(B, T_total, H, V_dim, dtype=dtype, device=device) * 0.1
    # g gate: always a 3-D tensor, matching the wrapper/HIP contract. cumsum is
    # along T; varlen has B=1 (flattened, N segments live in cu_seqlens).
    #   * use_g=False     -> None (USE_G=False path, validates padding masking)
    #   * g_head_major    -> head-major  [B, H, T_total]
    #   * not g_head_major-> token-major [B, T_total, H]  (default, == HIP)
    # The head-major base is generated first (cumsum along the last/T dim), then
    # transposed for the token-major layout so both layouts hold the same values.
    if not use_g:
        g = None
    else:
        gh = torch.randn(B, H, T_total, dtype=torch.float32, device=device).abs() * -0.5
        gh = gh.cumsum(dim=-1)
        g = gh.contiguous() if g_head_major else gh.transpose(1, 2).contiguous()

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
    g_head_major=False,
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

                    # g sequence for (batch b_idx, head i_h): g is always 3-D
                    # (or None). head-major [B,H,T] -> g[b_idx, i_h];
                    # token-major [B,T,H] -> g[b_idx, :, i_h].
                    if g is None:
                        g_seq = None
                    elif g_head_major:
                        g_seq = g[b_idx, i_h]
                    else:
                        g_seq = g[b_idx, :, i_h]

                    mask = torch.zeros(BT_dim, device=k.device)
                    mask[:actual_bt] = 1.0
                    if g_seq is None:
                        # No g: no gate decay; valid rows have gate=1 and padding
                        # rows are not in the chunk slice at all. Matches the
                        # kernel's pure padding masking under USE_G=False.
                        gate = mask[:actual_bt]
                    else:
                        last_idx = bos + t_end - 1
                        g_last = g_seq[last_idx].float()
                        g_chunk = g_seq[bos + t_start : bos + t_end].float()
                        gate = torch.where(
                            mask[:actual_bt].bool(),
                            torch.exp(g_last - g_chunk),
                            torch.zeros_like(g_chunk),
                        )
                        h_state = h_state * torch.exp(g_last)
                    b_v_gated = b_v * gate.unsqueeze(-1)

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
    except Exception:  # noqa: BLE001
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
    element. The mean abs error is what actually moves when an implementation
    regresses the *whole* distribution (e.g. a gating / accumulation bug)
    without yet tripping any single element past the elementwise tolerance.
    Bound it independently here.
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
    atol=2e-2,
    rtol=2e-2,
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
      * ``atol`` / ``rtol`` (2e-2): the per-element worst case.
      * ``mean_atol`` (5e-3): the mean abs error, which catches a regression
        that shifts the whole distribution before any single element trips
        the element tolerance. After natural-log gate alignment, the full
        54-shape gfx942 sweep (17B+ compared elements) has zero failures at
        2e-2/2e-2. Ten seeds of the worst no-g shape peak at mean abs 3.47e-3;
        5e-3 retains headroom for random input and cross-architecture variance.
        The next tighter elementwise candidate (1.5e-2/1.5e-2) already fails
        one final-state element in that multi-seed sweep.
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
    """Correctness and integration coverage for the FlyDSL mfma16 K5 backend."""

    @staticmethod
    def _minimal_inputs():
        """Smallest validated mfma16_hip input set for contract tests."""
        device = "cuda"
        B, T, Hg, H, K, V = 1, 64, 2, 4, 128, 128
        k = torch.zeros(B, T, Hg, K, dtype=torch.bfloat16, device=device)
        w = torch.zeros(B, H, T, K, dtype=torch.bfloat16, device=device)
        u = torch.zeros(B, H, T, V, dtype=torch.bfloat16, device=device)
        return k, w, u

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
            g_head_major=args.g_head_major,
            # ``g`` is generated in natural-log space (see ``_make_inputs``) and
            # the reference decays with ``exp``. Pass ``use_exp2=False`` so the
            # kernel's ``_fast_exp`` applies the LOG2E scale (exp2(x*LOG2E)==exp(x))
            # and both sides compare the SAME formula. With the default
            # ``use_exp2=True`` the kernel would treat ``g`` as log2-space and
            # compute ``exp2(x)``, a mismatch masked only by gates decaying to 0.
            use_exp2=False,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w_orig,
            u_orig,
            g=g,
            initial_state=h0,
            output_final_state=args.output_final_state,
            cu_seqlens=cu,
            g_head_major=args.g_head_major,
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

    @pytest.mark.parametrize("args", STATE_BF16_PARAMS, ids=STATE_BF16_TEST_IDS)
    def test_correctness_bf16_state(self, args: PrefillArgs):
        """Validate bf16 initial/final state on dense and varlen launch paths."""
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
            output_final_state=True,
            cu_seqlens=cu,
            g_head_major=args.g_head_major,
            use_exp2=False,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w_orig,
            u_orig,
            g=g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            g_head_major=args.g_head_major,
        )

        assert fs_fly.dtype == torch.bfloat16
        _assert_k5_outputs_match_ref(
            h_fly,
            vn_fly,
            fs_fly,
            h_ref,
            vn_ref,
            fs_ref,
            output_final_state=True,
            label="flydsl_mfma16_hip_bf16_state",
        )

    def test_e2e_dispatch_matches_triton(self):
        """Exercise K1-K6 with use_chunk_flydsl=True through public dispatch."""
        torch.manual_seed(42)
        B, T, H, D = 1, 64, 4, 128
        q = torch.randn(B, T, H, D, dtype=torch.bfloat16)
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1
        ).to(torch.bfloat16)
        v = torch.randn(B, T, H, D, dtype=torch.bfloat16)
        g = torch.nn.functional.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
        beta = torch.rand(B, T, H, dtype=torch.bfloat16).sigmoid()
        h0 = torch.randn(B, H, D, D, dtype=torch.float32)
        kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": D**-0.5,
            "initial_state": h0,
            "output_final_state": True,
            "use_exp2": True,
        }

        _, out_fly, fs_fly = chunk_gated_delta_rule_fwd_opt_vk(
            **kwargs, use_chunk_flydsl=True
        )
        _, out_tri, fs_tri = chunk_gated_delta_rule_fwd_opt_vk(
            **kwargs, use_chunk_flydsl=False
        )
        torch.testing.assert_close(
            out_fly.float(), out_tri.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(fs_fly.float(), fs_tri.float(), atol=2e-2, rtol=2e-2)

    def test_e2e_dispatch_rejects_k64(self):
        """K=64 is unsupported and must fail before launching any K1-K6 kernel."""
        B, T, H, D = 1, 64, 4, 64
        with pytest.raises(ValueError, match="K=128 and V=128"):
            chunk_gated_delta_rule_fwd_opt_vk(
                q=torch.zeros(B, T, H, D, dtype=torch.bfloat16),
                k=torch.zeros(B, T, H, D, dtype=torch.bfloat16),
                v=torch.zeros(B, T, H, 128, dtype=torch.bfloat16),
                g=torch.zeros(B, T, H, dtype=torch.float32),
                beta=torch.zeros(B, T, H, dtype=torch.bfloat16),
                scale=D**-0.5,
                initial_state=None,
                output_final_state=False,
                use_chunk_flydsl=True,
            )

    def test_natural_log_gate_formula(self):
        """Natural-log gates must use exp(x), not exp2(x).

        Only token 0 contributes to the state, and its gate is fixed at
        exp(g_last-g_0)=exp(-1). Using the wrong ``use_exp2=True`` contract with
        this unscaled natural-log gate produces exp2(-1)=0.5 instead, an explicit
        ~0.132 error that cannot be hidden by random decay or mean-error dilution.
        """
        device = "cuda"
        B, T, Hg, H, K, V = 1, 64, 2, 4, 128, 128

        k = torch.zeros(B, T, Hg, K, dtype=torch.bfloat16, device=device)
        w = torch.zeros(B, T, H, K, dtype=torch.bfloat16, device=device)
        u = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=device)
        k[:, 0, :, 0] = 1
        u[:, 0, :, 0] = 1

        # Token-major natural-log cumulative gate [B,T,H]: g_0=0 and
        # g_last=-1, so the only nonzero outer-product contribution is exp(-1).
        g = torch.full((B, T, H), -1.0, dtype=torch.float32, device=device)
        g[:, 0, :] = 0
        h0 = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)
        w_c = w.permute(0, 2, 1, 3).contiguous()
        u_c = u.permute(0, 2, 1, 3).contiguous()

        _, _, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=True,
            use_exp2=False,
        )
        _, _, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w,
            u,
            g=g,
            initial_state=h0,
            output_final_state=True,
        )

        expected = torch.tensor(
            math.exp(-1), dtype=torch.bfloat16, device=device
        ).float()
        torch.testing.assert_close(
            fs_ref[0, :, 0, 0],
            expected.expand(H),
            atol=0,
            rtol=0,
            msg="targeted gate setup no longer isolates bf16(exp(-1))",
        )
        torch.testing.assert_close(
            fs_fly.float(),
            fs_ref.float(),
            atol=2e-3,
            rtol=0,
            msg="natural-log gate path must compute exp(x), not exp2(x)",
        )

    def test_portable_rne_preserves_nan_and_inf(self):
        """RNE conversion must not turn low-payload f32 NaNs into bf16 Inf."""
        k, w, u = self._minimal_inputs()
        H, K, V = 4, 128, 128

        # Inject exact f32 bit patterns into h0. The first chunk snapshot converts
        # these f32 values to bf16 through the selected RNE converter before any
        # recurrence update can alter them.
        h0_bits = torch.zeros(1, H, V, K, dtype=torch.int32, device="cuda")
        h0_bits[0, 0, 0, 0] = 0x7F800001  # +NaN, mantissa only below bit 16
        h0_bits[0, 0, 0, 1] = -8388607  # 0xFF800001: -NaN, same low payload
        h0_bits[0, 0, 0, 2] = 0x7F800000  # +Inf
        h0_bits[0, 0, 0, 3] = -8388608  # 0xFF800000: -Inf
        h0 = h0_bits.view(torch.float32)

        h, _, _ = chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
            k,
            w,
            u,
            initial_state=h0,
            output_final_state=False,
            save_new_value=False,
            bf16_convert_trunc=False,
        )
        converted = h[0, 0, 0, 0, :4].float()
        assert torch.isnan(
            converted[:2]
        ).all(), "portable RNE converted a low-payload NaN to a non-NaN value"
        assert torch.isposinf(converted[2]), "portable RNE did not preserve +Inf"
        assert torch.isneginf(converted[3]), "portable RNE did not preserve -Inf"

    @pytest.mark.parametrize(
        "indices,index_dtype,match",
        [
            ([-1, 0], torch.int32, "out of range"),
            ([0, 3], torch.int64, "out of range"),
            ([1, 1], torch.int64, "duplicate initial_state_indices"),
            ([2**32, 1], torch.int64, "out of range"),
            ([0.0, 1.0], torch.float32, "must be int32 or int64"),
        ],
    )
    def test_initial_state_indices_validation(self, indices, index_dtype, match):
        """Indexed state-pool access validates before narrowing to int32."""
        k, w, u = self._minimal_inputs()
        H, V, K = w.shape[1], u.shape[-1], k.shape[-1]
        h0_pool = torch.zeros(3, H, V, K, dtype=torch.float32, device="cuda")
        cu = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        state_indices = torch.tensor(indices, dtype=index_dtype, device="cuda")

        with pytest.raises(ValueError, match=match):
            chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
                k,
                w,
                u,
                initial_state=h0_pool,
                output_final_state=True,
                cu_seqlens=cu,
                initial_state_indices=state_indices,
            )

    def test_initial_state_indices_rank_and_device_validation(self):
        """Indexed state-pool indices must be 1-D and colocated with the pool."""
        k, w, u = self._minimal_inputs()
        H, V, K = w.shape[1], u.shape[-1], k.shape[-1]
        h0_pool = torch.zeros(3, H, V, K, dtype=torch.float32, device="cuda")
        cu = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")

        with pytest.raises(ValueError, match="must be 1-D"):
            chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
                k,
                w,
                u,
                initial_state=h0_pool,
                output_final_state=True,
                cu_seqlens=cu,
                initial_state_indices=torch.tensor(
                    [[0, 1]], dtype=torch.int64, device="cuda"
                ),
            )

        with pytest.raises(ValueError, match="must be on the same device"):
            chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
                k,
                w,
                u,
                initial_state=h0_pool,
                output_final_state=True,
                cu_seqlens=cu,
                initial_state_indices=torch.tensor(
                    [0, 1], dtype=torch.int64, device="cpu"
                ),
            )

    def test_valid_int64_initial_state_indices(self):
        """Validated int64 indices narrow safely and execute the int32 kernel ABI."""
        k, w, u = self._minimal_inputs()
        H, V, K = w.shape[1], u.shape[-1], k.shape[-1]
        h0_pool = torch.zeros(3, H, V, K, dtype=torch.float32, device="cuda")
        cu = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        indices = torch.tensor([2, 0], dtype=torch.int64, device="cuda")

        h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
            k,
            w,
            u,
            initial_state=h0_pool,
            output_final_state=True,
            cu_seqlens=cu,
            initial_state_indices=indices,
        )
        assert final_state.data_ptr() == h0_pool.data_ptr()
        assert torch.count_nonzero(h) == 0
        assert torch.count_nonzero(v_new) == 0
        assert torch.count_nonzero(final_state) == 0

    @pytest.mark.parametrize(
        "case,match",
        [
            ("rank", "must be 4-D"),
            ("dtype", "dtype must match"),
            ("contiguous", "must be contiguous"),
            ("time_shape", "k T dim"),
            ("unsupported_v", "only V=128 is supported"),
            ("gk_dtype", "gk must be float32"),
            ("gk_shape", "gk must use token-major"),
            ("state_shape", "initial_state must have shape"),
            ("state_contiguous", "initial_state must be contiguous"),
            ("g_device", "g must be on k's device"),
        ],
    )
    def test_mfma16_input_validation(self, case, match):
        """Raw-buffer kernel inputs fail early on invalid dtype/layout/shape."""
        k, w, u = self._minimal_inputs()
        kwargs = {}
        if case == "rank":
            k = k.squeeze(0)
        elif case == "dtype":
            w = w.float()
        elif case == "contiguous":
            k = k.transpose(1, 2)
        elif case == "time_shape":
            w = w[:, :, :-1].contiguous()
            u = u[:, :, :-1].contiguous()
        elif case == "unsupported_v":
            u = u[..., :64].contiguous()
        elif case == "gk_dtype":
            kwargs["gk"] = torch.zeros(
                1, 64, 4, 128, dtype=torch.bfloat16, device="cuda"
            )
        elif case == "gk_shape":
            kwargs["gk"] = torch.zeros(1, 64, 4, 64, dtype=torch.float32, device="cuda")
        elif case == "state_shape":
            kwargs["initial_state"] = torch.zeros(
                1, 4, 64, 128, dtype=torch.float32, device="cuda"
            )
        elif case == "state_contiguous":
            kwargs["initial_state"] = torch.zeros(
                1, 4, 128, 128, dtype=torch.float32, device="cuda"
            ).transpose(-1, -2)
        elif case == "g_device":
            kwargs["g"] = torch.zeros(1, 64, 4, dtype=torch.float32, device="cpu")

        with pytest.raises(ValueError, match=match):
            chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(k, w, u, **kwargs)

    def test_gk_token_major_contract(self):
        """A valid contiguous float32 gk uses [B,T,H,K] and runs successfully."""
        k, w, u = self._minimal_inputs()
        gk = torch.zeros(1, 64, 4, 128, dtype=torch.float32, device="cuda")
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
            k,
            w,
            u,
            gk=gk,
            output_final_state=True,
        )
        assert torch.count_nonzero(h) == 0
        assert torch.count_nonzero(v_new) == 0
        assert torch.count_nonzero(final_state) == 0

    def test_reference_empty_tail_passthrough(self):
        """The FP32 reference must pass ``initial_state`` straight through to
        ``final_state`` for an empty (zero-length) trailing segment, not leave
        it at the zero-initialised buffer value. Guards the reference itself
        (independent of the kernel) so the empty-tail correctness check above
        cannot be silently satisfied by a wrong reference."""
        device = "cuda"
        BT = 64
        H, V, K, Hg = 4, 128, 128, 2
        # cu_seqlens=[0, BT, BT]: segment 0 has BT tokens, segment 1 is empty.
        _, cu = _build_cu_seqlens([BT, 0], device=device)
        T_total = int(cu[-1].item())
        k = torch.randn(1, T_total, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
        w = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
        u = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device=device) * 0.1
        h0 = torch.randn(2, H, V, K, dtype=torch.float32, device=device) * 0.01
        _, _, fs = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w,
            u,
            g=None,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        # Empty trailing segment: final_state must equal the passed-in h0.
        assert torch.equal(fs[1], h0[1]), "empty tail segment did not pass h0 through"
        # Non-empty segment must have been updated (differs from h0).
        assert not torch.equal(fs[0], h0[0]), "non-empty segment was not updated"


# -- Performance benchmark (flydsl-hip vs hip vs triton) -----------------

_perf_results: list[dict] = []


def _run_perf_comparison(args: PrefillArgs):
    """Bench the same shape on flydsl-hip / hip(C++) / triton(opt_vk) and record
    a row into ``_perf_results``; the session-scoped ``_print_summary_table``
    fixture prints an aligned table after all tests finish. hip/triton are
    mainline backends used only as references; hip is skipped for shapes it does
    not support (needs K=V=128, bf16, chunk_size=64)."""
    context_lens = args.resolve_context_lens()
    k, _w_orig, _u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)
    ofs = args.output_final_state
    total_tokens = int(cu[-1].item()) if cu is not None else sum(context_lens)

    # ``g`` from _make_inputs follows args.g_head_major. FlyDSL takes the layout
    # flag directly; the triton/hip reference backends here consume head-major
    # g, so hand them a head-major view (transpose the token-major [B,T,H] back
    # to [B,H,T]).
    g_hm = None
    if g is not None:
        g_hm = g if args.g_head_major else g.transpose(1, 2).contiguous()

    us_fly = _bench_fn(
        chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip,
        k,
        w_c,
        u_c,
        g=g,
        initial_state=h0,
        output_final_state=ofs,
        cu_seqlens=cu,
        g_head_major=args.g_head_major,
    )
    us_tri = _bench_fn(
        chunk_gated_delta_rule_fwd_h_opt_vk,
        k,
        w_c,
        u_c,
        g=g_hm,
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
            g=g_hm,
            initial_state=h0,
            output_final_state=ofs,
            cu_seqlens=cu,
        )
    else:
        us_hip = float("nan")

    has_hip = not math.isnan(us_hip)  # not NaN
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
            if math.isnan(val):  # NaN (hip skipped for unsupported shapes)
                return "-".rjust(width)
            return (f"{val:.2f}x" if "/" in key else f"{val:.1f}").rjust(width)
        return str(val).rjust(width)

    header = "|".join(disp.rjust(w) for disp, _, w in cols)
    sep = "+".join("-" * w for _, _, w in cols)
    border = "=" * len(header)
    lines = [
        "",
        border,
        (
            "K5 Prefill Perf Summary (mfma16_hip vs hip vs triton; K5 device kernel us via "
            "torch.profiler; fly/hip & tri/hip = speedup vs hip, >1 faster / <1 slower)"
        ),
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
