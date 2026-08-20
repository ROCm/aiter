# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention Prefill (chunk_gated_delta_h) regressions.

Grid: Qwen3.5-35B (Hv=32) and Qwen3.5-397B (Hv=64), TP 1/2/4/8.
Dense T=1k/2k/4k/8k/16k/32k/64k; varlen total T=16k/32k/64k with seqlen 1k/2k/4k/8k.

Filter cases (omit shape flags for the full 304-case grid)::

    python op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py \\
        TestPerformance --model 397b --tp 4 --t 8192 --n 8 --snapshot-dtype bf16 fp32

Plain ``pytest`` without these flags still runs the full grid.
"""

from __future__ import annotations

import argparse
import math
import zlib
from dataclasses import dataclass, replace

import pytest
import torch
import triton
from torch.profiler import ProfilerActivity, profile

from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.prefill_batch_metadata import (
    build_gated_delta_rule_prefill_metadata,
)

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL Linear Attention Prefill tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_flydsl_opt,
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
    # Per-chunk h-snapshot dtype, an independent policy from the SSM state
    # dtype. None (the default) resolves to k.dtype (bf16 here), which is the
    # bf16 store specialization; torch.float32 selects the fp32 one.
    snapshot_dtype: object = None  # torch.dtype | None
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
        if self.snapshot_dtype == torch.float32:
            tag += "_snapFP32"
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
    # Per-chunk h-snapshot dtype; None -> k.dtype (bf16). See PrefillArgs.
    snapshot_dtype: object = None  # torch.dtype | None
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
                                snapshot_dtype=g.snapshot_dtype,
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
                                    snapshot_dtype=g.snapshot_dtype,
                                    context_lens=context_lens,
                                    trace_tag=tag,
                                    dense_batch=g.dense_batch,
                                    use_g=g.use_g,
                                    g_head_major=g.g_head_major,
                                )
                            )
    return out


_DENSE_PROMPT_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
_VARLEN_SEQLENS = [1024, 2048, 4096, 8192]
_VARLEN_TOTAL_T = [8192, 16384, 32768, 65536]
_K5_TPS = [1, 2, 4, 8]


def _k5_dense_groups(model_name: str, hv: int) -> list[PrefillGroup]:
    """Dense prefill: TP x T sweep x bf16/fp32 per-chunk snapshot."""
    groups: list[PrefillGroup] = []
    for tp in _K5_TPS:
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-dense-tp{tp}-bf16snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_DENSE_PROMPT_LENS,
                is_varlen=False,
                output_final_state=False,
                max_num_batched_tokens="full_prompt_len",
            )
        )
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-dense-tp{tp}-fp32snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_DENSE_PROMPT_LENS,
                is_varlen=False,
                output_final_state=False,
                max_num_batched_tokens="full_prompt_len",
                snapshot_dtype=torch.float32,
            )
        )
    return groups


def _k5_varlen_groups(model_name: str, hv: int) -> list[PrefillGroup]:
    """Varlen prefill: TP x seqlen x total T x bf16/fp32 snapshot."""
    groups: list[PrefillGroup] = []
    for tp in _K5_TPS:
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-varlen-tp{tp}-bf16snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_VARLEN_SEQLENS,
                max_num_batched_tokens=_VARLEN_TOTAL_T,
            )
        )
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-varlen-tp{tp}-fp32snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_VARLEN_SEQLENS,
                max_num_batched_tokens=_VARLEN_TOTAL_T,
                snapshot_dtype=torch.float32,
            )
        )
    return groups


_PREFILL_GROUPS = [
    *_k5_dense_groups("Qwen3.5-35B", 32),
    *_k5_dense_groups("Qwen3.5-397B", 64),
    *_k5_varlen_groups("Qwen3.5-35B", 32),
    *_k5_varlen_groups("Qwen3.5-397B", 64),
]


def _model_key(hv: int) -> str:
    return "35b" if hv == 32 else "397b"


def _snapshot_key(args: PrefillArgs) -> str:
    return "fp32" if args.snapshot_dtype == torch.float32 else "bf16"


def _current_cli_opts():
    import sys

    return _build_prefill_cli_parser().parse_known_args(sys.argv[1:])[0]


def _cli_has_filters(opts) -> bool:
    return any(
        [
            opts.model,
            opts.tp is not None,
            opts.t is not None,
            opts.n is not None,
            opts.dense,
            opts.snapshot_dtype,
        ]
    )


def _case_matches_cli(args: PrefillArgs, opts) -> bool:
    if opts.model and _model_key(args.Hv) != opts.model:
        return False
    if opts.tp is not None and args.tp != opts.tp:
        return False
    if opts.dense:
        if args.is_varlen:
            return False
    elif opts.n is not None and not args.is_varlen:
        return False
    if opts.t is not None:
        if args.is_varlen:
            if opts.n is not None:
                if (
                    args.full_prompt_len != opts.t
                    or args.max_num_batched_tokens != opts.n * opts.t
                ):
                    return False
            elif (
                args.full_prompt_len != opts.t and args.max_num_batched_tokens != opts.t
            ):
                return False
        elif args.full_prompt_len != opts.t:
            return False
    return not (opts.snapshot_dtype and _snapshot_key(args) not in opts.snapshot_dtype)


def _filtered_prefill_params():
    opts = _current_cli_opts()
    if not _cli_has_filters(opts):
        return PREFILL_PARAMS
    filtered = [p for p in PREFILL_PARAMS if _case_matches_cli(p, opts)]
    if not filtered:
        raise RuntimeError(
            "No PrefillArgs cases matched --model/--tp/--t/--n/--dense/--snapshot-dtype."
        )
    return filtered


PREFILL_PARAMS = expand_groups(_PREFILL_GROUPS)

PREFILL_TEST_IDS = [repr(p) for p in PREFILL_PARAMS]


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ not in (
        "test_correctness_flydsl_opt",
        "test_perf_comparison",
    ):
        return
    params = _filtered_prefill_params()
    metafunc.parametrize("args", params, ids=[repr(p) for p in params])


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
        full_prompt_len=1024,
        model_name="Qwen3.5-35B-bf16state",
        is_varlen=False,
        output_final_state=True,
        max_num_batched_tokens=1024,
        ssm_state_dtype=torch.bfloat16,
    ),
    PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=64,
        tp=4,
        full_prompt_len=1024,
        model_name="Qwen3.5-397B-bf16state",
        is_varlen=True,
        output_final_state=True,
        max_num_batched_tokens=16384,
        ssm_state_dtype=torch.bfloat16,
    ),
]
STATE_BF16_TEST_IDS = [repr(p) for p in STATE_BF16_PARAMS]


# -- fp32 chunk-snapshot params (paired with TestSnapshotDtype below) ---

# The snapshot dtype is an independent policy from the SSM state dtype, so these
# reuse the small bf16-state shapes (one dense, one varlen launch route) with the
# default fp32 state and vary only ``snapshot_dtype``.
SNAPSHOT_DTYPE_PARAMS = [
    replace(
        p,
        model_name=f"{p.model_name}-fp32snapshot",
        ssm_state_dtype=torch.float32,
    )
    for p in STATE_BF16_PARAMS
]
SNAPSHOT_DTYPE_TEST_IDS = [repr(p) for p in SNAPSHOT_DTYPE_PARAMS]


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


def _case_seed(context_lens, args: PrefillArgs = None) -> int:
    """Per-case seed derived from the case identity.

    ``crc32`` (not the builtin ``hash``) so the value is stable across
    processes regardless of ``PYTHONHASHSEED``, and derived per case so a
    shape's inputs no longer depend on how many other cases ran before it.
    """
    tag = f"{args!r}|{list(context_lens)}"
    return zlib.crc32(tag.encode()) & 0x7FFFFFFF


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
    seed=None,
):
    torch.manual_seed(_case_seed(context_lens, args) if seed is None else seed)
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
    (``kv`` / ``opt`` / ``mfma16_2wave_opt1`` / ``mfma16_3wave_opt2``)
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
    snapshot_dtype=None,
    prefill_metadata=None,
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
        snapshot_dtype=snapshot_dtype,
        prefill_metadata=prefill_metadata,
    )


# -- Performance benchmark ----------------------------------------------


_K5_KERNEL_PREFIXES = [
    "chunk_gdn_fwd_h_flydsl_vk",
    "chunk_gdn_fwd_h_flydsl_kv",
    "chunk_gdn_fwd_h_flydsl_opt",
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


def _build_prefill_metadata(context_lens, cu_seqlens, chunk_size: int = 64):
    """Prebuild the reusable GDR chunk schedule for a benchmarked shape.

    Serving stacks build this once per forward pass and hand it to every GDR
    kernel; benchmarks that skip it make each wrapper rediscover the chunk
    counts with a blocking device-to-host copy. Returns None for the dense
    (``cu_seqlens is None``) shapes, where the wrappers take the batch layout
    straight from the tensor shapes.
    """
    if cu_seqlens is None:
        return None
    return build_gated_delta_rule_prefill_metadata(
        list(context_lens),
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


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


def _truncate_to_bf16(x):
    """Keep the high 16 bits of an fp32 tensor, i.e. the HIP ``float_to_bf16``
    truncation the bf16 snapshot specialization applies to its accumulators."""
    return (x.contiguous().view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)


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

    def test_correctness_flydsl_opt(self, args: PrefillArgs):
        """K5 opt FlyDSL K5 impl (formerly the "vk" fork): 16x16x16
        MFMA + HIP warp partition. Same VK public outputs as the baseline flydsl
        path; only the BV==64 configs exercise the kernel, others fall back."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_opt(
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
            snapshot_dtype=args.snapshot_dtype,
        )
        assert h_fly.dtype == (args.snapshot_dtype or k.dtype)
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
            label="flydsl_opt",
        )

    @pytest.mark.parametrize("args", STATE_BF16_PARAMS, ids=STATE_BF16_TEST_IDS)
    def test_correctness_bf16_state(self, args: PrefillArgs):
        """Validate bf16 initial/final state on dense and varlen launch paths."""
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_opt(
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
            label="flydsl_opt_bf16_state",
        )

    @pytest.mark.parametrize("args", SNAPSHOT_DTYPE_PARAMS, ids=SNAPSHOT_DTYPE_TEST_IDS)
    def test_correctness_fp32_snapshot(self, args: PrefillArgs):
        """fp32 per-chunk snapshots on the dense and varlen launch paths.

        The fp32 specialization stores the f32 accumulators straight from
        registers while the bf16 one truncates the very same registers through
        the [V][K] LDS transpose buffer, so truncating the fp32 snapshots must
        reproduce the bf16 ones bit for bit. Everything else the kernel writes
        (``v_new``, ``final_state``) must be untouched by the snapshot policy.
        """
        context_lens = args.resolve_context_lens()
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(
            context_lens, args=args
        )

        def run(snapshot_dtype):
            return chunk_gated_delta_rule_fwd_h_flydsl_opt(
                k,
                w_c,
                u_c,
                g=g,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu,
                g_head_major=args.g_head_major,
                use_exp2=False,
                snapshot_dtype=snapshot_dtype,
            )

        h_bf16, vn_bf16, fs_bf16 = run(torch.bfloat16)
        h_f32, vn_f32, fs_f32 = run(torch.float32)

        assert h_bf16.dtype == torch.bfloat16
        assert h_f32.dtype == torch.float32
        assert fs_bf16.dtype == torch.float32 and fs_f32.dtype == torch.float32
        assert torch.equal(vn_bf16, vn_f32), "snapshot dtype perturbed v_new"
        assert torch.equal(fs_bf16, fs_f32), "snapshot dtype perturbed final_state"
        assert torch.equal(_truncate_to_bf16(h_f32), h_bf16), (
            "fp32 snapshots do not truncate back to the bf16 specialization's "
            "snapshots; the two paths are storing different accumulators"
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
        _assert_k5_outputs_match_ref(
            h_f32,
            vn_f32,
            fs_f32,
            h_ref,
            vn_ref,
            fs_ref,
            output_final_state=True,
            label="flydsl_opt_fp32_snapshot",
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

    def test_e2e_dispatch_indexed_state_pool(self):
        """K5 gathers from / writes back into an SGLang-style pool via dispatch."""
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
            "output_final_state": True,
            "use_exp2": True,
            "use_chunk_flydsl": True,
        }

        _, out_ref, fs_ref = chunk_gated_delta_rule_fwd_opt_vk(
            **kwargs, initial_state=h0.clone()
        )

        pool_size = B + 5
        indices = torch.tensor([3], device=h0.device, dtype=torch.int32)
        pool = torch.randn(pool_size, H, D, D, dtype=torch.float32, device=h0.device)
        pool_before = pool.clone()
        pool[indices.long()] = h0

        _, out_pool, returned = chunk_gated_delta_rule_fwd_opt_vk(
            **kwargs, initial_state=pool, initial_state_indices=indices
        )

        assert returned is pool
        torch.testing.assert_close(
            out_pool.float(), out_ref.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            pool[indices.long()].float(), fs_ref.float(), atol=2e-2, rtol=2e-2
        )

        untouched = torch.ones(pool_size, dtype=torch.bool, device=pool.device)
        untouched[indices.long()] = False
        assert torch.equal(pool[untouched], pool_before[untouched])

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

        _, _, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl_opt(
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

    # Every backend takes the chunk schedule a serving stack builds once per
    # forward pass. Without it the wrappers recover the chunk counts with a
    # blocking chunk_offsets D2H, which stalls the launch stream and measures
    # host behaviour the production path does not have.
    metadata = _build_prefill_metadata(context_lens, cu)

    us_fly = _bench_fn(
        chunk_gated_delta_rule_fwd_h_flydsl_opt,
        k,
        w_c,
        u_c,
        g=g,
        initial_state=h0,
        output_final_state=ofs,
        cu_seqlens=cu,
        g_head_major=args.g_head_major,
        snapshot_dtype=args.snapshot_dtype,
        prefill_metadata=metadata,
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
        snapshot_dtype=args.snapshot_dtype,
        prefill_metadata=metadata,
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
            snapshot_dtype=args.snapshot_dtype,
            prefill_metadata=metadata,
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
            "snap": "fp32" if args.snapshot_dtype == torch.float32 else "bf16",
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
        ("snap", "snap", 4),
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
            "K5 Prefill Perf Summary (opt vs hip vs triton; K5 device kernel us via "
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
    def test_perf_comparison(self, args: PrefillArgs):
        _run_perf_comparison(args)


def _build_prefill_cli_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=["35b", "397b"], default=None)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=None)
    parser.add_argument("--t", type=int, default=None)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="varlen sequence count (mnbt = n * t); ignored with --dense",
    )
    parser.add_argument("--dense", action="store_true", default=False)
    parser.add_argument(
        "--snapshot-dtype",
        nargs="+",
        choices=["bf16", "fp32"],
        default=None,
    )
    return parser


if __name__ == "__main__":
    import sys

    _, pytest_argv = _build_prefill_cli_parser().parse_known_args(sys.argv[1:])
    raise SystemExit(pytest.main([__file__, *pytest_argv]))
