#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end benchmark: FlyDSL sparse MLA prefill (B2/GLM) vs Triton baselines.

Metric: **GPU-kernel-only time** (summed CUDA/HIP device self-time per iter, via
the torch profiler).  Model inference is GPU-bound and asynchronous — the host
enqueues kernels and only waits for the GPU at the end — so host launch/sync
overhead is not what we optimize; kernel device time is.  Wall time is ignored.

Fair comparison rule: still sum **all** prep kernels the production stack runs
before the core attention op.  The FlyDSL native path fuses gather/dequant/attn;
the Triton production prefill path does not — so ``triton_prefill_e2e`` counts
the dequant kernels + attention, not attention alone.  The per-kernel breakdown
printed under each result shows where the device time goes.

Run (gfx942, dsv4 preset: B2 main=512 + extra=128):

    cd /home/AMD/samremes/dev/aiter
    python op_tests/flydsl_tests/bench_sparse_mla_prefill.py
    python op_tests/flydsl_tests/bench_sparse_mla_prefill.py --T-sweep
    python op_tests/flydsl_tests/bench_sparse_mla_prefill.py --T 4096

Requires: flydsl (gfx942), triton (vendored Triton baselines in this directory).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import torch

# dsv4 sparse MLA prefill shapes (index_topk=512 compressed, sliding_window=128 SWA).
DSV4_TOPK_MAIN = 512
DSV4_TOPK_EXTRA = 128
# GLM-5 / DSv3.2: single global top-k over a flat fp8 576 latent cache.
GLM_TOPK = 2048
# Per-batch split cap for the persistent MLA sparse decode metadata. Mirrors
# test_mla_sparse.py's `max_split_per_batch=32`; bounds the number of partial
# reduce tiles so the decode's `logits`/`reduce_partial_map` buffers stay sized
# correctly (an unbounded split count overflows them -> HIP illegal access).
GLM_MAX_SPLIT_PER_BATCH = 32
T_SWEEP_DEFAULT: tuple[int, ...] = (4096, 8192, 16384)

_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 3)))
if os.path.isdir(os.path.join(_AITER_ROOT, "aiter")) and _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)
_flydsl_root = os.path.join(_DEV, "FlyDSL")

from op_tests.flydsl_tests.sparse_mla_prefill_ref import (  # noqa: E402
    GLM_HEAD_DIM,
    GLM_V_DIM,
    H_PROD,
    NOPE_HEAD_DIM,
    PACKED_HEAD_DIM,
    ROPE_HEAD_DIM,
    default_scale,
    default_scale_glm,
    gen_kv,
    gen_kv_glm,
    gen_q,
    gen_q_glm,
    gen_ragged_rows,
    identity_block_table,
    merge_two_region_csrs,
    pack_fp8_ds_mla_cache,
    pack_glm_fp8_cache,
    ragged_from_rows,
)
from op_tests.flydsl_tests.sparse_mla_prefill_triton_baseline import (  # noqa: E402
    is_fp8_fnuz,
    rocm_sparse_attn_decode_ragged_triton,
    rocm_sparse_attn_prefill_ragged_triton,
)


def _ensure_flydsl() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    try:
        import flydsl  # noqa: F401
        from flydsl._mlir import ir  # noqa: F401
    except Exception as exc:
        for c in (
            os.environ.get("FLYDSL_PKGS"),
            os.path.join(_DEV, ".r1_flydsl_pkgs"),
            os.path.join(_flydsl_root, "build-fly", "python_packages"),
            os.path.join(_flydsl_root, "python"),
        ):
            if c and os.path.isdir(os.path.join(c, "flydsl")) and c not in sys.path:
                sys.path.insert(0, c)
        try:
            import flydsl  # noqa: F401
            from flydsl._mlir import ir  # noqa: F401
        except Exception as exc2:
            raise RuntimeError(f"flydsl not importable: {exc2}") from exc
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower().split(":")[0]
    if not arch.startswith("gfx942"):
        raise RuntimeError(f"bench targets gfx942, got {arch}")


def _is_gfx942_fnuz() -> bool:
    return is_fp8_fnuz()


@dataclass
class TimingResult:
    name: str
    total_ms: float
    stages_ms: dict[str, float]
    notes: str = ""


class StageTimer:
    """CUDA-event timer; accumulates named stages within one forward pass."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._start: torch.cuda.Event | None = None
        self._cur: str | None = None

    def begin(self, name: str) -> None:
        if self._start is not None:
            self._end_stage()
        self._cur = name
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def _end_stage(self) -> None:
        assert self._start is not None and self._cur is not None
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        self.stages[self._cur] = self.stages.get(self._cur, 0.0) + self._start.elapsed_time(end)
        self._start = None
        self._cur = None

    def end(self) -> None:
        self._end_stage()

    def finish(self) -> dict[str, float]:
        if self._start is not None:
            self._end_stage()
        return dict(self.stages)


def _short_kernel(name: str) -> str:
    """Compact a CUDA/HIP kernel symbol for the per-kernel breakdown line."""
    base = name.split("(")[0].split("<")[0].strip()
    base = base.rsplit("::", 1)[-1].strip()
    return (base or name)[:30]


def _median_ms(fn: Callable[[], None], warmup: int, iters: int) -> tuple[float, dict[str, float]]:
    """Return GPU-kernel-only ms/iter (device self-time) and a per-kernel breakdown.

    Model inference is GPU-bound and asynchronous: the host enqueues kernels and
    only waits for the GPU at the end, so **host launch/sync overhead is not what
    we should optimize** — kernel device time is.  We therefore ignore wall time
    entirely and sum every CUDA/HIP kernel's *device self-time* over ``iters``
    profiled iterations (after ``warmup``), dividing by ``iters``.  The returned
    ``stages`` dict is the per-kernel device-time breakdown (descending), which
    replaces the old wall-clock StageTimer stages.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    per_kernel: dict[str, float] = {}
    total_us = 0.0
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0.0) or getattr(e, "self_cuda_time_total", 0.0)
        if us <= 0:
            continue  # host-only op (launch, etc.) — ignored on purpose
        total_us += us
        per_kernel[e.key] = per_kernel.get(e.key, 0.0) + us
    inv = 1.0 / (iters * 1000.0)  # us-total -> ms/iter
    stages = {
        _short_kernel(k): v * inv
        for k, v in sorted(per_kernel.items(), key=lambda kv: -kv[1])
    }
    return total_us * inv, stages


# ---------------------------------------------------------------------------
# Triton: dequant fp8_ds_mla packed cache -> dense bf16 [num_slots, 512]
# (production prefill materializes bf16 before tl.dot)
# ---------------------------------------------------------------------------
_dequant_module = None


def _get_dequant_kernel(is_extra: bool, block_size: int):
    global _dequant_module
    import triton
    import triton.language as tl

    key = (is_extra, block_size)
    if _dequant_module is not None and key in _dequant_module:
        return _dequant_module[key]

    IS_OCP = is_extra
    NOPE = NOPE_HEAD_DIM
    ROPE = ROPE_HEAD_DIM

    @triton.jit
    def _dequant_fp8_ds_mla_kernel(
        cache_ptr,
        out_ptr,
        cache_block_stride,
        num_slots,
        slot_ids_ptr,
        BLOCK_SIZE: tl.constexpr,
        NOPE_DIM: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        IS_OCP: tl.constexpr,
        USE_SLOT_LIST: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if USE_SLOT_LIST:
            slot = tl.load(slot_ids_ptr + pid).to(tl.int32)
        else:
            slot = pid
        if slot >= num_slots:
            return
        block_idx = slot // BLOCK_SIZE
        pos = slot % BLOCK_SIZE
        block_base = block_idx * cache_block_stride
        token_base = block_base + pos * 576
        scale_base = block_base + BLOCK_SIZE * 576 + pos * 8
        out_row = out_ptr + slot * (NOPE_DIM + ROPE_DIM)

        for qblock in tl.static_range(7):
            qstart = qblock * 64
            offs = qstart + tl.arange(0, 64)
            mask = offs < NOPE_DIM
            x_u8 = tl.load(cache_ptr + token_base + offs, mask=mask, other=0)
            if IS_OCP:
                x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True)
            else:
                x_fp8 = x_u8.to(tl.float8e4b8, bitcast=True)
            enc = tl.load(cache_ptr + scale_base + qblock).to(tl.float32)
            sc = tl.exp2(enc - 127.0)
            val = x_fp8.to(tl.bfloat16) * sc.to(tl.bfloat16)
            tl.store(out_row + offs, val, mask=mask)

        rope_base = token_base + NOPE_DIM
        for j in tl.static_range(ROPE_DIM // 16):
            roff = j * 16 + tl.arange(0, 16)
            rmask = roff < ROPE_DIM
            rope_ptr = (cache_ptr + rope_base).to(tl.pointer_type(tl.bfloat16))
            rv = tl.load(rope_ptr + roff, mask=rmask, other=0.0)
            tl.store((out_row + NOPE_DIM).to(tl.pointer_type(tl.bfloat16)) + roff, rv, mask=rmask)

    if _dequant_module is None:
        _dequant_module = {}
    _dequant_module[key] = _dequant_fp8_ds_mla_kernel
    return _dequant_fp8_ds_mla_kernel


def dequant_cache_to_bf16(
    cache: torch.Tensor,
    block_size: int,
    *,
    is_extra: bool = False,
    timer: StageTimer | None = None,
    slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize fp8_ds_mla cache rows to bf16.

    If ``slots`` is set, only those physical cache slots are dequantized and the
    returned tensor is indexed by slot id (shape ``[num_cache_slots, 512]`` with
    only referenced rows filled). This matches production ``dequantize_and_gather``
    better than materializing the entire cache when CSR hits a sparse subset.
    """
    num_blocks, blk, width = cache.shape
    assert width == 584 and blk == block_size
    num_slots = num_blocks * block_size
    use_slot_list = slots is not None
    if use_slot_list:
        slot_ids = slots.to(device=cache.device, dtype=torch.int32).unique()
        n_prog = int(slot_ids.numel())
    else:
        slot_ids = torch.empty(0, device=cache.device, dtype=torch.int32)
        n_prog = num_slots
    out = torch.zeros(num_slots, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=cache.device)
    if n_prog == 0:
        return out
    kernel = _get_dequant_kernel(is_extra, block_size)
    cache_flat = cache.contiguous().view(torch.uint8).reshape(-1)
    if timer is not None:
        timer.begin("dequant")
    kernel[(n_prog,)](
        cache_flat,
        out,
        cache.stride(0),
        num_slots,
        slot_ids,
        BLOCK_SIZE=block_size,
        NOPE_DIM=NOPE_HEAD_DIM,
        ROPE_DIM=ROPE_HEAD_DIM,
        IS_OCP=is_extra,
        USE_SLOT_LIST=use_slot_list,
        num_warps=4,
    )
    if timer is not None:
        timer.end()
    return out


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------
@dataclass
class BenchInputsB2:
    q: torch.Tensor
    main_cache: torch.Tensor
    main_indices: torch.Tensor
    main_indptr: torch.Tensor
    main_bt: torch.Tensor
    extra_cache: torch.Tensor
    extra_indices: torch.Tensor
    extra_indptr: torch.Tensor
    extra_bt: torch.Tensor
    block_size: int
    sink: torch.Tensor
    scale: float
    out: torch.Tensor
    main_rows: list[list[int]]
    extra_rows: list[list[int]]
    merged_indices: torch.Tensor
    merged_indptr: torch.Tensor
    main_unique_slots: torch.Tensor
    extra_unique_slots: torch.Tensor


def build_b2_inputs(
    T: int,
    topk_main: int,
    topk_extra: int,
    main_tokens: int,
    extra_tokens: int,
    block_size: int,
    seed: int,
    device: str = "cuda",
) -> BenchInputsB2:
    main_kv = gen_kv(main_tokens, seed=seed)
    extra_kv = gen_kv(extra_tokens, seed=seed + 1)
    main_cache = pack_fp8_ds_mla_cache(main_kv, block_size, is_extra=False)
    extra_cache = pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)
    main_rows = gen_ragged_rows(T, topk_main, main_tokens, seed=seed + 2)
    extra_rows = gen_ragged_rows(T, topk_extra, extra_tokens, seed=seed + 3)
    main_indices, main_indptr = ragged_from_rows(main_rows, torch.device(device))
    extra_indices, extra_indptr = ragged_from_rows(extra_rows, torch.device(device))
    q = gen_q(T, H_PROD, seed=seed + 4)
    sink = torch.randn(H_PROD, dtype=torch.float32, device=device) * 0.3
    out = torch.empty(T, H_PROD, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    main_slots = main_cache.shape[0] * main_cache.shape[1]
    merged_indices, merged_indptr = merge_two_region_csrs(main_rows, extra_rows, main_slots)
    return BenchInputsB2(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        main_bt=identity_block_table(main_tokens, block_size, torch.device(device)),
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        extra_bt=identity_block_table(extra_tokens, block_size, torch.device(device)),
        block_size=block_size,
        sink=sink,
        scale=default_scale(),
        out=out,
        main_rows=main_rows,
        extra_rows=extra_rows,
        merged_indices=merged_indices,
        merged_indptr=merged_indptr,
        main_unique_slots=main_indices.unique(),
        extra_unique_slots=extra_indices.unique(),
    )


@dataclass
class BenchInputsGLM:
    q: torch.Tensor
    cache: torch.Tensor
    indices: torch.Tensor
    indptr: torch.Tensor
    block_table: torch.Tensor
    block_size: int
    scale: float
    kv_scale: float
    out: torch.Tensor


def build_glm_inputs(
    T: int,
    topk: int,
    num_tokens: int,
    block_size: int,
    seed: int,
    kv_scale: float = 1.0,
    device: str = "cuda",
) -> BenchInputsGLM:
    kv = gen_kv_glm(num_tokens, seed=seed)
    cache = pack_glm_fp8_cache(kv, block_size, kv_scale=kv_scale)
    rows = gen_ragged_rows(T, topk, num_tokens, seed=seed + 1)
    indices, indptr = ragged_from_rows(rows, torch.device(device))
    q = gen_q_glm(T, H_PROD, seed=seed + 2)
    out = torch.empty(T, H_PROD, GLM_V_DIM, dtype=torch.bfloat16, device=device)
    return BenchInputsGLM(
        q=q,
        cache=cache,
        indices=indices,
        indptr=indptr,
        block_table=identity_block_table(num_tokens, block_size, torch.device(device)),
        block_size=block_size,
        scale=default_scale_glm(),
        kv_scale=kv_scale,
        out=out,
    )


def bench_flydsl_glm(inp: BenchInputsGLM, warmup: int, iters: int) -> TimingResult:
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    def run_once():
        timer = StageTimer()
        timer.begin("flydsl_kernel")
        flydsl_sparse_mla_prefill(
            inp.q,
            inp.cache,
            inp.indices,
            inp.indptr,
            inp.out,
            block_table=inp.block_table,
            block_size=inp.block_size,
            packed=True,
            scale_mode="per_tensor",
            kv_scale=torch.tensor([inp.kv_scale], dtype=torch.float32, device=inp.q.device),
        )
        return timer.finish()

    run_once()
    med, stages = _median_ms(run_once, warmup, iters)
    return TimingResult("flydsl_glm_e2e", med, stages, notes="native flat-576 fp8, fused, per-tensor scale")


# ---------------------------------------------------------------------------
# Production aiter MLA sparse decode path (matches vLLM rocm_aiter_mla_sparse:
# forward_mqa -> q fp8 quant -> _forward_mla -> aiter.mla.mla_decode_fwd with
# pre-built persistent split+reduce metadata).
#
# The aiter MLA decode kernel always runs with page_size=1 internally: the
# kv cache is flattened to [num_slots, 1, 1, 576] and CSR indices are global
# slot ids (see vllm rocm_aiter_mla.py "always operates with page_size=1").
# The persistent work-split / reduce metadata is a function of the bench shape,
# so it is built ONCE here (NOT timed) — the steady-state analogue of vLLM's
# fingerprint-cached metadata in ROCMAiterMLASparseMetadataBuilder.build().
# ---------------------------------------------------------------------------
@dataclass
class GlmDecodeState:
    kv_buffer: torch.Tensor  # [num_slots, 1, 1, 576] fp8 (flat view of inp.cache)
    qo_indptr: torch.Tensor  # [T+1] int32, arange (qseqlen=1 per token)
    kv_indptr: torch.Tensor  # [T+1] int32, per-query CSR lengths (= inp.indptr)
    kv_indices: torch.Tensor  # [nnz] int32, global slot ids (= inp.indices)
    kv_last_page_lens: torch.Tensor  # [T] int32, ones
    q_fp8: torch.Tensor  # [T, H, 576] fp8 reusable quant output buffer
    q_scale: torch.Tensor  # [1] f32
    kv_scale: torch.Tensor  # [1] f32
    out_asm: torch.Tensor  # [T, H, 512] bf16
    sm_scale: float
    nhead: int
    quant_backend: str
    num_kv_splits: int  # per-batch split cap (mla_decode_fwd + metadata must agree)
    # persistent split+reduce metadata (built once, reused every forward)
    work_meta_data: torch.Tensor
    work_indptr: torch.Tensor
    work_info_set: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor


def _quant_q_fp8(q: torch.Tensor, q_scale: torch.Tensor, out_fp8: torch.Tensor, backend: str) -> None:
    """Static per-tensor fp8 quant of q, matching vLLM ``scaled_fp8_quant(q.flatten(-2), q_scale)``.

    Writes into the pre-allocated ``out_fp8`` ([T, H, 576] fp8). ``backend`` is
    resolved once in :func:`build_glm_decode_state` ("aiter" fused kernel, else
    a torch ``(q / scale).to(fp8)`` fallback).
    """
    T = q.shape[0]
    q_flat = q.reshape(T, -1)
    out_flat = out_fp8.reshape(T, -1)
    if backend == "aiter":
        from aiter.ops.quant import static_per_tensor_quant

        static_per_tensor_quant(out_flat, q_flat, q_scale)
    else:
        out_flat.copy_((q_flat.to(torch.float32) / q_scale).to(out_fp8.dtype))


def build_glm_decode_state(
    inp: BenchInputsGLM, topk: int, *, q_scale: float = 1.0
) -> GlmDecodeState:
    import aiter
    from aiter import dtypes

    device = inp.q.device
    T, nhead, head_dim = inp.q.shape
    assert head_dim == GLM_HEAD_DIM, f"GLM decode expects head_dim={GLM_HEAD_DIM}, got {head_dim}"
    assert nhead % 8 == 0, f"aiter MLA metadata requires nhead % 8 == 0, got {nhead}"

    num_slots = inp.cache.shape[0] * inp.cache.shape[1]
    # Flatten cache to the page_size=1 layout the kernel expects (mirrors the
    # vLLM wrapper's kv_buffer.view(-1, 1, 1, H)).
    kv_buffer = inp.cache.reshape(num_slots, GLM_HEAD_DIM).view(num_slots, 1, 1, GLM_HEAD_DIM)

    qo_indptr = torch.arange(T + 1, dtype=torch.int32, device=device)
    kv_indptr = inp.indptr.to(device=device, dtype=torch.int32)
    kv_indices = inp.indices.to(device=device, dtype=torch.int32)
    kv_last_page_lens = torch.ones(T, dtype=torch.int32, device=device)

    q_fp8 = torch.empty((T, nhead, head_dim), dtype=dtypes.fp8, device=device)
    q_scale_t = torch.tensor([q_scale], dtype=torch.float32, device=device)
    kv_scale_t = torch.tensor([inp.kv_scale], dtype=torch.float32, device=device)
    out_asm = torch.empty((T, nhead, GLM_V_DIM), dtype=torch.bfloat16, device=device)

    # Resolve the q-quant backend once (fused aiter kernel if it builds/runs).
    quant_backend = "torch"
    try:
        from aiter.ops.quant import static_per_tensor_quant

        static_per_tensor_quant(q_fp8.reshape(T, -1), inp.q.reshape(T, -1), q_scale_t)
        quant_backend = "aiter"
    except Exception:
        quant_backend = "torch"

    # ---- persistent split+reduce metadata buffers (allocated + filled once) ----
    # The decode runs the sparse top-k MLA path (mirrors test_mla_sparse.py): the
    # metadata MUST be built in sparse mode (topk set) with a per-batch split cap,
    # and the buffer-sizing query MUST use the same cap so reduce_partial_map (and
    # the decode's logits buffer derived from it) is large enough. Omitting topk /
    # max_split_per_batch sizes the buffers for ~1 split per batch but then lets the
    # metadata kernel emit up to `num_clusters` splits, overflowing them and
    # triggering a HIP illegal memory access in the stage1 ASM kernel.
    num_kv_splits = GLM_MAX_SPLIT_PER_BATCH
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        T,
        1,
        nhead,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=True,
        fast_mode=True,
        num_kv_splits=num_kv_splits,
        max_split_per_batch=num_kv_splits,
    )
    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type, device=device)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device)
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
    )

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead,  # num_heads_per_head_k
        1,  # num_heads_k
        True,  # is_causal
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        topk=topk,
        max_split_per_batch=num_kv_splits,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
    )

    return GlmDecodeState(
        kv_buffer=kv_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        q_fp8=q_fp8,
        q_scale=q_scale_t,
        kv_scale=kv_scale_t,
        out_asm=out_asm,
        sm_scale=inp.scale,
        nhead=nhead,
        quant_backend=quant_backend,
        num_kv_splits=num_kv_splits,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )


def _run_mla_decode_once(inp: BenchInputsGLM, state: GlmDecodeState, timer: StageTimer | None = None) -> None:
    """One production forward: q fp8 quant + persistent split decode ASM + reduce."""
    import aiter

    if timer is not None:
        timer.begin("q_quant")
    _quant_q_fp8(inp.q, state.q_scale, state.q_fp8, state.quant_backend)
    if timer is not None:
        timer.begin("mla_decode_fwd")
    aiter.mla.mla_decode_fwd(
        state.q_fp8,
        state.kv_buffer,
        state.out_asm,
        state.qo_indptr,
        state.kv_indptr,
        state.kv_indices,
        state.kv_last_page_lens,
        1,  # max_seqlen_q
        page_size=1,
        nhead_kv=1,
        sm_scale=state.sm_scale,
        num_kv_splits=state.num_kv_splits,
        q_scale=state.q_scale,
        kv_scale=state.kv_scale,
        work_meta_data=state.work_meta_data,
        work_indptr=state.work_indptr,
        work_info_set=state.work_info_set,
        reduce_indptr=state.reduce_indptr,
        reduce_final_map=state.reduce_final_map,
        reduce_partial_map=state.reduce_partial_map,
        return_lse=False,
    )
    if timer is not None:
        timer.end()


def bench_mla_decode_fwd_glm(
    inp: BenchInputsGLM, state: GlmDecodeState, warmup: int, iters: int
) -> TimingResult:
    def run_once():
        timer = StageTimer()
        _run_mla_decode_once(inp, state, timer=timer)
        return timer.finish()

    # First call JIT-compiles the decode ASM; do it before the timed loop.
    run_once()
    med, stages = _median_ms(run_once, warmup, iters)
    return TimingResult(
        "aiter_mla_decode_glm_e2e",
        med,
        stages,
        notes=(
            f"prod path: q fp8 quant ({state.quant_backend}) + persistent "
            "split decode ASM + reduce (metadata pre-built)"
        ),
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).to(torch.float32)
    bf = b.reshape(-1).to(torch.float32)
    return torch.nn.functional.cosine_similarity(af, bf, dim=0).item()


def check_glm_paths(inp: BenchInputsGLM, state: GlmDecodeState) -> float:
    """Run flydsl and the decode path once on identical inputs; return cosine.

    Both paths read the SAME bf16 q, SAME fp8 cache and SAME CSR; only the
    output buffers differ. flydsl consumes bf16 q directly while the decode
    path quantizes q to fp8 first, so a small divergence is expected.
    """
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    fly_out = torch.empty_like(inp.out)
    flydsl_sparse_mla_prefill(
        inp.q,
        inp.cache,
        inp.indices,
        inp.indptr,
        fly_out,
        block_table=inp.block_table,
        block_size=inp.block_size,
        packed=True,
        scale_mode="per_tensor",
        kv_scale=state.kv_scale,
    )
    _run_mla_decode_once(inp, state)
    return _cosine(fly_out, state.out_asm)


def bench_flydsl_b2(inp: BenchInputsB2, warmup: int, iters: int) -> TimingResult:
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region

    def run_once():
        timer = StageTimer()
        timer.begin("flydsl_kernel")
        flydsl_sparse_mla_prefill_2region(
            inp.q,
            inp.out,
            inp.main_cache,
            inp.main_indices,
            inp.main_indptr,
            inp.main_bt,
            inp.extra_cache,
            inp.extra_indices,
            inp.extra_indptr,
            inp.extra_bt,
            block_size=inp.block_size,
            attn_sink=inp.sink,
            main_is_fnuz=_is_gfx942_fnuz(),
            extra_is_fnuz=False,
        )
        return timer.finish()

    run_once()
    med, stages = _median_ms(run_once, warmup, iters)
    return TimingResult("flydsl_b2_e2e", med, stages, notes="native 2-region fp8, fused")


def bench_triton_prefill_e2e_b2(
    inp: BenchInputsB2, warmup: int, iters: int, *, csr_dequant: bool
) -> TimingResult:
    """Production-like: dequant both caches, merged CSR prefill (CSR pre-built)."""
    main_slots = inp.main_cache.shape[0] * inp.main_cache.shape[1]
    main_slots_arg = inp.main_unique_slots if csr_dequant else None
    extra_slots_arg = inp.extra_unique_slots if csr_dequant else None

    def run_once():
        timer = StageTimer()
        kv_main = dequant_cache_to_bf16(
            inp.main_cache,
            inp.block_size,
            is_extra=False,
            timer=timer,
            slots=main_slots_arg,
        )
        timer.begin("dequant_extra")
        kv_extra = dequant_cache_to_bf16(
            inp.extra_cache,
            inp.block_size,
            is_extra=True,
            timer=None,
            slots=extra_slots_arg,
        )
        timer.end()
        kv_all = torch.cat([kv_main[:main_slots], kv_extra], dim=0)
        timer.begin("triton_prefill")
        rocm_sparse_attn_prefill_ragged_triton(
            q=inp.q,
            kv=kv_all,
            indices=inp.merged_indices,
            indptr=inp.merged_indptr,
            scale=inp.scale,
            attn_sink=inp.sink,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
        )
        return timer.finish()

    run_once()
    med, stages = _median_ms(run_once, warmup, iters)
    return TimingResult(
        "triton_prefill_e2e_b2",
        med,
        stages,
        notes=(
            "CSR-slot dequant×2 + bf16 prefill; CSR merge pre-built (not timed)"
            if csr_dequant
            else "full-cache dequant×2 + bf16 prefill (pessimistic)"
        ),
    )


def bench_triton_decode_kernel_b2(inp: BenchInputsB2, warmup: int, iters: int) -> TimingResult:
    """ROCm sparse decode kernel on fp8 caches (dequant inside kernel, bf16 dot).

    Not the production prefill stack, but the current ROCm sparse *decode* path.
    Included for context — do not compare its total directly to prod prefill E2E.
    """
    def run_once():
        timer = StageTimer()
        timer.begin("triton_decode")
        rocm_sparse_attn_decode_ragged_triton(
            q=inp.q,
            main_cache=inp.main_cache,
            main_indices=inp.main_indices,
            main_indptr=inp.main_indptr,
            scale=inp.scale,
            attn_sink=inp.sink,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            extra_cache=inp.extra_cache,
            extra_indices=inp.extra_indices,
            extra_indptr=inp.extra_indptr,
        )
        return timer.finish()

    run_once()
    med, stages = _median_ms(run_once, warmup, iters)
    return TimingResult(
        "triton_decode_kernel_b2",
        med,
        stages,
        notes="fp8 cache in-kernel dequant; decode kernel not prefill E2E",
    )


def _print_result(r: TimingResult, baseline_ms: float | None) -> None:
    # stages are per-kernel device times (descending); keep that order, not sorted by name.
    stage_str = ", ".join(f"{k}={v:.3f}ms" for k, v in r.stages_ms.items())
    ratio = ""
    if baseline_ms is not None and baseline_ms > 0:
        ratio = f"  ({r.total_ms / baseline_ms:.2f}x vs flydsl)"
    print(f"{r.name:28s}  gpu_kernel={r.total_ms:8.3f} ms{ratio}")
    if stage_str:
        print(f"{'':28s}  kernels: {stage_str}")
    if r.notes:
        print(f"{'':28s}  note: {r.notes}")


def run_bench(args: argparse.Namespace, T: int) -> None:
    topk_main = args.topk_main
    topk_extra = args.topk_extra
    csr_dequant = not args.full_cache_dequant

    print(
        f"sparse_mla_prefill bench  gfx942  T={T}  block_size={args.block_size}  "
        f"warmup={args.warmup} iters={args.iters}  metric=gpu_kernel_ms (device self-time)  "
        f"triton_dequant={'csr_slots' if csr_dequant else 'full_cache'}"
    )
    print()

    print(
        f"=== B2 two-region (topk_main={topk_main}, topk_extra={topk_extra}, "
        f"main_tokens={args.main_tokens}, extra_tokens={args.extra_tokens}) ==="
    )
    b2 = build_b2_inputs(
        T,
        topk_main,
        topk_extra,
        args.main_tokens,
        args.extra_tokens,
        args.block_size,
        args.seed + 100,
    )
    fly_b2 = bench_flydsl_b2(b2, args.warmup, args.iters)
    _print_result(fly_b2, None)
    if not args.skip_triton:
        triton_e2e_b2 = bench_triton_prefill_e2e_b2(b2, args.warmup, args.iters, csr_dequant=csr_dequant)
        _print_result(triton_e2e_b2, fly_b2.total_ms)
        triton_dec = bench_triton_decode_kernel_b2(b2, args.warmup, args.iters)
        _print_result(triton_dec, fly_b2.total_ms)
    print()
    print("All times are summed GPU kernel device self-time per iter (host overhead ignored).")
    print("Compare triton_prefill_e2e_* against flydsl_*_e2e for fair E2E; the per-kernel")
    print("breakdown shows where device time goes. triton_decode_kernel_* is a decode")
    print("subgraph reference only (not the prefill path this kernel replaces).")
    print()


def run_bench_glm(args: argparse.Namespace, T: int) -> None:
    print(
        f"sparse_mla_prefill bench  gfx942  preset=glm  T={T}  "
        f"block_size={args.block_size}  warmup={args.warmup} iters={args.iters}"
    )
    print()
    print(
        f"=== GLM/DSv3.2 single-region (head_dim={GLM_HEAD_DIM}, topk={args.topk}, "
        f"num_tokens={args.num_tokens}, kv_scale={args.kv_scale}) ==="
    )
    glm = build_glm_inputs(
        T, args.topk, args.num_tokens, args.block_size, args.seed, args.kv_scale
    )
    fly_glm = bench_flydsl_glm(glm, args.warmup, args.iters)
    _print_result(fly_glm, None)

    if args.skip_aiter_decode:
        print()
        print("(aiter mla_decode_fwd skipped via --skip-aiter-decode)")
        print()
        return

    try:
        state = build_glm_decode_state(glm, args.topk, q_scale=args.q_scale)
    except Exception as exc:  # ASM JIT / arch / metadata build failure
        print()
        print(f"aiter mla_decode_fwd unavailable, skipping decode peer: {exc}")
        print()
        return

    try:
        cos = check_glm_paths(glm, state)
        tag = "WARNING: paths diverge" if cos < 0.90 else "ok"
        print(f"correctness  flydsl vs mla_decode_fwd cosine={cos:.4f}  [{tag}]")
    except Exception as exc:
        print(f"correctness  check skipped: {exc}")

    dec_glm = bench_mla_decode_fwd_glm(glm, state, args.warmup, args.iters)
    _print_result(dec_glm, fly_glm.total_ms)
    print()
    print(
        "flydsl_glm_e2e          = fused gather/dequant/attn (bf16 q in, fp8 cache)."
    )
    print(
        "aiter_mla_decode_glm_e2e = production path (vLLM rocm_aiter_mla_sparse): "
        "q fp8 quant + persistent split decode ASM + reduce; metadata pre-built."
    )
    print("Ratio shown for the decode row is decode_total / flydsl_total (<1 => decode faster).")
    print()


def _parse_T_sweep(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return T_SWEEP_DEFAULT
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip().lower().replace("_", "")
        if not part:
            continue
        if part.endswith("k"):
            out.append(int(float(part[:-1]) * 1024))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("--T-sweep requires at least one size (e.g. 4096,8192,16384 or 4k,8k,16k)")
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse MLA prefill E2E benchmark (gfx942, DSv4 defaults)",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=4096,
        help="Prefill query tokens (num_queries) for a single run",
    )
    parser.add_argument(
        "--T-sweep",
        nargs="?",
        const="",
        default=None,
        metavar="SIZES",
        help=(
            "Run multiple T values. Default list: 4096,8192,16384. "
            "Optional comma list (supports 4k/8k/16k suffixes)."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("dsv4", "dsv32", "glm"),
        default="dsv4",
        help="dsv4: B2 two-region only. glm/dsv32: single-region flat fp8 head_dim=576, topk=2048.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="CSR length per query for glm/dsv32 single-region runs (default: 2048)",
    )
    parser.add_argument(
        "--kv-scale",
        type=float,
        default=1.0,
        help="GLM per-tensor KV dequant scalar (layer._k_scale); glm/dsv32 preset only",
    )
    parser.add_argument(
        "--q-scale",
        type=float,
        default=1.0,
        help="GLM q fp8 quant scalar (layer._q_scale); glm/dsv32 decode peer only "
        "(matches FlyDSL default q_scale=1.0)",
    )
    parser.add_argument(
        "--skip-aiter-decode",
        action="store_true",
        help="glm/dsv32: skip the production aiter.mla.mla_decode_fwd peer",
    )
    parser.add_argument(
        "--topk-main",
        type=int,
        default=DSV4_TOPK_MAIN,
        help="B2 main-region CSR length per query (DSv4 default: 512)",
    )
    parser.add_argument(
        "--topk-extra",
        type=int,
        default=DSV4_TOPK_EXTRA,
        help="B2 extra-region CSR length per query (DSv4 default: 128)",
    )
    parser.add_argument("--num-tokens", type=int, default=65536, help="glm/dsv32 cache slots")
    parser.add_argument("--main-tokens", type=int, default=65536, help="B2 SWA cache slots")
    parser.add_argument("--extra-tokens", type=int, default=32768, help="B2 compressed cache slots")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument(
        "--full-cache-dequant",
        action="store_true",
        help="Dequant entire cache (pessimistic). Default: CSR-referenced slots only.",
    )
    args = parser.parse_args()

    _ensure_flydsl()

    is_glm = args.preset in ("glm", "dsv32")
    if args.topk is None:
        args.topk = GLM_TOPK
    elif not is_glm:
        parser.error("--topk is only used with --preset glm/dsv32; use --topk-main/--topk-extra")
    bench_fn = run_bench_glm if is_glm else run_bench

    if args.T_sweep is not None:
        Ts = _parse_T_sweep(args.T_sweep if args.T_sweep else None)
        for i, T in enumerate(Ts):
            if i:
                print("=" * 72)
                print()
            bench_fn(args, T)
    else:
        bench_fn(args, args.T)


if __name__ == "__main__":
    main()
