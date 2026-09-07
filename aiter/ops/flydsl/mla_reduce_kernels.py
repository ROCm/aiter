# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL MLA decode reduce API.

Drop-in alternative for the HIP ``aiter.mla_reduce_v1`` (csrc/kernels/mla/reduce.cu):
same signature and in/out contract. Opt-in via ``AITER_MLA_REDUCE_FLYDSL=1``;
production keeps the HIP kernel by default. Kernels are compiled per
(H, Dv, out_dtype, tier, output_lse) and ``lru_cache``-backed; ``Tier.ALL`` (one
kernel, device-side per-tile tier selection) mirrors HIP.
"""

from __future__ import annotations

import collections
import functools
import weakref

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import T

from .kernels.mla_reduce import (
    LDS_MAX_SPLITS,
    NUM_THREADS,
    Tier,
    _get_splitk_scratch,
    _load_partial_out_narrow,
    _pointer_buffer_tensor,
    _store_final_out,
    compile_mla_reduce,
    compile_mla_reduce_splitk,
    derive_actual_max_splits,
    plan_splitk_capture_safe,
    should_use_persistent_launch,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg

__all__ = [
    "flydsl_mla_reduce_v1",
]


# Low-direct-pmap threshold: when actual_max_splits <= this, read
# reduce_partial_map directly instead of staging it to LDS + a barrier.
_LOW_DIRECT_PMAP_THR = 8


_MLA_REDUCE_GFX = ("gfx942", "gfx950")
_POINTER_DSL_DTYPES = {
    torch.float32: fx.Float32,
    torch.bfloat16: fx.BFloat16,
    torch.float16: fx.Float16,
    torch.int32: fx.Int32,
}


def _out_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise ValueError(
        f"flydsl_mla_reduce_v1 only supports bf16/fp16 output, got {dtype!r}"
    )


def _pointer_arg(tensor: torch.Tensor, expected_dtype: torch.dtype):
    """Pass a role-validated typed pointer through the FlyDSL launcher ABI."""
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"MLA pointer expects {expected_dtype}, got {tensor.dtype} for tensor "
            f"with shape {tuple(tensor.shape)}"
        )
    try:
        return ptr_arg(tensor, _POINTER_DSL_DTYPES[expected_dtype])
    except KeyError as exc:
        raise TypeError(f"Unsupported MLA pointer dtype: {expected_dtype!r}") from exc


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    device: torch.device,
    contiguous: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.device.type != "cuda":
        raise ValueError(f"`{name}` must be a CUDA/ROCm tensor, got {tensor.device}")
    if tensor.device != device:
        raise ValueError(f"`{name}` must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"`{name}` must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"`{name}` must be {ndim}D, got shape {tuple(tensor.shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"`{name}` must be contiguous")


def _validate_actual_max_splits(actual_max_splits: int | None) -> None:
    if actual_max_splits is None:
        return
    if isinstance(actual_max_splits, bool) or not isinstance(actual_max_splits, int):
        raise TypeError(
            f"`actual_max_splits` must be an int or None, got {actual_max_splits!r}"
        )
    if not 0 <= actual_max_splits <= LDS_MAX_SPLITS:
        raise ValueError(
            f"`actual_max_splits` must be in [0, {LDS_MAX_SPLITS}], "
            f"got {actual_max_splits}"
        )


def _validate_mla_reduce_inputs(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor | None,
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    final_lse: torch.Tensor | None,
    num_kv_splits: int,
    actual_max_splits: int | None,
) -> None:
    """Reject unsupported raw-pointer ABI inputs before a kernel can launch."""
    if not isinstance(final_output, torch.Tensor):
        raise TypeError(
            f"`final_output` must be a torch.Tensor, got {type(final_output)!r}"
        )
    if final_output.device.type != "cuda":
        raise ValueError(
            f"`final_output` must be a CUDA/ROCm tensor, got {final_output.device}"
        )
    if final_output.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "`final_output` must have dtype torch.bfloat16 or torch.float16, "
            f"got {final_output.dtype}"
        )
    if final_output.ndim != 3:
        raise ValueError(
            f"`final_output` must be 3D [rows, H, Dv], got {tuple(final_output.shape)}"
        )
    if final_output.stride(-1) != 1:
        raise ValueError(
            "`final_output` must have a packed last dimension (stride(-1) == 1), "
            f"got stride {tuple(final_output.stride())}"
        )

    device = final_output.device
    _require_tensor(
        "partial_output",
        partial_output,
        dtype=torch.float32,
        ndim=3,
        device=device,
    )
    _require_tensor(
        "partial_lse",
        partial_lse,
        dtype=torch.float32,
        ndim=2,
        device=device,
    )
    _require_tensor(
        "reduce_indptr",
        reduce_indptr,
        dtype=torch.int32,
        ndim=1,
        device=device,
    )
    _require_tensor(
        "reduce_partial_map",
        reduce_partial_map,
        dtype=torch.int32,
        ndim=1,
        device=device,
    )
    if reduce_final_map is not None:
        _require_tensor(
            "reduce_final_map",
            reduce_final_map,
            dtype=torch.int32,
            ndim=2,
            device=device,
        )
    if final_lse is not None:
        _require_tensor(
            "final_lse",
            final_lse,
            dtype=torch.float32,
            ndim=2,
            device=device,
        )

    rows, H, Dv = partial_output.shape
    if tuple(partial_lse.shape) != (rows, H):
        raise ValueError(
            "`partial_lse` must have shape [rows, H] matching `partial_output`, "
            f"got {tuple(partial_lse.shape)} vs {tuple(partial_output.shape)}"
        )
    if tuple(final_output.shape[1:]) != (H, Dv):
        raise ValueError(
            "`final_output` must have shape [final_rows, H, Dv] matching "
            f"`partial_output`, got {tuple(final_output.shape)} vs H={H}, Dv={Dv}"
        )
    if final_lse is not None and tuple(final_lse.shape) != tuple(
        final_output.shape[:2]
    ):
        raise ValueError(
            "`final_lse` must have shape [final_rows, H] matching `final_output`, "
            f"got {tuple(final_lse.shape)} vs {tuple(final_output.shape[:2])}"
        )

    num_reduce_tile = reduce_indptr.numel() - 1
    if reduce_final_map is not None and tuple(reduce_final_map.shape) != (
        num_reduce_tile,
        2,
    ):
        raise ValueError(
            "`reduce_final_map` must have shape [num_reduce_tile, 2], "
            f"got {tuple(reduce_final_map.shape)} for num_reduce_tile={num_reduce_tile}"
        )
    if isinstance(max_seqlen_q, bool) or not isinstance(max_seqlen_q, int):
        raise TypeError(f"`max_seqlen_q` must be an int, got {max_seqlen_q!r}")
    if max_seqlen_q <= 0:
        raise ValueError(f"`max_seqlen_q` must be positive, got {max_seqlen_q}")
    if isinstance(num_kv_splits, bool) or not isinstance(num_kv_splits, int):
        raise TypeError(f"`num_kv_splits` must be an int, got {num_kv_splits!r}")
    if not 0 <= num_kv_splits <= LDS_MAX_SPLITS:
        raise ValueError(
            f"`num_kv_splits` must be in [0, {LDS_MAX_SPLITS}], got {num_kv_splits}"
        )
    if Dv % NUM_THREADS != 0:
        raise ValueError(
            f"`final_output.size(-1)` ({Dv}) must be divisible by NUM_THREADS "
            f"({NUM_THREADS})"
        )
    arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
    if arch not in _MLA_REDUCE_GFX:
        raise ValueError(
            "flydsl_mla_reduce_v1 supports gfx942; gfx950 is permitted but "
            f"unvalidated; got {arch}"
        )
    _validate_actual_max_splits(actual_max_splits)


# Cache mapping reduce_indptr buffer identity -> derived actual_max_splits.
#
# The key carries data_ptr, but a raw pointer is not an identity: once a buffer
# is freed the allocator hands the same address to the next tensor, and a stale
# entry would then answer for a different CSR -- silently supplying the wrong
# split bound to a launch that cannot re-derive it under graph capture. Each
# entry therefore also holds a weak reference to the storage the pointer came
# from; a hit is only honoured while that storage is still alive and still at
# that address.
_ACTUAL_MAX_SPLITS_CACHE: collections.OrderedDict = collections.OrderedDict()
_ACTUAL_MAX_SPLITS_CACHE_CAP = 512


def _storage_ref(t: torch.Tensor):
    """Weak reference to the tensor's storage, or None if it cannot be taken."""
    try:
        return weakref.ref(t.untyped_storage())
    except (AttributeError, TypeError):
        return None


# Cache the complete 33-split x 2-topology domain. Recompiling an evicted entry
# costs about 20 ms even with the on-disk cache warm.
@functools.lru_cache(maxsize=128)
def _compile_sparse_decode_direct_combine(ni: int, fine: bool):
    if not 1 <= ni <= 33:
        raise ValueError(f"sparse decode split count must be 1..33, got {ni}")

    # One wave owns one head. Fine mode splits Dv into four independent slices;
    # coarse mode keeps eight values per lane while retaining wave-local scales.
    dv_slices_per_head = 4 if fine else 1
    blocks_per_row = 16 * dv_slices_per_head
    values_per_lane = 2 if fine else 8
    topology = "fine" if fine else "coarse"

    @flyc.kernel(
        name=f"flydsl_sparse_mla_decode_combine_ni{ni}_{topology}",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        partial_output: fx.Pointer,
        partial_lse: fx.Pointer,
        final_output: fx.Pointer,
        seq: fx.Int32,
    ):
        lane = fx.Int32(fx.thread_idx.x)
        block = fx.Int32(fx.block_idx.x)
        block_in_row = block % fx.Int32(blocks_per_row)
        row = block // fx.Int32(blocks_per_row)
        head = block_in_row // fx.Int32(dv_slices_per_head)
        dv_slice = block_in_row % fx.Int32(dv_slices_per_head)
        out_lane = dv_slice * fx.Int32(64) + lane
        partial_buf = _pointer_buffer_tensor(
            partial_output,
            fx.BFloat16,
            (seq * fx.Int32(ni), 16, 512),
            (16 * 512, 512, 1),
        )
        final_buf = _pointer_buffer_tensor(
            final_output,
            fx.BFloat16,
            (seq, 16, 512),
            (16 * 512, 512, 1),
        )

        in_split = lane < fx.Int32(ni)
        safe_split = in_split.select(lane, fx.Int32(0))
        lse = fx.Float32(
            fx.ptr_load(
                partial_lse
                + ((fx.Int64(row) * ni + fx.Int64(safe_split)) * 16 + fx.Int64(head))
            )
        )
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        lse = in_split.select(lse, neg_inf)
        max_lse = lse
        for off in [32, 16, 8, 4, 2, 1]:
            peer = fx.Float32(max_lse).shuffle_xor(fx.Int32(off), fx.Int32(64))
            max_lse = fx.Float32(max_lse).maximumf(peer)
        scale = fx.Float32(fx.rocdl.exp2(T.f32, (lse - max_lse).ir_value()))
        scale = in_split.select(scale, zero)
        denom = scale
        for off in [32, 16, 8, 4, 2, 1]:
            peer = fx.Float32(denom).shuffle_xor(fx.Int32(off), fx.Int32(64))
            denom = denom + peer
        inv = fx.Float32(fx.rocdl.rcp(T.f32, denom.ir_value()))
        scale = scale * inv

        acc = [fx.Float32(0.0) for _ in fx.range_constexpr(values_per_lane)]
        for split in fx.range_constexpr(ni):
            vals = _load_partial_out_narrow(
                partial_buf,
                row * fx.Int32(ni) + fx.Int32(split),
                head,
                out_lane,
                values_per_lane,
                fx.BFloat16,
            )
            split_scale = fx.Float32(
                fx.rocdl.readlane(
                    T.f32,
                    scale.ir_value(),
                    fx.Int32(split).ir_value(),
                )
            )
            acc = [
                acc[i] + vals[i] * split_scale
                for i in fx.range_constexpr(values_per_lane)
            ]
        _store_final_out(
            final_buf,
            row,
            head,
            out_lane,
            acc,
            values_per_lane,
            fx.BFloat16,
        )

    @flyc.jit
    def launch(
        partial_output: fx.Pointer,
        partial_lse: fx.Pointer,
        final_output: fx.Pointer,
        seq: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(partial_output, partial_lse, final_output, seq).launch(
            grid=(seq * fx.Int32(blocks_per_row), 1, 1),
            block=(64, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": 4},
        )

    return launch


def _use_fine_decode_combine(seq: int, ni: int, num_cu: int) -> bool:
    """Select the Dv-sliced reducer only over its measured occupancy window."""
    fine_ctas = seq * 64
    return ni > 16 and num_cu <= fine_ctas <= 3 * num_cu


def _flydsl_sparse_mla_decode_combine(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    final_output: torch.Tensor,
) -> None:
    """Combine exact sparse-decode BF16 partials and log2 FP32 LSEs."""
    if not torch.cuda.is_available():
        raise RuntimeError("FlyDSL sparse MLA decode requires ROCm")
    tensors = (partial_output, partial_lse, final_output)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("sparse MLA decode tensors must be contiguous ROCm tensors")
    if partial_output.dtype != torch.bfloat16:
        raise TypeError(f"partial_output must be bf16, got {partial_output.dtype}")
    if partial_lse.dtype != torch.float32 or final_output.dtype != torch.bfloat16:
        raise TypeError("partial_lse must be fp32 and final_output must be bf16")
    if partial_output.ndim != 5 or partial_lse.ndim != 4 or final_output.ndim != 4:
        raise ValueError("expected partial_output/partial_lse/final_output ranks 5/4/4")
    b, seq, ni, H, Dv = partial_output.shape
    if b != 1 or not 1 <= seq <= 96 or not 1 <= ni <= 33 or (H, Dv) != (16, 512):
        raise ValueError(
            "supported sparse decode scope is b=1, 1<=seq<=96, "
            f"1<=splits<=33, H=16, Dv=512; got {tuple(partial_output.shape)}"
        )
    if tuple(partial_lse.shape) != (b, seq, ni, H):
        raise ValueError("partial_lse shape does not match partial_output")
    if tuple(final_output.shape) != (b, seq, H, Dv):
        raise ValueError("final_output shape does not match partial_output")
    if not (partial_output.device == partial_lse.device == final_output.device):
        raise ValueError("all sparse decode tensors must be on the same device")
    arch = str(torch.cuda.get_device_properties(final_output.device).gcnArchName).split(
        ":"
    )[0]
    if arch != "gfx950":
        raise ValueError(
            f"sparse decode BF16-direct reduce is gated to gfx950, got {arch}"
        )

    # A one-split softmax is already fully reduced by the partial kernel.  The
    # generic reducer is tuned for multiple splits and its Dv-split launch
    # intentionally has no single-split specialization.  A device copy keeps
    # this edge case exact and remains safe under graph capture/replay.
    if ni == 1:
        final_output.copy_(partial_output[:, :, 0])
        return

    num_cu = torch.cuda.get_device_properties(
        final_output.device.index
    ).multi_processor_count
    fine = _use_fine_decode_combine(seq, ni, num_cu)
    direct = _compile_sparse_decode_direct_combine(ni, fine)
    _run_compiled(
        direct,
        _pointer_arg(partial_output, torch.bfloat16),
        _pointer_arg(partial_lse, torch.float32),
        _pointer_arg(final_output, torch.bfloat16),
        int(seq),
        fx.Stream(torch.cuda.current_stream(final_output.device)),
    )


def _resolve_actual_max_splits(reduce_indptr: torch.Tensor) -> int | None:
    """Resolve ``actual_max_splits`` (true ``max_t(n_splits)``) capture-safely.

    Deriving the value needs a device read that is illegal under CUDA-graph capture.
    The eager/warmup pass reads the CSR and caches by buffer identity; under capture
    it returns the cached value or ``None`` on a miss. The launch path rejects a
    cache miss before launch rather than using an unvalidated split bound.
    """
    ptr = reduce_indptr.data_ptr()
    key = (ptr, int(reduce_indptr.numel()))
    cache = _ACTUAL_MAX_SPLITS_CACHE

    def _live(entry):
        """True while the cached entry still describes this exact allocation."""
        ref, cached_ptr, _ = entry
        if ref is None:
            return False
        storage = ref()
        return storage is not None and storage.data_ptr() == cached_ptr

    if torch.cuda.is_current_stream_capturing():
        entry = cache.get(key)
        if entry is None:
            return None
        if not _live(entry):
            # The pointer was recycled: the buffer this value was derived from
            # is gone. Report a miss -- the launch path rejects that rather than
            # using an unvalidated bound.
            del cache[key]
            return None
        return entry[2]

    val = derive_actual_max_splits(reduce_indptr)
    if key in cache:
        cache.move_to_end(key)
    cache[key] = (_storage_ref(reduce_indptr), ptr, val)
    if len(cache) > _ACTUAL_MAX_SPLITS_CACHE_CAP:
        cache.popitem(last=False)
    return val


def flydsl_mla_reduce_v1(
    partial_output: torch.Tensor,  # fp32 [rows, H, Dv] contiguous
    partial_lse: torch.Tensor,  # fp32 [rows, H] contiguous
    reduce_indptr: torch.Tensor,  # i32  [num_reduce_tile + 1]
    reduce_final_map: torch.Tensor | None,  # i32 [num_reduce_tile, 2] (optional)
    reduce_partial_map: torch.Tensor,  # i32 [reduce_indptr[-1]]
    max_seqlen_q: int,
    final_output: torch.Tensor,  # bf16/fp16 [bs, H, Dv]
    final_lse: torch.Tensor | None = None,  # fp32 [bs, H] (optional)
    num_kv_splits: int = 0,  # HIP parity and split-K engagement budget
    actual_max_splits: int | None = None,  # true max tile width; gates DA split-K
    *,
    stream: torch.cuda.Stream | None = None,
    waves_per_eu: int = 4,
) -> None:
    """FlyDSL drop-in replacement for ``aiter.mla_reduce_v1`` (write-in-place).

    Results land in ``final_output`` (and ``final_lse`` if provided); no return.

    Args:
        partial_output / partial_lse: fp32 per-split partials.
        reduce_indptr / reduce_partial_map: CSR over splits (+ gather map).
        reduce_final_map: optional [tile, 2] {q_start, q_end}; ``None`` derives the
            q-range from the partial map (HIP ``use_reduce_final_map = false`` path).
        max_seqlen_q: q-positions per decode token group; drives grid-y (NTG).
        final_output: output tensor (bf16/fp16); strides read at runtime.
        final_lse: optional merged LSE output (fp32).
        num_kv_splits: HIP-signature parity and split-K engagement budget.
        actual_max_splits: true ``max_t(n_splits)`` gating split-K + low-direct-pmap.
            ``None`` auto-resolves from ``reduce_indptr`` eagerly or from its
            warmup cache during capture; a cache miss must pass it explicitly.
        stream: launch stream; defaults to the device's current stream.
        waves_per_eu: compile-time occupancy hint for FlyDSL kernels.
    """
    _validate_mla_reduce_inputs(
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_q,
        final_output,
        final_lse,
        num_kv_splits,
        actual_max_splits,
    )
    if reduce_indptr.numel() < 2:
        return

    if actual_max_splits is None:
        actual_max_splits = _resolve_actual_max_splits(reduce_indptr)
    if actual_max_splits is None:
        raise RuntimeError(
            "Cannot validate `actual_max_splits` during CUDA-graph capture; "
            "warm up this reduce_indptr buffer first or pass it explicitly"
        )
    _validate_actual_max_splits(actual_max_splits)

    H = partial_output.size(-2)
    Dv = final_output.size(-1)
    out_dtype_str = _out_dtype_str(final_output.dtype)
    num_reduce_tile = reduce_indptr.numel() - 1
    output_lse = final_lse is not None
    use_reduce_final_map = reduce_final_map is not None

    if final_lse is None:
        final_lse = torch.empty(1, dtype=torch.float32, device=final_output.device)
    if reduce_final_map is None:
        reduce_final_map = torch.empty(1, dtype=torch.int32, device=final_output.device)
    if stream is None:
        stream = torch.cuda.current_stream(final_output.device)

    num_cu = torch.cuda.get_device_properties(
        final_output.device.index
    ).multi_processor_count

    use_persistent = should_use_persistent_launch(
        H=H,
        max_seqlen_q=max_seqlen_q,
        num_reduce_tile=num_reduce_tile,
        num_cu=num_cu,
    )

    # Device-adaptive split-K: for low-tile/high-split decode, split each active
    # tile's reduction across K cooperating blocks then combine. Needs
    # reduce_final_map for the combine's per-tile q-range.
    if use_reduce_final_map:
        engage_sk, sk_K, sk_slots = plan_splitk_capture_safe(
            num_final_rows=int(final_output.size(0)),
            H=H,
            max_seqlen_q=max_seqlen_q,
            num_kv_splits=int(num_kv_splits),
            num_cu=num_cu,
            actual_max_splits=actual_max_splits,
        )
        if engage_sk:
            launch_partial, launch_combine = compile_mla_reduce_splitk(
                H=H,
                Dv=Dv,
                out_dtype=out_dtype_str,
                K=sk_K,
                output_lse=output_lse,
                waves_per_eu=waves_per_eu,
            )
            sk_acc, sk_ml = _get_splitk_scratch(
                sk_slots, sk_K, Dv, final_output.device.index
            )
            fx_stream = fx.Stream(stream)
            _run_compiled(
                launch_partial,
                _pointer_arg(partial_output, torch.float32),
                _pointer_arg(partial_lse, torch.float32),
                _pointer_arg(reduce_indptr, torch.int32),
                _pointer_arg(reduce_partial_map, torch.int32),
                _pointer_arg(sk_acc, torch.float32),
                _pointer_arg(sk_ml, torch.float32),
                int(partial_output.size(0)),
                sk_slots * sk_K,
                fx_stream,
            )
            _run_compiled(
                launch_combine,
                _pointer_arg(reduce_indptr, torch.int32),
                _pointer_arg(reduce_final_map, torch.int32),
                _pointer_arg(sk_acc, torch.float32),
                _pointer_arg(sk_ml, torch.float32),
                _pointer_arg(final_output, final_output.dtype),
                _pointer_arg(final_lse, torch.float32),
                int(final_output.stride(-3)),
                int(final_output.stride(-2)),
                int(final_output.size(0)),
                sk_slots,
                fx_stream,
            )
            return

    # Tier.ALL: one kernel with a device-side per-tile tier branch (always correct).
    tier = Tier.ALL

    # Adaptive launch: on a sparse grid, launch one block per active
    # (tile, head, q-group) instead of the persistent grid-stride kernel.
    # Multi-tile decode only (single-tile is split-K's domain).
    num_final_rows = int(final_output.size(0))
    use_adaptive = (
        use_reduce_final_map
        and num_final_rows > 1
        and num_final_rows < num_reduce_tile
        and num_final_rows * H * max_seqlen_q <= 4 * num_cu
    )

    low_direct_pmap_thr = (
        _LOW_DIRECT_PMAP_THR
        if use_adaptive
        and actual_max_splits is not None
        and int(actual_max_splits) <= _LOW_DIRECT_PMAP_THR
        else 0
    )

    kernel = compile_mla_reduce(
        H=H,
        Dv=Dv,
        out_dtype=out_dtype_str,
        tier=tier,
        persistent=use_persistent and not use_adaptive,
        output_lse=output_lse,
        use_reduce_final_map=use_reduce_final_map,
        waves_per_eu=waves_per_eu,
        adaptive=use_adaptive,
        low_direct_pmap_thr=low_direct_pmap_thr,
    )

    _run_compiled(
        kernel,
        _pointer_arg(partial_output, torch.float32),
        _pointer_arg(partial_lse, torch.float32),
        _pointer_arg(reduce_indptr, torch.int32),
        _pointer_arg(reduce_partial_map, torch.int32),
        _pointer_arg(reduce_final_map, torch.int32),
        _pointer_arg(final_output, final_output.dtype),
        _pointer_arg(final_lse, torch.float32),
        int(final_output.stride(-3)),
        int(final_output.stride(-2)),
        int(num_cu),
        int(num_reduce_tile),
        int(max_seqlen_q),
        int(partial_output.size(0)),
        int(final_output.size(0)),
        fx.Stream(stream),
    )
