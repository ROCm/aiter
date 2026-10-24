# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL FlashAttention forward for gfx1151 (GFX11, wave32).

Separate from ``flash_attn_func_gfx1201`` because the GFX11 BF16 WMMA takes
replicated ``vector<16 x i16>`` A/B operands (GFX12: ``vector<8 x i16>``) and
owns its ``v8f32`` result as ``row = 2 * element + half, column = lane``.

Contract: BF16 in/out, FP32 accumulation, non-causal, no dropout, ``head_dim``
a multiple of 16, independent query/key lengths. BSHD with ``(head_dim, 1)``
head/dim strides; batch and token strides are runtime, so packed-QKV views work.
Sequence lengths are runtime arguments and do not trigger a recompile.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, Vector
from flydsl.expr.utils.arith import ArithValue

from .dpp_utils import update_dpp_i32
from .tensor_shim import GTensor, _run_compiled, _to_raw

_TILE = 16
_WAVE_SIZE = 32
_ACC = 8
# One 128-bit BF16 access.
_VEC = 8
_LOG2E = math.log2(math.e)

# Tuned on gfx1151 (Radeon 8060S).
_DEFAULT_WAVES = 4
_DEFAULT_WAVES_PER_EU = 8
_DEFAULT_KEY_TILES = 1
_DEFAULT_HEAD_GROUP = 8

_SUPPORTED_ARCH_PREFIX = "gfx1151"


# DPP row_xmask: read lane ``lane ^ N`` within a 16-lane row.
_DPP_ROW_XMASK = 0x160


def _row_xor_f32(value, offset: int):
    if offset not in (1, 2, 4, 8):
        raise ValueError(f"row_xmask offset must be 1, 2, 4 or 8, got {offset}")
    bits = fx.Float32(value).bitcast(fx.Int32)
    moved = update_dpp_i32(bits, bits, _DPP_ROW_XMASK | offset)
    return fx.Int32(moved).bitcast(fx.Float32)


def _store_element(tensor, offset, value):
    # Inside a dynamic ``if``, the AST rewriter treats ``tensor.store(...)`` as a
    # carried value; a free-function call keeps ``tensor`` a plain capture.
    tensor.store(offset, value)


def _arch(device: torch.device) -> str:
    try:
        name = torch.cuda.get_device_properties(device.index).gcnArchName
    except Exception:  # noqa: BLE001 - probing an arbitrary device
        return ""
    return name.lower().split(":")[0]


def flydsl_flash_attn_gfx1151_supported(
    device: torch.device | None = None,
    head_dim: int = 128,
) -> bool:
    """Return whether this kernel can serve ``head_dim`` on ``device``."""
    if not torch.cuda.is_available():
        return False
    if head_dim <= 0 or head_dim % _TILE != 0:
        return False
    resolved = device if device is not None else torch.device("cuda")
    if resolved.type != "cuda":
        return False
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if not _arch(resolved).startswith(_SUPPORTED_ARCH_PREFIX):
        return False
    return torch.cuda.get_device_properties(resolved.index).warp_size == _WAVE_SIZE


@functools.cache
def build_flash_attn_func_gfx1151_module(
    num_heads: int,
    head_dim: int,
    waves: int,
    waves_per_eu: int,
    key_tiles: int,
    head_group: int,
):
    block_threads = waves * _WAVE_SIZE
    head_group = math.gcd(head_group, num_heads)
    block_m = waves * _TILE
    block_n = key_tiles * _TILE
    d_tiles = head_dim // _TILE
    channels = num_heads * head_dim
    scale = 1.0 / math.sqrt(head_dim)
    # Unpadded: padding P rows to 24 lowered bank conflicts but was 2-5% slower.
    p_stride = _TILE
    p_per_wave = _TILE * p_stride

    # V is staged transposed ([dim][key]) so a PV B fragment is two 128-bit LDS
    # reads. Each staging item is 2 keys x 8 dims, re-paired in registers.
    stage_key_pairs = block_n // 2
    stage_dim_chunks = head_dim // _VEC
    stage_items = stage_key_pairs * stage_dim_chunks
    # Tail rounds wrap and rewrite identical values, keeping staging branch-free.
    stage_rounds = -(-stage_items // block_threads)
    # 4-dword row padding makes PV reads conflict-free; physical row
    # ``row ^ ((row >> 4) & 1)`` removes the remaining 2-way staging-write
    # conflict at block_n=16.
    vt_stride = block_n + _VEC

    @fx.struct
    class SharedStorage:
        # One 16x16 P tile per wave per key sub-tile.
        probabilities: fx.Array[fx.BFloat16, waves * key_tiles * p_per_wave, 16]
        # This step's V, transposed and shared by all waves in the block.
        v_tile: fx.Array[fx.BFloat16, head_dim * vt_stride, 16]

    @flyc.kernel(
        name=(
            "flydsl_flash_attn_gfx1151_bf16"
            f"_h{num_heads}_d{head_dim}_w{waves}_n{block_n}"
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        q_mem: fx.Tensor,
        k_mem: fx.Tensor,
        v_mem: fx.Tensor,
        o_mem: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        q_batch_stride: fx.Int32,
        k_batch_stride: fx.Int32,
        v_batch_stride: fx.Int32,
        o_batch_stride: fx.Int32,
        q_token_stride: fx.Int32,
        k_token_stride: fx.Int32,
        v_token_stride: fx.Int32,
        seq_q_tiles: fx.Int32,
    ):
        q = GTensor(q_mem, dtype=T.bf16, shape=(-1,))
        k = GTensor(k_mem, dtype=T.bf16, shape=(-1,))
        v = GTensor(v_mem, dtype=T.bf16, shape=(-1,))
        o = GTensor(o_mem, dtype=T.bf16, shape=(-1,))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        p_lds = shared.probabilities.ptr
        v_lds = shared.v_tile.ptr

        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(_WAVE_SIZE)
        wave_lane = tid % fx.Int32(_WAVE_SIZE)
        lane = wave_lane % fx.Int32(_TILE)
        half = wave_lane // fx.Int32(_TILE)

        # Block order: head group, query tile, head in group. Resident blocks then
        # share K/V rows through L2. Head-fastest re-read K/V once per query tile
        # (1.9 GiB at seq 4096); groups of 8 were fastest for both layouts.
        block = fx.Int32(fx.block_idx.x)
        sample = fx.Int32(fx.block_idx.y)
        group_blocks = seq_q_tiles * fx.Int32(head_group)
        head_group_index = block // group_blocks
        in_group = block % group_blocks
        q_tile = in_group // fx.Int32(head_group)
        head = head_group_index * fx.Int32(head_group) + in_group % fx.Int32(head_group)
        q_base_row = q_tile * fx.Int32(block_m) + wave * fx.Int32(_TILE)

        def safe_row(row, sequence):
            return (row < sequence).select(row, fx.Int32(0))

        def input_offset(row, column, batch_stride, token_stride):
            return (
                sample * batch_stride
                + row * token_stride
                + head * fx.Int32(head_dim)
                + column
            )

        def load_dim_fragment(tensor, row, d_base, batch_stride, token_stride):
            """16 contiguous dims of one row as two 128-bit loads."""
            base = input_offset(row, fx.Int32(d_base), batch_stride, token_stride)
            low = Vector(tensor.load(base, vec_size=_VEC))
            high = Vector(tensor.load(base + fx.Int32(_VEC), vec_size=_VEC))
            return low.shuffle(high, list(range(_TILE)))

        def wmma(a_fragment, b_fragment, accumulator):
            return rocdl.wmma_f32_16x16x16_bf16(
                Vector.make_type(_ACC, fx.Float32),
                _to_raw(a_fragment.bitcast(fx.Int16)),
                _to_raw(b_fragment.bitcast(fx.Int16)),
                _to_raw(accumulator),
            ).result

        def maximum(a, b):
            return fx.Float32(ArithValue(a > b).select(a, b))

        def reduce_lanes(value, combine):
            """Reduce the 16 lanes of a WMMA result row with DPP, not LDS."""
            result = value
            for shuffle_offset in (1, 2, 4, 8):
                result = combine(result, _row_xor_f32(result, shuffle_offset))
            return result

        initial_running_max = [fx.Float32(float("-inf")) for _ in range(_ACC)]
        initial_running_sum = [fx.Float32(0.0) for _ in range(_ACC)]
        initial_accumulators = [
            Vector.filled(_ACC, 0.0, fx.Float32) for _ in range(d_tiles)
        ]
        init_args = [
            *[_to_raw(value) for value in initial_running_max],
            *[_to_raw(value) for value in initial_running_sum],
            *[_to_raw(value) for value in initial_accumulators],
        ]

        query_row = safe_row(q_base_row + lane, seq_len_q)

        # Q is loop-invariant; keep it in registers.
        query_fragments = [
            load_dim_fragment(
                q, query_row, d_tile * _TILE, q_batch_stride, q_token_stride
            )
            for d_tile in range_constexpr(d_tiles)
        ]

        # Consecutive lanes take consecutive key pairs of one dim chunk.
        stage_slots = []
        for stage_round in range_constexpr(stage_rounds):
            stage_item = (tid + fx.Int32(stage_round * block_threads)) % fx.Int32(
                stage_items
            )
            stage_pair = stage_item % fx.Int32(stage_key_pairs)
            stage_dim = (stage_item // fx.Int32(stage_key_pairs)) * fx.Int32(_VEC)
            # Row swizzle: in odd 16-row groups, even dims move down one row and
            # odd dims up one.
            row_shift = ((stage_dim >> fx.Int32(4)) & fx.Int32(1)) * fx.Int32(vt_stride)
            stage_base = stage_dim * fx.Int32(vt_stride) + stage_pair * fx.Int32(2)
            stage_slots.append(
                (
                    stage_pair * fx.Int32(2),
                    stage_dim,
                    (stage_base + row_shift, stage_base - row_shift),
                )
            )

        p_wave_base = wave * fx.Int32(key_tiles * p_per_wave)
        lane_swapped = lane ^ fx.Int32(1)
        bf16_vec = Vector.make_type(_VEC, fx.BFloat16)

        loop_results = init_args
        for key_base, inner_args in range(
            0, fx.Index(seq_len_k), block_n, init=init_args
        ):
            running_max = [
                fx.Float32(inner_args[element]) for element in range_constexpr(_ACC)
            ]
            running_sum = [
                fx.Float32(inner_args[_ACC + element])
                for element in range_constexpr(_ACC)
            ]
            accumulators = [
                Vector(inner_args[2 * _ACC + d_tile])
                for d_tile in range_constexpr(d_tiles)
            ]
            key_base_row = fx.Int32(key_base)

            # Issue V before QK to overlap its latency. Out-of-range keys read
            # row 0; their probabilities are masked to zero.
            for stage_key, stage_dim, stage_lds in stage_slots:
                key_rows = [
                    Vector(
                        v.load(
                            input_offset(
                                safe_row(
                                    key_base_row + stage_key + fx.Int32(pair_half),
                                    seq_len_k,
                                ),
                                stage_dim,
                                v_batch_stride,
                                v_token_stride,
                            ),
                            vec_size=_VEC,
                        )
                    )
                    for pair_half in range_constexpr(2)
                ]
                for dim_offset in range_constexpr(_VEC):
                    fx.ptr_store(
                        Vector(
                            key_rows[0].shuffle(
                                key_rows[1], [dim_offset, _VEC + dim_offset]
                            )
                        ),
                        v_lds
                        + stage_lds[dim_offset % 2]
                        + fx.Int32(dim_offset * vt_stride),
                    )

            # S = Q K^T.
            sub_scores = []
            sub_valid = []
            for sub in range_constexpr(key_tiles):
                sub_base = key_base_row + fx.Int32(sub * _TILE)
                sub_valid.append(sub_base + lane < seq_len_k)
                sub_key_row = safe_row(sub_base + lane, seq_len_k)
                scores = Vector.filled(_ACC, 0.0, fx.Float32)
                for d_tile in range_constexpr(d_tiles):
                    k_fragment = load_dim_fragment(
                        k,
                        sub_key_row,
                        d_tile * _TILE,
                        k_batch_stride,
                        k_token_stride,
                    )
                    scores = Vector(wmma(query_fragments[d_tile], k_fragment, scores))
                sub_scores.append(scores)

            # Online softmax; rescale once per step across all sub-tiles.
            sub_probabilities = [[] for _ in range(key_tiles)]
            corrections = []
            for element in range_constexpr(_ACC):
                scaled = []
                for sub in range_constexpr(key_tiles):
                    value = fx.Float32(sub_scores[sub][element]) * fx.Float32(scale)
                    scaled.append(
                        sub_valid[sub].select(value, fx.Float32(float("-inf")))
                    )

                row_max = None
                for sub in range_constexpr(key_tiles):
                    sub_max = reduce_lanes(scaled[sub], maximum)
                    row_max = sub_max if row_max is None else maximum(row_max, sub_max)

                new_max = maximum(running_max[element], row_max)
                correction = fx.math.exp2(
                    (running_max[element] - new_max) * fx.Float32(_LOG2E)
                )

                step_sum = None
                for sub in range_constexpr(key_tiles):
                    probability = fx.math.exp2(
                        (scaled[sub] - new_max) * fx.Float32(_LOG2E)
                    )
                    probability = sub_valid[sub].select(probability, fx.Float32(0.0))
                    sub_probabilities[sub].append(probability)
                    tile_sum = reduce_lanes(probability, lambda a, b: a + b)
                    step_sum = (
                        tile_sum
                        if step_sum is None
                        else step_sum + fx.Float32(tile_sum)
                    )

                running_sum[element] = running_sum[element] * correction + step_sum
                running_max[element] = new_max
                corrections.append(correction)

            correction_vector = Vector.from_elements(corrections, fx.Float32)
            for d_tile in range_constexpr(d_tiles):
                accumulators[d_tile] = Vector(accumulators[d_tile] * correction_vector)

            # Re-layout P through LDS: WMMA owns [2*element+half, lane], the PV
            # A operand needs row [lane, 0:16].
            for sub in range_constexpr(key_tiles):
                sub_p_base = p_wave_base + fx.Int32(sub * p_per_wave)
                for element in range_constexpr(_ACC):
                    result_row = fx.Int32(2 * element) + half
                    fx.ptr_store(
                        fx.BFloat16(sub_probabilities[sub][element]),
                        p_lds + sub_p_base + result_row * fx.Int32(p_stride) + lane,
                    )
            fx.gpu.barrier()

            for sub in range_constexpr(key_tiles):
                sub_p_base = p_wave_base + fx.Int32(sub * p_per_wave)
                p_row_base = p_lds + sub_p_base + lane * fx.Int32(p_stride)
                p_fragment = Vector(
                    fx.ptr_load(p_row_base, result_type=bf16_vec)
                ).shuffle(
                    Vector(
                        fx.ptr_load(p_row_base + fx.Int32(_VEC), result_type=bf16_vec)
                    ),
                    list(range(_TILE)),
                )
                for d_tile in range_constexpr(d_tiles):
                    # Swizzle bit is d_tile's parity.
                    v_row = (
                        v_lds
                        + (
                            fx.Int32(d_tile * _TILE)
                            + (lane_swapped if d_tile % 2 else lane)
                        )
                        * fx.Int32(vt_stride)
                        + fx.Int32(sub * _TILE)
                    )
                    v_fragment = Vector(
                        fx.ptr_load(v_row, result_type=bf16_vec)
                    ).shuffle(
                        Vector(
                            fx.ptr_load(v_row + fx.Int32(_VEC), result_type=bf16_vec)
                        ),
                        list(range(_TILE)),
                    )
                    accumulators[d_tile] = Vector(
                        wmma(p_fragment, v_fragment, accumulators[d_tile])
                    )
            fx.gpu.barrier()

            loop_results = yield [
                *[_to_raw(value) for value in running_max],
                *[_to_raw(value) for value in running_sum],
                *[_to_raw(value) for value in accumulators],
            ]

        running_max = [
            fx.Float32(loop_results[element]) for element in range_constexpr(_ACC)
        ]
        running_sum = [
            fx.Float32(loop_results[_ACC + element])
            for element in range_constexpr(_ACC)
        ]
        accumulators = [
            Vector(loop_results[2 * _ACC + d_tile])
            for d_tile in range_constexpr(d_tiles)
        ]

        # Output rows follow WMMA ownership; the tail tile is predicated.
        for d_tile in range_constexpr(d_tiles):
            d_base = fx.Int32(d_tile * _TILE)
            for element in range_constexpr(_ACC):
                output_row = q_base_row + fx.Int32(2 * element) + half
                normalized = fx.BFloat16(
                    fx.Float32(accumulators[d_tile][element]) / running_sum[element]
                )
                output_offset = (
                    sample * o_batch_stride
                    + output_row * fx.Int32(channels)
                    + head * fx.Int32(head_dim)
                    + d_base
                    + lane
                )
                if output_row < seq_len_q:
                    _store_element(o, output_offset, normalized)

    @flyc.jit
    def launch(
        q_mem: fx.Tensor,
        k_mem: fx.Tensor,
        v_mem: fx.Tensor,
        o_mem: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        q_batch_stride: fx.Int32,
        k_batch_stride: fx.Int32,
        v_batch_stride: fx.Int32,
        o_batch_stride: fx.Int32,
        q_token_stride: fx.Int32,
        k_token_stride: fx.Int32,
        v_token_stride: fx.Int32,
        q_tiles: fx.Int32,
        batch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            q_mem,
            k_mem,
            v_mem,
            o_mem,
            seq_len_q,
            seq_len_k,
            q_batch_stride,
            k_batch_stride,
            v_batch_stride,
            o_batch_stride,
            q_token_stride,
            k_token_stride,
            v_token_stride,
            q_tiles,
        ).launch(
            grid=(q_tiles * fx.Int32(num_heads), batch, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "waves_per_eu": waves_per_eu,
        "unsafe_fp_math": True,
        "fast_fp_math": True,
    }
    launch.block_m = block_m
    return launch


def _check_operand(name: str, tensor: torch.Tensor, head_dim: int) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"`{name}` must be a CUDA/HIP tensor")
    if tensor.dtype is not torch.bfloat16:
        raise ValueError(f"`{name}` must be bfloat16, got {tensor.dtype}")
    if tensor.dim() != 4:
        raise ValueError(f"`{name}` must be a 4D BSHD tensor, got rank {tensor.dim()}")
    if tensor.shape[3] != head_dim:
        raise ValueError(f"`{name}` head_dim must be {head_dim}, got {tensor.shape[3]}")
    if tensor.stride(3) != 1 or tensor.stride(2) != head_dim:
        raise ValueError(
            f"`{name}` must have head/dim strides ({head_dim}, 1), got "
            f"{tensor.stride()[2:]}"
        )


def flydsl_flash_attn_func_gfx1151(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    waves: int = _DEFAULT_WAVES,
    waves_per_eu: int = _DEFAULT_WAVES_PER_EU,
    key_tiles: int = _DEFAULT_KEY_TILES,
    head_group: int = _DEFAULT_HEAD_GROUP,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run FlyDSL FlashAttention on gfx1151 (Strix Halo, RDNA 3.5, wave32).

    Args:
        q: query tensor ``[batch, seq_len_q, num_heads, head_dim]`` (BSHD).
        k: key tensor ``[batch, seq_len_k, num_heads, head_dim]``.
        v: value tensor, same shape as ``k``.
        waves: waves per block, i.e. ``BLOCK_M / 16``.
        waves_per_eu: occupancy hint forwarded to the FlyDSL builder.
        key_tiles: key tiles consumed per loop step, i.e. ``BLOCK_N / 16``.
        head_group: heads whose blocks are scheduled together; reduced to its
            largest divisor of ``num_heads``.
        out: optional destination with ``q``'s shape; allocated when omitted.
        stream: stream to launch on. Defaults to the current stream.

    Q/K/V need only ``(head_dim, 1)`` head/dim strides, so packed-QKV views are
    accepted. Returns contiguous BSHD. Defaults are tuned on gfx1151.

    Raises:
        ValueError: on an unsupported device, dtype, rank, layout or shape
            mismatch.
    """
    if q.dim() != 4:
        raise ValueError(f"`q` must be a 4D BSHD tensor, got rank {q.dim()}")
    head_dim = q.shape[3]
    if not flydsl_flash_attn_gfx1151_supported(q.device, head_dim):
        raise ValueError(
            "flydsl_flash_attn_func_gfx1151 requires a gfx1151 wave32 device "
            f"and a head_dim that is a multiple of {_TILE}; got "
            f"arch={_arch(q.device)!r} head_dim={head_dim}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_operand(name, tensor, head_dim)
    if not (q.device == k.device == v.device):
        raise ValueError(
            "q/k/v must share a device, got " f"q={q.device} k={k.device} v={v.device}"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must share a shape, got k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    batch, seq_len_q, num_heads, _ = q.shape
    if k.shape[0] != batch or k.shape[2] != num_heads:
        raise ValueError(
            "q and k must share batch and num_heads, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)}"
        )
    seq_len_k = k.shape[1]
    if seq_len_q <= 0 or seq_len_k <= 0:
        raise ValueError(
            f"sequence lengths must be positive, got q={seq_len_q} k={seq_len_k}"
        )
    if waves <= 0 or waves_per_eu <= 0 or key_tiles <= 0 or head_group <= 0:
        raise ValueError(
            "waves, waves_per_eu, key_tiles and head_group must be positive, got "
            f"{waves}/{waves_per_eu}/{key_tiles}/{head_group}"
        )

    if out is None:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
    else:
        _check_operand("out", out, head_dim)
        if out.shape != q.shape:
            raise ValueError(
                f"`out` must have q's shape {tuple(q.shape)}, got {tuple(out.shape)}"
            )
        if out.stride(1) != num_heads * head_dim:
            raise ValueError("`out` must be contiguous along the token axis")

    with torch.cuda.device(q.device.index):
        executable = build_flash_attn_func_gfx1151_module(
            num_heads, head_dim, waves, waves_per_eu, key_tiles, head_group
        )
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError(
                f"`stream` must be on {q.device}, got {launch_stream.device}"
            )
        q_tiles = (seq_len_q + executable.block_m - 1) // executable.block_m
        _run_compiled(
            executable,
            q,
            k,
            v,
            out,
            seq_len_q,
            seq_len_k,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            out.stride(0),
            q.stride(1),
            k.stride(1),
            v.stride(1),
            q_tiles,
            batch,
            launch_stream,
        )
    return out


__all__ = [
    "build_flash_attn_func_gfx1151_module",
    "flydsl_flash_attn_func_gfx1151",
    "flydsl_flash_attn_gfx1151_supported",
]
