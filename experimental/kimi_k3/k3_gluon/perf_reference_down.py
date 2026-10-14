"""Isolated expert-down buffer-load ablation; original arithmetic retained."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .perf_packed_scale import mask_zero_scale_packed, mask_zero_scale_words


@g.jit
def down_tile_reference(
    Mid,
    IDs,
    Weights,
    W,
    Scale,
    sample,
    tile,
    INTER: gl.constexpr,
    AITER_LAYOUT: gl.constexpr = False,
    BUFFER: gl.constexpr = False,
    CHUNK: gl.constexpr = 32,
    UNROLL: gl.constexpr = 1,
    LDS: gl.constexpr = False,
    ZERO_GUARD: gl.constexpr = True,
    DIRECT_ACC: gl.constexpr = False,
    MFMA_LANES: gl.constexpr = False,
):
    # One wave per route, two rounds for top16, following FlyDSL stage7.
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [8, 1, 1])
    al: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 16, 4], [8, 1, 1], [1, 2, 0] if MFMA_LANES else [2, 1, 0]
    )
    ul: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 16, 4], [8, 1, 1], [1, 2, 0] if MFMA_LANES else [2, 1, 0]
    )
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])
    aw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ar = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ak = gl.arange(0, CHUNK // 2, layout=gl.SliceLayout(0, gl.SliceLayout(0, al)))
    uw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, ul)))
    ur = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, ul)))
    uk = gl.arange(0, CHUNK, layout=gl.SliceLayout(0, gl.SliceLayout(0, ul)))
    bw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    bk = gl.arange(0, CHUNK, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    bn = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    weight_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, mma))
    mw = gl.arange(0, 8, layout=weight_layout)
    stride: gl.constexpr = ((INTER // 32 + 7) // 8) * 8 if AITER_LAYOUT else INTER // 32
    gl.static_assert((INTER // CHUNK) % UNROLL == 0)
    if LDS:
        linear: gl.constexpr = gl.BlockedLayout([8], [64], [8], [0])
        indices = gl.arange(0, triton.next_power_of_2(16 * INTER), layout=linear)
        staged = gl.allocate_shared_memory(
            gl.bfloat16,
            [triton.next_power_of_2(16 * INTER)],
            gl.SwizzledSharedLayout(1, 1, 1, [0]),
            gl.load(Mid + sample * 16 * INTER + indices, indices < 16 * INTER, 0),
        )
    total = gl.full((8, 16, 16), 0.0, gl.float32, mma)
    for round in gl.static_range(2):
        ae = gl.load(IDs + sample * 16 + round * 8 + aw).to(gl.int64)
        ue = gl.load(IDs + sample * 16 + round * 8 + uw).to(gl.int64)
        route_weight = gl.load(Weights + sample * 16 + round * 8 + mw)
        acc = gl.full((8, 16, 16), 0.0, gl.float32, mma)
        for outer in range(INTER // CHUNK // UNROLL):
            for inner in gl.static_range(UNROLL):
                step = outer * UNROLL + inner
                rows = tile * 16 + ar[None, :, None]
                k = step * CHUNK + gl.expand_dims(gl.expand_dims(ak, 0), 0) * 2
                off = (
                    (
                        ((rows // 16 * (INTER // 128) + k // 128) * 4 + (k % 32) // 8)
                        * 16
                        + rows % 16
                    )
                    * 4
                    + (k % 128) // 32
                ) * 4 + (k % 8) // 2
                if AITER_LAYOUT:
                    off = (
                        ((rows // 16 * (INTER // 128) + k // 128) * 4 + (k % 128) // 32)
                        * 16
                        + rows % 16
                    ) * 16 + (k % 32) // 2
                if BUFFER:
                    off = gl.max_contiguous(gl.multiple_of(off, (1, 1, 4)), (1, 1, 4))
                    raw = gl.amd.cdna3.buffer_load(
                        W, ae[:, None, None].to(gl.int32) * (3584 * INTER // 2) + off
                    )
                else:
                    raw = gl.load(W + ae[:, None, None] * (3584 * INTER // 2) + off)
                scale_row = tile * 16 + ur[None, :, None]
                group = (step * CHUNK + gl.expand_dims(gl.expand_dims(uk, 0), 0)) // 32
                if AITER_LAYOUT:
                    soff = (
                        (scale_row // 32 * (stride // 8) + group // 8) * 4 + group % 4
                    ) * 64
                    soff += (
                        (scale_row % 16) * 4
                        + ((group // 4) % 2) * 2
                        + (scale_row // 16) % 2
                    )
                else:
                    soff = scale_row * stride + group
                if BUFFER:
                    scales = gl.amd.cdna3.buffer_load(
                        Scale, ue[:, None, None].to(gl.int32) * (3584 * stride) + soff
                    )
                else:
                    scales = gl.load(Scale + ue[:, None, None] * (3584 * stride) + soff)
                if ZERO_GUARD == 3:
                    raw = mask_zero_scale_words(raw, scales)
                if ZERO_GUARD == 2:
                    raw = mask_zero_scale_packed(raw, scales)
                a = gl.amd.cdna4.scaled_upcast(raw, scales, gl.bfloat16, axis=2)
                if ZERO_GUARD == 1:
                    a = gl.where(scales == 0, 0.0, a).to(gl.bfloat16)
                a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
                input_k = (
                    (round * 8 + bw[:, None, None]) * INTER
                    + step * CHUNK
                    + bk[None, :, None]
                    + gl.expand_dims(gl.expand_dims(bn, 0), 0) * 0
                )
                if LDS:
                    inp = gl.reshape(
                        staged.gather(gl.reshape(input_k, (8 * CHUNK * 16,)), 0),
                        (8, CHUNK, 16),
                    )
                elif BUFFER:
                    inp = gl.amd.cdna3.buffer_load(Mid, sample * 16 * INTER + input_k)
                else:
                    inp = gl.load(Mid + sample * 16 * INTER + input_k)
                inp = gl.convert_layout(inp, gl.DotOperandLayout(1, mma, 8))
                if DIRECT_ACC:
                    acc = gl.amd.cdna3.mfma(a, inp, acc)
                else:
                    acc += gl.amd.cdna3.mfma(
                        a, inp, gl.full((8, 16, 16), 0.0, gl.float32, mma)
                    )
        total += acc * route_weight[:, None, None]
    if LDS:
        staged._keep_alive()
    plain: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 4], [8, 1, 1], [1, 2, 0])
    shared = gl.allocate_shared_memory(
        gl.float32,
        [2048],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        gl.reshape(gl.convert_layout(total, plain), (2048,)),
    )
    fl: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    row = gl.arange(0, 16, layout=fl)
    value = gl.full((16,), 0.0, gl.float32, fl)
    for wave in gl.static_range(8):
        value += shared.gather(wave * 256 + row * 16, 0)
    shared._keep_alive()
    return value
