# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402
from .coherent_memory import coherent_load

"""Single-sample tail projection: retain MFMA math, remove unused sample loads."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def project_tile_single(
    X, W, slot, sample_base, row_base, B: gl.constexpr, K: gl.constexpr
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[2, 4, 1]
    )
    al: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 16, 4], [2, 4, 1], [1, 2, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 16], [2, 1, 4], [2, 1, 0])
    sa = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ra = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ka = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, al)))
    sb = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    row = row_base + ra
    sample = sample_base + nb * 0
    acc = gl.full((2, 64, 16), 0, gl.float32, mma)
    for step in range(K // 64):
        ak = (
            gl.expand_dims(gl.expand_dims(sa, 1), 2) * (K // 2)
            + step * 32
            + gl.expand_dims(gl.expand_dims(ka, 0), 0)
        )
        rows = gl.expand_dims(gl.expand_dims(row, 0), 2)
        offset = (
            (((rows // 16 * (K // 64) + ak // 64) * 2 + ak // 32 % 2) * 4 + ak // 8 % 4)
            * 16
            + rows % 16
        ) * 8 + ak % 8
        a = gl.load(W + offset)
        bk = (
            gl.expand_dims(gl.expand_dims(sb, 1), 2) * (K // 2)
            + step * 32
            + gl.expand_dims(gl.expand_dims(kb, 0), 2)
        )
        samples = gl.expand_dims(gl.expand_dims(sample, 0), 0)
        b = coherent_load(X + (slot * B + samples) * K + bk)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna3.mfma(a, b, acc)
    return gl.sum(acc, 0)


@g.jit
def project_tile_rows(
    X,
    W,
    slot,
    sample_base,
    row_base,
    B: gl.constexpr,
    K: gl.constexpr,
    ROWS: gl.constexpr,
):
    SPLITS: gl.constexpr = 128 // ROWS
    gl.static_assert(ROWS == 16 or ROWS == 32 or ROWS == 64)
    gl.static_assert(K % (SPLITS * 32) == 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[SPLITS, ROWS // 16, 1],
    )
    al: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 16, 4], [SPLITS, ROWS // 16, 1], [1, 2, 0]
    )
    bl: gl.constexpr = gl.BlockedLayout(
        [1, 8, 1], [1, 4, 16], [SPLITS, 1, ROWS // 16], [2, 1, 0]
    )
    sa = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, gl.SliceLayout(2, al)))
    ra = gl.arange(0, ROWS, layout=gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ka = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, al)))
    sb = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(0, bl)))
    row = row_base + ra
    sample = gl.minimum(sample_base + nb, gl.minimum(sample_base + 3, B - 1))
    acc = gl.full((SPLITS, ROWS, 16), 0, gl.float32, mma)
    for step in range(K // (SPLITS * 32)):
        ak = (
            gl.expand_dims(gl.expand_dims(sa, 1), 2) * (K // SPLITS)
            + step * 32
            + gl.expand_dims(gl.expand_dims(ka, 0), 0)
        )
        rows = gl.expand_dims(gl.expand_dims(row, 0), 2)
        offset = (
            (((rows // 16 * (K // 64) + ak // 64) * 2 + ak // 32 % 2) * 4 + ak // 8 % 4)
            * 16
            + rows % 16
        ) * 8 + ak % 8
        a = gl.load(W + offset)
        bk = (
            gl.expand_dims(gl.expand_dims(sb, 1), 2) * (K // SPLITS)
            + step * 32
            + gl.expand_dims(gl.expand_dims(kb, 0), 2)
        )
        samples = gl.expand_dims(gl.expand_dims(sample, 0), 0)
        b = coherent_load(X + (slot * B + samples) * K + bk)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna3.mfma(a, b, acc)
    return gl.sum(acc, 0)
