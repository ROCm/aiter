# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus batched-BMM Python bindings.

This module is intentionally separate from `gemm_op_a16w16.py`: BMM callers use
batch-in-the-middle or grouped layouts (for example DSV4 `wo_a`) while the
underlying kernels still live in the shared opus GEMM backend.
"""

import functools

import torch

from ...jit.core import compile_ops


def _gen_bmm_a8w8_scale_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 2,
    kernelId: int = 0,
) -> None:
    # In-place mutation of ``Y``; fake must mirror the void C++ op (full arg
    # list + None return) so torch.compile registers a mutating op, not a
    # tensor-producing one.
    return None


# mmajor fp8 e8m0 mxscale BMM raw binding: x/Y are [M, batch, *], wo_a + w_scale
# batch-major (zero-copy DSV4 wo_a). kid-dispatched; driven by
# bmm_a8w8_mxscale_opus below.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_bpreshuffle",
    gen_fake=_gen_bmm_a8w8_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_bpreshuffle_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 1,
    kernelId: int = 0,
) -> None: ...


def _gen_bmm_a8w8_cc_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    ws: torch.Tensor,
    bias: torch.Tensor,
    splitK: int = 1,
    mClusterWg: int = 1,
    kernelId: int = 0,
) -> None:
    # Mutates Y (and scribbles on ws); mirrors the void C++ op so torch.compile
    # registers a mutating op rather than a tensor-producing one.
    return None


# Cluster-launch, fused split-K sibling of the bpreshuffle binding above.
#
# ws / bias are NOT optional at the binding level -- pass an EMPTY tensor for
# "absent", which is what the C++ side tests for. An Optional[Tensor] would have
# to survive the compile_ops signature and the fake-tensor path, and an empty
# tensor costs nothing and cannot be None-dereferenced by accident.
#
# Size ws with opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_ws_numel (see
# bmm_a8w8_mxscale_bpreshuffle_cc_ws below) and allocate it with Y's dtype.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch",
    gen_fake=_gen_bmm_a8w8_cc_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    ws: torch.Tensor,
    bias: torch.Tensor,
    splitK: int = 1,
    mClusterWg: int = 1,
    kernelId: int = 0,
) -> None: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_ws_numel",
    develop=True,
)
def _opus_bmm_a8w8_mxscale_bpreshuffle_cc_ws_numel(
    m: int,
    n: int,
    batch: int,
    splitK: int,
    kernelId: int = 0,
) -> int: ...


def bmm_a8w8_mxscale_bpreshuffle_cc_ws(
    m: int,
    n: int,
    batch: int,
    splitK: int,
    kernelId: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Allocate the partial workspace the cluster-launch split-K path needs.

    The size comes from the C++ side (which owns the tile's B_M/B_N) rather than
    from a table repeated here: the launcher re-checks the byte count and throws
    on a shortfall, so a duplicated formula would turn a tile change into a
    launch error instead of just working.

    ``dtype`` must be float32 regardless of Y's dtype: the kernel stores partials
    as fp32 for BOTH C dtypes. bf16 partials were measured to push 0.05%-1.3% of
    cells outside the atol=0.5 gate at splitK>1 (cancellation amplifies the
    per-partial rounding), so the extra workspace bytes buy real accuracy. A
    bf16 buffer here is half the required BYTES and the launcher will reject it.

    Returns an EMPTY tensor for splitK <= 1, which is exactly what the raw
    binding reads as "no workspace".
    """
    numel = int(
        _opus_bmm_a8w8_mxscale_bpreshuffle_cc_ws_numel(
            int(m), int(n), int(batch), int(splitK), int(kernelId)
        )
    )
    return torch.empty(numel, dtype=dtype, device=device)


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale",
    gen_fake=_gen_bmm_a8w8_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 2,
    kernelId: int = 0,
) -> None:
    # In-place: result written into ``Y``, void return (``-> None`` keeps it
    # torch.compile-safe as a mutating op). Callers read ``Y``.
    ...


# ---- Shape-driven mxscale flatmm BMM (tuned row + heuristic fallback) ------
# The raw binding has no tuning of its own (kernelId=0 -> slow k32 fused). This
# wrapper adds selection: explicit kernelId -> verbatim; else the tuned row the
# family entry looked up; else M-split for large unaligned M; else a coarse M/G
# heuristic.


@functools.cache
def _mxscale_kid_m_align() -> dict[int, int]:
    """kid -> M multiple its launcher requires (1 == it masks a partial M tile).

    Comes from the codegen instance table, which is also what the tuner filters
    candidates on. This used to be a hand-kept kid allowlist here and a second
    hand-kept m_align column in the tuner, and the two disagreed: kid326 was
    dispatched at unaligned M by this file while the tuner never tuned it there,
    which cost ~9% at the wo_a decode shapes.
    """
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): int(inst.m_align)
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
    }


def _kid_runs_m(kid: int, m: int) -> bool:
    """True iff kid's launcher accepts this M (unknown kid -> assume it does not).

    Only a tuned row found at a padded M can name a kernel that rejects the
    real, smaller M, so this is what an incoming id is checked against below.
    No tuned winner needs alignment today (all 11 mask their partial M tile),
    but 10 of the 45 codegen instances require M % 128 or % 256, so a re-tune
    can put one in the CSV.
    """
    align = _mxscale_kid_m_align().get(int(kid))
    return align is not None and m % align == 0


def _heuristic_mxscale_kid(g: int, m: int, n: int, k: int) -> int:
    """Coarse M/G kid picker for shapes not in the tuned CSV.

    kid 158 (512x256 preload pipeline) for large-M/high-G, falling back to kid 150
    (256x256 plain) for K>8192 where 158 early-returns; kid 320/640 for small-M;
    kid 653 the general strong mid/small-M pick; kid 0 (k32 fused) for shapes that
    are not tile-aligned in N or K.
    """

    def div(a: int, b: int) -> bool:
        return a % b == 0

    if div(n, 256) and div(k, 128) and (m >= 2048 or (m >= 1024 and g >= 8)):
        # Large M: the preload pipeline (kid158) is the tuned winner across this
        # whole region (CSV picks 158 for every aligned m>=2048). No M alignment
        # needed -- the pipeline family masks its partial trailing tile via buffer
        # OOB. kid158 stages the SFA/SFB scales into LDS and early-returns for
        # K>8192 (SFA_K_MAX), so gate the preload pick at K<=8192 and fall back to
        # the plain 256x256 (kid150) for K>8192. Measured on g=2,n=1024,k=4096:
        # kid150 was 34-51% slower than 158 at the untuned m=2560/3072/3584
        # buckets, and on unaligned M a single kid158 launch beats the sub-tile
        # kid653 by 13-34% (g2/m2624, g8/m1000, g16/m600).
        return 158 if 4096 <= k <= 8192 else 150
    # Sub-tile M: B_M=32/64 tiles mask partial M via buffer OOB, so run any M
    # (no m-alignment needed -- verified 653/321/... run arbitrary unaligned M).
    if m < 64:
        return 640 if (div(n, 64) and div(k, 256)) else 653
    if m <= 256 and k <= 1024 and div(n, 32) and div(k, 256):
        return 320
    if div(n, 64) and div(k, 128):
        return 653
    return 0  # nothing tile-aligned: k32 fused runs arbitrary shapes


def bmm_a8w8_mxscale_opus(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    kernelId: int | None = None,
    splitK: int | None = None,
) -> torch.Tensor:
    """Opus fp8 e8m0 mxscale (block-scale) BMM by kernel id.

    mmajor DSV4 wo_a layout: ``x`` [M, G, K] fp8, ``wo_a`` [G, N, K] fp8,
    ``x_scale`` [M, G, K/128], ``w_scale`` [G, N/128, K/128], ``out`` optional
    [M, G, N]. Returns the [M, G, N] output.

    ``kernelId`` None falls back to the shape heuristic: the tuned CSV is read
    one layer up, in batched_gemm_a8w8_mxscale, which hands the tuned id down.
    An id this backend cannot run at this M gets the heuristic too, so the
    caller never has to know the alignment rules; _opus_bmm_a8w8_mxscale_raw is
    the entry that launches an id verbatim. ``splitK`` defaults to 1.
    """
    m, g, k = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    n = int(wo_a.shape[1])

    if out is not None:
        Y = out
    else:
        Y = torch.empty((m, g, n), dtype=dtype, device=x.device)

    # A tuned row found at a padded M can name a kernel whose launcher rejects
    # the real, smaller M; drop its splitK along with it and let the heuristic
    # pick instead of letting the launcher throw.
    if kernelId is not None and not _kid_runs_m(int(kernelId), m):
        kernelId = splitK = None
    if kernelId is None:
        kernelId = _heuristic_mxscale_kid(g, m, n, k)
    if splitK is None:
        splitK = 1

    _opus_bmm_a8w8_mxscale_raw(x, wo_a, Y, x_scale, w_scale, int(splitK), int(kernelId))
    return Y


# The bpreshuffle raw binding takes a kernelId verbatim and defaults it to 0,
# the 128x128 prefill tile. That default costs 1.8x at the DSV4 decode shapes,
# so the selection below is what a caller should go through.


@functools.cache
def _cu_count() -> int:
    return int(torch.cuda.get_device_properties(0).multi_processor_count)


# The three bpreshuffle tiles the heuristic picks among, as (B_M, B_N). The
# full list (kid 0..10, 13, 14, 17, 18) is documented in csrc/opus_gemm/
# opus_bmm.cu; the rest are either A/B controls for a tile that won, tiles for
# the 128x128 blocked w_scale, or -- kid 2 and 3 -- known broken.
_BPRESHUF_TILE_BN = {
    0: (128, 128), 6: (16, 128), 7: (16, 256),
    27: (256, 256),                      # per-column prefill, wide
    28: (256, 256), 29: (128, 128),      # 128x128 blocked prefill
}
# Largest m the decode tiles were swept at. Past it, kid0.
_BPRESHUF_DECODE_M_MAX = 256


def _heuristic_bpreshuffle_kid(
    batch: int, m: int, n: int, blocked: bool = False
) -> int:
    """Tile picker for the bpreshuffle BMM.

    ``blocked`` selects the w_scale family: False is a per-column scale
    ([G, N, K/128], GROUP_N=1), True the DSV4 128x128 block scale
    ([G, N/128, K/128], GROUP_N=128). THE BLOCKED FAMILY IS THE ONE THE MODEL
    RUNS -- the tuned table is dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned
    and FlyDSL's batched_gemm_a8w8_mxscale_bpreshuffle takes exactly that shape.
    Measured on the wide prefill tile, the blocked scale is worth 1.27x over the
    per-column one (1609 vs 1262 TFLOP/s at b=8 m=2048): at GROUP_N=128 with
    kExpN=4 the wave's 64-column span sits inside one scale block, so
    kSfBUniformOverN goes true and the per-lane gather collapses to one
    broadcast read per WMMA.

    Measured on gfx1250 (256 CU) over batch 1..16 x m 1..256 at n=1024 k=4096,
    on KERNEL time -- a host dispatch costs ~8 us there and the kernels are
    9..16, so wall time reads every tile as identical and cannot see this at
    all. Winner map: kid6 takes 26 of 35 cells, kid7 a narrow band above it,
    kid0 the large-b*m corner.

    This rule names the measured winner in 30 of 35 cells. All five misses are
    near-ties it declines to chase: four are cells where the 16x64 tile (kid4)
    edged kid6 by 0.5%-2.3%, and one is a 0.2% split at batch=4 m=256. The
    sweep's own agreement with rocprofv3 is 2.7% at worst, so a branch cut to
    catch them would be fitting the measurement, not the machine. Worst regret
    against a perfect oracle is therefore 2.3%, against 1.8x for the kernelId=0
    the raw binding defaults to.

    The shape of the rule is a workgroup count, not an M threshold, because the
    two effects that set the optimum both key on it. Below the CU count, time
    falls as the grid SHRINKS -- 64 -> 32 -> 16 workgroups all get faster --
    because a narrower B_N means more workgroups each re-reading the same
    B_M=16 rows of A and the same per-WMMA scales, and that duplication, not
    occupancy, is what binds. This inverts the premise the decode tiles were
    added under (kid0 "leaves 94% of the CUs idle", so add workgroups): kid1,
    the 16x32 tile that premise produced, does not win a single cell and is
    6x off the best at batch=16 m=256. Above the CU count the grid no longer
    buys anything, wasted M rows stop being free, and kid0's B_M=128 -- which
    amortises the A and scale reads over 128 rows instead of 16 -- takes over.

    n and k enter only through the workgroup count: every tile masks its
    partial M/N/K tiles, so nothing here needs an alignment check.
    """
    # The sweep this is fitted to stops at m=256, and the workgroup rule alone
    # extrapolates badly past it: at batch=2 m=512 it lands in the kid7 band and
    # picks a tile that is 1.52x off kid0 (50.5 us against 33.3). Measured at
    # every m>256 boundary cell -- (2,512), (4,512), (8,256), (16,128), (16,256)
    # -- kid0 is the winner, which is what a 128-row tile should be once m fills
    # it several times over. So the decode tiles are confined to the region they
    # were actually measured in rather than trusted outside it.
    cus = _cu_count()
    if blocked:
        # Blocked-scale family. Only the prefill pair has been swept; the decode
        # tiles at GROUP_N=128 (kid 8/9/10) are 16-row and were never measured
        # against these, so a small m gets the narrow prefill tile rather than a
        # guess at a decode one.
        #
        # kid28 is 256x256x128 and kid29 128x128x256, both non-specialized on 8
        # waves. Swept over batch 1..16 x m 256..8192: every cell whose kid28
        # grid reaches the CU count is a kid28 win by 1.50x-2.02x. At exactly
        # half the CU count the band splits on batch -- kid28 still wins at
        # b<=2 (1.62x at b=1 m=8192, 1.23x at b=2 m=4096) and loses at b>=8 --
        # so that half is taken only for the narrow batches.
        #
        # The batch term is fitted to two cells and is NOT understood: b=1
        # m=8192 and b=16 m=512 have the same kid29 grid, the same kid28 grid
        # and the same total rows, yet kid29 runs 119.0 us on the first and
        # 59.6 on the second. Something about a tall single-batch A is costing
        # 2x and no counter here explains it. The margin is large enough to take
        # and the mechanism is an open question.
        bm28, bn28 = _BPRESHUF_TILE_BN[28]
        wg28 = -(-m // bm28) * -(-n // bn28) * batch
        if wg28 >= cus or (wg28 * 2 >= cus and batch <= 2):
            return 28
        return 29
    if m > _BPRESHUF_DECODE_M_MAX:
        # Prefill. Two tiles, split on kid27's own workgroup count.
        #
        # kid27 is 256x256x128 with NO producer/consumer split -- 8 waves that
        # all load and all run WMMA. It is four times kid24's output tile, and
        # it only exists because dropping the specialization fits 8 consumers
        # into 256 threads (2 waves/SIMD, 512-VGPR ceiling); the specialized
        # attempt at the same tile needed 320 threads, landed 3 waves on a SIMD,
        # and spilled 1353 registers.
        #
        # Its grid is a quarter of kid24's per shape, so it is entirely gated on
        # having enough of it. Swept over batch 1..16 x m 256..8192: every cell
        # whose kid27 grid reaches the CU count is a kid27 win (1.08x-1.76x over
        # kid24), and below that it loses, by as much as 2.5x at b=8 m=512 where
        # a 256x256 tile leaves 64 workgroups on 256 CUs.
        #
        # One cell disagrees and is deliberately not fitted: b=1 m=8192 has only
        # 128 kid27 workgroups yet kid27 wins 1.38x there, while b=2 m=4096 --
        # same kid27 grid, same kid24 grid, same total rows -- goes the other way
        # by 3%. Two cells that agree on every count we can name and disagree on
        # the answer is not a rule, so the band is left to kid24.
        bm27, bn27 = _BPRESHUF_TILE_BN[27]
        wg27 = -(-m // bm27) * -(-n // bn27) * batch
        if wg27 >= cus:
            return 27
        # kid24: kid0's 128x128x256 tile at 192 threads with a 2x2 consumer-wave
        # grid. Measured 1.33x faster than kid0 at healthy clocks (111.75 vs
        # 148.36 us, b=8 m=2048). The mechanism is occupancy 1 -> 2 (538 -> 314
        # VGPR, zero scratch) from the extra consumer waves and the squared grid.
        return 24
    bm6, bn6 = _BPRESHUF_TILE_BN[6]
    wg6 = -(-m // bm6) * -(-n // bn6) * batch
    if wg6 <= cus:
        return 6
    if wg6 <= 2 * cus:
        return 7
    return 24


def bmm_a8w8_mxscale_bpreshuffle_opus(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    kernelId: int | None = None,
    splitK: int | None = None,
) -> torch.Tensor:
    """Opus fp8 e8m0 mxscale BMM with a PRESHUFFLED B, tile picked by shape.

    ``x`` [M, batch, K] fp8 (a [batch, M, K] buffer is passed as
    ``.transpose(0, 1)``, which is free -- the strides carry the layout),
    ``wo_a`` [batch, N, K] fp8 run through ``shuffle_weight(w, layout=(16,16))``,
    ``x_scale`` [M, batch, K/128], ``w_scale`` [batch, N, K/128] e8m0,
    ``out`` optional [M, batch, N]. Returns the [M, batch, N] output.

    ``kernelId`` None picks by shape (_heuristic_bpreshuffle_kid); pass an id to
    run it verbatim, which is what a tile sweep wants.
    _opus_bmm_a8w8_mxscale_bpreshuffle_raw is the entry with no selection at all.

    Both w_scale families are dispatched: the row count of ``w_scale`` selects
    between the per-column tiles and the 128x128 blocked ones, which is the
    family the model actually runs.
    """
    # Only the grid is decided here; K and every layout rule are the
    # launcher's, which checks them against the tile it is handed.
    m, batch = int(x.shape[0]), int(x.shape[1])
    n = int(wo_a.shape[1])

    Y = out if out is not None else torch.empty(
        (m, batch, n), dtype=dtype, device=x.device
    )

    if kernelId is None:
        # The w_scale's row count IS the scale family: N rows is per-column,
        # ceil(N/128) is the 128x128 block scale. Deriving it beats a flag --
        # the launcher checks the same quantity against the tile's GROUP_N, so a
        # disagreement here would surface as a shape error rather than silently
        # reading the wrong scale.
        rows = int(w_scale.shape[1])
        if rows == n:
            blocked = False
        elif rows == -(-n // 128):
            blocked = True
        else:
            raise ValueError(
                "bmm_a8w8_mxscale_bpreshuffle_opus: w_scale.shape[1] is "
                f"{rows}, which is neither N={n} (per-column scale) nor "
                f"ceil(N/128)={-(-n // 128)} (128x128 block scale)."
            )
        kernelId = _heuristic_bpreshuffle_kid(batch, m, n, blocked)

    _opus_bmm_a8w8_mxscale_bpreshuffle_raw(
        x, wo_a, Y, x_scale, w_scale, int(splitK or 1), int(kernelId)
    )
    return Y


__all__ = [
    "_opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_raw",
    "_opus_bmm_a8w8_mxscale_bpreshuffle_raw",
    "_opus_bmm_a8w8_mxscale_raw",
    "bmm_a8w8_mxscale_bpreshuffle_cc_ws",
    "bmm_a8w8_mxscale_bpreshuffle_opus",
    "bmm_a8w8_mxscale_opus",
]
