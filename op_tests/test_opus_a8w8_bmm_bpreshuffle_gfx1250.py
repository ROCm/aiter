# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Probe / correctness gate for gfx1250 a8w8 mxscale BMM with preshuffled B.

The cases form a ladder: each one adds exactly one axis on top of the previous,
so a failure localises to the thing that was just turned on rather than to "the
kernel". In order: identity B (frag_a / frag_b / store_c mapping), dense B (the
K reduction), non-unit e8m0 scales (dequant + the per-row/per-col scale layout),
multi-K-step (the NUM_SLOTS=3 software pipeline's steady state and drain), a
partial K tail, batch > 1 (the five *_batch strides), multi-tile M/N (grid.x and
grid.y), and an fp32 Y (the second kernel instantiation).

Inputs are small integers and the e8m0 exponents are +-2 around 2^0, so every
product and every partial sum is exactly representable in fp32; the reference is
computed in fp64 and the only permitted error is the single rounding of the bf16
store. That is what makes a tight rtol meaningful here.

NOT covered: the kScaleBcast=false (GROUP_K=32 / true MX) branch of
pack_sf_word_bmm_mx. No trait instance sets GROUP_K=32, so that branch is parsed
but never instantiated -- reaching it needs a new tile, not a new test case.

Usage (on gfx1250 hardware, inside the dev container):
    rm -f aiter/jit/module_deepgemm_opus.so
    rm -rf aiter/jit/build/module_deepgemm_opus
    ROCM_HOME=/opt/rocm-260803 GPU_ARCHS=gfx1250 PYTHONPATH=. \
        python3 op_tests/test_opus_a8w8_bmm_bpreshuffle_gfx1250.py

ROCM_HOME is not optional on a host that also has the rocm-sdk pip wheels
installed: aiter's _find_rocm_home() (aiter/jit/utils/cpp_extension.py) checks
ROCM_HOME/ROCM_PATH first and otherwise prefers site-packages/_rocm_sdk_devel,
whose LLVM is a different revision -- its __builtin_amdgcn_tensor_load_to_lds
takes 5 arguments where opus.hpp passes 6, so the whole module fails to build
with "too many arguments to function call" in files unrelated to this test.
"""

from __future__ import annotations

import sys

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_bpreshuffle_raw
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")

GROUP_K = (
    128  # DS V4 / gfx950 flatmm: 1x128 blockscale (pack_e8m0x4 broadcast in kernel)
)
E8M0_ONE = 127  # 0x7F -> 2^0 scale factor
E8M0_SPREAD = 2  # exponents drawn from ONE +- this -> scale factors 0.25 .. 4

# bf16 has 8 mantissa bits, so a correct result may differ from the exact sum by
# one rounding (2^-9 ~ 0.2%); 0.02 leaves 10x headroom without hiding real bugs.
BF16_RTOL, BF16_ATOL = 0.02, 0.5
# The fp32 store should reproduce the reference bit-for-bit (see module docstring).
FP32_RTOL, FP32_ATOL = 1e-6, 1e-3


def _fp8_ints(shape: tuple[int, ...], lo: int, hi: int) -> torch.Tensor:
    """Random integers in [lo, hi], exact in fp8 e4m3 and so free of input rounding."""
    v = torch.randint(lo, hi + 1, shape, dtype=torch.int32)
    return v.to(dtypes.fp32).to(dtypes.fp8)


def _e8m0(shape: tuple[int, ...], varied: bool) -> torch.Tensor:
    """e8m0 exponent bytes: all 2^0, or spread so a mis-indexed scale is a wrong number."""
    if not varied:
        return torch.full(shape, E8M0_ONE, dtype=torch.uint8)
    off = torch.randint(-E8M0_SPREAD, E8M0_SPREAD + 1, shape, dtype=torch.int32)
    return (off + E8M0_ONE).to(torch.uint8)


def _ref_bmm(
    O_fp8: torch.Tensor,
    W_fp8: torch.Tensor,
    x_scale_u8: torch.Tensor,
    w_scale_u8: torch.Tensor,
) -> torch.Tensor:
    """[G,M,K] x [G,N,K] -> [G,M,N] with 1x128 e8m0 blockscale dequant.

    fp64 throughout: an fp32 einsum here could legitimately reduce in a different
    order (or on tf32 hardware, at lower precision) and blunt the comparison.
    """
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    act = O_fp8.to(torch.float64).view(G, M, K // GROUP_K, GROUP_K)
    act = act * torch.exp2(x_scale_u8.to(torch.float64) - 127.0).unsqueeze(-1)
    act = act.reshape(G, M, K)
    w = W_fp8.to(torch.float64).view(G, N, K // GROUP_K, GROUP_K)
    w = w * torch.exp2(w_scale_u8.to(torch.float64) - 127.0).unsqueeze(-1)
    w = w.reshape(G, N, K)
    return torch.einsum("gmk,gnk->gmn", act, w)


def run_case(
    name: str,
    m: int,
    n: int,
    k: int,
    g: int = 1,
    *,
    dense_w: bool = False,
    dense_a: bool = False,
    varied_scales: bool = False,
    out_dtype: torch.dtype | None = None,
    seed: int = 0,
    group_n: int = 1,
    kid: int = 0,
) -> bool:
    assert m % 16 == 0 and n % 16 == 0 and k % 32 == 0, "shape alignment"
    assert k % GROUP_K == 0, "K must be a whole number of scale groups"
    assert n % group_n == 0, "N must be a whole number of B-scale blocks"
    out_dtype = dtypes.bf16 if out_dtype is None else out_dtype
    torch.manual_seed(seed)

    if dense_a:
        O = _fp8_ints((g, m, k), 1, 6)
    else:
        # A[m,k] = (m % 16) + 1 -- constant along K, unique within a 16x16 tile.
        O = torch.zeros(g, m, k, dtype=dtypes.fp8)
        for mi in range(m):
            O[:, mi, :] = float((mi % 16) + 1)

    if dense_w:
        W = _fp8_ints((g, n, k), -3, 3)
    else:
        # Identity before shuffle: W[g,n,k] = 1 when n==k else 0.
        W = torch.zeros(g, n, k, dtype=dtypes.fp8)
        for i in range(min(n, k)):
            W[:, i, i] = torch.tensor(1.0, dtype=dtypes.fp8)

    W_shuf = shuffle_weight(W, layout=(16, 16)).contiguous()

    xs = _e8m0((g, m, k // GROUP_K), varied_scales)
    # A's scale is always per-row (the "1x" of 1x128). B's granularity along N is
    # the tile's GROUP_N: 1 gives a per-column scale, 128 the DSV4 block scale
    # whose tensor has only N/128 rows. The kernel is handed the blocked tensor;
    # the reference expands it back to per-column so _ref_bmm stays one code path
    # and the expansion -- not the reference -- is what the kernel must match.
    ws = _e8m0((g, n // group_n, k // GROUP_K), varied_scales)
    ws_ref = ws.repeat_interleave(group_n, dim=1) if group_n > 1 else ws

    # Axis order is fixed at (M, batch, *) for A/Y/x_scale; wo_a and w_scale stay
    # batch-major. The transposes are what make the launcher's stride_*_batch and
    # the asymmetric stride_sfa / stride_sfb reads meaningful, so they must not be
    # "simplified" away.
    O_in = O.transpose(0, 1).contiguous()  # [m,g,k]
    xs_in = xs.transpose(0, 1).contiguous()  # [m,g,k/GROUP_K]
    ref = _ref_bmm(O, W, xs, ws_ref).transpose(0, 1)  # [m,g,n]

    Y = torch.empty(m, g, n, dtype=out_dtype)
    _opus_bmm_a8w8_mxscale_bpreshuffle_raw(O_in, W_shuf, Y, xs_in, ws, 1, kid)

    rtol, atol = (
        (FP32_RTOL, FP32_ATOL) if out_dtype == dtypes.fp32 else (BF16_RTOL, BF16_ATOL)
    )
    got = Y.to(torch.float64)
    err_ratio = checkAllclose(
        got, ref, msg=f"{name}: ", rtol=rtol, atol=atol, printLog=True
    )
    ok = err_ratio == 0

    denom = ref.abs().clamp_min(1.0)
    max_abs = (got - ref).abs().max().item()
    max_rel = ((got - ref).abs() / denom).max().item()
    print(
        f"{'PASS' if ok else 'FAIL'}  {name:<26} "
        f"m={m:<4} n={n:<4} k={k:<4} g={g} out={str(out_dtype).split('.')[-1]:<8} "
        f"kid={kid} gn={group_n:<3} "
        f"max_abs={max_abs:.4g} max_rel={max_rel:.3g}"
    )
    return ok


def run_shape_guard_case() -> bool:
    """The launcher must REJECT a per-column w_scale handed to a GROUP_N=128 tile.

    This is the one mismatch that cannot be caught numerically after the fact: a
    per-column scale is a valid, contiguous, in-bounds e8m0 tensor, and indexing
    it by block number just reads a real exponent belonging to the wrong columns.
    Without the launcher's size(1) check the result is plausible and wrong, so the
    check is part of the contract and gets its own test.
    """
    m, n, k, g = 32, 256, 256, 1
    torch.manual_seed(7)
    O = _fp8_ints((g, m, k), 1, 6)
    W = _fp8_ints((g, n, k), -3, 3)
    W_shuf = shuffle_weight(W, layout=(16, 16)).contiguous()
    xs_in = _e8m0((g, m, k // GROUP_K), True).transpose(0, 1).contiguous()
    ws_per_column = _e8m0((g, n, k // GROUP_K), True)  # GROUP_N=1 shaped
    Y = torch.empty(m, g, n, dtype=dtypes.bf16)
    O_in = O.transpose(0, 1).contiguous()

    try:
        _opus_bmm_a8w8_mxscale_bpreshuffle_raw(
            O_in, W_shuf, Y, xs_in, ws_per_column, 1, 8  # kid8 is GROUP_N=128
        )
    except Exception as e:
        msg = str(e)
        ok = "GROUP_N" in msg or "w_scale.size(1)" in msg
        print(
            f"{'PASS' if ok else 'FAIL'}  {'gn shape guard':<26} "
            f"rejected as expected: {msg.splitlines()[0][:70]}"
        )
        return ok
    print(
        f"FAIL  {'gn shape guard':<26} "
        f"kid8 ACCEPTED a per-column w_scale (size(1)={n}, expected {n // 128})"
    )
    return False


def main() -> int:
    gfx = get_gfx_runtime()
    print(f"runtime gfx = {gfx}")
    if gfx != "gfx1250":
        print(
            "SKIP: opus_bmm_a8w8_mxscale_bpreshuffle is gfx1250-only; "
            f"this node reports {gfx}.",
            file=sys.stderr,
        )
        return 2

    results: list[tuple[str, bool]] = []

    def case(name: str, *args, **kwargs) -> None:
        results.append((name, run_case(name, *args, **kwargs)))

    # Tile is 128x128x256. Layout-only cases first: an identity B makes each
    # output element one input element, so a frag/store mistake is unmissable.
    case("identity full tile", 128, 128, 256)
    case("identity minimal", 16, 16, 128)
    # Dense B turns on the K reduction; dense A also varies A along K, which the
    # identity cases above cannot see (their A is constant along K).
    case("dense A/B unit scale", 128, 128, 256, dense_a=True, dense_w=True)
    # Non-unit exponents: dequant, plus the per-row (sfa) / per-col (sfb) layouts.
    case(
        "varied e8m0 scales",
        128,
        128,
        256,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        seed=1,
    )
    # K = 3 * B_K: first K step the pipeline actually reaches steady state + drain.
    case(
        "multi-K-step",
        128,
        128,
        768,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        seed=2,
    )
    # 256 + 128: a full K step followed by a partial one.
    case(
        "partial K tail",
        128,
        128,
        384,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        seed=3,
    )
    # g > 1: until here every stride_*_batch was multiplied by zero, including the
    # deliberately asymmetric stride_sfa / stride_sfb pair.
    case(
        "batch g=3",
        128,
        128,
        256,
        g=3,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        seed=4,
    )
    # grid.x > 1 and grid.y > 1, on top of batch and multi-K-step.
    case(
        "multi-tile M/N",
        256,
        256,
        512,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        seed=5,
    )
    # M/N below one tile (masked edges) and the fp32 Y instantiation.
    case(
        "masked edges, fp32 Y",
        48,
        48,
        384,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        out_dtype=dtypes.fp32,
        seed=6,
    )

    # -- GROUP_N = 128, the DSV4 block-scale granularity ---------------------
    # n = 256 is the minimum that means anything here: it gives TWO scale blocks,
    # so a wrong block index is a wrong number. At n = 128 there is one block,
    # every block index folds to 0, and any indexing bug is invisible.
    #
    # The two tiles probe different halves of the indexing. kid8 (B_N = 64) makes
    # each workgroup sit wholly inside one block, exercising the tile_col -> block
    # mapping across 4 tiles. kid9 (B_N = 256) puts ONE workgroup across both
    # blocks, so waves 0-1 must read block 0 and waves 2-3 block 1 -- the
    # intra-tile boundary, which is where kSfBUniformOverN is actually load-bearing.
    case(
        "gn128 kid8 B_N=64",
        64,
        256,
        512,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        group_n=128,
        kid=8,
        seed=8,
    )
    case(
        "gn128 kid9 B_N=256",
        64,
        256,
        512,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        group_n=128,
        kid=9,
        seed=9,
    )
    # kid10 (B_N=192, kExpN=3) is the ONLY tile where the scale tensor is blocked
    # but a wave's 48-column span does not divide 128: wave 2 covers columns
    # 96..143, straddling block 0 and block 1, so its scale differs between lanes
    # despite GROUP_N=128. n=384 gives 3 blocks and 2 N-tiles, so the straddle
    # happens at two different block pairs. A layout branch keyed on uniformity
    # rather than on GROUP_N reads this case as if w_scale had n rows -- in bounds,
    # plausible, and wrong, which the launcher's shape check cannot catch.
    case(
        "gn128 kid10 B_N=192",
        64,
        384,
        512,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        group_n=128,
        kid=10,
        seed=10,
    )
    # -- A-scale staged in LDS (SF_A_LDS) ------------------------------------
    # kid13 is kid0 with the panel on, so this case is aimed at what the panel
    # changes rather than at the maths again. What is new is the row indexing:
    # the global path reads an ABSOLUTE row clamped to m-1, the panel reads a
    # TILE-LOCAL row with no clamp. Forgetting to subtract tile_row makes every
    # tile but the first read the wrong rows, so m spans several M tiles; m=208
    # with B_M=128 also leaves 48 rows of the second tile outside the fill's
    # buffer bound, which is the partial-M path.
    #
    # g=2 is deliberate: the fill takes stride_sfa as its source row pitch and
    # the batch offset through ptr_sfa, and x_scale's [M, batch, K/GK] layout
    # makes those two different numbers (g*K/GK vs K/GK). Swapping them is in
    # bounds at g=1 and only shows up here.
    #
    # k=768 gives 6 K-groups against stride_sfa=12, so 6|12 = 14 fails the 16-byte
    # test and the fill drops to the VEC=1 rung -- the narrow rung is the one a
    # divisibility bug would hide in, and no shape reaches it by accident.
    case(
        "sfa kid13 prefill",
        208,
        256,
        768,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=13,
        seed=13,
    )
    # kid14 adds B's panel, whose row is a kGroupN BLOCK index rather than a row
    # index, so it needs its own case even though the fill code is shared.
    #
    # At GROUP_N=1 the panel base is tile_col itself, so n must span more than one
    # N tile or the subtraction is a no-op and a missing one passes. n=208 with
    # B_N=128 gives two N tiles, the second holding 80 live columns: the panel's
    # rows past nb_max=207 come back zero from the buffer bound, and the read side
    # clamps onto row 207. Those two have to agree about which row is last, and
    # they are written in different places (the fill's bound vs sfb_nb's min).
    # 208 and not a rounder partial because shuffle_weight(16,16) needs n % 16.
    #
    # g=2 again, because w_scale's [batch, N, K/GK] layout makes stride_sfb and
    # stride_sfb_batch different numbers and the fill uses one of each.
    case(
        "sfab kid14 prefill",
        208,
        208,
        768,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=14,
        seed=14,
    )
    # The WIDE rung of the fill ladder, which nothing above reaches: it needs 16
    # to divide both K/GROUP_K and the source row stride, so k must be at least
    # 2048 (k/128 = 16, stride = g*16). Every case above runs at VEC=1.
    #
    # This is the rung the benchmarks have always used and the one no correctness
    # case covered -- a 16-byte LDS store lands on a row start of
    # round16(K/GK)+16, so it is only aligned if the panel's LDS base is too, and
    # that base is a sum of ring segment sizes. A static_assert now guards it,
    # but the assert is an argument about arithmetic and this is the measurement.
    case(
        "sfab kid14 wide fill",
        208,
        208,
        2048,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=14,
        seed=15,
    )
    # Same rung on the A-only tile, so a fault here separates which panel broke.
    case(
        "sfa kid13 wide fill",
        208,
        208,
        2048,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=13,
        seed=16,
    )
    # k=4096 is the shape every performance number is taken at, and it is not
    # covered by k=2048: there K/GROUP_K is 16, so a row is exactly ONE 16-byte
    # chunk and the fill's intra-row offset kt is always 0. At k=4096 a row is
    # two chunks, so kt reaches 16 and the store offset r*sf_pitch + kt is the
    # only place the padded pitch and the chunk index have to compose correctly.
    case(
        "sfab kid14 k4096",
        208,
        208,
        4096,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=14,
        seed=17,
    )
    # kid17/kid18 fill the same A panel by DMA rather than cooperatively, so what
    # needs covering is the descriptor's clamping rather than a store offset.
    # k=4096 is the geometry kid17 is built for: width 32 == K/GROUP_K exactly,
    # and its pitch of 48 is the one kid13 computes at runtime.
    case(
        "sfa kid17 tdm w32 k4096",
        208,
        208,
        4096,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=17,
        seed=18,
    )
    # Width WIDER than the K it is given: sf_kg is 16 here against a tile width
    # of 32, so half of every descriptor row is out of range. The read side never
    # looks past sf_kg, so this must pass -- and if the engine were writing those
    # columns at the wrong pitch instead of clamping, the rows below would be the
    # ones to break.
    case(
        "sfa kid17 tdm w32 k2048",
        208,
        208,
        2048,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=17,
        seed=19,
    )
    # kid18 is the 4x over-fetch: width 128 against sf_kg=32, and a pad of 4 that
    # makes the pitch 132 rather than a 16-aligned 48. The pitch the reader uses
    # comes from the same constant as the D#'s pad, so this case is what proves
    # the two spellings agree.
    case(
        "sfa kid18 tdm w128 k4096",
        208,
        208,
        4096,
        g=2,
        dense_a=True,
        dense_w=True,
        varied_scales=True,
        kid=18,
        seed=20,
    )
    # The same two tiles on the per-column path must still be rejected, not
    # silently mis-read.
    results.append(("gn shape guard", run_shape_guard_case()))

    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} cases passed")
    if failed:
        print("FAILED: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
