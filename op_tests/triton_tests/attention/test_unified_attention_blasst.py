# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block skipping in unified_attention's 2D Triton kernel.

Block skipping is approximate on purpose: it drops K/V tiles whose scores sit
far enough below the running maximum that their softmax weights cannot matter.
So these tests are not "does it match a reference" -- at a threshold above zero
it deliberately does not. They pin down the properties that must hold anyway:

  * with the feature off, NOTHING changes -- bitwise
  * with the feature on but nothing elided, nothing is APPROXIMATED
  * the paths that cannot support it fall back to dense rather than half-working
  * no threshold produces NaN, including the > 1.0 range that once produced
    100% NaN by comparing against the post-fold maximum instead of the running
    one (see `test_no_nan_above_unity`)
  * the scheduling that rides along reorders work without changing a single bit
  * the elision counter reports something a caller can act on

Reuses generate_data / ref_paged_attn from test_unified_attention.py so the
inputs and the reference are exactly the ones the dense tests use.
"""
import pytest
import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils._triton import arch_info

from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)

# The shape has to be big enough to ROUTE to the 2D kernel, which is the only
# one this feature touches. use_2d_kernel() picks 2D when
# total_num_q_blocks * num_kv_heads > num_sms * 4; below that it takes the 3D
# path, where the wrapper disables skipping outright.
#
# 8192 tokens at 32q/8kv gives ~2056 programs, comfortably over the target on
# both MI300X (304 CUs) and MI355X (256).
SEQ_LENS = [(8192, 8192)]
NUM_HEADS = (32, 8)
HEAD_SIZE = 128
NUM_BLOCKS = 8192

# Measured on these exact inputs (generate_data, seed 0, bf16):
#
#   lambda   0.001   0.01    0.1     0.3     1.0     1.5     2.0     4.0    12.0
#   elide     0.0%   0.0%   0.0%    0.0%   8.26%  38.16%  58.12%  85.15%  96.87%
#   reldiff      0      0      0       0   0.116   0.427   0.728   1.705   4.060
#
# The shape of that table is the specification. The skip decision is per TILE:
# a tile is dropped only if every row votes to, and a tile that survives the
# vote is computed exactly as dense. So below the threshold at which whole
# tiles start dropping, the kernel computes dense values -- reldiff is 0 up to
# floating-point reassociation. `test_no_elision_is_dense_within_rounding`
# pins that down.
#
# Random data needs lambda >= 1 before anything elides at all. A calibrated
# threshold on real activations is far smaller; these are mechanism tests, not
# representative operating points.
THRESHOLDS_DEGRADE = (1.0, 1.5, 2.0, 4.0)
THRESHOLDS_NO_ELISION = (1e-9, 1e-3, 1e-2, 1e-1, 3e-1)
THRESHOLDS_ELIDE = (1e-9, 1.0, 2.0, 4.0, 12.0)


def _build(block_size=16, shuffled_kv_cache=False, sliding_window=None,
           seq_lens=None, num_heads=None):
    return generate_data(
        seq_lens=seq_lens or SEQ_LENS,
        num_blocks=NUM_BLOCKS,
        block_size=block_size,
        head_size=HEAD_SIZE,
        num_heads=num_heads or NUM_HEADS,
        sliding_window=sliding_window,
        shuffled_kv_cache=shuffled_kv_cache,
        device="cuda",
    )


def _run(data, threshold, out=None, skip_counter=None, sliding_window=None,
         backend="triton"):
    """One unified_attention call on prepared data, returning the output."""
    (query, _kc_orig, _vc_orig, key_cache, value_cache, sinks, output,
     cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, window_size,
     block_tables, _mq, _qs, q_descale, k_descale, v_descale,
     output_scale) = data
    dst = output if out is None else out
    unified_attention(
        q=query, k=key_cache, v=value_cache, out=dst,
        cu_seqlens_q=cu_query_lens, seqused_k=kv_lens,
        max_seqlen_q=max_query_len, max_seqlen_k=max_kv_len,
        softmax_scale=scale, causal=True, window_size=window_size,
        block_table=block_tables, softcap=0,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        sinks=sinks, output_scale=output_scale,
        block_skip_threshold=threshold,
        skip_counter=skip_counter,
        backend=backend,
    )
    return dst.clone()


def _skipif_gfx12():
    """Deprecated: skipping by ARCHITECTURE was too broad.

    The threshold is rejected on the GLUON backend, not on gfx1250 as such, and
    every test here drives `backend="triton"`. Skipping the whole architecture
    therefore deleted all coverage of the supported Triton path on gfx1250.
    Kept as a no-op so call sites stay readable; remove once they are all gone.
    """
    return


def _require_2d_path(data):
    """Fail loudly if this shape does not reach the 2D kernel.

    The 3D path disables skipping, so a test that lands there passes while
    testing nothing -- which is exactly what an earlier version of this file
    did. The counter is the cheapest honest probe: if the 2D kernel ran with
    skipping on, it saw tiles.
    """
    buf = torch.zeros(2, dtype=torch.int64, device="cuda")
    _run(data, 1.0, out=torch.empty_like(data[6]), skip_counter=buf)
    seen = int(buf[0])
    assert seen > 0, (
        "this shape did not reach the 2D kernel (the counter saw 0 tiles), so "
        "block skipping was disabled and the test would prove nothing. "
        "use_2d_kernel() needs total_num_q_blocks * num_kv_heads > num_sms * 4."
    )


# ---------------------------------------------------------------------------
# 1. the feature off must change nothing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("shuffled_kv_cache", [False, True])
def test_threshold_zero_is_dense(block_size, shuffled_kv_cache):
    """threshold=0 must match the reference AND be bitwise identical to not
    passing the argument at all. The second half is the one that matters: it is
    what says the conditional folded away rather than merely evaluating to the
    same numbers by luck."""
    _skipif_gfx12()
    if shuffled_kv_cache:
        pytest.skip("shuffled_kv_cache requires the gfx1250 gluon 2d kernel")
    data = _build(block_size=block_size, shuffled_kv_cache=shuffled_kv_cache)
    query, kc_orig, vc_orig = data[0], data[1], data[2]
    kv_lens_list = [x[1] for x in SEQ_LENS]

    explicit_zero = _run(data, 0.0, out=torch.empty_like(data[6]))
    omitted = torch.empty_like(data[6])
    unified_attention(
        q=query, k=data[3], v=data[4], out=omitted,
        cu_seqlens_q=data[7], seqused_k=data[8], max_seqlen_q=data[9],
        max_seqlen_k=data[10], softmax_scale=data[11], causal=True,
        window_size=data[12], block_table=data[13], softcap=0,
        q_descale=data[16], k_descale=data[17], v_descale=data[18],
        sinks=data[5], output_scale=data[19], backend="triton",
    )
    assert torch.equal(explicit_zero, omitted), (
        "threshold=0.0 is not bitwise identical to omitting the argument"
    )

    ref = ref_paged_attn(
        query=query, key_cache=kc_orig, value_cache=vc_orig,
        query_lens=[x[0] for x in SEQ_LENS], kv_lens=kv_lens_list,
        block_tables=data[13], scale=data[11], out_dtype=torch.bfloat16,
        sliding_window=None, soft_cap=None, sinks=data[5],
        q_descale=data[16], k_descale=data[17], v_descale=data[18],
        output_scale=data[19],
    )
    torch.testing.assert_close(
        explicit_zero.to(torch.float32), ref.to(torch.float32),
        atol=1.5e-2, rtol=1e-2,
    )


# ---------------------------------------------------------------------------
# 2. no elision means no approximation
# ---------------------------------------------------------------------------
def test_no_elision_is_dense_within_rounding():
    """A threshold that drops no tiles must not APPROXIMATE anything.

    NOT bitwise, deliberately. Enabling the threshold puts the P@V dot inside
    an scf.if, and the AMD backend's chain-dot detection only looks within one
    MLIR region -- so the warp layout it picks for that dot can differ from the
    dense path's, which reassociates the MFMA accumulation. Measured on gfx950
    without the chain-dot patch: max absolute difference 1.95e-3, exactly one
    bf16 ULP, with mean relative difference 3.4e-7. With the patch both paths
    get the same layout and it is bitwise identical.

    The tolerance below sits five orders of magnitude above reassociation and
    five below the per-row-masking bug it exists to catch (>= 0.03), so it
    cannot be satisfied by that bug returning.
    """
    _skipif_gfx12()
    data = _build()
    _require_2d_path(data)
    dense = _run(data, 0.0, out=torch.empty_like(data[6]))

    checked = 0
    for thr in THRESHOLDS_NO_ELISION:
        buf = torch.zeros(2, dtype=torch.int64, device="cuda")
        out = _run(data, thr, out=torch.empty_like(data[6]), skip_counter=buf)
        if int(buf[1]):
            continue  # this threshold does elide here; not what this tests
        checked += 1
        rel = ((out.float() - dense.float()).abs().mean()
               / dense.float().abs().mean().clamp_min(1e-6)).item()
        assert rel < 1e-5, (
            f"threshold={thr} elided 0 tiles yet moved the output by "
            f"{rel:.3g} mean relative -- far beyond floating-point "
            "reassociation. Rows are being masked individually, which costs "
            "accuracy and saves no work."
        )

    # Without this the test passes vacuously: if a bug made every threshold
    # elide something, every iteration would `continue` and nothing would ever
    # be compared.
    assert checked, (
        "no threshold in THRESHOLDS_NO_ELISION elided zero tiles, so the "
        "dense-equivalence assertion never ran. Lower the thresholds, or a "
        "regression has made everything elide."
    )


# ---------------------------------------------------------------------------
# 3. degradation is bounded and monotonic in the threshold
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [16, 64])
def test_degrades_monotonically(block_size):
    """A larger threshold skips more and so must move further from dense. If
    this is flat, the threshold is not reaching the kernel; if it is
    non-monotonic, the skip predicate is not doing what it claims."""
    _skipif_gfx12()
    data = _build(block_size=block_size)
    _require_2d_path(data)
    dense = _run(data, 0.0, out=torch.empty_like(data[6]))

    prev = -1.0
    for thr in THRESHOLDS_DEGRADE:
        out = _run(data, thr, out=torch.empty_like(data[6]))
        assert torch.isfinite(out).all(), f"threshold={thr} produced non-finite output"
        rel = ((out.float() - dense.float()).abs().mean()
               / dense.float().abs().mean().clamp_min(1e-6)).item()
        assert rel >= prev - 1e-6, (
            f"threshold={thr} moved LESS far from dense ({rel:.6g}) than the "
            f"previous threshold ({prev:.6g}); skipping is not monotonic"
        )
        prev = rel
    assert prev > 0, "no threshold changed the output -- the flag is dead"


# ---------------------------------------------------------------------------
# 4. the NaN regression
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("threshold", [1.001, 2.0, 4.0, 12.0])
def test_no_nan_above_unity(threshold):
    """Thresholds above 1.0 mean log2(threshold) > 0, so a row can skip even
    when this tile's maximum EXCEEDS the running maximum.

    This is the regression guard for comparing against the post-fold maximum
    (`m_j`) rather than the running one (`M`). That form gives `m_j - m_j == 0`
    for the leading row, so it skips, and its accumulator is then rescaled by
    exp2(-inf - -inf) -> NaN. It produced 100% NaN for every threshold above
    1.0. These thresholds are not useful operating points; they exist purely to
    keep that bug from coming back."""
    _skipif_gfx12()
    data = _build()
    out = _run(data, threshold, out=torch.empty_like(data[6]))
    assert torch.isfinite(out).all(), (
        f"threshold={threshold} produced NaN/Inf -- the skip predicate is "
        "comparing against the folded maximum instead of the running maximum"
    )


# ---------------------------------------------------------------------------
# 5. paths that cannot support it fall back to dense, bitwise
# ---------------------------------------------------------------------------
def test_sliding_window_falls_back_to_dense():
    """Sliding window can mask an entire row, leaving its running maximum at
    -inf which the sanitizer rewrites to 0.0 -- changing what the skip
    comparison means. Until that has a test of its own the wrapper disables
    skipping, and this checks it does so completely rather than partially."""
    _skipif_gfx12()
    data = _build(sliding_window=256)
    dense = _run(data, 0.0, out=torch.empty_like(data[6]))
    asked = _run(data, 0.3, out=torch.empty_like(data[6]))
    assert torch.equal(dense, asked), (
        "sliding_window + block_skip_threshold did not fall back to dense"
    )


def test_3d_routed_shape_falls_back_to_dense():
    """A shape that routes to the 3D kernel must run dense and leave the
    counter alone.

    The 3D kernel splits the key sequence into segments and restarts the
    running maximum in each one, so `tile_max - running_max` means something
    different there and the first tile of every segment could never skip. The
    wrapper force-disables skipping on that path; this checks that it does.

    ROUTING, which this test has to PROVE rather than assume: use_2d_kernel()
    picks 2D when total_num_q_blocks * num_kv_heads exceeds num_sms * 4. At 256
    query tokens with 1 KV head and BLOCK_Q=32 that is ~9 programs against a
    target of ~1024 on MI355X and ~1216 on MI300X, so this lands on 3D by a
    wide margin. max_seqlen_k is 1024, above the 512 cutoff that would
    otherwise force 2D regardless of program count.

    `tiles visited == 0` IS the routing proof. On the 2D path COUNT_SKIPS is on
    whenever the threshold is positive and a counter is passed, and it counts
    every tile it inspects -- including ones it does not elide. So a non-zero
    count can only mean the 2D kernel ran, i.e. this test is not exercising
    what it claims. The 3D path never writes the counter at all. That single
    assertion therefore covers both the routing and the fact that the 3D
    branch drops skip_counter.
    """
    _skipif_gfx12()
    data = _build(seq_lens=[(256, 1024)], num_heads=(4, 1))
    buf = torch.zeros(2, dtype=torch.int64, device="cuda")

    dense = _run(data, 0.0, out=torch.empty_like(data[6]))
    asked = _run(data, 0.3, out=torch.empty_like(data[6]), skip_counter=buf)

    assert int(buf[0]) == 0, (
        f"the 2D kernel ran ({int(buf[0])} tiles counted), so this shape no "
        "longer routes to 3D and the test is not exercising the 3D fallback. "
        "Shrink the shape until it does."
    )
    assert torch.equal(dense, asked), (
        "a 3D-routed shape changed its output once a threshold was set, so the "
        "force-disable on the 3D path is incomplete"
    )


def test_decode_falls_back_to_dense():
    """One query row per sequence: there is no prior maximum to be far below,
    so nothing can skip and the check would be pure cost. Disabled silently
    rather than raising, because a caller that switches this on once for a
    whole model still reaches this wrapper on every decode step."""
    _skipif_gfx12()
    data = _build(seq_lens=[(1, 1024)])
    dense = _run(data, 0.0, out=torch.empty_like(data[6]))
    asked = _run(data, 0.3, out=torch.empty_like(data[6]))
    assert torch.equal(dense, asked), "decode did not fall back to dense"


@pytest.mark.skipif(not IS_DEVICE_ARCH_GFX12, reason="gfx1250 only")
def test_gluon_backend_rejects_threshold():
    """The Gluon kernel cannot honour the threshold. Ignoring it silently would
    return dense results to a caller who believed they had asked for sparsity,
    so any speedup they then measured would be against the wrong baseline."""
    data = _build()
    with pytest.raises(AssertionError, match="not supported on the Gluon backend"):
        # backend="gluon" explicitly: _run defaults to Triton, so without this
        # the call takes the Triton path and the test passes for the wrong
        # reason -- it never exercises the Gluon rejection it is named for.
        _run(data, 0.3, out=torch.empty_like(data[6]), backend="gluon")


# ---------------------------------------------------------------------------
# 6. the scheduling reorders work without changing it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("threshold", [0.1, 0.3])
def test_scheduling_is_bitwise_identical(threshold, monkeypatch):
    """Descending-q ordering and the atomic work ticket only change WHICH
    workgroup computes WHICH tile. Permuting that cannot move a bit.

    If this fails, the ticket is not a bijection onto the work items -- some
    tile computed twice and another never -- which would still produce
    plausible-looking output that no accuracy metric would flag."""
    _skipif_gfx12()
    data = _build()
    monkeypatch.setenv("AITER_UA_BLASST_SCHED", "1")
    with_sched = _run(data, threshold, out=torch.empty_like(data[6]))
    monkeypatch.setenv("AITER_UA_BLASST_SCHED", "0")
    without = _run(data, threshold, out=torch.empty_like(data[6]))
    assert torch.equal(with_sched, without), (
        f"{int((with_sched != without).sum())} elements differ with the "
        "scheduling on vs off; the work ticket is not covering the grid exactly"
    )


# ---------------------------------------------------------------------------
# 7. the elision counter
# ---------------------------------------------------------------------------
def test_skip_counter_reports_elision():
    """The counter is how a caller discovers whether their threshold elides
    anything on their own data, so a silently-zero counter would be worse than
    none: it would read as "this threshold is useless" when the wiring was at
    fault."""
    _skipif_gfx12()
    data = _build()

    prev = -1.0
    for thr in THRESHOLDS_ELIDE:
        buf = torch.zeros(2, dtype=torch.int64, device="cuda")
        _run(data, thr, out=torch.empty_like(data[6]), skip_counter=buf)
        seen, elided = (int(x) for x in buf.cpu())
        assert seen > 0, f"threshold={thr}: counter recorded 0 tiles visited"
        assert 0 <= elided <= seen, f"elided={elided} outside [0, {seen}]"
        frac = elided / seen
        assert frac >= prev - 1e-9, (
            f"threshold={thr} elided a SMALLER fraction ({frac:.4f}) than the "
            f"lower threshold before it ({prev:.4f})"
        )
        prev = frac
    assert prev > 0, "the largest threshold elided nothing -- counter is dead"


def test_skip_counter_untouched_when_disabled():
    """With skipping off there is nothing to count, and the kernel should not
    be paying for atomics it does not need."""
    _skipif_gfx12()
    data = _build()
    buf = torch.zeros(2, dtype=torch.int64, device="cuda")
    _run(data, 0.0, out=torch.empty_like(data[6]), skip_counter=buf)
    assert int(buf.sum()) == 0, "counter was written with block skipping disabled"


def test_skip_counter_rejects_bad_buffer():
    """Wrong dtype or too small: fail at the call, not with a corrupted count
    or an out-of-bounds atomic."""
    _skipif_gfx12()
    data = _build()
    for bad, why in (
        (torch.zeros(2, dtype=torch.float32, device="cuda"), "dtype"),
        # int32 is specifically rejected: the count is global and a long prefill
        # overflows it, so accepting one would hand back a wrong ratio.
        (torch.zeros(2, dtype=torch.int32, device="cuda"), "narrow dtype"),
        (torch.zeros(1, dtype=torch.int64, device="cuda"), "too small"),
        # Expanded view: numel() == 2 but both elements share ONE element of
        # storage, so the kernel's ptr+1 atomic would write out of bounds.
        (torch.zeros(1, dtype=torch.int64, device="cuda").expand(2), "non-contiguous"),
    ):
        with pytest.raises(AssertionError, match="skip_counter"):
            _run(data, 0.3, out=torch.empty_like(data[6]), skip_counter=bad)
