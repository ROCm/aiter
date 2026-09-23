# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layer P for the MiniMax-M3 top-k-index DECODE BLOCK-SCORING pass
(bf16 index-Q x fp8 index-K).

SCOPE, stated first because the previous module name over-claimed it: this is
the *index block-score* half of ``minimax_m3_index_topk_decode`` -- the part
that produces ``score[h, b*Q + tok, blk]``. The top-k selector itself
(``_topk_index_packed_kernel`` / ``pa_sparse_block_topk``) is NOT here and is
not touched.

This is a NEW operator beside ``pa_sparse_block_score_decode``, not a variant of
it. The two differ in Q dtype, in scale handling and in sentinels, so they do
not share a signature -- matching the incumbent's signature is how a wrong
dispatch gets written. Nothing in ``msa_block_select.py`` is imported, called or
modified here: the incumbent fp8-Q path keeps its behaviour byte-for-byte.

Layer P owns what needs tensor metadata (integration.md section 8.2). The C
entry point receives raw addresses and CANNOT see a dtype, so every dtype
decision lives here and only here -- most importantly the
``== torch.float8_e4m3fn`` pin (N17).

Frozen contract (spec.md section 1)::

    score[h, b*Q + tok, blk]
        = max_p( fp32dot(bf16(K[page][p][:]), Q[b*Q+tok][h][:]) * sm_scale*log2e )

Q is bf16 and is never quantized; K is fp8 e4m3 and is lifted to bf16 exactly;
``sm_scale`` is applied IN-KERNEL (ledger D1). bf16 K is excluded by human
decision (ledger D0): the fallback for it IS the rejection below.
"""

import contextlib
import math

import torch

from aiter.jit.core import compile_ops


@compile_ops(
    "module_topk_index_score",
    fc_name="topk_index_score_decode",
    ffi_type="ctypes",
)
def _topk_index_score_raw(
    q_idx: int,
    key_cache_idx: int,
    score: int,
    block_table: int,
    seq_lens: int,
    q_numel: int,
    key_cache_numel: int,
    score_numel: int,
    block_table_numel: int,
    batch: int,
    num_chunks: int,
    chunk_blocks: int,
    stride_q_n: int,
    stride_q_h: int,
    stride_ik_blk: int,
    stride_s_h: int,
    stride_s_b: int,
    stride_bt_b: int,
    sm_scale: float,
    block_size: int,
    head_dim: int,
    num_idx_heads: int,
    query_len: int,
    aux_k: int,
    cert_bypass: int,
) -> None: ...


def _ptr(t) -> int:
    """Device address of a tensor; 0 is the null the kernels test for."""
    return 0 if t is None else t.data_ptr()


def _addressed_span(t) -> int:
    """Elements a view can ADDRESS from its base pointer: 1 + sum((n-1)*stride).

    This is what the buffer descriptors must be built from, and it is NOT
    numel(). For a non-contiguous view they differ: score = empty(1,32,96)[:,:,:32]
    has numel 1024 but addresses element (0,31,31) at offset 3007, and
    block_table = empty(8,48)[:, :32] has numel 256 but addresses offset 367.

    Passing numel() there makes the descriptor SHORT, and a short descriptor
    does not fault -- the hardware bounds check silently drops the accesses past
    it (that check is exactly what invariant I3 keeps alive by putting the page
    offset in voffset). Dropped stores leave the score buffer at its pre-fill;
    dropped block-table loads return ZERO, i.e. page 0, so the kernel scores the
    WRONG PAGES with plausible numbers and no crash. Found by writing the
    stride-generality cases L1/L2 of the S9 sweep, before they were ever run.

    A zero-element tensor addresses nothing; the entry's extent guard rejects a
    zero extent, which is the correct outcome for a tensor with no data.
    """
    if t is None or t.numel() == 0:
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))


# Build constants. Not defaults: the LDS staging layout and invariant I5's 1.00
# L1 accesses per 128 B line are both derived for 128-byte rows.
OPUS_HEAD_DIM = 128
OPUS_BLOCK_SIZE = 128

# Columns of the MFMA, each holding one (token, head) pair.
OPUS_MFMA_COLS = 16

# Cells the C++ table BUILDS, and the subset dispatch may SELECT. The C++
# predicate opus_idx_score_cell_certified is the authority; this mirrors it so
# the refusal can carry a Python-level explanation, and the two must be edited
# together. Widening either requires per-cell bitwise evidence at the accepted
# config by the reviewer who owns the certification evidence -- it is not an
# integration decision.
# WIDENED 2026-09-23 on the configuration-coverage measurement
# (derived from the model config and the caller's shard geometry; see the
# coverage note below,
# 7064-7066). Q = 2 is the DEFAULT MTP width and was refused on every TP; Q = 8
# is the model's full num_mtp_modules = 7. H = 2 (TP2) is still deliberately
# absent (N15, a human decision). (4, 8) is H*Q = 32 -- an MFMA column-budget
# wall, not a table gap, and the H*Q check above rejects it first.
OPUS_BUILT_CELLS = ((1, 1), (1, 2), (1, 4), (1, 8), (4, 1), (4, 2), (4, 4))
# ALL FOUR BUILT CELLS ARE CERTIFIED, on a per-cell bitwise certification sweep
# 32/32 per cell, 16 constructions x 2 aux legs, non-vacuous, bit-pattern
# equality against the frozen Triton reference, 102 -> 124 -> 128 with every
# original failure case-side and NOT ONE LINE of the kernel or of this layer
# changed to make a case pass.
#
# CERTIFIED UNDER BYPASS -- the record says so, with per-record receipts. The
# N16 certification bypass is now RETIRED: with the built set equal to the
# certified set it has nothing left to unlock, and a run that uses it would be
# a bug. The hook stays because it is needed exactly once per new cell.
#
# Edited TOGETHER with opus_idx_score_cell_certified in
# csrc/kernels/sparse_attn/topk_index/topk_index_score.hpp.
# The six NEW instantiations are BUILT BUT NOT CERTIFIED: N16 refuses them until
# the certification sweep produces their per-cell bitwise evidence. That also
# makes the N16 arm reachable again, which it had stopped being when the built
# set and the certified set coincided.
# Widened to SIX on the certification sweep re-run against the refactored
# build): the original four RE-ANCHORED -- the refactor changed the code objects
# so the 7014/7017 bindings were stale -- plus (1,2) and (4,2) at 32/32 each.
# (1,2) is the production speculative-decode cell: Q = 2 is the default MTP
# width. (1,8) stays OUT until its re-run under the r5 expect derivation.
# ALL SEVEN BUILT CELLS CERTIFIED (certification sweep:
# 224/224, every cell 16/16 on both legs, non-vacuous under the r5 derivation).
# The built set and the certified set coincide again, so the N16 bypass RETIRES
# entirely -- no built cell needs it. The mechanism stays in the code: it is
# needed exactly once per future cell, and a gate deleted when it stops firing
# is a gate that is absent when it should fire.
OPUS_CERTIFIED_CELLS = ((1, 1), (1, 2), (1, 4), (1, 8), (4, 1), (4, 2), (4, 4))

# The one tuned template axis. Both legs of the deployed cell carry phase-1
# correctness evidence; 3 is the accepted performance point on the traced shape.
OPUS_AUX_K_BUILT = (0, 3)

# Blocks per workgroup on the block axis. Swept 1..128 over the whole 80-point
# surface on three exclusive nodes, then validated on a held-out set sharing no
# shape with it; see _grid_lever for the derivation and the invalidation
# conditions. This is a MEASURED constant, not a default.
OPUS_CHUNK_BLOCKS = 4

# The banded arm. A working set in [OPUS_LLC_BAND_LO_MIB, OPUS_LLC_BAND_HI_MIB]
# takes a much larger chunk and the control aux leg; everything else takes
# OPUS_CHUNK_BLOCKS / OPUS_AUX_K_DEFAULT. See _grid_lever for the derivation,
# the three held-out validations, and the fact that this arm was twice REJECTED
# by its own pre-registered rules and is here on an explicit human override.
# OFF. The band arm is implemented, validated and DISABLED -- the code stays so
# that turning it on is a one-line, reviewable change with its evidence beside
# it, and so that the next person does not have to re-derive it.
#
# WHY OFF, with the human override already granted: held-out set 3 (jobs
# 7021/7022/7023, three exclusive nodes, 44 shapes, DENSE AT BOTH EDGES) was
# pre-registered to judge the EDGES, and the band failed clause (c) of its own
# rule -- no in-band shape worse than T1 by more than the 2.0% measured
# position bound, on any node. Two shapes violated it, and both are AT AN EDGE:
#   133 MiB  worse by 2.5%      253 MiB  worse by 5.8%
# That is the designed test firing at the designed place, not a noise
# rejection: the band's INTERIOR is clearly right (12 of 14 shapes, up to 7.8%
# better than T1) and its EDGES are not.
#
# The aggregate case remains strong -- geomean 0.9401 vs T1's 0.9568, max 1.086
# vs 1.110, 8 shapes slower than Triton vs 15. A NARROWER band would be a
# fourth fit, and the pre-registration forbids it in terms: three fits is not a
# boundary, it is a search. If this is to ship, it needs a mechanism (PMC), not
# another boundary.
OPUS_LLC_BAND_ENABLED = False

OPUS_LLC_BAND_LO_MIB = 128
OPUS_LLC_BAND_HI_MIB = 256
OPUS_LLC_BAND_CHUNK_BLOCKS = 24
OPUS_LLC_BAND_AUX_K = 0
OPUS_INDEX_K_PAGE_BYTES = OPUS_BLOCK_SIZE * OPUS_HEAD_DIM  # fp8: 1 byte/element

# AUX_K for a bucket with no tuning-table entry. See _aux_k_for.
OPUS_AUX_K_DEFAULT = 3

# Waves per SIMD the ACCEPTED build achieves, read from its code object
# (VGPR 61 -> 8 slots; LDS 16,384 B -> 9; the wave64 slot cap gives 8).
#
# NO LONGER FEEDS THE GRID. Until the three-node shape survey the launch
# geometry was derived from this number via resident capacity; it is not any
# more (see _grid_lever, which ignores capacity by measurement). The figure is
# kept because _resident_workgroups still reports it to harnesses and because
# it is the occupancy the accepted build actually achieves.
#
# INVALIDATION CONDITION, live rather than theoretical (integration.md section
# 8.4): config 3 sits THREE registers from the >64 threshold. A future build
# that moves VGPR above 64 or LDS above 20,480 B per workgroup drops this
# figure -- and because the grid no longer tracks it automatically, the chunk
# size in _grid_lever must then be RE-SWEPT rather than inferred.
OPUS_WAVES_PER_CU = 8

# --------------------------------------------------------------------------- #
# N16 certification-mode bypass (lead ruling, team-message-7dff0e55)
# --------------------------------------------------------------------------- #
# The scope ruling "build eight, dispatch only cells with bitwise evidence" had
# no way to PRODUCE that evidence: the certification sweep's uncertified cells
# are exactly the cells N16 refuses, so 96 of its 128 runs refused. This hook is
# the way out, and it is narrow on purpose:
#
#   * N16 ONLY. N16 is a dispatch-POLICY gate -- it says "we have not proven
#     this cell yet", which is a statement about EVIDENCE. The SAFETY gates
#     (arch, the 128/128 build shapes, the dtype pins including the e4m3fn pin,
#     and the built-cell check) stay live and are NOT bypassable. Bypassing
#     those would be a hole, not a harness.
#   * Off by default and awkward to enable on purpose: a scoped context manager
#     demanding an exact acknowledgement string. Not a default argument, not an
#     environment variable -- nothing a deployment can inherit and nothing a
#     stray truthy value can trip.
#   * Mirrored in the C++ entry (layer C4) with a magic token, because a bypass
#     in one layer and a refusal in the next is the "passes one layer, unchecked
#     by the next" hole integration.md section 8.2 forbids.
#   * Every bypassed call is RECORDED so a certification record can state that
#     its evidence was produced under bypass. "Certified" must never silently
#     mean "certified through a hook". Once a cell's evidence lands and the
#     predicate widens, that cell no longer needs the hook -- it is needed
#     exactly once per cell, and the record shows it.
#
# Mirrored by OPUS_IDX_SCORE_CERT_TOKEN in topk_index_score_entry.cu;
# the two must be edited together.
OPUS_CERT_BYPASS_TOKEN = 0x5339C0DE

_CERT_ACK = (
    "I am producing per-cell bitwise certification evidence and I accept that "
    "this bypasses the N16 dispatch-policy gate only"
)

# None = the hook is off. A list = the hook is on, and receipts go there.
_cert_receipts = None


@contextlib.contextmanager
def certification_bypass_n16(acknowledgement):
    """Scoped, acknowledged bypass of N16 -- for certification harnesses ONLY.

    Yields the receipt list: one entry per call that actually used the bypass,
    so the harness can tag exactly those records. Leaving the block always turns
    the hook off, including on an exception, so it cannot be left on.

    This does NOT make an uncertified cell certified. It makes the evidence
    obtainable; the evidence is what certifies, and widening
    OPUS_CERTIFIED_CELLS on that evidence belongs to whoever reviews the
    certification, not to this hook and not to the author of the change.
    """
    global _cert_receipts
    if acknowledgement != _CERT_ACK:
        raise ValueError(
            "certification_bypass_n16: refused. The acknowledgement must be the "
            "exact string at aiter/ops/topk_index_score.py:_CERT_ACK. It is "
            "verbose so that enabling this is deliberate and greppable."
        )
    if _cert_receipts is not None:
        raise RuntimeError(
            "certification_bypass_n16: already active; nesting is not permitted "
            "(a nested block would restore the wrong state on exit)"
        )
    _cert_receipts = []
    try:
        yield _cert_receipts
    finally:
        _cert_receipts = None

# Tuning table: bucket -> AUX_K. Still DELIBERATELY EMPTY.
#
# It is empty for a different reason than it used to be. N14 required evidence
# before any bucket could take AUX_K = 3, and the fallback was the control leg
# 0. That evidence now exists -- but it is SURFACE-WIDE rather than per-bucket
# (three exclusive nodes, 80 shapes, both legs at the accepted chunking), so it
# belongs in the DEFAULT, not in a list of buckets invented to carry it.
# Inventing bucket boundaries to house a surface-wide result would be exactly
# the tuning-by-assertion this project refuses; so would leaving the default at
# a leg the surface says is worse.
#
# An entry here overrides the default for one exact
# (num_idx_heads, query_len, batch, max_blk). Adding one requires PER-BUCKET
# evidence that disagrees with the surface, not a hunch.
_OPUS_AUX_TUNING_TABLE: dict = {}

_RESIDENT_WG_CACHE: dict = {}


def _resident_workgroups(device) -> int:
    """Workgroups resident across the device: CU count x waves per CU.

    The CU count is a device property, read once per device -- a host query at
    capture time, never a device-memory read, so cudagraph invariant G2 holds.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _RESIDENT_WG_CACHE.get(index)
    if cached is None:
        cached = (
            torch.cuda.get_device_properties(index).multi_processor_count
            * OPUS_WAVES_PER_CU
        )
        _RESIDENT_WG_CACHE[index] = cached
    return cached


def _grid_lever(max_blk: int, batch: int, capacity: int) -> tuple[int, int]:
    """Chunk count and chunk size for the block axis.

    **OPUS_CHUNK_BLOCKS blocks per workgroup, unconditionally.** The block
    axis is split into ceil(max_blk / 4) chunks and every chunk gets one
    workgroup per request. ``batch`` and ``capacity`` are accepted and
    deliberately ignored; see below.

    HOW THIS WAS CHOSEN.
    Not asserted: chunk_blocks was FORCED across 1..128 over the whole 80-point
    shape surface on three exclusive nodes, and the two
    candidate policies read off that surface were then judged on a held-out set
    sharing no context and no batch with it (6992/6993/6994), against a rule
    committed before those jobs were submitted. 4 was the best chunk size on 33
    of 80 fitting shapes and within noise of the best on most of the rest.

    WHAT IT REPLACED, and why. The previous lever sized chunks against RESIDENT
    CAPACITY: target = clamp(capacity // batch, 1, max_blk). Measured, that was
    worse on the surface in two distinct ways:

      * Its "batch at or past capacity -> chunk_blocks = 1" arm cost 32-33
        points: (4,4) ctx 32768 x b8 ran at 1.174 x Triton and goes to 0.839 at
        chunk_blocks 4; ctx 2048 x b128 goes 1.170 -> 0.850. That edge case was
        documented as "decided, not accidental". It was decided wrongly.
      * Across the held-out set it left a geometric mean of 1.0262 x Triton
        with 24 of 36 shapes SLOWER than Triton. This policy leaves 0.9584
        with 12 slower. (Both at aux_k = 3; see _aux_k_for.)

    CUDAGRAPH (G1/G2): strictly safer than what it replaced. The chunking now
    depends only on max_blk, i.e. on the caller's context bound -- a
    capture-time constant. It no longer reads the device CU count and no longer
    varies with batch, so there is one less way for the launch dimensions to
    move between capture and replay.

    KNOWN LIMITATION, not smoothed over. This policy's worst measured shape is
    1.362 x Triton, worse than the old lever's worst of 1.199. The loss
    concentrates where the index-K working set is roughly 128-256 MiB, i.e.
    comfortably inside the 256 MB last-level cache. A banded policy that
    special-cases exactly that region measured better on every aggregate --
    geometric mean, median, maximum, and 13 of 14 in-band shapes -- and was
    REJECTED because it failed the "every in-band shape, every node" clause of
    its own pre-registered decision rule (
    PREREG-shape-survey-3node.md, addendum 2). It can be revisited by a human
    decision; it is not in this policy.

    INVALIDATION CONDITIONS (live, not theoretical):
      * A different SKU. Every number above is MI355X / gfx950, 256 CU.
      * A change to the LDS staging layout or the page hoist. 4 is the MODE of
        a broad, shape-dependent distribution of best chunk sizes (best on 33
        of 80 fitting shapes; 1, 2, 3, 8, 16, 24, 32 and 64 are each best
        somewhere) -- the best single CONSTANT, not an optimum, and no
        mechanism is claimed for it. An earlier note called it "a quarter of
        the 16-block page hoist"; that is a red herring, because resizing the
        hoist measured as a 0.2% null. Re-sweep on any layout change; do not
        scale.
      * A build whose occupancy differs from VGPR 61 / LDS 16,384 B / 8 waves
        per SIMD. The old lever read resident capacity and would have tracked
        that automatically; this one will not, which is the price of a policy
        that does not depend on it.
    """
    if _in_llc_band(max_blk, batch):
        chunk_blocks = min(OPUS_LLC_BAND_CHUNK_BLOCKS, max(1, max_blk))
    else:
        chunk_blocks = min(OPUS_CHUNK_BLOCKS, max(1, max_blk))
    num_chunks = math.ceil(max_blk / chunk_blocks)
    return num_chunks, chunk_blocks


def _in_llc_band(max_blk: int, batch: int) -> bool:
    """Is the index-K working set inside the last-level-cache band?

    batch * max_blk pages of OPUS_INDEX_K_PAGE_BYTES each. Every input is a
    capture-time constant, so this does not move under cudagraph capture
    (G1/G2) -- the same property the rest of the lever has.

    THE MECHANISM IS NOT ESTABLISHED. 128-256 MiB sits inside the 256 MB
    last-level cache, and that is the only reason to think the band is a cache
    effect; no PMC evidence has been taken. What IS established is the effect:
    in this region the unbanded policy lost up to 36% to Triton and the banded
    arm recovers it to about parity. The edges are EMPIRICAL, and the
    invalidation conditions in _grid_lever bind a re-sweep on any SKU change --
    a different LLC size moves this band and nothing here would notice.
    """
    if not OPUS_LLC_BAND_ENABLED:
        return False
    mib = batch * max_blk * OPUS_INDEX_K_PAGE_BYTES / (1024.0 * 1024.0)
    return OPUS_LLC_BAND_LO_MIB <= mib <= OPUS_LLC_BAND_HI_MIB


def _aux_k_for(num_idx_heads: int, query_len: int, batch: int, max_blk: int) -> int:
    """AUX_K for a bucket: the tuning table if it has an entry, else
    OPUS_AUX_K_DEFAULT.

    The default is 3, and that is a change from the control leg 0 which N14
    required evidence for. The evidence is in
    Across the 80-point
    surface at the accepted chunking, aux_k = 3 is better on 55 of 80 shapes,
    and the 25 where aux_k = 0 wins are almost all ties (the 0.90-vs-0.92
    class). On the held-out set the geometric mean is 0.9584 x Triton at
    aux_k = 3 against 1.0262 for what shipped before.

    The table is still consulted FIRST, so a bucket can be pinned without
    touching this default -- and it is still empty, because no bucket has
    per-bucket evidence that disagrees with the surface-wide answer.
    """
    pinned = _OPUS_AUX_TUNING_TABLE.get(
        (num_idx_heads, query_len, batch, max_blk))
    if pinned is not None:
        return pinned
    # The banded arm changes BOTH knobs together. They were fitted together and
    # neither is valid alone: at chunk_blocks 24 the aux3 cache policy is a
    # LOSS, which is exactly why the band takes the control leg.
    if _in_llc_band(max_blk, batch):
        return OPUS_LLC_BAND_AUX_K
    return OPUS_AUX_K_DEFAULT


def _fp8_e4m3fnuz():
    # Absent on some torch builds; None simply means the fnuz arm is
    # unreachable, and the pin below still rejects it as "not e4m3fn".
    return getattr(torch, "float8_e4m3fnuz", None)


def _check_devices(*tensors) -> None:
    """D02 (review): every tensor is forwarded as a raw data_ptr, so nothing
    downstream can tell a host pointer from a device one, and aiter.jit.core
    takes the stream from the CURRENT device rather than from the tensors. A CPU
    tensor therefore reached HIP as a host address, and a valid cuda:1 batch
    launched on cuda:0's stream.

    aiter/ops/msa_block_select.py has the same shape; this fixes it here rather
    than claiming the convention is safe.
    """
    named = [(n, t) for n, t in tensors if t is not None]
    for n, t in named:
        if not t.is_cuda:
            raise ValueError(
                "topk_index_score_decode: %s is on %s; every tensor must be a "
                "CUDA tensor because they are forwarded as raw pointers" % (n, t.device)
            )
    devs = {t.device for _, t in named}
    if len(devs) != 1:
        raise ValueError(
            "topk_index_score_decode: tensors span more than one device (%s); "
            "the launch takes one stream and cannot straddle them"
            % sorted(str(d) for d in devs)
        )


def _check_dtypes(q_idx, key_cache_idx, score) -> None:
    """Dtype routing. Every arm either accepts or raises -- the space is covered.

    N1  fp8 Q -> the incumbent's operator, never this one: this kernel would
        read the fp8 bytes as bf16 and return wrong numbers rather than fail.
    N2/N3/D0  bf16 K -> the deliberate exclusion. aiter has no bf16-K
        implementation of this operator, so preserving the rejection IS the
        fallback.
    N17 e4m3fnuz -> refused by a pin to == float8_e4m3fn. The incumbent's
        checker accepts fnuz; routing fnuz onto a kernel built for e4m3fn is a
        different exponent bias and different NaN handling -- a WRONG-NUMBERS
        path, not a crash. The pin is deliberate and must not be relaxed to a
        generic is_fp8 predicate.
    """
    fnuz = _fp8_e4m3fnuz()
    if q_idx.dtype != torch.bfloat16:
        if q_idx.dtype == torch.float8_e4m3fn or (
            fnuz is not None and q_idx.dtype == fnuz
        ):
            raise ValueError(
                "topk_index_score_decode: q_idx is fp8; this operator takes an "
                "UNQUANTIZED bf16 q_idx. The fp8-Q operator is "
                "pa_sparse_block_score_decode -- use it instead of this path."
            )
        raise ValueError(
            f"topk_index_score_decode: q_idx must be bf16, got {q_idx.dtype}"
        )

    if key_cache_idx.dtype == torch.bfloat16:
        raise ValueError(
            "topk_index_score_decode: bf16 key_cache_idx is not supported "
            "(deliberate exclusion, integration.md ledger D0). No bf16-K "
            "implementation of this operator exists in aiter; this rejection is "
            "the documented fallback, not a missing feature."
        )
    if key_cache_idx.dtype != torch.float8_e4m3fn:
        if fnuz is not None and key_cache_idx.dtype == fnuz:
            raise ValueError(
                "topk_index_score_decode: key_cache_idx is float8_e4m3fnuz; this "
                "path is built for float8_e4m3fn only. The two differ in exponent "
                "bias and NaN handling, so dispatching fnuz here would return "
                "wrong numbers rather than fail."
            )
        raise ValueError(
            "topk_index_score_decode: key_cache_idx must be float8_e4m3fn, got "
            f"{key_cache_idx.dtype}"
        )

    if score.dtype != torch.float32:
        raise ValueError(
            f"topk_index_score_decode: score must be fp32, got {score.dtype}"
        )

def topk_index_score_decode_supported(
    q_idx,
    key_cache_idx,
    score,
    query_len: int = 1,
    max_seq_len: int = 0,
):
    """Can this operator serve these inputs? Returns ``(bool, reason)``.

    WHY THIS EXISTS. ATOM's index-K cache for MiniMax-M3 follows
    ``--kv-cache-dtype``, whose default is **bf16** (atom/config.py:1755 and
    :2341 -- MiniMax-M3 is not DeepseekV4, so it takes the kv-cache-dtype
    branch). The Triton kernel serves both dtypes; this operator serves fp8 K
    only, by human decision (ledger D0). So on a DEFAULT-configured server the
    unsupported case is the common one, not the rare one.

    An integration must therefore be able to ASK, and route: fp8 K here, bf16 K
    to the Triton path. Without this predicate the only way to find out is to
    call and catch, which turns a supported-configuration question into
    exception-driven control flow and makes a naive swap crash a default
    server instead of falling back.

    This is a PURE predicate: it allocates nothing, launches nothing and
    mutates nothing. It deliberately mirrors the entry's own checks rather than
    relaxing them -- if it says True, the call must not raise for a reason this
    predicate could have seen. The entry keeps every check regardless: this is
    a routing aid, never a substitute for validation (the "passes one layer,
    unchecked by the next" hole this project forbids).
    """
    fnuz = _fp8_e4m3fnuz()
    if getattr(q_idx, "dtype", None) != torch.bfloat16:
        return False, "q_idx must be bf16 (this operator never quantises Q)"
    if key_cache_idx.dtype == torch.bfloat16:
        return False, ("bf16 key_cache_idx is not supported (ledger D0); use "
                       "the Triton index-score path, which serves both dtypes")
    if key_cache_idx.dtype != torch.float8_e4m3fn:
        if fnuz is not None and key_cache_idx.dtype == fnuz:
            return False, "key_cache_idx is float8_e4m3fnuz; built for e4m3fn only"
        return False, "key_cache_idx must be float8_e4m3fn, got %s" % (
            key_cache_idx.dtype,)
    if score.dtype != torch.float32:
        return False, "score must be fp32"
    if q_idx.dim() != 3 or key_cache_idx.dim() != 3 or score.dim() != 3:
        return False, "q_idx, key_cache_idx and score are 3-D"
    if not (q_idx.is_contiguous() and key_cache_idx.is_contiguous()):
        return False, "q_idx and key_cache_idx must be contiguous"
    if score.stride(2) != 1:
        return False, "score must be contiguous along the block axis"
    # D03/D01 mirrored: the core refuses these, so the predicate must too or it
    # routes a caller into a ValueError it was added to prevent.
    if score.stride(0) == 0 or score.stride(1) == 0:
        return False, "score has a zero stride on the head or row axis; outputs alias"
    if _addressed_span(score) * score.element_size() >= 2**32:
        return False, "score exceeds the 32-bit buffer-descriptor extent"
    # D02 mirrored: every tensor is forwarded as a raw pointer.
    for _n, _t in (("q_idx", q_idx), ("key_cache_idx", key_cache_idx),
                   ("score", score)):
        if not _t.is_cuda:
            return False, "%s is on %s; every tensor must be a CUDA tensor" % (
                _n, _t.device)
    if len({q_idx.device, key_cache_idx.device, score.device}) != 1:
        return False, "tensors span more than one device"
    total_q, num_idx_heads, head_dim = q_idx.shape
    block_size = key_cache_idx.size(1)
    if head_dim != OPUS_HEAD_DIM or block_size != OPUS_BLOCK_SIZE:
        return False, "built for head_dim %d and block_size %d, got %d and %d" % (
            OPUS_HEAD_DIM, OPUS_BLOCK_SIZE, head_dim, block_size)
    if key_cache_idx.size(2) != head_dim:
        return False, "key_cache_idx head dim must match q_idx"
    if query_len < 1 or total_q % query_len != 0:
        return False, "q_idx rows must be a positive multiple of query_len"
    if num_idx_heads * query_len > OPUS_MFMA_COLS:
        return False, "num_idx_heads * query_len exceeds the %d MFMA columns" % (
            OPUS_MFMA_COLS,)
    cell = (num_idx_heads, query_len)
    if cell not in OPUS_BUILT_CELLS:
        return False, "no build for (num_idx_heads, query_len) = %s" % (cell,)
    if cell not in OPUS_CERTIFIED_CELLS:
        return False, "cell %s is built but not certified (N16)" % (cell,)
    if max_seq_len < 1:
        return False, "max_seq_len is required so the grid is fixed at capture"
    # D07 / rule A3 (review, upheld at YELLOW): the core rejects a mismatched
    # score outer shape at its own shape check, and this predicate did not, so
    # it answered "supported" for a case the core then raised on -- the exact
    # failure it exists to prevent.
    if score.size(0) != num_idx_heads or score.size(1) != total_q:
        return False, "score must be [num_idx_heads, total_q, S], got %s for " \
                      "num_idx_heads=%d total_q=%d" % (
                          tuple(score.shape), num_idx_heads, total_q)
    if key_cache_idx.size(0) < 1:
        return False, "key_cache_idx holds no pages"
    max_blk = math.ceil(max_seq_len / block_size)
    if score.size(2) < max_blk:
        return False, "score width is below cdiv(max_seq_len, %d)" % block_size
    return True, "supported"

def topk_index_score_decode(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    seq_lens,
    sm_scale: float,
    query_len: int = 1,
    max_seq_len: int = 0,
    aux_k=None,
):
    """Score every block of the fp8 index key cache against a bf16 index query.

    Args:
        q_idx: ``[num_reqs * query_len, num_idx_heads, head_dim]`` **bf16**,
            contiguous. Never quantized -- an fp8 MFMA would round it.
        key_cache_idx: ``[num_pages, 128, 128]`` **float8_e4m3fn**, contiguous.
        score: ``[num_idx_heads, num_reqs * query_len, S]`` fp32, written in
            place. Blocks past ``cdiv(seq_len, 128)`` are left untouched, exactly
            as the incumbent and the frozen reference leave them, so pre-fill
            with ``-inf`` unless the consumer bounds each row itself
            (``pa_sparse_block_topk`` does).
        block_table: ``[num_reqs, >= max_block]`` int32.
        seq_lens: ``[num_reqs]`` int32.
        sm_scale: applied in-kernel as ``sm_scale * log2e`` AFTER the dot.
        query_len: query tokens per request. ``num_idx_heads * query_len`` must
            fit the MFMA's 16 columns.
        max_seq_len: upper bound on the context, required. The grid comes from
            this and never from ``seq_lens``: reading live lengths would move
            the grid between steps and break cudagraph capture.
        aux_k: index-K load cache policy. ``None`` takes the tuning table,
            whose default is ``OPUS_AUX_K_DEFAULT = 3`` -- see ``_aux_k_for``
            for the surface-wide evidence behind that default and
            the sweep that produced it for its
            validity range. Pass 0 or 3 explicitly to pin a leg.

    Returns:
        ``score``, written in place.
    """
    _check_dtypes(q_idx, key_cache_idx, score)
    _check_devices(
        ("q_idx", q_idx), ("key_cache_idx", key_cache_idx), ("score", score),
        ("block_table", block_table), ("seq_lens", seq_lens),
    )

    if q_idx.dim() != 3 or key_cache_idx.dim() != 3 or score.dim() != 3:
        raise ValueError(
            "topk_index_score_decode: q_idx, key_cache_idx and score are 3-D"
        )
    if not (q_idx.is_contiguous() and key_cache_idx.is_contiguous()):
        raise ValueError(
            "topk_index_score_decode: q_idx and key_cache_idx must be contiguous"
        )
    if score.stride(2) != 1:
        raise ValueError(
            "topk_index_score_decode: score must be contiguous along the block axis"
        )
    # D03 / rule D8 (review, upheld): stride(2) == 1 was the ONLY output-layout
    # gate, so an expanded view -- torch.empty(1,1,1).expand(4,1,1) -- passed with
    # stride(0) == 0 and every head stored to one address concurrently. A zero
    # stride on either non-contiguous axis means two logical outputs share one
    # physical element, which no amount of care inside the kernel can make safe.
    if score.stride(0) == 0 or score.stride(1) == 0:
        raise ValueError(
            "topk_index_score_decode: score has a zero stride on the head or row "
            "axis (strides %s), so different outputs alias one element; pass a "
            "materialised tensor rather than an expanded view" % (score.stride(),)
        )
    # D01 (review): the addressed span is forwarded as the buffer-descriptor
    # extent and the kernel narrows span*4 to 32 bits, so a span at or past
    # 2**30 float32 elements wraps the extent to zero and silently drops every
    # store. Refused here, where the number is still a Python int and the
    # arithmetic cannot wrap.
    _span_bytes = _addressed_span(score) * score.element_size()
    if _span_bytes >= 2**32:
        raise ValueError(
            "topk_index_score_decode: score spans %d bytes, which does not fit "
            "the 32-bit buffer-descriptor extent the kernel uses; the limit is "
            "%d bytes" % (_span_bytes, 2**32 - 1)
        )
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError(
            "topk_index_score_decode: block_table and seq_lens must be int32"
        )
    if seq_lens.dim() != 1 or seq_lens.stride(0) != 1:
        # The kernel builds the seq_lens descriptor as batch*sizeof(int) and
        # indexes it by request, so a strided or multi-dim seq_lens would be
        # read at the wrong offsets. Same class as the span defect above:
        # refused rather than mis-addressed.
        raise ValueError(
            "topk_index_score_decode: seq_lens must be 1-D and contiguous"
        )
    if block_table.dim() != 2 or block_table.stride(1) != 1:
        raise ValueError(
            "topk_index_score_decode: block_table must be 2-D with contiguous rows"
        )

    total_q, num_idx_heads, head_dim = q_idx.shape
    block_size = key_cache_idx.size(1)
    if key_cache_idx.size(2) != head_dim:
        raise ValueError(
            "topk_index_score_decode: key_cache_idx head dim must match q_idx"
        )
    if head_dim != OPUS_HEAD_DIM or block_size != OPUS_BLOCK_SIZE:
        raise ValueError(
            f"topk_index_score_decode: built for head_dim {OPUS_HEAD_DIM} and "
            f"block_size {OPUS_BLOCK_SIZE}, got {head_dim} and {block_size}"
        )
    if score.size(0) != num_idx_heads or score.size(1) != total_q:
        raise ValueError(
            "topk_index_score_decode: score must be [num_idx_heads, total_q, S]"
        )

    if query_len < 1:
        raise ValueError("topk_index_score_decode: query_len must be >= 1")
    if total_q % query_len != 0:
        raise ValueError(
            f"topk_index_score_decode: q_idx rows {total_q} not a multiple of "
            f"query_len {query_len}"
        )
    num_reqs = total_q // query_len

    if num_idx_heads * query_len > OPUS_MFMA_COLS:
        raise ValueError(
            f"topk_index_score_decode: num_idx_heads * query_len = "
            f"{num_idx_heads * query_len} exceeds the {OPUS_MFMA_COLS} MFMA columns"
        )

    cell = (num_idx_heads, query_len)
    if cell not in OPUS_BUILT_CELLS:
        # N15 lives here: H = 2 is in the INCUMBENT's table and not in this
        # path's, and no bf16-Q implementation exists for it. An honest refusal,
        # not a silent route onto a neighbouring cell.
        raise ValueError(
            f"topk_index_score_decode: no build for (num_idx_heads, query_len) = "
            f"{cell}; built cells are {OPUS_BUILT_CELLS}"
        )
    cert_token = 0
    if cell not in OPUS_CERTIFIED_CELLS:
        # N16. The cell IS built -- that is what keeps the gap visible -- but it
        # has no per-cell bitwise evidence at the accepted config, so it may not
        # be dispatched on the 'same code path, compile-time constants only'
        # argument alone. The C++ entry refuses it again at layer C4.
        if _cert_receipts is None:
            raise ValueError(
                f"topk_index_score_decode: variant built but not certified -- "
                f"(num_idx_heads, query_len) = {cell} has no per-cell bitwise "
                f"evidence at the accepted config. Certified cells: "
                f"{OPUS_CERTIFIED_CELLS}."
            )
        # Certification mode: produce the evidence. The receipt is written
        # below, once aux_k has been resolved -- at this point it may still be
        # None (the tuning-table default is applied later), and a receipt naming
        # the wrong leg would be worse than none.
        cert_token = OPUS_CERT_BYPASS_TOKEN

    if seq_lens.size(0) != num_reqs or block_table.size(0) != num_reqs:
        raise ValueError(
            "topk_index_score_decode: seq_lens and block_table need one entry "
            "per request"
        )

    # N13: an empty batch is a no-op, and no zero-extent grid reaches capture.
    if num_reqs == 0:
        return score

    if max_seq_len < 1:
        raise ValueError(
            "topk_index_score_decode: pass max_seq_len so the launch dimensions "
            "are fixed at capture"
        )
    max_blk = math.ceil(max_seq_len / block_size)
    if score.size(2) < max_blk:
        raise ValueError(
            f"topk_index_score_decode: score width {score.size(2)} is below "
            f"cdiv(max_seq_len, {block_size}) = {max_blk}"
        )
    if block_table.size(1) < max_blk:
        raise ValueError(
            f"topk_index_score_decode: block_table width {block_table.size(1)} is "
            f"below cdiv(max_seq_len, {block_size}) = {max_blk}"
        )

    if not math.isfinite(sm_scale):
        # Mirrors the C-side guard. A non-finite scale multiplies every dot, so
        # it NaNs the whole score buffer and the top-k consumer then ranks that
        # -- a silently-wrong-answer path, not a crash path.
        raise ValueError(
            f"topk_index_score_decode: sm_scale must be finite, got {sm_scale}"
        )

    if aux_k is None:
        aux_k = _aux_k_for(num_idx_heads, query_len, num_reqs, max_blk)
    if aux_k not in OPUS_AUX_K_BUILT:
        raise ValueError(
            f"topk_index_score_decode: aux_k {aux_k} is not built; the tuned axis "
            f"carries {OPUS_AUX_K_BUILT}"
        )

    num_chunks, chunk_blocks = _grid_lever(
        max_blk, num_reqs, _resident_workgroups(q_idx.device)
    )

    if cert_token:
        # Receipt, with the leg resolved. One per call that actually used the
        # hook, so the certification record can name exactly which evidence was
        # produced under bypass.
        _cert_receipts.append(
            {"cell": list(cell), "aux_k": aux_k, "bypassed_gate": "N16",
             "batch": num_reqs, "max_blk": max_blk}
        )

    _topk_index_score_raw(
        _ptr(q_idx),
        _ptr(key_cache_idx),
        _ptr(score),
        _ptr(block_table),
        _ptr(seq_lens),
        # Descriptor extents: the ADDRESSED SPAN of each view, never numel().
        # See _addressed_span -- a short descriptor drops accesses silently.
        _addressed_span(q_idx),
        _addressed_span(key_cache_idx),
        _addressed_span(score),
        _addressed_span(block_table),
        num_reqs,
        num_chunks,
        chunk_blocks,
        q_idx.stride(0),
        q_idx.stride(1),
        key_cache_idx.stride(0),
        score.stride(0),
        score.stride(1),
        block_table.stride(0),
        float(sm_scale),
        block_size,
        head_dim,
        num_idx_heads,
        query_len,
        aux_k,
        cert_token,
    )
    return score
