# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Framework tuner for the opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a).

Wired into the canonical :class:`GemmCommonTuner`, so it runs like the other
aiter GEMM tuners: multi-GPU via ``mp_tuner``, standard ``-i/--untune_file`` /
``-o/--tune_file`` CLI, batching, and the shared post-process / CSV writer.

The candidate pool lives here (``_TUNE_POLICY``). Per kid it holds only the
split-K factors to sweep; tile geometry, kernelName and the M alignment come from
the codegen instance table, so a kid cannot be tuned on a shape its launcher
rejects. That alignment used to be a second hand-maintained column and was wrong
in both directions -- it hid kid326, which is really arbitrary-M, from every
unaligned shape while the runtime dispatched it there anyway.

Runtime schema (what the tuner emits, and what the runtime reads back):
    gfx,b,m,n,k,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
``aiter/ops/batched_gemm_op_a8w8.py:lookup_mxscale_bmm_config`` indexes on
``["gfx","b","m","n","k"]``, dispatches to a backend on the winning row's
``libtype``, and ``bmm_op.py`` reads ``kernelId`` / ``splitK`` off that row, so
those columns must match exactly.

Verification (the part that catches column-transpose / scale defects):
  * inputs are *signed* and have *per-128-K-block varied magnitude*
    (``randn * 2**randint(-4,4)`` per block) so the e8m0 128-block scales span
    many exponents. Uniform non-negative ``rand()/10`` data hides a pure output
    column permutation (kid312/313 measured ~0.007 there but ~0.7-1.0 on real
    signed data) -- see the opus_bmm.md root-cause note.
  * reference is a dequantized fp32 einsum.
  * gate: ``mp_tuner`` runs ``checkAllclose(rtol=1e-2, atol=1e-2)`` and
    ``post_process`` keeps the fastest candidate whose mismatch fraction is
    ``<= --errRatio`` (default 0.02). A still-broken tileN COM_REP_N>1 kernel
    measures ~0.5 here and is rejected; the fp8 e8m0 quant floor is ~1e-4.

Usage (gfx950 only; the repo root must be on PYTHONPATH so the edited/rebuilt
tree wins over any installed aiter):
    cd <repo> && PYTHONPATH=$PWD \\
        python3 csrc/opus_gemm/opus_bmm_mxscale_tune.py -g 16 -m 1,16,64 -n 1024 -k 4096

    # re-tune every shape already in the shipped CSV, write a diffable copy:
    ... opus_bmm_mxscale_tune.py

    # overwrite the shipped tuned CSV in place:
    ... opus_bmm_mxscale_tune.py --apply

    # from an untuned CSV (columns: b,m,n,k -- or g,m,n,k), 8-way parallel:
    ... opus_bmm_mxscale_tune.py -i my_untuned.csv -o /tmp/out.csv --mp 8

    # the shipped shapes retuned into the second, preshuffle-inclusive table
    # (BPRESHUFFLE_CSV below; leaves the shipped one alone):
    ... opus_bmm_mxscale_tune.py --bpreshuffle --all --mp 8

    # what preshuffling B is worth: the same shapes twice, pool split by B layout,
    # then compare the two tables cell by cell:
    ... opus_bmm_mxscale_tune.py -i shapes.csv -o /tmp/rowb.csv --pool rowb --mp 8
    ... opus_bmm_mxscale_tune.py -i shapes.csv -o /tmp/preb.csv --pool preb --mp 8
"""

import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import dtypes, logger
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_raw
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner, TunerCommon
from aiter.utility.mp_tuner import mp_tuner

# Neither op_tests nor this directory is a package, so put both on sys.path. This
# also has to hold in the spawned mp_tuner subprocesses, which re-import this
# module top-to-bottom.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OPTESTS = os.path.join(_REPO, "op_tests")
for _p in (_HERE, _OPTESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# opus_gemm_common is pure python (stdlib only), so importing the codegen kid
# table here does not pull in the build.
from opus_gemm_common import (
    _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF,
    a8w8_mxscale_bmm_kernel_lists,
)
from test_opus_a8w8_bmm import (
    GROUP,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    run_torch,
)

# kid -> OpusGemmInstance. Kids are disjoint across the BMM families today;
# assert so a future collision (which the codegen dedups by launcher name
# downstream) is caught here instead of silently tuning one of the two.
_CODEGEN_BMM = {}
for _fam in a8w8_mxscale_bmm_kernel_lists:
    for _kid, _inst in _fam.items():
        assert (
            _kid not in _CODEGEN_BMM
        ), f"bmm kid {_kid} collides across codegen families; disambiguate by name"
        _CODEGEN_BMM[_kid] = _inst

# Split-K sweep for the flatmm_splitk family. Small-M / few-tile shapes (the G16
# wo_a decode: 16 batch * n1024 * k4096) underfill the CUs at splitK=1, so split-K
# (fp32-workspace partials + fused reduce tail) can win by exposing parallelism
# along K. The correctness gate drops any combo a kernel mishandles, so an
# over-broad sweep is safe, just slower.
_SK = [1, 2, 4, 8]

# Tuning policy: kid -> splitK list. The ONLY hand-maintained per-kid metadata --
# it decides which kids to sweep and with which split-K factors, not their
# geometry and not their M alignment. Tile shape, kernelName and m_align all come
# from the codegen instance, so this cannot drift from what compiles. kid 0 (the
# heuristic default) is intentionally not tuned.
_TUNE_POLICY = {
    # flatmm_splitk family: the M=16/32 last-mile tiles, the mid-M SFA/SFB-preload
    # tiles and the 64x* tiles. All are split-K capable via the fused reduce tail,
    # except kid646 whose persistent DIRECT_ONLY schedule requires splitK == 1.
    32: _SK,
    64: _SK,
    138: _SK,
    139: _SK,
    256: _SK,
    311: _SK,
    312: _SK,
    313: _SK,
    314: _SK,
    316: _SK,
    317: _SK,
    318: _SK,
    319: _SK,
    320: _SK,
    321: _SK,
    322: _SK,
    323: _SK,
    324: _SK,
    326: _SK,
    327: _SK,
    640: _SK,
    642: _SK,
    646: [1],
    650: _SK,
    653: _SK,
    # fused single-tile launcher.
    100: [1],
    # pipeline family; kid158 preloads both the per-token SFA and the block SFB
    # panel into LDS.
    149: [1],
    150: [1],
    151: [1],
    152: [1],
    158: [1],
    # kid158's scale preload at the two half tiles. Narrow-N wo_a (n1024) gives
    # the 256x256 tile only 4 N-tiles, so mid-M shapes leave half the CUs idle;
    # these fill them from the M side (159) and the N side (164). kid164 also
    # covers n128, which the 256-wide tiles reject outright.
    159: [1],
    164: [1],
    # The preshuffled-B families, all of them, splitK=1 only: none carries the
    # flatmm_splitk launcher's fused reduce tail. Nothing here was ever actually
    # evaluated before -- B arrived row-major (see gen_bmm_mxscale_data) so every
    # one of them failed the correctness gate silently, which is why the shipped
    # table picks none of them despite three of them having been listed here.
    #
    # wave8n4, the 2x4 eight-wave grid: 256x256 for large M, 128x256 mid, and the
    # 128x64x256 tile that carries the small-M end.
    168: [1],
    175: [1],
    194: [1],
    # kid194 plus the banded tile map, the wave8n4 answer to what kid205 is for
    # wavetm1. Worth the sweep on that precedent alone: kid203 and kid205 are the
    # same tile differing only in the map, and the shipped table gives kid205 22
    # rows against kid203's 6. kid194 holds 16, all of them m>=2048 shapes that
    # fill the machine, which is the regime a tile map acts on at all.
    346: [1],
    # The 128x128 tile this family never had, at both K depths. Swept because they
    # win 4 cells of the m=128..512 band by 1.025-1.057x; they do not do what they
    # were added to do, which was to close a 1.27x gap to Triton's swept kernel at
    # g16/m256/k4096 at equal geometry. The kid348 note in opus_gemm_common.py has
    # the numbers and where that gap actually lives.
    #
    # Worth sweeping only from g8 up: at g2/m128 they are 1.8x off the incumbent,
    # since a 128-row eight-wave tile on a 16-workgroup grid is mostly idle machine.
    348: [1],
    349: [1],
    # wavetm1, the 1x8 / 1x4 grids. kid205 is kid203 plus the banded tile map.
    202: [1],
    203: [1],
    205: [1],
    # bdirect, B straight to registers with no LDS hop: the 16x32 and 64x32
    # last-mile tiles, and the 128x128 tile that owns the mid band.
    171: [1],
    172: [1],
    173: [1],
    179: [1],
    184: [1],
    # The plain-scale halves of the kid334/335 ablation pairs (see the term table
    # in opus_gemm_common.py). Built to attribute a layout ratio rather than to
    # ship, but they are ordinary preload_sf kids at tiles the pool otherwise has
    # only at a different B_K or WG_PER_CU, so a table sweep is the cheapest way
    # to find out whether one of them owns a cell.
    #
    # kid344 stays out on purpose. Its B_N=128 geometry runs 105,000-133,000us
    # against 19us for the same family's baseline tile, so sweeping it would cost
    # more than the rest of the pool put together and it cannot win a cell.
    336: [1],
    338: [1],
    342: [1],
    # kid158's pipeline with only B's layout flipped -- the direct A/B comparison
    # for what preshuffling B is worth at the tile the shipped table leans on.
    196: [1],
    # monolithic mouter / wave pipelines.
    131: [1],
    132: [1],
    134: [1],
    142: [1],
    144: [1],
    148: [1],
    160: [1],
    161: [1],
    # minterleave only exists in split-K form.
    162: [2, 4, 8],
    163: [2, 4, 8],
    # 128x128x128 tiles, splitK=1 only and deliberately so. They are the largest
    # BMM tile (COM_REP_M=4 x COM_REP_N=8 -> 32 C fragments, 128 fp32 C values per
    # lane) at 512 VGPRs / occupancy 1. At splitK=1 they run the Cbf16
    # direct-output kernel and are strong -- kid325 wins 5 shipped wo_a rows. At
    # splitK>1 they switch to the Cvoid fp32-workspace kernel, which spills and has
    # never won (g2/m256: best kid325 split-K is 23.6us against the 14.5us winner),
    # and which is also where the clang-22 gfx950 greedy-VGPR miscompile lives (one
    # C-fragment dword left unmaterialized under --amdgpu-mfma-vgpr-form).
    128: [1],
    137: [1],
    325: [1],
}

# The blds twins, derived rather than listed: a twin is its plain kid's tile with
# B preshuffled and B's LDS staging kept, so wherever the plain kid is worth trying
# the twin is the preshuffled pool's answer to that cell. Deriving it here is what
# keeps the two in step -- the catalog asserts every plain flatmm tile has a twin,
# and this makes every one of those twins a candidate, so neither half of the pair
# can be added without the other reaching the sweep.
#
# splitK stays at 1 even where the plain kid sweeps the full _SK. Split-K has never
# won a flatmm cell on this envelope (the only sk>1 winners are kid163's minterleave
# rows), and sweeping four factors over 27 more kids is the bulk of the tuning time.
# A plain kid that wins a cell at splitK>1 is therefore still a cell the preshuffled
# pool cannot answer; widen this if one ever appears.
_TUNE_POLICY.update(
    {
        twin: [1]
        for plain, twin in _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF.items()
        if plain in _TUNE_POLICY
    }
)

# Deliberately not candidates: kid208 (mpack_sfa) and kid210/213/214/215/216/217
# (shuffle_scale) need A's scale panel, and for the shuffle_scale kids B's too, relaid out by
# whatever produces them. That layout is fixed once per model, so a table meant
# for a model whose quantiser emits plain scales cannot dispatch to them at all --
# see the shuffle_scale notes in opus_gemm_common.py for what the layout is.
#
# Priced, in case a quantiser can emit it, and the answer is no. The axis measures on
# its own because every shuffle_scale kid has a plain-scale twin at the same tile/wg/xcd:
# over the 133-cell preshuffle envelope the seven wave8 pairs come out at 1.004x median
# and 496/931 cells won, i.e. a wash, and strongly per-tile (kid216/172 1.063x and
# kid214/168 1.023x against kid217/184 0.860x). kid208's A-only mpack layout is a
# separate decision and a clear no at 0.903x.
#
# The pool-level number to trust is best-shuffle_scale against best-plain-scale, both
# sides drawing the whole pool. Comparing against the *table's* pick instead prices the
# table's sub-optimality along with the layout, which is what made an earlier pass read
# 25 of 133 cells better by >1%. Full pool both sides, 12 shapes x 3 K, order rotated:
# median -0.80% at K=1024, -4.25% at K=4096, -5.74% at K=8192, 4 of 36 cells better by
# >1% and none at K=8192.
#
# Two things that number is not. It is not a property of the layout uniformly: measured
# twin against twin, the shuffled read beats the LDS panel at every K on kid334's
# 64x32x256 (1.108x/1.041x/1.038x against kid172) and loses a fifth on kid335's
# 128x128x128 (0.984x/0.805x/0.793x against kid184), so the pool reads negative because
# the pool's top is the tiles where it loses. Which tile property that is remains open --
# those two kids differ in four parameters, and kid215 rules out the obvious answer by
# being B_K=256/COM_REP_K=2 and losing anyway; see opus_gemm_common.py's kid334/335
# entry. And it is not measured with the scale prefetch missing any more: kid334/335 are
# kid216/217 with PREFETCH_SCALE on, worth only 1-2% here against flatmm's 1.139x, but
# their absence was half the apparent decay with K (the same sweep without them reads
# -0.43% / -5.97% / -10.82%). The wave8n4/wavetm1 kids cannot be measured with it at
# all -- that pipeline static_asserts !PREFETCH_SCALE.
#
# Past K=8192 the panel does not fit and 21 of 99 kids stop dispatching at splitK=1,
# which looks like the layout's opening: at splitK=1 the fastest kid at K=16384/32768 is
# a shuffle_scale kid in all 10 cells measured. But the bound is per split -- the
# launcher checks ceil(total_iters/split_k) <= SF_PRELOAD_K_MAX/B_K -- so a plain-scale
# preload kid reaches K=32768 at split_k>=4 today. Swept over split_k in {1,2,4,8} the
# advantage is gone: median -3.1%, 0 of 10 cells better by >1%, kid194 taking 6 of 10.
#
# That held only because every shape in the sweep leaves the machine half empty -- the
# largest, g2/m4096, is 128 workgroups at B_M=256 against 256 CUs, so split-K was partly
# buying parallelism the shape lacked. On shapes that fill it (g16/m4096 is 2048 WGs) at
# K=16384/32768, split_k=1 takes 6 of 6 cells and no panel kid dispatches there at all,
# so kid213 (shuffle_scale) wins every cell with the best panel kid 1.068-1.144x behind.
# The full kid x split_k grid says that win is split_k=1's and not the layout's: at every
# split_k both families reach, the panel kid is 0.75-0.83x the shuffle_scale kid. So the
# 7-14% is the K bound charging rent, collectable either by emitting shuffle_scale (worth
# it only in this regime) or by letting the faster family into the split_k=1 column.
#
# Which is worth doing, because the bound is not each kid's. The panel is
# (SFA rows + SFB rows) * K/GROUP_K bytes with SFA rows == B_M, and reading
# .group_segment_fixed_size out of the built code objects puts 19 of 25
# panel kids at 3-884x of unused headroom: kid208 (mpack_sfa, SFA from global, 2 SFB rows)
# could take K=7.2M per split, kid205 111,360, kid194 30,976. Only 158/196/228/230/324/326
# are genuinely full, and 151,680 -- the figure the traits comment justifies 8192 with --
# is kid158/196's, a pipeline whose staging is the 2*(B_M+B_N)*B_K double buffer and not
# the flatmm/wave8 families the constant also governs. Chunked refills are forced only for
# those 6.
#
# So the wave8 traits now derives SF_PRELOAD_K_MAX from the LDS its staging leaves over
# (capped at 32768, with a 256-byte reserve for allocator padding), which gives kid194
# 30,848 and the B_M=128 kids 30,464 -- the latter is not the cap because the budget is
# the LDS share that keeps the kernel's workgroups resident, not the whole CU. Spending
# the whole CU is what the first cut did, and it cost kid203/kid205 1.19-1.20x at m>=1536:
# they are 256-thread workgroups that fit a CU twice at 59,012 bytes and once at 83,972.
# See the SF_PANEL_LDS_CEILING note in the traits. At split_k=1 and K=16384 on machine-filling shapes
# that is worth 3.0-4.0% over kid213, 8/8 paired draws in each of three shapes, bit-exact.
# Less than the 17-25% the shared-split_k columns suggest, because the panel's edge falls
# from 0.79x at sk2/sk4 to 0.97x at sk1 -- kid213 gains more from that column than the
# panel kids do. K=32768 stays kid213's: a 256x256 tile needs a 66,048-byte panel against
# 62,208 of headroom, the one cell where a chunked refill is the only lever left (it would
# have to find 5.3%). No shipped row moves -- this table is K in {1024, 4096} at splitK=1
# -- so the gain is available to whoever tunes large-K rows, and re-running all 133 rows
# shows no drift beyond the machine's own (1.011x affected over control).
#
# split_k 2-8 being free is measured, not assumed: the reduce sums an fp32 workspace in an
# fp32 accumulator and casts once, so splitting K is blocked summation. Against an fp64
# reference at K=32768 with positive operands the max relative error falls monotonically,
# 6.18e-06 at sk1 to 4.75e-06 at sk8; with bf16 output every split_k reads 1.327e-03 and
# the difference is invisible. The ceiling on split_k is the ">= 3 K-tiles per split"
# launcher check, not precision.
#
# One follow-on still declined: a deeper register prefetch cannot replace the panel. The
# shared BMM pipeline already stages scales in v_sfa[2][2] one to two K tiles ahead with
# SFA_VM in every vmcnt immediate, so latency is covered, and what the panel buys is
# SFA_VM==0 -- the scale load leaving every vmcnt wait, vmcnt being one in-order counter.
# More stages attack latency, not ordering.
# See the kid328-333 block in opus_gemm_common.py.
#
# One methodology note from that work, for anyone re-running a pool sweep here:
# stepping the candidate pool in kid order every draw is worth several percent to
# whoever runs first. Rescanning the 39-kid preshuffle pool that way produced three
# mis-ranks that all evaporated under a small pool with the order rotated and 12
# draws (kid196 at g4/m3072/k4096 read 64.09us in kid order and 73.35us rotated).
# Rank on a handful of candidates with rotated order and per-draw values, not on a
# median over a full-pool pass.
_RELAYOUT_KIDS = sorted(
    kid
    for kid, inst in _CODEGEN_BMM.items()
    if inst.needs_mpacked_sfa is not None or inst.needs_shuffle_scale is not None
)
assert not (set(_TUNE_POLICY) & set(_RELAYOUT_KIDS)), (
    f"kids {sorted(set(_TUNE_POLICY) & set(_RELAYOUT_KIDS))} need a producer-side "
    "scale relayout and cannot be tuned against plain scales"
)

# A policy entry for a kid the codegen no longer emits used to KeyError inside
# _applicable on the first shape, i.e. after the data was built. kid165, kid174
# and kid192 sat here that way. Fail at import instead.
_dead = sorted(set(_TUNE_POLICY) - set(_CODEGEN_BMM))
assert not _dead, f"_TUNE_POLICY lists kids the codegen does not emit: {_dead}"

# Only the flatmm_splitk (non-direct) and minterleave launchers honor splitK>1.
# Any other family sweeping it is a policy bug, so fail loudly at import.
for _kid, _sks in _TUNE_POLICY.items():
    if any(s > 1 for s in _sks):
        _tag = _CODEGEN_BMM[_kid].kernel_tag
        assert (
            _tag == "a8w8_mxscale_bmm_flatmm_splitk"
            and not getattr(_CODEGEN_BMM[_kid], "direct_only", False)
        ) or _tag == "a8w8_mxscale_bmm_minterleave", (
            f"kid {_kid} ({_tag}) is not split-K capable but sweeps {_sks}"
        )


POOLS = ("all", "preb", "rowb")


def _applicable(kid, g, m, n, k, pool="all"):
    """Split-K factors worth trying for this kid on this shape ([] == skip it).

    ``pool`` restricts by B's layout: "preb" to the preshuffled-B kids, "rowb" to
    the row-major ones, "all" to both. A shipped table has to be all one layout to
    be usable -- the weight is shuffled offline, so a deployment holds one form of
    it, and a mixed table would need both resident. The first cut of the preshuffle
    table did mix them, row-major kids winning 15 of 77 shapes on merit, and those
    rows are exactly the ones a preshuffled deployment cannot dispatch. Tuning the
    two pools separately over one shape set is also how the layouts get compared
    at all: per shape, best row-major against best preshuffled.
    """
    k_inst = _CODEGEN_BMM[kid]
    if pool == "preb" and not k_inst.needs_preshuffled_b:
        return []
    if pool == "rowb" and k_inst.needs_preshuffled_b:
        return []
    if n % k_inst.B_N or k % k_inst.B_K or m % k_inst.m_align:
        return []
    return _TUNE_POLICY[kid]


SHIPPED_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv",
)
DEFAULT_OUT = os.path.join(_REPO, "dsv4_bmm_mxscale_retuned.csv")

# The same dsv4 shapes retuned with the preshuffled-B families in the pool, kept
# as a second table rather than applied over the first.
#
# The name is load-bearing in a way worth spelling out. get_config_file globs
# model_configs/ for "*batched_gemm_a8w8_blockscale_mxscale_tuned*.csv" and merges
# every hit; two tables covering the same shapes would collide on the (gfx,b,m,n,k)
# key, and update_config_files answers a collision by rewriting the source files
# down to the lowest-us row each and then raising. Putting "bpreshuffle" before
# "tuned" breaks the substring, so this file is invisible to that glob and is
# instead the default table of the separate _BPRESHUFFLE config entry, which
# batched_gemm_a8w8_mxscale reads only under b_preshuffled=True. Nothing has to
# be set to pick it up; override that entry's env var to try another one.
#
# Four of the 133 cells are slower here than the shipped table is with row-major
# B, and they stay in anyway: a b_preshuffled=True caller has no row-major kernel
# to fall back to, and each row already names the fastest preshuffled kid the
# entry can dispatch (re-swept over the whole pool at re-drawn placements). Two of
# them are twin-vs-twin -- g16/m128/k4096 is kid326 against kid230 (+11%) and
# g16/m256/k4096 is kid325 against kid229 (+14%) -- and neither is a cost of the
# layout. Those two kids compile to identical VGPR/AGPR/LDS with no spill, and the
# preshuffled one issues fewer instructions with the same 210 ds_read / 86
# buffer_load / 288 MFMA, so they move the same bytes doing the same work. Run as
# a pair across a g x m grid at both K they are a wash on 39 of 40 cells; the
# exception is the cell where the grid is exactly one occupancy wave (256
# workgroups on 256 CUs), where all workgroups march through K in phase and the
# memory pipe is already at its deepest queue. Profiled, the two are identical on
# every volume counter (L2 requests, hit rate, EA read requests all within 0.1%)
# and both spread perfectly evenly over the 128 memory channels, so it is not
# camping; the preshuffled side even takes 3x fewer tag stalls. It holds the
# channels 14% longer (TCC_BUSY), and that is the only counter that tracks the
# gap -- the +11-15% EA read latency it also carries is present at 2 and 4 waves
# too, where preshuffle wins anyway. Thread trace puts the extra wave-time on
# s_barrier and takes it off the B loads, i.e. the consumers wait longer at the
# rendezvous for B to reach LDS, and shows workgroup durations spreading 13%
# across CUs inside one dispatch -- at one wave the kernel is the max of that
# spread, so a 1-3% shift in it is a 4-6% kernel. What the counters cannot see is
# the L2 set index: 16 channels x 128 B means it advances per 2 KiB and wraps at
# 256 KiB, so the 64 KiB panel stride puts the tile's 8 chunks on 4 sets, and the
# n-tile and batch strides are whole multiples of the wrap so every workgroup
# picks the same 4. Padding stride_b (a kargs field taken from wo_a.stride(1), so
# no kernel change) to 72 KiB takes the cell from 0.87x to 0.98x and does nothing
# at the strides and wave counts that were already fine; across a wider pad sweep
# every stride landing on 8 sets runs 0.98-1.00x and every one landing on <=4 runs
# 0.87-0.94x. This is the stride and not the shuffle -- the shuffle only multiplies
# B's stride by 16 (K -> 16*K, four bits out of the set index), and forcing a
# 16 KiB row stride on the row-major baseline costs it 20%, more than preshuffle
# ever loses. Not shipped: 12.5% of weight memory for one cell. split_k>1 breaks
# the lockstep but costs more than it saves.
#
# The other two are g2/m512/k1024 (+6.4%) and g2/m1024/k1024 (+4.2%), where the
# row-major side is the heuristic, not a row.
#
# g2/m32768/k4096 was a fifth at +2.4%, and was not a layout cost at all: kid196
# (kid158's own pipeline reading a preshuffled B) and kid205 both land within 1%
# of row-major there, and the row named kid194, the slowest of the three. It now
# names kid205, and g16/m4096/k4096 -- same family, same mis-rank -- now names
# kid196. Neither was visible to the sweep that wrote them, because the candidates
# sit inside 2% of each other and a single pass ranks them by luck. The other 15
# rows naming kid194 were re-checked and it is the right pick on all of them.
#
# Two measurement traps here, each of which inverted an answer before it was found:
#   * run_perftest deep-copies the arguments it is handed into rotate_args sets and
#     cycles them, so the timed kernel reads a weight it did not just read.
#     Operands captured in a zero-argument closure are not arguments, and all 101
#     iterations then hit one cache-resident copy -- worth 14% to the 8-wave kids,
#     enough to make kid175 look like it beat row-major at g16/m128 by 1.4% when
#     it and kid230 are a wash.
#   * the placement note in opus_gemm_common.py, that at K=4096 a kernel's time
#     depends on where its weight buffer landed, does not cover these cells, but
#     had to be ruled out rather than assumed: the two sides necessarily hold
#     different buffers, so a single allocation bakes one placement difference into
#     the comparison. Over 8 draws that move both buffers, every gap above holds
#     its sign and no kernel varies by more than 4%.
BPRESHUFFLE_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv",
)


# ---------------------------------------------------------------------------
# mp_tuner hooks (module-level so the spawn workers can import them by name).
# ---------------------------------------------------------------------------
def _gen_varied(shape, k, device):
    """Signed, per-128-K-block varied-magnitude bf16 (mirrors _block_varied)."""
    x = torch.randn(shape, dtype=dtypes.fp32, device=device)
    amp = torch.exp2(torch.randint(-4, 4, (k // GROUP,), device=device).float())
    return (x * amp.repeat_interleave(GROUP)).to(dtypes.bf16)


def gen_bmm_mxscale_data(batch, m, n, k, seed, out_dtype, device="cuda"):
    """Return the 7-tuple mp_tuner indexes into:

    0 O_in   [m,g,k]     fp8 (mmajor transposed view, K contiguous)
    1 W_mx   [g,n,k]     fp8 (batch-major)
    2 Y       [m,g,n]     out_dtype output buffer
    3 xs_in  [m,g,k/128] uint8 e8m0 per-token scale (mmajor view)
    4 ws_mx  [g,n/128,k/128] uint8 e8m0 128x128-block scale
    5 ref     [m,g,n]     out_dtype dequant fp32 einsum reference
    6 W_sh   [g,n,k]     the same B in the (16,16) preshuffled layout
    """
    torch.manual_seed(seed)
    O_bf16 = _gen_varied((batch, m, k), k, device)
    W_bf16 = _gen_varied((batch, n, k), k, device)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)
    O_in = O_mx.transpose(0, 1)  # [m,g,k]
    xs_in = xs_mx.transpose(0, 1)  # [m,g,k/128]
    Y = torch.empty((m, batch, n), dtype=out_dtype, device=device)
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1).to(out_dtype)
    # The preshuffled-B kids read B through the (16,16) MFMA-fragment layout that
    # a serving stack bakes into the weight offline. It is a permutation, so the
    # reference above covers both forms. Building it here rather than in the bench
    # keeps it out of what is timed.
    W_sh = shuffle_weight(W_mx, layout=(16, 16))
    return (O_in, W_mx, Y, xs_in, ws_mx, ref, W_sh)


def run_bmm_mxscale_bench(O_in, W_mx, Y, xs_in, ws_mx, W_sh, kernelId, splitK):
    """Tuner bench func: run the kid in-place, return Y for checkAllclose.

    B's layout is per kid, so it is selected here rather than by the caller: a
    preshuffled-B kid handed row-major B reads the right bytes in the wrong order
    and fails the gate, which is how three of these kids sat in _TUNE_POLICY
    without ever being able to win a shape.
    """
    Wb = W_sh if _CODEGEN_BMM[kernelId].needs_preshuffled_b else W_mx
    _opus_bmm_a8w8_mxscale_raw(O_in, Wb, Y, xs_in, ws_mx, splitK, kernelId)
    return Y


def _bmm_ref_passthrough(ref):
    """ref_func: the fp32 reference is precomputed in gen_data (slot 5)."""
    return ref


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------
class OpusBmmMxscaleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": DEFAULT_OUT,
        "untune_file": "",
        # Fraction-of-mismatch (rtol=atol=1e-2) accept threshold. Correct kids
        # sit at the ~1e-4 fp8 e8m0 quant floor; a column-transposed kid is ~0.5.
        "errRatio": 0.02,
        "batch": 100,
    }

    KEYS: ClassVar[list[str]] = ["gfx", "b", "m", "n", "k"]
    RESULTS: ClassVar[list[str]] = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]

    def __init__(self):
        # Bypass GemmCommonTuner.__init__ (it force-swaps "M"/"N" in the key,
        # which assumes the uppercase gptoss schema). Go straight to the
        # grandparent with our lowercase batched schema.
        TunerCommon.__init__(
            self,
            "OpusBmmMxscaleTuner",
            self.KEYS,
            self.RESULTS,
            description="Tune opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a)",
        )
        # sort N before M like the GEMM tuners (cosmetic ordering of the CSV).
        self.sort_keys = ["gfx", "b", "n", "m", "k"]

    # --- schema helpers -----------------------------------------------------
    def getKernelName(self, kernelId):
        k_inst = _CODEGEN_BMM.get(int(kernelId))
        return k_inst.name if k_inst else None

    def calculate(self, results, bpes=None):
        info, time, _err = results
        if time == self.INVALID_TIME:
            return 0, 0
        _gfx, b, m, n, k = info[0]
        us_s = time * 1e-6
        tflops = round(2 * b * m * n * k / us_s / 1e12, 1)
        # fp8 A + fp8 W + bf16 out.
        bw = round((b * m * k + b * n * k + 2 * b * m * n) / us_s / 1e9, 2)
        return tflops, bw

    def result_to_df(self, results):
        rows = []
        for info, time, err in results:
            keys, kernelId, splitK, kernelName = info
            resolved = kernelName or self.getKernelName(kernelId)
            tflops, bw = self.calculate((info, time, err))
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "libtype": "opus",
                    "kernelId": int(kernelId),
                    "splitK": int(splitK),
                    "us": time,
                    "kernelName": "None" if resolved is None else str(resolved),
                    "tflops": tflops,
                    "bw": bw,
                    "errRatio": err,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    # --- CLI ----------------------------------------------------------------
    def _setup_specific_arguments(self):
        # Free the base "-k/--splitK" store_true so we can reuse -k for the K dim.
        for action in list(self.parser._actions):
            if "-k" in action.option_strings or "--splitK" in action.option_strings:
                self.parser._actions.remove(action)
                for s in action.option_strings:
                    self.parser._option_string_actions.pop(s, None)
                for grp in self.parser._action_groups:
                    if action in grp._group_actions:
                        grp._group_actions.remove(action)
                break

        def _intlist(s):
            return [int(x) for x in str(s).split(",") if x != ""]

        self.parser.add_argument(
            "-g",
            "--batch_g",
            type=_intlist,
            default=None,
            help="comma list of batch g (e.g. 2,8,16)",
        )
        self.parser.add_argument(
            "-m",
            "--M",
            type=_intlist,
            default=None,
            help="comma list of M (e.g. 1,16,64)",
        )
        self.parser.add_argument(
            "-n",
            "--N",
            type=_intlist,
            default=[1024],
            help="comma list of N (default 1024)",
        )
        self.parser.add_argument(
            "-k",
            "--K",
            type=_intlist,
            default=[4096],
            help="comma list of K (default 4096)",
        )
        self.parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="overwrite the shipped tuned CSV in place",
        )
        self.parser.add_argument(
            "--bpreshuffle",
            action="store_true",
            default=False,
            help="write to the preshuffle-inclusive table (BPRESHUFFLE_CSV)",
        )
        self.parser.add_argument(
            "--pool",
            choices=POOLS,
            default="all",
            help="restrict candidates by B layout (--bpreshuffle implies preb)",
        )

    # --- shape sourcing -----------------------------------------------------
    def _shapes_from_shipped(self):
        try:
            df = pd.read_csv(SHIPPED_CSV)
        except FileNotFoundError:
            return []
        return sorted(
            {(int(r.b), int(r.m), int(r.n), int(r.k)) for _, r in df.iterrows()}
        )

    def pre_process(self, args):
        if args.apply and args.bpreshuffle:
            raise SystemExit("--apply and --bpreshuffle write different tables")
        if args.apply:
            args.tune_file = SHIPPED_CSV
        elif args.bpreshuffle:
            args.tune_file = BPRESHUFFLE_CSV
            if args.pool == "all":
                args.pool = "preb"

        gfx = self.get_gfx()
        if args.batch_g and args.M:
            shapes = [
                (g, m, n, k)
                for g in args.batch_g
                for m in args.M
                for n in args.N
                for k in args.K
            ]
        elif args.untune_file and os.path.exists(args.untune_file):
            df = pd.read_csv(args.untune_file)
            df.columns = [c.strip().lower() for c in df.columns]
            bcol = "b" if "b" in df.columns else "g"
            shapes = [
                (int(r[bcol]), int(r["m"]), int(r["n"]), int(r["k"]))
                for _, r in df.iterrows()
            ]
        else:
            logger.info(
                "no -g/-m and no untune_file; re-tuning shapes from %s", SHIPPED_CSV
            )
            shapes = self._shapes_from_shipped()

        self.untunedf = pd.DataFrame(
            [{"gfx": gfx, "b": g, "m": m, "n": n, "k": k} for (g, m, n, k) in shapes],
            columns=self.keys,
        )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)

        # Skip shapes already present in the tuned CSV (unless --all forces retune).
        if not args.all and len(self.tunedf) and len(self.untunedf):
            td = self.tunedf
            if "gfx" not in td.columns:
                td = td.assign(gfx=gfx)
            have = set(td[self.keys].apply(lambda r: tuple(r), axis=1).tolist())
            mask = self.untunedf.apply(lambda r: tuple(r) in have, axis=1)
            if args.verbose and mask.any():
                logger.info("skipping %d already-tuned shapes", int(mask.sum()))
            self.untunedf = self.untunedf[~mask].reset_index(drop=True)

    # --- tuning -------------------------------------------------------------
    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        out_dtype = dtypes.bf16
        perf_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}

        task = []
        tasks_data = []
        for seed, i in enumerate(range(len(untunedf)), start=1):
            b = int(untunedf.loc[i, "b"])
            m = int(untunedf.loc[i, "m"])
            n = int(untunedf.loc[i, "n"])
            k = int(untunedf.loc[i, "k"])
            info_keys = (gfx, b, m, n, k)

            n_cand = 0
            for kid in _TUNE_POLICY:
                for sk in _applicable(kid, b, m, n, k, args.pool):
                    info = (info_keys, kid, sk, "")
                    task.append(
                        (
                            info,
                            gen_bmm_mxscale_data,
                            (b, m, n, k, seed, out_dtype),
                            run_bmm_mxscale_bench,
                            ([0, 1, 2, 3, 4, 6], kid, sk),
                            perf_kwargs,
                            _bmm_ref_passthrough,
                            ([5],),
                            {},
                            None,
                            1e-2,  # rtol
                            1e-2,  # atol
                            None,  # compare_fn
                            None,  # max_abs_delta
                            [2],  # output_keys: NaN-init Y to catch partial writes
                        )
                    )
                    n_cand += 1
            tasks_data.append((n_cand, ()))

        if not task:
            return []
        return mp_tuner(
            task,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    tuner = OpusBmmMxscaleTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
