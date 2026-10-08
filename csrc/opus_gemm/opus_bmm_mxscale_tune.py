# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 BMM tuning through the shared GEMM tuner framework.

The CSV key is (gfx, b, m, n, k, w_scale_block); existing scale rows are
preserved. Shape-only inputs may specify --w_scale_block 32x32,128x128. Tuning uses
BF16 output, token-major contiguous activations/scales, and the per-kid
weight layout. All operands are passed to mp_tuner for buffer rotation.

Use -i/-o for input/output CSVs, --all to retune existing rows, --run_config
for saved or default configurations, and --compare --update_improved for
production-operator comparisons. The joint OPUS/FlyDSL entry point is
csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py.
"""

import argparse
import math
import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import dtypes, logger
from aiter.ops.opus.gemm_op_a8w8 import _opus_gemm_a8w8_mxscale_bmm_launch_raw
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner, TunerCommon
from aiter.utility.mp_tuner import mp_tuner
from csrc.opus_gemm._scale_layouts import shuffle_scale_a, shuffle_scale_b

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
    BMM_MXSCALE_KID_OFFSET,
    MX32_KID_STRIDE,
    _opus_sf_shuf_sub,
    a8w8_mxscale_bmm_kernel_lists,
    bmm_mxscale_global_kid,
)

# The quantisation block is a property of the kid, not of the shape: a 32-block
# kid needs 32-block scales and its tuned row is meaningless for a 128 one. So
# it is read off the instance here rather than taken from the module constant.
_KID_INSTANCE = {
    kid: inst
    for family in a8w8_mxscale_bmm_kernel_lists
    for kid, inst in family.items()
}
# Resolve both local IDs and global codegen IDs when deriving scale groups.
for _k, _v in list(_KID_INSTANCE.items()):
    _loc = _k - BMM_MXSCALE_KID_OFFSET
    if _loc >= 0 and _loc not in _KID_INSTANCE:
        _KID_INSTANCE[_loc] = _v


def _kid_groups(kid):
    # _TUNE_POLICY is keyed on global ids, so take the id as given and only fall
    # back to globalising a local one.
    inst = _KID_INSTANCE.get(kid) or _KID_INSTANCE[bmm_mxscale_global_kid(kid)]
    return inst.GROUP_N, inst.GROUP_K


def _kid_group(kid):
    return _kid_groups(kid)[1]


def _kid_block(kid):
    return "{}x{}".format(*_kid_groups(kid))


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

# Accept local IDs while assembling policies; launchers use global IDs.
for _kid, _inst in list(_CODEGEN_BMM.items()):
    _local = _kid - BMM_MXSCALE_KID_OFFSET
    if _local >= 0 and _local not in _CODEGEN_BMM:
        _CODEGEN_BMM[_local] = _inst

# Split-K factors considered for supported flatmm tiles.
_SK = [1, 2, 4, 8]

# Tuning policy: kernel ID -> split-K candidates.
# Shape and layout constraints come from the registered OpusGemmInstance.
_TUNE_POLICY = {
    # flatmm_splitk family: the M=16/32 last-mile tiles, the mid-M SFA/SFB-preload
    # tiles and the 64x* tiles. All are split-K capable via the fused reduce tail,
    # except kid646 whose persistent DIRECT_ONLY schedule requires splitK == 1.
    8032: _SK,
    8064: _SK,
    8138: _SK,
    8139: _SK,
    8256: _SK,
    8311: _SK,
    8312: _SK,
    8313: _SK,
    8314: _SK,
    8316: _SK,
    8317: _SK,
    8318: _SK,
    8319: _SK,
    8320: _SK,
    8321: _SK,
    8322: _SK,
    8323: _SK,
    8324: _SK,
    8326: _SK,
    8327: _SK,
    8640: _SK,
    8642: _SK,
    8646: [1],
    8650: _SK,
    8653: _SK,
    # fused single-tile launcher.
    8100: [1],
    # pipeline family; kid158 preloads both the per-token SFA and the block SFB
    # panel into LDS.
    8149: [1],
    8150: [1],
    8151: [1],
    8152: [1],
    8158: [1],
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
    # Preshuffled-B compute-wave candidates.
    168: [1],
    175: [1],
    194: [1],
    # wave8n4 tiles with banded workgroup mapping.
    346: [1],
    # 128x128 wave8n4 tiles at both K depths.
    348: [1],
    349: [1],
    # wavetm1, the 1x8 / 1x4 grids. kid205 is kid203 plus the banded tile map.
    202: [1],
    203: [1],
    205: [1],
    # The banded wave8 tiles (kid401/402 = 175/348 + band 4; kid404 = 202 +
    # band 4); they only win at m512-2048.
    401: [1],
    402: [1],
    404: [1],
    # kid205's 1x4 grid on the mid-M tiles (see
    # _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_1X4_TILES).
    405: [1],
    406: [1],
    407: [1],
    408: [1],
    409: [1],
    410: [1],
    411: [1],
    # All-compute 2x2 wave grids with A and B staged through LDS.
    # ID421 is not registered.
    420: [1],
    422: [1],
    423: [1],
    424: [1],
    425: [1],
    426: [1],
    427: [1],
    # bdirect, B straight to registers with no LDS hop: the 16x32 and 64x32
    # last-mile tiles, and the 128x128 tile that owns the mid band.
    171: [1],
    172: [1],
    173: [1],
    179: [1],
    184: [1],
    # Plain-scale panel variants paired with the shuffled-scale bdirect tiles.
    # The wider 64x128x256 variant is excluded because of register pressure.
    336: [1],
    338: [1],
    342: [1],
    # bdirect_tilen variants remain registered outside the default policy.
    196: [1],
    # monolithic mouter / wave pipelines.
    8131: [1],
    8132: [1],
    8134: [1],
    8142: [1],
    8144: [1],
    8148: [1],
    8160: [1],
    8161: [1],
    # minterleave only exists in split-K form.
    8162: [1],
    8163: [1],
    # 128x128x128 flatmm tiles use splitK=1 in the default policy.
    8128: [1],
    8137: [1],
    8325: [1],
    # Single-wave decode kernels load operands directly into registers.
    # The default policy considers splitK=1 and splitK=4.
    8440: [1, 4],
    8441: [1, 4],
    8442: [1, 4],
    8443: [1, 4],
    8444: [1, 4],
    8445: [1, 4],
    8446: [1, 4],
    8447: [1, 4],
    # B_N=16 decode variants with additional split-K choices.
    8448: [1, 2, 4],
    8449: [1, 2, 4],
}

# Derive preshuffled-B/LDS counterparts from the plain flatmm policy.
# Normalize IDs before membership checks and preserve each tile's split factors.
_TUNE_POLICY.update(
    {
        bmm_mxscale_global_kid(twin): [1]
        for plain, twin in _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF.items()
        if bmm_mxscale_global_kid(plain) in _TUNE_POLICY
    }
)
# Small-M plain-scale panels use COM_REP_M=1 so the source layout
# matches the LDS panel without M packing.
_TUNE_POLICY[398] = [1]
_TUNE_POLICY[399] = [1]

# The default policy consumes ordinary scales. M-packed and shuffled-scale
# kernels require a matching scale producer and are handled by separate pools.
_RELAYOUT_KIDS = sorted(
    kid
    for kid, inst in _CODEGEN_BMM.items()
    if inst.needs_mpacked_sfa is not None or inst.needs_shuffle_scale is not None
)
assert not (set(_TUNE_POLICY) & set(_RELAYOUT_KIDS)), (
    f"kids {sorted(set(_TUNE_POLICY) & set(_RELAYOUT_KIDS))} need a producer-side "
    "scale relayout and cannot be tuned against plain scales"
)


# Shuffled-scale candidate pool. Match each instance to its plain counterpart
# so the two variants use the same split-K choices.
def _plain_twin_key(inst):
    # WG_PER_CU, not wg_per_cu. The lowercase field is the gfx1250 cluster-TDM
    # co-residency knob and is the dataclass default (2) for every kid in this
    # file, so keying on it is blind to occupancy and pairs a wg1 kid with a wg2
    # one -- which is the one comparison this table exists to make.
    return (
        inst.kernel_tag,
        inst.BLOCK_SIZE,
        inst.B_M,
        inst.B_N,
        inst.B_K,
        inst.WG_PER_CU,
        inst.xcd_wgm,
    )


_PLAIN_BY_TILE = {}
for _kid in _TUNE_POLICY:
    _inst = _CODEGEN_BMM[_kid]
    if _inst.needs_shuffle_scale is None and _inst.needs_mpacked_sfa is None:
        _PLAIN_BY_TILE.setdefault(_plain_twin_key(_inst), _kid)

# shuffled kid -> the plain kid it is the twin of. Both directions are used: the
# twin supplies the splitK sweep, and reporting a shuf result next to its twin's
# is the only comparison that isolates the layout.
_SHUF_TWIN_OF = {}
for _kid, _inst in _CODEGEN_BMM.items():
    if _inst.needs_shuffle_scale is None:
        continue
    _plain = _PLAIN_BY_TILE.get(_plain_twin_key(_inst))
    if _plain is not None:
        _SHUF_TWIN_OF[_kid] = _plain

_SHUF_POLICY = {kid: _TUNE_POLICY[plain] for kid, plain in _SHUF_TWIN_OF.items()}

# Collect plain candidates before adding the shuffled-scale pool.
_TUNE_POLICY.update(
    {
        twin: factors
        for mirror, factors in list(_TUNE_POLICY.items())
        if (twin := mirror + MX32_KID_STRIDE) in _KID_INSTANCE
    }
)


# Normalize every policy ID to the global codegen namespace.
def _globalise_policy(policy):
    return {
        (bmm_mxscale_global_kid(k) if k < BMM_MXSCALE_KID_OFFSET else k): v
        for k, v in policy.items()
    }


# All seven production compact schedules enter both policy and exhaustive sweeps.
_TUNE_POLICY.update(
    {
        kid: [1]
        for kid, inst in _CODEGEN_BMM.items()
        if kid >= BMM_MXSCALE_KID_OFFSET
        and inst.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_compact"
    }
)
_TUNE_POLICY = _globalise_policy(_TUNE_POLICY)
_TUNE_POLICY.update(
    {kid: [1, 2, 4, 8] for kid in _CODEGEN_BMM if _kid_block(kid) == "1x32"}
)
_SHUF_POLICY = _globalise_policy(_SHUF_POLICY)
_CANDIDATE_KIDS = tuple(_TUNE_POLICY) + tuple(_SHUF_POLICY)
assert len(set(_CANDIDATE_KIDS)) == len(_CANDIDATE_KIDS), (
    "a kid is in both _TUNE_POLICY and _SHUF_POLICY; it would be benched twice "
    "per shape, once against each scale layout, under one kid id"
)


def _shuf_arm(inst):
    """Return "reg" or "lds" according to the instance's shuffled-scale storage."""
    return "lds" if inst.sf_shuf_in_lds else "reg"


# A policy entry for a kid the codegen no longer emits used to KeyError inside
# _applicable on the first shape, i.e. after the data was built. kid165, kid174
# and kid192 sat here that way. Fail at import instead.
_dead = sorted(set(_TUNE_POLICY) - set(_CODEGEN_BMM))
assert not _dead, f"_TUNE_POLICY lists kids the codegen does not emit: {_dead}"

# Families whose launcher honors splitK>1: they write fp32 partials to the
# caller-owned workspace and carry the fused reduce tail (per-tile counter,
# last arrival sums in split order). Everything else is splitK==1 only, so a
# policy entry sweeping splitK>1 on another family is a bug -- fail at import
# rather than silently benchmark a wrong result.
_SPLITK_FAMILIES = frozenset(
    {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_bpreshuffle_wave1",
    }
)

for _kid, _sks in _TUNE_POLICY.items():
    if any(s > 1 for s in _sks):
        _tag = _CODEGEN_BMM[_kid].kernel_tag
        assert (
            _tag in _SPLITK_FAMILIES and not _CODEGEN_BMM[_kid].direct_only
        ), f"kid {_kid} ({_tag}) is not split-K capable but sweeps {_sks}"


# Families whose kernel implements the SHUFFLE_SCALE template bool at all. The
# other BMM families (pipeline, pipeline_bpreshuffle, minterleave, mouter,
# wave4m2_selfload, ...) never see the flag, so no tile of theirs can read the
# layout however it is shaped.
_SHUF_FAMILIES = frozenset(
    {
        "a8w8_mxscale_bmm_bpreshuffle_wave8n4",
        "a8w8_mxscale_bmm_bpreshuffle_wavetm1",
        "a8w8_mxscale_bmm_bpreshuffle_bdirect",
        "a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen",
        "a8w8_mxscale_bmm_bpreshuffle_blds",
        "a8w8_mxscale_bmm_flatmm_splitk",
    }
)


# The A-scale layout's `sub`, read from the traits header rather than restated:
# host, kernel and tuner disagreeing about it is a silent wrong-answer bug, not a
# build failure. See opus_gemm_common._opus_sf_shuf_sub.
SHUF_SUB = _opus_sf_shuf_sub()


def _shuf_sub(inst):
    """``SHUF_SUB`` if this kid's tile can read the shipped layout, else None.

    ``sub`` is a property of the *layout*, which the quantize kernel emits once
    per launch, so a deployment holds exactly one -- the same all-or-nothing
    property `pool` already encodes for B's layout, one axis over. What is
    per-tile is only the *capability*, and this function is a transcription of
    ``opus_sf_shuf_geom``'s ``OK`` / ``KD``.

    Keep the two in step. A predicate looser than the header's turns a pool entry
    into a build error; one tighter silently under-reports what the layout can
    do, which is what this did before 2a-ii by two separate clauses. Note the
    B_K clause is {1,2,4} rather than "any even" only because op_sel is two bits
    -- the index algebra itself generalises -- so the tight direction is here.

    kid388/389/390 (bdirect_tilen, B_M=64 at T_M=1) are the one geometry this
    admits that no *shuffled* kid exercises: A_SLOTS=2 on the flatmm pipeline.
    They are plain-scale, so returning a sub for them is correct -- that is what
    the subok pool is -- but a shuffled tilen kid would need the traits geometry
    and twin bit-exactness gates before this answer is trusted for it.
    """
    if inst.kernel_tag not in _SHUF_FAMILIES:
        return None
    com_rep_k = inst.B_K // inst.GROUP_K
    if com_rep_k not in (1, 2, 4):
        return None
    kd = com_rep_k // 2 if com_rep_k >= 2 else 1
    assert kd * 2 >= com_rep_k, "SFG::KD must cover every K block the tile owns"
    sf_mb = inst.T_M * inst.W_M
    paired = sf_mb <= SHUF_SUB and SHUF_SUB % sf_mb == 0
    wide = sf_mb == 2 * SHUF_SUB
    if not (paired or wide) or inst.B_M % sf_mb:
        return None
    if SHUF_SUB < inst.B_M < 2 * SHUF_SUB:
        return None
    return SHUF_SUB


# subok holds the plain-scale kids whose tiles could read the layout, so it
# prices the tile restriction alone; shuf_reg / shuf_lds / shuf hold the shuffled
# kids and price the layout itself. The two numbers add. `shuf` lets the tuner
# see both arms and pick, which is the only one of the four that answers the
# shipping question -- the reg/lds split just says which mechanism earned it.
# (There is one layout, at whatever OPUS_SF_SHUF_SUB_VALUE says; there is no
# per-pool sub any more.)
POOLS = ("all", "preb", "rowb", "subok", "shuf", "shuf_reg", "shuf_lds")


def _applicable(kid, g, m, n, k, pool="all", *, split_ks=None):
    """Return legal split-K candidates for this shape, or an empty list.

    The B-layout pools are "preb" (preshuffled), "rowb" (row-major), and
    "all". "subok" selects plain-scale preshuffled kernels whose geometry
    also supports the shuffled-scale layout. The "shuf" pools select the
    corresponding shuffled-scale kernels and their register/LDS variants."""
    k_inst = _CODEGEN_BMM[kid]
    shuf = kid in _SHUF_POLICY
    # A shuffled kid is a candidate only where the pool asked for one, and a
    # plain-scale kid only where it did not. The two cannot be mixed in one table
    # for the same reason preb and rowb cannot: the quantize kernel emits one A
    # scale layout per launch, so a table is all one form or it is undispatchable.
    if pool in ("shuf", "shuf_reg", "shuf_lds"):
        if not shuf:
            return []
        if pool != "shuf" and _shuf_arm(k_inst) != pool.split("_")[1]:
            return []
    elif shuf:
        return []
    if pool == "preb" and not k_inst.needs_preshuffled_b:
        return []
    if pool == "rowb" and k_inst.needs_preshuffled_b:
        return []
    if pool == "subok":
        if not k_inst.needs_preshuffled_b:
            return []
        if _shuf_sub(k_inst) is None:
            return []
    if n % k_inst.B_N or k % k_inst.B_K or m % k_inst.m_align:
        return []
    if k_inst.GROUP_N == 1:
        tiles = k // k_inst.B_K
        return [
            s
            for s in (split_ks if split_ks is not None else _TUNE_POLICY[kid])
            if s >= 1 and tiles - (s - 1) * ((tiles + s - 1) // s) >= 1
        ]
    if k_inst.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_compact":
        if not (
            0 < m < 2048
            and 0 < g <= 16
            and 0 < n <= (1 << 31) - 1
            and 0 < k <= (1 << 31) - 1
        ):
            return []
        if n % 128 or max(m * g * k, n * k, m * g * n * 2) > (1 << 31) - 1:
            return []
        return [1] if split_ks is None or 1 in split_ks else []
    if split_ks is not None:
        return split_ks
    return _SHUF_POLICY[kid] if shuf else _TUNE_POLICY[kid]


SHIPPED_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv",
)
DEFAULT_OUT = os.path.join(_REPO, "dsv4_bmm_mxscale_retuned.csv")
# The w_scale block every OPUS MXFP8 BMM kernel reads (a tuned-CSV key column).
W_SCALE_BLOCK = "128x128"

# Preshuffled-weight model configurations use a separate CSV.
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
def _gen_varied(shape, k, device, group=GROUP):
    """Signed, per-128-K-block varied-magnitude bf16 (mirrors _block_varied)."""
    x = torch.randn(shape, dtype=dtypes.fp32, device=device)
    amp = torch.exp2(
        torch.randint(-4, 4, (*shape[:-1], k // group), device=device).float()
    )
    return (x * amp.repeat_interleave(group, dim=-1)).to(dtypes.bf16)


def _workspace_numel(kernel_id, split_k, batch, m, n):
    instance = _CODEGEN_BMM[int(kernel_id)]
    if split_k <= 1 or instance.kernel_tag not in {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
        "a8w8_mxscale_bmm_bpreshuffle_bdirect",
        "a8w8_mxscale_bmm_bpreshuffle_blds",
        "a8w8_mxscale_bmm_bpreshuffle_wave1",
        # The wavetm1 kids emit the same split-K branch as every other family in
        # this list -- the impl stub AITER_CHECKs for the workspace -- but the
        # tags were never added here, so asking one for split_k > 1 raised
        # instead of running. It went unnoticed because the family's tiles were
        # all B_M >= 64 large-M kids that never wanted a split.
        "a8w8_mxscale_bmm_bpreshuffle_wavetm1",
        "a8w8_mxscale_bmm_bpreshuffle_wavetm1_blds",
    }:
        return 0
    tiles_m = (m + instance.B_M - 1) // instance.B_M
    tiles_n = (n + instance.B_N - 1) // instance.B_N
    partial_numel = split_k * batch * tiles_m * instance.B_M * tiles_n * instance.B_N
    if instance.kernel_tag != "a8w8_mxscale_bmm_fused":
        return partial_numel
    counter_offset = (partial_numel * 4 + 255) & ~255
    counter_bytes = batch * tiles_m * tiles_n * 4
    return (counter_offset + counter_bytes + 3) // 4


def gen_bmm_mxscale_data(
    batch, m, n, k, seed, out_dtype, kernel_id, split_k, device="cuda"
):
    """Generate operands and reference for mp_tuner; q is the kernel scale group.

    0 A_mx    [m,g,k]       contiguous token-major FP8 input
    1 W_mx    [g,n,k]       row-major FP8 weight
    2 Y       [m,g,n]       contiguous output buffer
    3 A_scale [m,g,k/q]     E8M0 activation scales
    4 ws_mx   [g,n/q,k/q]   E8M0 weight scales
    5 workspace            caller-owned FP32 split-K buffer, or None
    6 ref     [m,g,n]       reference from dequantized FP32 operands
    7 W_sh    [g,n,k]       weight in the (16,16) preshuffled layout
    8 xs_sh                shuffled A scales for group128
    9 ws_sh                shuffled B scales for group128

    Prepare layouts and workspace outside the timed region. Pass all operands
    as benchmark arguments so run_perftest can rotate their buffers."""
    torch.manual_seed(seed)
    group_n, group = _kid_groups(kernel_id)
    O_bf16 = _gen_varied((batch, m, k), k, device, group)
    W_bf16 = _gen_varied((batch, n, k), k, device, group)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16, group=group)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16, group=group, group_n=group_n)
    Y = torch.empty((m, batch, n), dtype=out_dtype, device=device)

    workspace_numel = _workspace_numel(kernel_id, split_k, batch, m, n)
    workspace = (
        torch.empty(workspace_numel, dtype=torch.float32, device=device)
        if workspace_numel
        else None
    )
    ref = (
        run_torch(O_mx, W_mx, xs_fp32, ws_fp32, group=group, group_n=group_n)
        .transpose(0, 1)
        .to(out_dtype)
    )
    # The preshuffled-B kids read B through the (16,16) MFMA-fragment layout that
    # a serving stack bakes into the weight offline. It is a permutation, so the
    # reference above covers both forms. Building it here rather than in the bench
    # keeps it out of what is timed.
    W_sh = shuffle_weight(W_mx, layout=(16, 16))
    # The shuffled A-scale view has stride(0)=0: kernel addressing uses only
    # its per-batch slab. Buffer rotation preserves that view's storage/strides.
    # This packed layout is defined only for group128; group32 candidates keep
    # ordinary scales in these tuple slots.
    if group == 128:
        _slab = shuffle_scale_a(xs_mx, k, SHUF_SUB)
        xs_sh = _slab.as_strided((m, batch, _slab.shape[1]), (0, _slab.shape[1], 1))
        # Kept 3-D with the N-block axis in the middle so stride(0) is the
        # per-batch slab, the only term the shuffled path reads.
        ws_sh = shuffle_scale_b(ws_mx, n, k).view(batch, n // 128, -1)
    else:
        xs_sh, ws_sh = xs_mx, ws_mx
    # Materialize token-major A and scales after reference/shuffle preparation.
    # Both backends receive the same contiguous layout, with copies outside timing.
    A_mx = O_mx.transpose(0, 1).contiguous()
    A_scale = xs_mx.transpose(0, 1).contiguous()
    return (A_mx, W_mx, Y, A_scale, ws_mx, workspace, ref, W_sh, xs_sh, ws_sh)


def run_bmm_mxscale_bench(
    A_mx, W_mx, Y, A_scale, ws_mx, workspace, W_sh, xs_sh, ws_sh, kernelId, splitK
):
    """Select the kernel's weight/scale layouts, launch it, and return Y.

    All layouts and any split-K workspace are prepared by the data generator;
    this function performs no layout conversion or workspace allocation."""
    inst = _CODEGEN_BMM[kernelId]
    Wb = W_sh if inst.needs_preshuffled_b else W_mx
    # A, its plain scale and Y all arrive [M, batch, ...] contiguous, which is
    # both what the launcher takes and what a serving stack holds. The shuffled
    # scale is the exception: shuffle_scale_a builds its slab batch-first, so
    # that one is still flipped into place here.
    if inst.needs_shuffle_scale:
        sfa, sfb = xs_sh.transpose(0, 1), ws_sh
    else:
        sfa, sfb = A_scale, ws_mx
    _opus_gemm_a8w8_mxscale_bmm_launch_raw(
        A_mx,
        Wb,
        Y,
        sfa,
        sfb,
        workspace=workspace,
        kid=kernelId,
        split_k=splitK,
    )
    return Y


def _bmm_ref_passthrough(ref):
    """ref_func: the fp32 reference is precomputed in gen_data (slot 6)."""
    return ref


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------
BMM_BENCH_KEYS = (0, 1, 2, 3, 4, 5, 7, 8, 9)
BMM_DATA_KIDS = {128: 8179, 32: 9179}
BMM_BLOCK_DATA_KIDS = {"128x128": 8179, "32x32": 9179, "1x32": 9704}


def make_bmm_tuning_task(
    info, generate, gen_args, run, bench_args, perf_kwargs, *, ref_index, output_index
):
    """Shared mp_tuner contract: rotate inputs and NaN-initialize output."""
    return (
        info,
        generate,
        gen_args,
        run,
        bench_args,
        perf_kwargs,
        _bmm_ref_passthrough,
        ([ref_index],),
        {},
        None,
        1e-2,
        1e-2,
        None,
        None,
        [output_index],
    )


class OpusBmmMxscaleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": "",
        "untune_file": "",
        # Fraction-of-mismatch (rtol=atol=1e-2) accept threshold. Correct kids
        # sit at the ~1e-4 fp8 e8m0 quant floor; a column-transposed kid is ~0.5.
        "errRatio": 0.02,
        "batch": 100,
        "config_env_name": "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
    }

    KEYS: ClassVar[list[str]] = ["gfx", "b", "m", "n", "k", "w_scale_block"]
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
            type(self).__name__,
            self.KEYS,
            self.RESULTS,
            description="Tune opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a)",
        )
        # sort N before M like the GEMM tuners (cosmetic ordering of the CSV),
        # then w_scale_block, so a shape's 32x32 and 128x128 rows land together.
        self.sort_keys = ["gfx", "b", "n", "m", "k", "w_scale_block"]

    # --- schema helpers -----------------------------------------------------
    def getKernelName(self, kernelId):
        k_inst = _CODEGEN_BMM.get(int(kernelId))
        return k_inst.name if k_inst else None

    def calculate(self, results, bpes=None):
        info, time, _err = results
        if time == self.INVALID_TIME:
            return 0, 0
        shape = dict(zip(self.keys, info[0]))
        b, m, n, k = (int(shape[name]) for name in ("b", "m", "n", "k"))
        us_s = time * 1e-6
        tflops = round(2 * b * m * n * k / us_s / 1e12, 1)
        # fp8 A + fp8 W + bf16 out.
        bw = round((b * m * k + b * n * k + 2 * b * m * n) / us_s / 1e9, 2)
        return tflops, bw

    def result_to_df(self, results):
        rows = []
        for info, time, err in results:
            keys, kernelId, splitK, kernelName = info
            # Invalid timings must not win the per-shape ranking.
            if not (time > 0):
                logger.warning(
                    "%s: kid %s timed at %s, not ranking it", keys, kernelId, time
                )
                time = float("inf")
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
        self.parser.add_argument(
            "--input_file",
            dest="untune_file",
            default=argparse.SUPPRESS,
            help="Input CSV (alias for -i)",
        )
        self.parser.add_argument(
            "--tuned_file",
            dest="tune_file",
            default=argparse.SUPPRESS,
            help="Output CSV (alias for -o)",
        )
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

        def _blocklist(s):
            blocks = tuple(dict.fromkeys(part.strip().lower() for part in s.split(",")))
            allowed = ("1x32", "32x32", "128x128")
            if any(block not in allowed for block in blocks):
                raise argparse.ArgumentTypeError(
                    "expected comma-separated blocks from " + ", ".join(allowed)
                )
            return blocks

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
            "--w_scale_block",
            type=_blocklist,
            default=None,
            help="comma-separated weight-scale blocks (1x32,32x32,128x128); "
            "filter CSV rows or expand shape-only inputs",
        )
        self.parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="retune the selected shipped table in place (alias for -o TABLE --all)",
        )
        self.parser.add_argument(
            "--bpreshuffle",
            action="store_true",
            default=False,
            help="use preshuffled-B candidates and the preshuffle table by default",
        )
        self.parser.add_argument(
            "--pool",
            choices=POOLS,
            default="all",
            help="restrict candidates by B layout (--bpreshuffle implies preb)",
        )
        self.parser.add_argument(
            "--graph_m_max",
            type=int,
            default=0,
            help="use graph timing for M <= this value (0 disables it)",
        )

    def get_arg_defaults(self):
        defaults = super().get_arg_defaults()
        if getattr(self, "_bpreshuffle", False):
            defaults["config_env_name"] = (
                "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE"
            )
        return defaults

    def _normalize_rows(self, df, *, default_blocks=("128x128",), blocks=None):
        """Preserve CSV shape/scale keys; expand scales only when unspecified."""
        aliases = {name: name for name in self.keys}
        aliases["g"] = "b"
        df = df.rename(
            columns={c: aliases.get(str(c).strip().lower(), c) for c in df.columns}
        ).copy()
        if df.columns.duplicated().any():
            raise ValueError("MXFP8 BMM CSV has duplicate shape columns")
        missing = {"b", "m", "n", "k"}.difference(df.columns)
        if missing:
            raise ValueError(f"MXFP8 BMM CSV is missing columns: {sorted(missing)}")
        for name in ("b", "m", "n", "k"):
            values = pd.to_numeric(df[name], errors="raise")
            if (values.isna() | (values <= 0) | (values % 1 != 0)).any():
                raise ValueError(f"{name} must contain positive integers")
            df[name] = values.astype("int64")
        if "gfx" not in df:
            df["gfx"] = self.get_gfx()
        if "w_scale_block" not in df:
            df = pd.concat(
                [
                    df.assign(w_scale_block=block)
                    for block in (blocks or default_blocks)
                ],
                ignore_index=True,
            )
        df["w_scale_block"] = df["w_scale_block"].astype(str).str.strip().str.lower()
        if not df["w_scale_block"].isin(BMM_BLOCK_DATA_KIDS).all():
            raise ValueError("w_scale_block must be 1x32, 32x32 or 128x128")
        n_align = df["w_scale_block"].map({"1x32": 16, "32x32": 128, "128x128": 128})
        if ((df["n"] % n_align != 0) | (df["k"] % 128 != 0)).any():
            raise ValueError(
                "MXFP8 BMM requires aligned N and K: N16/K128 for 1x32, N128/K128 otherwise"
            )
        if blocks is not None:
            df = df[df["w_scale_block"].isin(blocks)]
        return df.reset_index(drop=True)

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        return self._normalize_rows(df) if not df.empty else df

    def pre_process(self, args):
        gfx = self.get_gfx()
        if gfx != "gfx950":
            raise RuntimeError(f"MXFP8 BMM tuning is gfx950-only; detected {gfx!r}")
        self._bpreshuffle = args.bpreshuffle or args.pool in (
            "preb",
            "subok",
            "shuf",
            "shuf_reg",
            "shuf_lds",
        )
        if args.bpreshuffle and args.pool == "all":
            args.pool = "preb"
        source = BPRESHUFFLE_CSV if self._bpreshuffle else SHIPPED_CSV
        if args.apply:
            args.tune_file, args.all = source, True
        elif not args.tune_file:
            args.tune_file = source if self._bpreshuffle else DEFAULT_OUT
        if (args.batch_g is None) != (args.M is None):
            raise ValueError("-g/--batch_g and -m/--M must be provided together")
        if args.batch_g is not None:
            df = pd.DataFrame(
                [
                    (b, m, n, k)
                    for b in args.batch_g
                    for m in args.M
                    for n in args.N
                    for k in args.K
                ],
                columns=["b", "m", "n", "k"],
            )
        else:
            path = (
                args.run_config
                if isinstance(args.run_config, str)
                else args.untune_file or source
            )
            df = (
                self.get_tuned_gemm_list(path)
                if isinstance(args.run_config, str)
                else self.get_untuned_gemm_list(path)
            )
        self.untunedf = self._normalize_rows(
            df, default_blocks=("32x32", "128x128"), blocks=args.w_scale_block
        )
        if not self._bpreshuffle and self.untunedf["w_scale_block"].eq("1x32").any():
            raise ValueError("1x32 tuning requires --bpreshuffle")
        self.untunedf = (
            self.untunedf[self.untunedf["gfx"] == gfx][self.keys]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not (args.all or args.run_config or args.compare) and not self.tunedf.empty:
            have = set(self.tunedf[self.keys].apply(tuple, axis=1))
            self.untunedf = self.untunedf[
                ~self.untunedf.apply(tuple, axis=1).isin(have)
            ].reset_index(drop=True)
        self.opus_policy = {
            kid: _TUNE_POLICY.get(kid, _SHUF_POLICY.get(kid)) for kid in _CANDIDATE_KIDS
        }

    def _clear_op_caches(self):
        from aiter.jit.core import AITER_CONFIGS
        from aiter.ops import batched_gemm_op_a8w8
        from aiter.ops.opus import policy

        AITER_CONFIGS.get_config_file.cache_clear()
        policy._load_mxscale_bmm_tuned.cache_clear()
        policy.lookup_mxscale_bmm_config.cache_clear()
        batched_gemm_op_a8w8._get_mxscale_bmm_launch_plan.cache_clear()

    def _set_config_env_for_run_config(self, args, config_file=None):
        # OPUS dispatches exact compiled kids; changing the CSV needs no rebuild.
        from aiter.jit import core

        env_name = self.get_arg_defaults()["config_env_name"]
        previous = os.environ.get(env_name)
        os.environ[env_name] = str(config_file or self.get_out_file(args.tune_file))
        self._clear_op_caches()
        return previous, core.AITER_REBUILD

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        self._clear_op_caches()

    def _perf_kwargs(self, args, m):
        return {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
            "testGraph": bool(args.graph_m_max and m <= args.graph_m_max),
        }

    def _saved_benchmark(self, row, seed):
        if str(row["libtype"]).lower() != "opus":
            raise ValueError(
                "The OPUS tuner requires libtype=opus; use the joint preshuffle tuner for FlyDSL"
            )
        kid = int(row["kernelId"])
        if kid not in _CODEGEN_BMM:
            kid = bmm_mxscale_global_kid(kid)
        if kid not in _CODEGEN_BMM:
            raise ValueError(f"Unregistered MXFP8 BMM kid {row['kernelId']}")
        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        split = int(row["splitK"])
        if split not in _applicable(kid, b, m, n, k, split_ks=[split]):
            raise ValueError(
                f"MXFP8 BMM kid {kid} cannot run {(b, m, n, k)} with splitK={split}"
            )
        if row["w_scale_block"] != _kid_block(kid):
            raise ValueError(f"Scale block does not match kid {kid}")
        data = gen_bmm_mxscale_data(b, m, n, k, seed, dtypes.bf16, kid, split)
        data[2].fill_(float("nan"))
        return (
            run_bmm_mxscale_bench,
            tuple(data[i] for i in BMM_BENCH_KEYS) + (kid, split),
            data[6],
        )

    def _default_benchmark(self, row, seed):
        from aiter.ops.batched_gemm_op_a8w8 import (
            batched_gemm_a8w8_mxscale,
            batched_gemm_a8w8_mxscale_bpreshuffle,
        )

        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        data = gen_bmm_mxscale_data(
            b, m, n, k, seed, dtypes.bf16, BMM_BLOCK_DATA_KIDS[row["w_scale_block"]], 1
        )
        run = (
            batched_gemm_a8w8_mxscale_bpreshuffle
            if self._bpreshuffle
            else batched_gemm_a8w8_mxscale
        )
        weight = data[7] if self._bpreshuffle else data[1]
        return run, (data[0], weight, data[3], data[4]), data[6]

    def run_config(self, args):
        from aiter.test_common import checkAllclose, run_perftest

        # The shared run() reloads saved configs after pre_process. Reapply
        # the requested scale filter before reporting standalone benchmarks.
        if isinstance(args.run_config, str) and args.w_scale_block is not None:
            self.untunedf = self._normalize_rows(
                self.untunedf, blocks=args.w_scale_block
            )
        required = {"libtype", "kernelId", "splitK"}
        present = required.intersection(self.untunedf.columns)
        if present and present != required:
            raise ValueError(
                f"Saved BMM config is missing {sorted(required - present)}"
            )
        results = []
        for seed, (_, row) in enumerate(self.untunedf.iterrows(), start=1):
            shape = ",".join(f"{name}={row[name]}" for name in self.keys)
            allowed, description = self._get_run_config_err_ratio_limit(row, args)
            try:
                prepare = self._saved_benchmark if present else self._default_benchmark
                run, operands, ref = prepare(row, seed)
                out, us = run_perftest(
                    run, *operands, **self._perf_kwargs(args, int(row["m"]))
                )
                err = checkAllclose(
                    out,
                    ref,
                    rtol=1e-2,
                    atol=1e-2,
                    tol_err_ratio=allowed,
                    msg=f"run_config {shape}",
                    printLog=args.verbose,
                )
                if (
                    not math.isfinite(us)
                    or us <= 0
                    or not math.isfinite(err)
                    or err > allowed
                ):
                    raise RuntimeError(
                        f"us={us}, errRatio={err}, allowed={description}"
                    )
                results.append(
                    {"shape": shape, "e2e_us": us, "errRatio": err, "status": "ok"}
                )
            except Exception as exc:  # noqa: BLE001
                results.append({"shape": shape, "e2e_us": -1, "status": f"error:{exc}"})
        return results

    def _iter_opus_tasks(self, row, seed, args):
        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        for kid, splits in self.opus_policy.items():
            if _kid_block(kid) != row["w_scale_block"]:
                continue
            for split in _applicable(kid, b, m, n, k, args.pool, split_ks=splits):
                yield make_bmm_tuning_task(
                    (tuple(row[name] for name in self.keys), kid, split, ""),
                    gen_bmm_mxscale_data,
                    (b, m, n, k, seed, dtypes.bf16, kid, split),
                    run_bmm_mxscale_bench,
                    (BMM_BENCH_KEYS, kid, split),
                    self._perf_kwargs(args, m),
                    ref_index=6,
                    output_index=2,
                )

    def _iter_tuning_tasks(self, row, seed, args):
        yield from self._iter_opus_tasks(row, seed, args)

    def tune(self, untunedf, tunedf, args):
        tasks, tasks_data = [], []
        for seed, (_, row) in enumerate(untunedf.iterrows(), start=1):
            candidates = list(self._iter_tuning_tasks(row, seed, args))
            if args.verbose:
                logger.info("%s: %d candidates", tuple(row[self.keys]), len(candidates))
            tasks.extend(candidates)
            tasks_data.append((len(candidates), ()))
        if not tasks:
            return []
        return mp_tuner(
            tasks,
            tasks_data,
            mp_num=args.mp,
            shape_grouped=args.shape_grouped,
            err_ratio=args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    tuner = OpusBmmMxscaleTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
