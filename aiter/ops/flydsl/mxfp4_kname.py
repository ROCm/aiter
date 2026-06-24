# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure mxfp4 kernel-name parsing (no torch / enum / JIT deps). Kept in its own
# lightweight module so the AOT pre-compile collection (setup.py -> start_aot ->
# aiter.aot.flydsl.mxfp4_moe.parse_csv) can import these helpers without pulling
# in aiter.fused_moe, whose top-level imports trigger JIT module loads that are
# not yet available during the build.

import re

# FlyDSL a4w4 port names follow the native flydsl_ convention: a lowercase,
# "_"-joined token string led by the tile triple <BM>x<BN>x<BK> (BN=BK=256, fixed
# by the kernel). The problem shape (experts / hidden / inter) is NOT encoded in
# the name -- it lives in the CSV columns -- so the name carries only the tile and
# the variant flags. Parsing walks the tokens: the tile triple sets BM, remaining
# tokens match a numeric registry (e.g. topk9) or a boolean flag set (e.g. nt,
# inlinequant, atomic). Matching is case-insensitive so the legacy
# mxfp4_moe_g{1,2}_a4w4_NE..._BM... names (which embed shape) still parse.
_MXFP4_NUMERIC_TOKENS = {
    # token-prefix -> result-field. Value is int(token[len(prefix):]).
    "TOPK": "TOPK",
    "SK": "kSplitK",
    "XCD": "xcd_swizzle",
    # legacy-only shape tokens; captured for back-compat, unused at runtime.
    "NE": "NE",
    "H": "H",
    "E": "D_INTER",
    "BM": "BM",  # legacy BM token; new names carry BM in the tile triple
}
# Flag tokens shared/!specific to a stage. Each present token sets its field True.
_MXFP4_G1_FLAG_TOKENS = {"NT", "CACHED", "INLINEQUANT"}
_MXFP4_G2_FLAG_TOKENS = {"NT", "ATOMIC", "NONATOMIC", "MXFP4OUT", "CSHUFFLE"}
_MXFP4_NUMERIC_RE = re.compile(r"^([A-Z]+)(\d+)$")
_MXFP4_TILE_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")  # <BM>x<BN>x<BK>

# (canonical flydsl_ prefix, legacy alias) per stage.
_MXFP4_PREFIXES = {
    1: ("flydsl_mxfp4_g1_a4w4_", "mxfp4_moe_g1_a4w4_"),
    2: ("flydsl_mxfp4_g2_a4w4_", "mxfp4_moe_g2_a4w4_"),
}


def _tokenize_mxfp4_kname(kname: str, stage: int, flag_tokens: set) -> dict:
    """Split a mxfp4 kernel name into {numeric fields} + {flags present}.

    Strips the ``_FLYDSL`` backend token (routing-only) and the stage prefix
    (FlyDSL ``flydsl_mxfp4_g{1,2}_a4w4_`` or legacy ``mxfp4_moe_g{1,2}_a4w4_``),
    then classifies each remaining token: the ``<BM>x<BN>x<BK>`` tile triple sets
    BM, others match the numeric registry or the boolean flag set (case-insensitive).
    """
    kname = (kname or "").replace("_FLYDSL", "")
    body = None
    for pfx in _MXFP4_PREFIXES[stage]:
        if kname.startswith(pfx):
            body = kname[len(pfx) :]
            break
    if body is None:
        raise ValueError(
            f"bad mxfp4 kernel name: {kname!r} "
            f"(expected prefix in {_MXFP4_PREFIXES[stage]})"
        )
    nums: dict = {}
    flags: set = set()
    for tok in body.split("_"):
        if not tok:
            continue
        mt = _MXFP4_TILE_RE.match(tok)
        if mt:
            nums["BM"] = int(mt.group(1))  # tile = BM x BN(256) x BK(256)
            continue
        utok = tok.upper()
        if utok in flag_tokens:
            flags.add(utok)
            continue
        m = _MXFP4_NUMERIC_RE.match(utok)
        field = _MXFP4_NUMERIC_TOKENS.get(m.group(1)) if m else None
        if field is None:
            raise ValueError(f"bad mxfp4 kernel name {kname!r}: unknown token {tok!r}")
        nums[field] = int(m.group(2))
    return {"nums": nums, "flags": flags}


def _parse_mxfp4_g1_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, 1, _MXFP4_G1_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    inline_quant = "INLINEQUANT" in flags
    if inline_quant:
        # bare _inlinequant = NT (read-once); _inlinequant_cached = cached.
        use_nt = "CACHED" not in flags
    else:
        use_nt = "NT" in flags  # BM=32 cshuffle: _nt vs _cached
    # NE / H / D_INTER are not encoded in FlyDSL names (they come from the CSV
    # shape columns); .get() keeps legacy shape-bearing names working.
    return {
        "BM": nums["BM"],
        "NE": nums.get("NE"),
        "H": nums.get("H"),
        "D_INTER": nums.get("D_INTER"),
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "inline_quant": inline_quant,
        "use_nt": use_nt,
    }


def _parse_mxfp4_g2_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, 2, _MXFP4_G2_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    atomic = "ATOMIC" in flags
    mxfp4out = "MXFP4OUT" in flags
    cshuffle = "CSHUFFLE" in flags
    # _mxfp4out / _cshuffle are nonatomic-only epilogs: atomic accumulates straight
    # into the (M, D_HIDDEN) output buffer, while these stage flat_out at max_sorted
    # rows. Reject the contradiction -- a malformed name like ..._atomic_cshuffle
    # would size the buffer for atomic but run the nonatomic epilog, an OOB write.
    if atomic and (mxfp4out or cshuffle):
        bad = "mxfp4out" if mxfp4out else "cshuffle"
        raise ValueError(
            f"illegal mxfp4 g2 kernel name {kname!r}: atomic is incompatible with "
            f"{bad} (nonatomic-only epilog)"
        )
    return {
        "BM": nums["BM"],
        "NE": nums.get("NE"),
        "H": nums.get("H"),
        "D_INTER": nums.get("D_INTER"),
        "TOPK": nums.get("TOPK"),
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "atomic": atomic,
        "use_nt": "NT" in flags,  # non-temporal B load (atomic only)
        # _mxfp4out (nonatomic only): gemm2 stages flat_out as packed fp4+e8m0 and
        # scatter_reduce reads it back as mxfp4 (the mxfp4-intermediate path).
        "mxfp4out": mxfp4out,
        "cshuffle": cshuffle,
    }


def _is_mxfp4_kname(kname: str) -> bool:
    return bool(kname) and (
        kname.startswith("flydsl_mxfp4_g") or kname.startswith("mxfp4_moe_")  # legacy
    )
