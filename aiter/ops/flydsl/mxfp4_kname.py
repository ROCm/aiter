# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure mxfp4 kernel-name parsing (no torch / enum / JIT deps). Kept in its own
# lightweight module so the AOT pre-compile collection (setup.py -> start_aot ->
# aiter.aot.flydsl.mxfp4_moe.parse_csv) can import these helpers without pulling
# in aiter.fused_moe, whose top-level imports trigger JIT module loads that are
# not yet available during the build.

import re

# mxfp4 kernel names are token registries, not a regex grammar: a name is
# "_"-joined tokens emitted by gen_instances.py. Parsing walks the tokens and
# matches each against two registries -- numeric fields (LETTERS+digits, e.g.
# NE385, BM32, XCD2) and boolean flag tokens (e.g. NT, INLINEQUANT, ATOMIC).
# Adding a new variant = add one registry entry, no grammar to re-derive.
# Keep these in sync with gen_instances.py (enumerate_g{1,2}_instances).
_MXFP4_NUMERIC_TOKENS = {
    # token-prefix -> result-field. Value is int(token[len(prefix):]).
    "NE": "NE",
    "H": "H",
    "E": "D_INTER",  # historical "E" tag = per-shard inter_dim, NOT expert count
    "BM": "BM",
    "TOPK": "TOPK",
    "SK": "kSplitK",
    "XCD": "xcd_swizzle",
}
# Flag tokens shared/!specific to a stage. Each present token sets its field True.
_MXFP4_G1_FLAG_TOKENS = {"NT", "CACHED", "INLINEQUANT"}
_MXFP4_G2_FLAG_TOKENS = {"NT", "ATOMIC", "NONATOMIC", "MXFP4OUT", "CSHUFFLE"}
_MXFP4_NUMERIC_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _tokenize_mxfp4_kname(kname: str, prefix: str, flag_tokens: set) -> dict:
    """Split a mxfp4 kernel name into {numeric fields} + {flags present}.

    Strips the `_FLYDSL` backend token (routing-only) and the fixed
    ``mxfp4_moe_g{1,2}_a4w4`` prefix, then classifies each remaining token as a
    numeric field (via _MXFP4_NUMERIC_TOKENS) or a boolean flag (via flag_tokens).
    """
    kname = (kname or "").replace("_FLYDSL", "")
    if not kname.startswith(prefix):
        raise ValueError(
            f"bad mxfp4 kernel name: {kname!r} (expected prefix {prefix!r})"
        )
    nums: dict = {}
    flags: set = set()
    for tok in kname[len(prefix) :].split("_"):
        if not tok:
            continue
        if tok in flag_tokens:
            flags.add(tok)
            continue
        m = _MXFP4_NUMERIC_RE.match(tok)
        field = _MXFP4_NUMERIC_TOKENS.get(m.group(1)) if m else None
        if field is None:
            raise ValueError(f"bad mxfp4 kernel name {kname!r}: unknown token {tok!r}")
        nums[field] = int(m.group(2))
    return {"nums": nums, "flags": flags}


def _parse_mxfp4_g1_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, "mxfp4_moe_g1_a4w4_", _MXFP4_G1_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    inline_quant = "INLINEQUANT" in flags
    if inline_quant:
        # bare _INLINEQUANT = NT (read-once); _INLINEQUANT_CACHED = cached.
        use_nt = "CACHED" not in flags
    else:
        use_nt = "NT" in flags  # BM=32 cshuffle: _NT vs _CACHED
    return {
        "BM": nums["BM"],
        "NE": nums["NE"],
        "H": nums["H"],
        "D_INTER": nums["D_INTER"],
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "inline_quant": inline_quant,
        "use_nt": use_nt,
    }


def _parse_mxfp4_g2_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, "mxfp4_moe_g2_a4w4_", _MXFP4_G2_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    atomic = "ATOMIC" in flags
    mxfp4out = "MXFP4OUT" in flags
    cshuffle = "CSHUFFLE" in flags
    # _MXFP4OUT / _CSHUFFLE are nonatomic-only epilogs: atomic accumulates straight
    # into the (M, D_HIDDEN) output buffer, while these stage flat_out at max_sorted
    # rows. Reject the contradiction -- a malformed CSV row like ..._ATOMIC_CSHUFFLE
    # would size the buffer for atomic but run the nonatomic epilog, an OOB write.
    if atomic and (mxfp4out or cshuffle):
        bad = "MXFP4OUT" if mxfp4out else "CSHUFFLE"
        raise ValueError(
            f"illegal mxfp4 g2 kernel name {kname!r}: ATOMIC is incompatible with "
            f"{bad} (nonatomic-only epilog)"
        )
    return {
        "BM": nums["BM"],
        "NE": nums["NE"],
        "H": nums["H"],
        "D_INTER": nums["D_INTER"],
        "TOPK": nums.get("TOPK"),
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "atomic": atomic,
        "use_nt": "NT" in flags,  # non-temporal B load (atomic only)
        # _MXFP4OUT (nonatomic only): gemm2 stages flat_out as packed fp4+e8m0 and
        # scatter_reduce reads it back as mxfp4 (the mxfp4-intermediate path).
        "mxfp4out": mxfp4out,
        "cshuffle": cshuffle,
    }


def _is_mxfp4_kname(kname: str) -> bool:
    return bool(kname) and kname.startswith("mxfp4_moe_")
