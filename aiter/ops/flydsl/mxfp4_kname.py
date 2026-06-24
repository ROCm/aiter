# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure mxmoe kernel-name parsing (no torch / JIT deps) so the AOT pre-compile
# pass can import it without triggering JIT module loads.
#
# Name: flydsl_mxmoe_g{1,2}_a4w4_<BM>x256x256[_flag...], lowercase. Shape is in
# the CSV columns, not the name. g1 flags: f16in (inline act quant), nt (else
# cached). g2 flags: atomic (else nonatomic), nt (atomic only), f4out / cshuffle.

import re

_MXMOE_NUMERIC_TOKENS = {"SK": "kSplitK", "XCD": "xcd_swizzle"}
_MXMOE_G1_FLAG_TOKENS = {"NT", "F16IN"}
_MXMOE_G2_FLAG_TOKENS = {"NT", "ATOMIC", "F4OUT", "CSHUFFLE"}
_MXMOE_NUMERIC_RE = re.compile(r"^([A-Z]+)(\d+)$")
_MXMOE_TILE_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")  # <BM>x<BN>x<BK>
_MXMOE_PREFIX = {1: "flydsl_mxmoe_g1_a4w4_", 2: "flydsl_mxmoe_g2_a4w4_"}


def _tokenize_mxfp4_kname(kname: str, stage: int, flag_tokens: set) -> dict:
    kname = (kname or "").replace("_FLYDSL", "")
    pfx = _MXMOE_PREFIX[stage]
    if not kname.startswith(pfx):
        raise ValueError(f"bad mxmoe kernel name: {kname!r} (expected prefix {pfx!r})")
    nums: dict = {}
    flags: set = set()
    for tok in kname[len(pfx) :].split("_"):
        if not tok:
            continue
        mt = _MXMOE_TILE_RE.match(tok)
        if mt:
            nums["BM"] = int(mt.group(1))
            continue
        utok = tok.upper()
        if utok in flag_tokens:
            flags.add(utok)
            continue
        m = _MXMOE_NUMERIC_RE.match(utok)
        field = _MXMOE_NUMERIC_TOKENS.get(m.group(1)) if m else None
        if field is None:
            raise ValueError(f"bad mxmoe kernel name {kname!r}: unknown token {tok!r}")
        nums[field] = int(m.group(2))
    return {"nums": nums, "flags": flags}


def _parse_mxfp4_g1_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, 1, _MXMOE_G1_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    return {
        "BM": nums["BM"],
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "inline_quant": "F16IN" in flags,
        "use_nt": "NT" in flags,
    }


def _parse_mxfp4_g2_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, 2, _MXMOE_G2_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    atomic = "ATOMIC" in flags
    mxfp4out = "F4OUT" in flags
    cshuffle = "CSHUFFLE" in flags
    # f4out/cshuffle are nonatomic-only; atomic sizes a different output buffer.
    if atomic and (mxfp4out or cshuffle):
        bad = "f4out" if mxfp4out else "cshuffle"
        raise ValueError(f"illegal mxmoe g2 name {kname!r}: atomic incompatible with {bad}")
    return {
        "BM": nums["BM"],
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "atomic": atomic,
        "use_nt": "NT" in flags,
        "mxfp4out": mxfp4out,
        "cshuffle": cshuffle,
    }


def _is_mxfp4_kname(kname: str) -> bool:
    return bool(kname) and kname.startswith("flydsl_mxmoe_g")
