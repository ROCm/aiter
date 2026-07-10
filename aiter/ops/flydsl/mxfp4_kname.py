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
_FLYDSL_V2_GEMM2_RE = re.compile(
    r"^flydslv2_moe2_a(?P<a>\w+?)_w(?P<b>\w+?)_(?P<out>\w+?)_"
    r"t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)_(?P<epilog>atomic|reduce)"
    r"(?P<persist>_persist)?(?P<nt>_nt)?(?:_sbm(?P<sbm>\d+))?$"
)


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
        "xcd_swizzle": nums.get("xcd_swizzle", 0),
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
        raise ValueError(
            f"illegal mxmoe g2 name {kname!r}: atomic incompatible with {bad}"
        )
    return {
        "BM": nums["BM"],
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "atomic": atomic,
        "use_nt": "NT" in flags,
        "mxfp4out": mxfp4out,
        "cshuffle": cshuffle,
        "xcd_swizzle": nums.get("xcd_swizzle", 0),
    }


def _is_mxfp4_kname(kname) -> bool:
    # CSV tune files leave kernelName empty for 1-stage configs; pandas loads
    # those cells as float('nan'), and bool(nan) is True, so guard on str type.
    return isinstance(kname, str) and kname.startswith("flydsl_mxmoe_g")


def parse_flydsl_v2_gemm2_kernel(name):
    m = _FLYDSL_V2_GEMM2_RE.match(name or "")
    if not m:
        return None
    return {
        "a_dtype": m.group("a"),
        "b_dtype": m.group("b"),
        "out_dtype": m.group("out"),
        "tile_m": int(m.group("tm")),
        "tile_n": int(m.group("tn")),
        "tile_k": int(m.group("tk")),
        "epilog": m.group("epilog"),
        "persist": bool(m.group("persist")),
        "use_nt": bool(m.group("nt")),
        "sort_block_m": int(m.group("sbm")) if m.group("sbm") else 0,
    }
