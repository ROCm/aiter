# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import re

_MXMOE_NUMERIC_TOKENS = {
    "SK": "kSplitK",
    "KW": "k_wave",
    "XCD": "xcd_swizzle",
}
_MXMOE_G1_FLAG_TOKENS = {
    "NT",
    "F16IN",
    "HPF",
    "FP8OUT",
    "IL",
    "SITUV2",
    "SWIGLU",
    "BIAS",
    "W2",
}
_MXMOE_G2_FLAG_TOKENS = {"NT", "ATOMIC", "F4OUT", "CSHUFFLE"}
_MXMOE_NUMERIC_RE = re.compile(r"^([A-Z]+)(\d+)$")
_MXMOE_TILE_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")
_MXMOE_PREFIX = {1: "flydsl_mxmoe_g1_a4w4_", 2: "flydsl_mxmoe_g2_a4w4_"}
_MXMOE_G1_PREFIX_RE = re.compile(r"^flydsl_mxmoe_g1_a(?P<a>[48])w4_")
MXFP4_G1_VARIANTS = {
    "fp4": {
        (32, True, False),
        (32, False, False),
        (64, True, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
    },
    "fp8": {
        (32, True, False),
        (32, False, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
    },
}


def native_scale_layout_for(BM: int, out_dtype: str) -> bool:
    """The A-scale layout GEMM1 must emit for a block_m and output dtype.

    This is a GEMM1/GEMM2 contract, not a tuning knob: BM16 FP4 output writes
    the native scale layout and the matching GEMM2 reads it back. FP8 output
    uses the regular scale layout. Every caller must agree, so the rule lives
    here.
    """
    return int(BM) == 16 and str(out_dtype).lower() == "fp4"


_FLYDSL_V2_GEMM2_RE = re.compile(
    r"^flydsl_moe2_layout_a(?P<a>\w+?)_w(?P<b>\w+?)_(?P<out>\w+?)_"
    r"t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)_(?P<epilog>atomic|reduce)"
    r"(?P<persist>_persist)?(?P<nt>_nt)?(?:_sbm(?P<sbm>\d+))?"
    r"(?P<bf16lds>_bf16lds)?(?:_sp(?P<sp>\d+))?$"
)


def _tokenize_mxfp4_kname(kname: str, stage: int, flag_tokens: set) -> dict:
    kname = (kname or "").replace("_FLYDSL", "")
    mode = {}
    if stage == 1:
        prefix_match = _MXMOE_G1_PREFIX_RE.match(kname)
        pfx = prefix_match.group(0) if prefix_match else ""
        if prefix_match:
            mode["a_dtype"] = "fp8" if prefix_match.group("a") == "8" else "fp4"
    else:
        pfx = _MXMOE_PREFIX[stage]
    if not pfx or not kname.startswith(pfx):
        raise ValueError(f"bad mxmoe kernel name: {kname!r} (expected prefix {pfx!r})")
    nums: dict = {}
    flags: set = set()
    for tok in kname[len(pfx) :].split("_"):
        if not tok:
            continue
        mt = _MXMOE_TILE_RE.match(tok)
        if mt:
            nums["BM"] = int(mt.group(1))
            nums["BN"] = int(mt.group(2))
            nums["BK"] = int(mt.group(3))
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
    return {"nums": nums, "flags": flags, "mode": mode}


def _parse_mxfp4_g1_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, 1, _MXMOE_G1_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    act = "situv2" if "SITUV2" in flags else ("swiglu" if "SWIGLU" in flags else "silu")
    return {
        "BM": nums["BM"],
        "BN": nums["BN"],
        "BK": nums["BK"],
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "inline_quant": "F16IN" in flags,
        "prefetch_hidden": "HPF" in flags,
        "use_nt": "NT" in flags,
        "xcd_swizzle": nums.get("xcd_swizzle", 0),
        "a_dtype": parsed["mode"].get("a_dtype", "fp4"),
        "out_dtype": "fp8" if "FP8OUT" in flags else "fp4",
        "interleave": "IL" in flags,
        "act": act,
        "enable_bias": "BIAS" in flags,
        "num_waves": 2 if "W2" in flags else 4,
        "k_wave": nums.get("k_wave", 1),
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
        "BN": nums["BN"],
        "BK": nums["BK"],
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
        "bf16_lds": True if m.group("bf16lds") else None,
        "spart": int(m.group("sp")) if m.group("sp") else None,
    }


def parse_g2_kname_any(kname) -> dict:
    """Parse either gemm2 name family into the fields the stage2 dispatch needs.

    ``v2`` tells path B (flydsl_moe2_layout gemm2 behind the mxmoe front-end)
    apart from the native mxmoe gemm2; the other keys mean the same for both.
    """
    v2 = parse_flydsl_v2_gemm2_kernel(kname)
    if v2 is not None:
        return {
            "v2": True,
            "BM": v2["tile_m"],
            "atomic": v2["epilog"] == "atomic",
            "use_nt": v2["use_nt"],
            "mxfp4out": False,
            "cshuffle": False,
        }
    p2 = _parse_mxfp4_g2_kname(kname)
    return {
        "v2": False,
        "BM": p2["BM"],
        "atomic": p2["atomic"],
        "use_nt": p2["use_nt"],
        "mxfp4out": p2["mxfp4out"],
        "cshuffle": p2["cshuffle"],
    }
