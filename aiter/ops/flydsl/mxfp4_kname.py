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


def _select_mxfp4_a4w4_kernels(*, token: int, expert: int, topk: int) -> dict:
    """Select the canonical MXFP4 GEMM1/GEMM2 pair for a routed-M shape."""
    routed_rows = int(token) * int(topk)
    expert = int(expert)
    average_rows = (routed_rows + expert - 1) // expert

    # BM16's fused inline quantization has excessive error for a single token.
    # Use the prequantized BM32 path for this decode corner case.
    if int(token) == 1:
        block_m = 32
    elif int(token) <= 128:
        block_m = 16
    elif average_rows <= 32:
        block_m = 32
    elif average_rows <= 64:
        block_m = 64
    else:
        block_m = 128

    block_n = 256
    total_m_blocks = (routed_rows + block_m - 1) // block_m
    use_nt = block_m in (16, 32, 64) and total_m_blocks < expert
    xcd_swizzle = 2 if block_m == 64 and use_nt else 0

    g1 = f"{_MXMOE_PREFIX[1]}{block_m}x{block_n}x256"
    if block_m == 16:
        g1 += "_f16in_nt"
    elif use_nt:
        g1 += "_nt"
    if xcd_swizzle:
        g1 += f"_xcd{xcd_swizzle}"

    g2 = f"flydsl_moe2_afp4_wfp4_bf16_t{block_m}x128x256_reduce"

    return {"BM": block_m, "kernelName1": g1, "kernelName2": g2}


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
    return {"nums": nums, "flags": flags}


def _parse_mxfp4_g1_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, 1, _MXMOE_G1_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    return {
        "BM": nums["BM"],
        "BN": nums["BN"],
        "BK": nums["BK"],
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
