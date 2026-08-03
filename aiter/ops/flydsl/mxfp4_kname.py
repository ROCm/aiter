# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure mxmoe kernel-name parsing (no torch / JIT deps) so the AOT pre-compile
# pass can import it without triggering JIT module loads.
#
# Legacy name: flydsl_mxmoe_g{1,2}_a4w4_<BM>x256x256[_flag...], lowercase.
# New GEMM1 names use an a8w4 prefix plus fp8out/situv2 flags. SiTUv2 beta
# values are encoded by their exact IEEE-754 bits so kernel names, AOT jobs,
# and FlyDSL cache entries cannot alias after lossy decimal formatting.

import math
import re
import struct

_MXMOE_NUMERIC_TOKENS = {"SK": "kSplitK", "XCD": "xcd_swizzle"}
_MXMOE_G1_FLAG_TOKENS = {"NT", "F16IN", "FP8OUT", "IL", "SITUV2"}
_MXMOE_G2_FLAG_TOKENS = {"NT", "ATOMIC", "F4OUT", "CSHUFFLE"}
_MXMOE_NUMERIC_RE = re.compile(r"^([A-Z]+)(\d+)$")
_MXMOE_FLOAT_RE = re.compile(r"^(SB|SLB)([0-9A-F]{16})$")
_MXMOE_TILE_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")  # <BM>x<BN>x<BK>
_MXMOE_PREFIX = {1: "flydsl_mxmoe_g1_a4w4_", 2: "flydsl_mxmoe_g2_a4w4_"}
_MXMOE_G1_PREFIX_RE = re.compile(r"^flydsl_mxmoe_g1_a(?P<a>[48])w4_")


def _encode_mxfp4_float(value: float) -> str:
    """Encode a compile-time float without changing its binary value."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"mxmoe compile-time float must be finite, got {value!r}")
    return struct.pack(">d", value).hex()


def _decode_mxfp4_float(bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def _select_mxfp4_block_m(*, token: int, expert: int, topk: int) -> int:
    routed_rows = int(token) * int(topk)
    expert = int(expert)
    average_rows = (routed_rows + expert - 1) // expert

    # BM16's fused inline quantization has excessive error for a single token.
    if int(token) == 1:
        return 32
    if int(token) <= 128:
        return 16
    if average_rows <= 32:
        return 32
    if average_rows <= 64:
        return 64
    return 128


def _make_mxfp4_g1_kname(
    *,
    BM: int,
    BN: int = 256,
    BK: int = 256,
    a_dtype: str = "fp4",
    out_dtype: str = "fp4",
    act: str = "silu",
    inline_quant: bool = False,
    use_nt: bool = False,
    interleave: bool = False,
    kSplitK: int = 0,
    xcd_swizzle: int = 0,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
) -> str:
    """Build a cache-safe GEMM1 name; legacy a4w4 names remain byte-for-byte."""
    a_dtype = str(a_dtype).lower()
    out_dtype = str(out_dtype).lower()
    act = str(act).lower()
    if a_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported mxmoe GEMM1 a_dtype: {a_dtype!r}")
    if out_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported mxmoe GEMM1 out_dtype: {out_dtype!r}")
    if act not in ("silu", "situv2"):
        raise ValueError(f"unsupported mxmoe GEMM1 activation: {act!r}")
    if act == "situv2" and (float(situ_beta) <= 0.0 or float(situ_linear_beta) <= 0.0):
        raise ValueError("SiTUv2 beta values must be positive")

    family = "a8w4" if a_dtype == "fp8" else "a4w4"
    name = f"flydsl_mxmoe_g1_{family}_{int(BM)}x{int(BN)}x{int(BK)}"
    if inline_quant:
        name += "_f16in"
    if use_nt:
        name += "_nt"
    if interleave:
        name += "_il"
    if out_dtype == "fp8":
        name += "_fp8out"
    if act == "situv2":
        name += "_situv2"
    if kSplitK:
        name += f"_sk{int(kSplitK)}"
    if xcd_swizzle:
        name += f"_xcd{int(xcd_swizzle)}"
    if act == "situv2":
        name += (
            f"_sb{_encode_mxfp4_float(situ_beta)}"
            f"_slb{_encode_mxfp4_float(situ_linear_beta)}"
        )
    return name


def _select_mxfp4_g1_kernel(
    *,
    token: int,
    expert: int,
    topk: int,
    block_m: int | None = None,
    BN: int = 256,
    BK: int = 256,
    a_dtype: str = "fp4",
    out_dtype: str = "fp4",
    act: str = "silu",
    interleave: bool = False,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
) -> dict:
    """Select an MXMOE GEMM1 while retaining a tuned block_m when supplied."""
    routed_rows = int(token) * int(topk)
    expert = int(expert)
    block_m = (
        _select_mxfp4_block_m(token=token, expert=expert, topk=topk)
        if block_m is None
        else int(block_m)
    )
    total_m_blocks = (routed_rows + block_m - 1) // block_m
    use_nt = block_m in (16, 32, 64) and total_m_blocks < expert
    # The FP8-input port intentionally has no BM64 non-temporal specialization.
    if a_dtype == "fp8" and block_m == 64:
        use_nt = False
    xcd_swizzle = 2 if block_m == 64 and use_nt else 0
    return {
        "BM": block_m,
        "kernelName1": _make_mxfp4_g1_kname(
            BM=block_m,
            BN=BN,
            BK=BK,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
            act=act,
            inline_quant=block_m == 16,
            use_nt=True if block_m == 16 else use_nt,
            interleave=interleave,
            xcd_swizzle=xcd_swizzle,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        ),
    }


def _select_mxfp4_a4w4_kernels(*, token: int, expert: int, topk: int) -> dict:
    """Select the canonical MXFP4 GEMM1/GEMM2 pair for a routed-M shape."""
    selected = _select_mxfp4_g1_kernel(
        token=token,
        expert=expert,
        topk=topk,
    )
    block_m = selected["BM"]
    g2 = f"flydsl_moe2_afp4_wfp4_bf16_t{block_m}x128x256_reduce"
    return {**selected, "kernelName2": g2}


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
    floats: dict = {}
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
        fm = _MXMOE_FLOAT_RE.match(utok)
        if fm:
            field = "situ_beta" if fm.group(1) == "SB" else "situ_linear_beta"
            floats[field] = _decode_mxfp4_float(fm.group(2))
            continue
        m = _MXMOE_NUMERIC_RE.match(utok)
        field = _MXMOE_NUMERIC_TOKENS.get(m.group(1)) if m else None
        if field is None:
            raise ValueError(f"bad mxmoe kernel name {kname!r}: unknown token {tok!r}")
        nums[field] = int(m.group(2))
    return {"nums": nums, "flags": flags, "floats": floats, "mode": mode}


def _parse_mxfp4_g1_kname(kname: str) -> dict:
    parsed = _tokenize_mxfp4_kname(kname, 1, _MXMOE_G1_FLAG_TOKENS)
    nums, flags = parsed["nums"], parsed["flags"]
    floats = parsed["floats"]
    act = "situv2" if "SITUV2" in flags else "silu"
    if act == "silu" and floats:
        raise ValueError(f"illegal mxmoe GEMM1 name {kname!r}: SiLU has beta tokens")
    if act == "situv2" and set(floats) != {"situ_beta", "situ_linear_beta"}:
        raise ValueError(
            f"illegal mxmoe GEMM1 name {kname!r}: "
            "SiTUv2 requires sb/slb bit encodings"
        )
    if act == "situv2" and any(
        not math.isfinite(value) or value <= 0.0 for value in floats.values()
    ):
        raise ValueError(
            f"illegal mxmoe GEMM1 name {kname!r}: SiTUv2 betas must be finite and positive"
        )
    return {
        "BM": nums["BM"],
        "BN": nums["BN"],
        "BK": nums["BK"],
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "inline_quant": "F16IN" in flags,
        "use_nt": "NT" in flags,
        "xcd_swizzle": nums.get("xcd_swizzle", 0),
        "a_dtype": parsed["mode"].get("a_dtype", "fp4"),
        "out_dtype": "fp8" if "FP8OUT" in flags else "fp4",
        "interleave": "IL" in flags,
        "act": act,
        "situ_beta": floats.get("situ_beta", 1.0),
        "situ_linear_beta": floats.get("situ_linear_beta", 1.0),
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
