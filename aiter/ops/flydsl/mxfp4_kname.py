# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure mxmoe kernel-name parsing (no torch / JIT deps) so the AOT pre-compile
# pass can import it without triggering JIT module loads.
#
# Legacy name: flydsl_mxmoe_g{1,2}_a4w4_<BM>x256x256[_flag...], lowercase.
# New GEMM1 names use activation flags; activation-specific scalar values are
# fixed by the activation contract and remain in FlyDSL's closure cache key.

import re

_MXMOE_NUMERIC_TOKENS = {"SK": "kSplitK", "XCD": "xcd_swizzle"}
_MXMOE_G1_FLAG_TOKENS = {
    "NT",
    "F16IN",
    "FP8OUT",
    "IL",
    "SITUV2",
    "SWIGLU",
    "BIAS",
}
_MXMOE_G2_FLAG_TOKENS = {"NT", "ATOMIC", "F4OUT", "CSHUFFLE"}
_MXMOE_NUMERIC_RE = re.compile(r"^([A-Z]+)(\d+)$")
_MXMOE_TILE_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")  # <BM>x<BN>x<BK>
_MXMOE_PREFIX = {1: "flydsl_mxmoe_g1_a4w4_", 2: "flydsl_mxmoe_g2_a4w4_"}
_MXMOE_G1_PREFIX_RE = re.compile(r"^flydsl_mxmoe_g1_a(?P<a>[48])w4_")
_SITUV2_BETA = 2.0
_SITUV2_LINEAR_BETA = 1.5
_SWIGLU_LIMIT = 7.0


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
    situ_beta: float = _SITUV2_BETA,
    situ_linear_beta: float = _SITUV2_LINEAR_BETA,
    swiglu_limit: float = _SWIGLU_LIMIT,
    enable_bias: bool = False,
) -> str:
    """Build a cache-safe GEMM1 name; legacy a4w4 names remain byte-for-byte."""
    a_dtype = str(a_dtype).lower()
    out_dtype = str(out_dtype).lower()
    act = str(act).lower()
    if a_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported mxmoe GEMM1 a_dtype: {a_dtype!r}")
    if out_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported mxmoe GEMM1 out_dtype: {out_dtype!r}")
    if act not in ("silu", "swiglu", "situv2"):
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
    elif act == "swiglu":
        if float(swiglu_limit) <= 0.0:
            raise ValueError("Swiglu limit must be positive")
        name += "_swiglu"
        if enable_bias:
            name += "_bias"
    if kSplitK:
        name += f"_sk{int(kSplitK)}"
    if xcd_swizzle:
        name += f"_xcd{int(xcd_swizzle)}"
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
    situ_beta: float = _SITUV2_BETA,
    situ_linear_beta: float = _SITUV2_LINEAR_BETA,
    swiglu_limit: float = _SWIGLU_LIMIT,
    enable_bias: bool = False,
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
            swiglu_limit=swiglu_limit,
            enable_bias=enable_bias,
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


_FLYDSL_V2_GEMM2_RE = re.compile(
    r"^flydsl_moe2_layout_a(?P<a>\w+?)_w(?P<b>\w+?)_(?P<out>\w+?)_"
    r"t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)_(?P<epilog>atomic|reduce)"
    r"(?P<persist>_persist)?(?P<nt>_nt)?(?:_sbm(?P<sbm>\d+))?$"
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
        "use_nt": "NT" in flags,
        "xcd_swizzle": nums.get("xcd_swizzle", 0),
        "a_dtype": parsed["mode"].get("a_dtype", "fp4"),
        "out_dtype": "fp8" if "FP8OUT" in flags else "fp4",
        "interleave": "IL" in flags,
        "act": act,
        "situ_beta": _SITUV2_BETA if act == "situv2" else 1.0,
        "situ_linear_beta": _SITUV2_LINEAR_BETA if act == "situv2" else 1.0,
        "swiglu_limit": _SWIGLU_LIMIT,
        "enable_bias": "BIAS" in flags,
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
