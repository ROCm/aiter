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
#: ``(BM, use_nt, inline_quant)`` triples that are compiled for each A dtype.
#:
#: ``(BM, use_nt, inline_quant)`` triples that are compiled for each A dtype.
#:
#: ``(16, True, False)`` -- BM16 on a *pre-quantized* operand -- is absent
#: because it computes wrong numbers, with the cause **not** established.
#:
#: It compiles and is correct at glm5 M=8/16/64 and kimi3 M=8/16; it is wrong at
#: glm5 M=128 (NaN) and at kimi3 M=64 paired with a reduce GEMM2 (rel_l2 0.43),
#: in the ordinary three-kernel path, not only inside the merged kernel. NaN
#: from garbage bytes decoded as E8M0 exponents fits an out-of-range read.
#:
#: The boundary is sharp and reproducible: with BM16 forced, kimi3 passes at
#: M=8/32/128 and returns NaN from M=192 upward.
#:
#: Cause NOT found, after three hypotheses were tested and all three failed.
#: Recorded so the next attempt does not re-walk them.
#:
#: 1. Wrong chunk index. ``issue_a_scale_load`` uses
#:    ``chunk_base = m_row // 32``, which does put a BM16 block on chunk
#:    ``b // 2``. Not it.
#:
#: 2. Odd blocks read the wrong half of their chunk. The A-scale layout is
#:    32-row chunked and the MFMA scale selector picks which 16-row half
#:    applies; ``mfma_cluster`` hardwires BM16 (``kMChunks == 1``) to selectors
#:    0/2, i.e. the lower half, while BM32 uses all of 0/1/2/3. So an odd BM16
#:    block looked like it must be reading its predecessor's scales.
#:    **Disproved by measurement**: dumping ``moe_sorting`` output shows odd
#:    blocks carry real rows at *every* M for kimi3 (64 of them at M=8, 448 at
#:    M=128), and those shapes are correct. If this were the mechanism they
#:    would fail too.
#:
#: 3. The over-wide buffer bound. ``_asc_per_mb`` is
#:    ``max(BM // 32, 1) * kAS_per_chunk_dw * 4``, handing BM16 the BM32 stride
#:    over twice the blocks -- an 8x view against BM32's designed-in 4x, which
#:    would disable the hardware clamp on a stray read. Tightening it to the
#:    real chunk count changed nothing: kimi3 still failed from M=192.
#:
#: The one hard fact to design the next experiment around: with BM16 forced,
#: kimi3 is correct at M=8/32/128 and returns NaN from M=192 up, while the sort
#: output (valid rows, block count, odd/even occupancy) is *identical* for
#: M=128 and M=192 -- both 14336 valid rows in 896 blocks. So whatever changes
#: is not the sorted layout.
#:
#: A note on method: an earlier round "ruled out" the odd-block hypothesis using
#: runs that had ``AITER_TP_MEGA_PIN_BM=16`` set but a tuned CSV row present --
#: which shadows it entirely (see ``pin_tuned_csv_path``). Those runs never
#: exercised BM16. Redirect the CSV to a nonexistent path when probing.
#:
#: Worth chasing: without BM16 the FP4 wire cannot reach it and borrows a larger
#: bucket's tuned row, measuring 1.37x on glm5 M=8 against the BF16 wire's
#: 1.93x. Do it as its own task, with a full shape sweep.
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
    # Gate/up layout comes from the runtime gate_mode; the same config name
    # can compile both interleaved and separated GPU kernels.
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
