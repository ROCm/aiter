# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 MoE kernel names, default dispatch rows and layout limits, shared by the op
(aiter/ops/moe_nvfp4.py) and the kernel generator
(csrc/kernels/nvfp4_moe/gen_instances.py). Python side of Cfg in moe_nvfp4_common.h.
No torch import.

Kernel names: kernel_name / parse_kernel (the name is the config; nvfp4_moe_s1_ /
nvfp4_moe_s2_ prefix: the stage). Dtypes: "bf16" / "f16".
"""

import re

# Cfg fields of moe_nvfp4_common.h, in declaration order, with their defaults
CFG_DEFAULTS = {
    "s1": False,
    "x": "xreg",
    "m_tiles": 0,
    "n_tiles": 0,
    "waves": 1,
    "ksplit": 1,
    "chunk_k_tiles": 1,
    "wbuf": 1,
    "xbuf": 0,
    "k_tiles": 0,
    "gs": 2,
    "upfront": False,
    "tight": False,
    "inbf16": False,
    "xcdremap": False,
    "staged": False,
    "occ": 1,
    "outbf16": False,
}
_NAME_RE = re.compile(
    r"nvfp4_moe_s([12])_([1-9]\d*)x([1-9]\d*)_(xreg|xstream|xres|xdma)((?:_\w+)?)"
)
_TOKEN_RE = re.compile(r"([a-z]+)([1-9]\d*)(?:x([1-9]\d*))?")  # word number [x number]
_NUM_TOKENS = ("ksplit", "waves", "xbuf", "gs", "occ")
_FLAGS = ("upfront", "tight", "xcdremap", "staged", "inbf16", "outbf16")


def kernel_name(c: dict) -> str:
    """Name of kernel config c: stage, BMxBN tile and X placement, then one token
    per non-default field, in this order."""
    bm, bn = 16 * c["m_tiles"], 16 * c["n_tiles"] * c["waves"]
    t = [f"nvfp4_moe_s{1 if c['s1'] else 2}_{bm}x{bn}_{c['x']}"]
    if c["ksplit"] > 1:
        t.append(f"ksplit{c['ksplit']}")
    if c["waves"] > 1:
        t.append(f"waves{c['waves']}")
    if c["wbuf"] > 1 or c["chunk_k_tiles"] > 1:
        t.append(f"wbuf{c['wbuf']}x{128 * c['chunk_k_tiles']}")
    if c["xbuf"]:
        t.append(f"xbuf{c['xbuf']}")
    if c["k_tiles"]:
        t.append(f"kdim{128 * c['k_tiles']}")
    if c["gs"] != 2:
        t.append(f"gs{c['gs']}")
    t += [f for f in ("upfront", "tight") if c[f]]
    if c["occ"] > 1:
        t.append(f"occ{c['occ']}")
    t += [f for f in ("xcdremap", "staged", "inbf16", "outbf16") if c[f]]
    return "_".join(t)


def parse_kernel(name: str) -> dict:
    """Config of a kernel name (CFG_DEFAULTS keys). ValueError unless the name is
    kernel_name of it: one name per config (token order, no default values)."""
    bad = ValueError(f"bad NVFP4 MoE kernel name {name!r}")
    m = _NAME_RE.fullmatch(name)
    if not m:
        raise bad
    c = dict(CFG_DEFAULTS, s1=m[1] == "1", x=m[4])
    for tok in m[5].split("_")[1:]:
        t = _TOKEN_RE.fullmatch(tok)
        k, a, b = t.groups() if t else (None, None, None)
        if tok in _FLAGS:
            c[tok] = True
        elif k in _NUM_TOKENS and b is None:
            c[k] = int(a)
        elif k == "wbuf" and b:
            c["wbuf"], c["chunk_k_tiles"] = int(a), int(b) // 128
        elif k == "kdim" and b is None:
            c["k_tiles"] = int(a) // 128
        else:
            raise bad
    c["m_tiles"], c["n_tiles"] = int(m[2]) // 16, int(m[3]) // 16 // c["waves"]
    if kernel_name(c) != name:
        raise bad
    return c


# Largest T * topk and number of experts the fused sort kernel handles
# (SORT_THREADS * SORT_PER_THREAD, SORT_MAX_EXPERTS).
MAX_SORT_ROWS = 4096
MAX_SORT_EXPERTS = 512
# kernel name suffix of the GS-n scale layout
GS_SFX = {2: "", 4: "_gs4", 6: "_gs6"}


def uses_fallback(T: int, topk: int, E: int) -> bool:
    """Whether moe_nvfp4 takes the fallback path by default for this shape."""
    return T * topk > MAX_SORT_ROWS or E > MAX_SORT_EXPERTS


def scale_gs(kt: int) -> int:
    """Scale group size GS (K-tiles of 128 per group) of shuffle_scale_nvfp4_moe for
    kt K-tiles: 6 if 6 divides kt, else 4 if kt <= 16 and 4 divides kt, else 2.
    An odd kt takes the GS of kt + 1."""
    kt += kt % 2
    if kt % 6 == 0:
        return 6
    return 4 if kt <= 16 and kt % 4 == 0 else 2


def inbf16_twin(name: str) -> str:
    """bf16-X (_inbf16) twin of a stage-1 kernel: same tile and math (stage-1 output
    bit-identical); an X_DMA twin stages 3 X buffers (its bf16 X register ring
    needs them)."""
    if name.endswith("_inbf16"):
        return name
    if "_xdma_" in name:
        name = re.sub(r"_xbuf\d+(?=_|$)", "_xbuf3", name)
    return name + "_inbf16"


def outbf16(name: str) -> str:
    """bf16-output (_outbf16) build of a stage-2 kernel."""
    return name if name.endswith("_outbf16") else name + "_outbf16"


def default_rows(H: int, I: int, x_dtype: str, out_dtype: str, fb: bool) -> tuple:
    """Default rule: ((T, [stage-1 names], [stage-2 names]), ...); moe_nvfp4 takes
    the first name of a list that fits the shape and drops a row without one.

    16-row tiles up to T 8, 32-row tiles above; stage 2 at odd KT 3: the KT-3 xres
    kernels. Fallback path: _inbf16 stage 1 for bf16 x, and on GS-6 stage-1 scales
    a 16-wave 32x512 stage-1 tile above T 1024 (3 N blocks: half the X loads and
    casts per token; 0.93-0.98x of the 32x256 kernel at T 4096-16384). Stage 2
    reads f16 x_inter; _outbf16 for bf16 output.
    """
    kt2 = I // 128
    g1, g2 = GS_SFX[scale_gs(H // 128)], GS_SFX[scale_gs(kt2)]
    if kt2 % 2:
        x2 = f"_xres_waves4_wbuf3x128_kdim{I}_gs4_upfront"
        small2, large2 = ["16x128" + x2], ["32x256" + x2 + "_occ3_staged"]
    else:
        small2 = ["16x16_xreg_wbuf2x256" + g2]
        large2 = [
            "32x256_xstream_waves4_wbuf2x256_xbuf2" + g2,
            "32x256_xstream_waves8_wbuf3x256_xbuf2" + g2,
            "32x64_xreg_wbuf2x256" + g2,
        ]
    rows = [
        (8, ["16x32_xreg_wbuf2x256" + g1], small2),
        (1024, [f"32x256_xdma_waves8_wbuf2x128_xbuf3{g1}_xcdremap"], large2),
    ]
    if fb and g1 == "_gs6":
        big = "32x512_xdma_waves16_wbuf2x128_xbuf3_gs6_xcdremap"
        rows.append((2048, [big], large2))
    out = []
    for t, s1, s2 in rows:
        s1 = ["nvfp4_moe_s1_" + n for n in s1]
        s2 = ["nvfp4_moe_s2_" + n for n in s2]
        if fb and x_dtype == "bf16":
            s1 = [inbf16_twin(n) for n in s1]
        if out_dtype == "bf16":
            s2 = [outbf16(n) for n in s2]
        out.append((t, s1, s2))
    return tuple(out)


def default_names() -> list:
    """Every kernel name default_rows can return (one K-tile count per GS layout and
    KT 3, both paths and dtypes)."""
    names = set()
    for H in (256, 384, 512, 768):
        for I in (256, 384, 512, 768):
            for x in ("bf16", "f16"):
                for o in ("bf16", "f16"):
                    for fb in (False, True):
                        for _, s1, s2 in default_rows(H, I, x, o, fb):
                            names.update(s1 + s2)
    return sorted(names)
