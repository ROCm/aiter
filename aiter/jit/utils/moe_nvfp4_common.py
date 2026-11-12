# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 MoE kernel names, scale layouts and sort limits, shared by the op
(aiter/ops/moe_nvfp4.py, moe_nvfp4_dispatch.py), the weight shuffle and the kernel
generator (csrc/kernels/nvfp4_moe/gen_instances.py). Python side of Cfg in
moe_nvfp4_common.h. No torch import.

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
    "reduce": False,
}
_NAME_RE = re.compile(
    r"nvfp4_moe_s([12])_([1-9]\d*)x([1-9]\d*)_(xreg|xstream|xres|xdma)((?:_\w+)?)"
)
_TOKEN_RE = re.compile(r"([a-z]+)([1-9]\d*)(?:x([1-9]\d*))?")  # word number [x number]
_NUM_TOKENS = ("ksplit", "waves", "xbuf", "gs", "occ")
_FLAGS = ("upfront", "tight", "xcdremap", "staged", "inbf16", "outbf16", "reduce")


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
    t += [f for f in ("xcdremap", "staged", "inbf16", "outbf16", "reduce") if c[f]]
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


def reduce_twin(name: str) -> str:
    """Top-k reduce (_reduce) build of a stage-2 kernel (any output dtype)."""
    name = name.removesuffix("_outbf16")
    return name if name.endswith("_reduce") else name + "_reduce"
