# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 MoE dispatch policy (aiter/ops/moe_nvfp4.py): path selection, the default
kernel rule and the tuned CSV. No torch import (gen_instances.py loads this file by
path to build every default and CSV kernel into module_moe_nvfp4).

Tuned CSV (aiter/configs/moe_nvfp4_dispatch.csv, AITER_CONFIG_MOE_NVFP4), the
tuned_fmoe.csv column names:
    cu_num,token,model_dim,inter_dim,expert,topk,dtype,us1,kernelName1,err1,
    us2,kernelName2,err2,us,tflops,bw
- dtype: torch.bfloat16 / torch.float16, of the activations and the output.
- token: dispatch upper bound. A call takes the first row of its (cu_num,
  model_dim, inter_dim, expert, topk, dtype) key and path (uses_fallback of the
  row token) with T <= token; above the last row or without rows, default_rows.
- kernelName1 / kernelName2: stage-1 / stage-2 kernel (kernel_name in
  aiter/jit/utils/moe_nvfp4_common.py; _inbf16 stage 1 iff bf16 x on the fallback
  path; stage 2: _reduce (any output dtype), else _outbf16 iff bf16 output). A
  kernel outside module_moe_nvfp4 is JIT-built in its own module.
- us: the whole moe_nvfp4 call at T = token in a CUDA graph (sort + cast +
  zeroing, both stages, top-k reduce or f32-to-f16 copy); us1 / us2: the stage
  kernels alone. err1 / err2: share of stage outputs not within atol = rtol =
  1e-2 (checkAllclose) of the fp32 reference rounded to the output dtype (stage 2:
  on the kernel's own stage-1 output; bf16 atomics round each top-k partial sum).
  tflops / bw: of us, the tuned_fmoe formula with 0.5-byte weights (bw counts the
  weights of all experts). Unmeasured: empty. The op reads no metric.
"""

import csv

try:
    from ..jit.utils.moe_nvfp4_common import (
        GS_SFX,
        MAX_SORT_EXPERTS,
        MAX_SORT_ROWS,
        inbf16_twin,
        outbf16,
        parse_kernel,
        reduce_twin,
        scale_gs,
    )
except ImportError:  # loaded by path (gen_instances.py, aiter/jit/utils on sys.path)
    from moe_nvfp4_common import (
        GS_SFX,
        MAX_SORT_EXPERTS,
        MAX_SORT_ROWS,
        inbf16_twin,
        outbf16,
        parse_kernel,
        reduce_twin,
        scale_gs,
    )

CSV_DTYPES = {"torch.bfloat16": "bf16", "torch.float16": "f16"}


def uses_fallback(T: int, topk: int, E: int) -> bool:
    """Whether moe_nvfp4 takes the fallback path by default for this shape."""
    return T * topk > MAX_SORT_ROWS or E > MAX_SORT_EXPERTS


def default_rows(H: int, I: int, x_dtype: str, out_dtype: str, fb: bool) -> tuple:
    """Default rule: ((T, [stage-1 names], [stage-2 names]), ...); moe_nvfp4 takes
    the first name of a list that fits the shape and drops a row without one.

    16-row tiles up to T 8, 32-row tiles above; stage 2 at odd KT 3: the KT-3 xres
    kernels. Fallback path: _inbf16 stage 1 for bf16 x, and on GS-6 stage-1 scales
    a 16-wave 32x512 stage-1 tile above T 1024 (3 N blocks: half the X loads and
    casts per token; 0.93-0.98x of the 32x256 kernel at T 4096-16384). Stage 2
    reads f16 x_inter; _outbf16 for bf16 output; f16 output: _reduce above T 8
    (f32 atomics: 1.02-1.05x of it up to T 8, 1.07-1.9x from T 256-512).
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
        elif t > 8:
            s2 = [reduce_twin(n) for n in s2]
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


def csv_names(path: str) -> list:
    """Kernels of the tuned CSV rows, as written."""
    with open(path, newline="") as f:
        return [r[k] for r in csv.DictReader(f) for k in ("kernelName1", "kernelName2")]


def read_dispatch_csv(path: str, cu_num: int) -> dict[tuple, tuple]:
    """Tuned CSV rows of cu_num: {(H, I, E, topk, dtype): ((token, stage-1,
    stage-2), ...)}, dtype "bf16" / "f16". Raises ValueError for an unknown dtype, a
    kernel name that is not kernel_name of a config of its stage, or one that does
    not match the row dtype and path."""
    out: dict[tuple, list] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["cu_num"]) != cu_num:
                continue
            t, s1, s2 = int(r["token"]), r["kernelName1"], r["kernelName2"]
            dtype = CSV_DTYPES.get(r["dtype"])
            if dtype is None:
                raise ValueError(f"{path}: token={t}: unsupported dtype {r['dtype']}")
            key = (int(r["model_dim"]), int(r["inter_dim"]), int(r["expert"]))
            key += (int(r["topk"]), dtype)
            for stage, name in ((1, s1), (2, s2)):
                if parse_kernel(name)["s1"] != (stage == 1):
                    raise ValueError(f"{path}: {name!r} is not a stage-{stage} kernel")
            fb = uses_fallback(t, key[3], key[2])
            ok1 = s1.endswith("_inbf16") == (fb and dtype == "bf16")
            atomics = not s2.endswith("_reduce")
            ok2 = not atomics or s2.endswith("_outbf16") == (dtype == "bf16")
            if not (ok1 and ok2):
                raise ValueError(
                    f"{path}: {key} token={t}: kernel names do not match the dtype"
                    f" and path ({'fallback' if fb else 'main'})"
                )
            out.setdefault(key, []).append((t, s1, s2))
    return {k: tuple(sorted(v)) for k, v in out.items()}
