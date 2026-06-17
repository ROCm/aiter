# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ahead-of-time prebuild for the cpp_itfs ``pa_ps`` paged-attention *reduce* kernel.

Scope note: this prebuilds ONLY the pa_ps partition-combine/softmax-rescale step
(the C++ ``pa_decode_ps_reduce_hip_kernel`` in pa_ps.cuh). It does NOT prebuild the
paged-attention *decode* / main-attention compute -- that runs through the separate
gluon/triton path (aiter/ops/triton/gluon/pa_decode_gluon.py) and is unaffected
here. "pa_ps" == the reduce kernel, not the whole decode kernel.

``pa_ps`` is a cpp_itfs template op: each (head_size, query_group_size,
context_partition_num, dtypes, use_sinks) combination is rendered from a jinja
template and hipcc-compiled lazily on the FIRST inference that needs it. On a
freshly (re)built container that compilation happens on the request critical
path, stalling the first benchmark runs (gpt-oss-120b: ~24k -> ~31k tok/s).

The module_* kernels avoid this because setup.py's PREBUILD_KERNELS compiles
them AOT into the packaged jit dir; the cpp_itfs family was never wired in.
This module closes that gap for pa_ps. ``compile()`` only runs hipcc (no GPU
needed -- only GPU_ARCHS, the same constraint PREBUILD_KERNELS already imposes),
so it is safe to call from the wheel build.

Variants are derived from, and md5-verified against, the set gpt-oss-120b
actually triggers (head_size=64, query_group_size=8, bf16, attention sinks,
context_partition_num growing with context length). To add another model, append
a recipe dict below.

Coverage / caveats (a prebuilt variant is hit only on an EXACT signature match):
  * Scope is gpt-oss-120b only. Other head_size / query_group_size, or
    use_sinks=False decode paths, are not covered and still JIT on first use.
  * All three runtime dtypes must be bf16: out_dtype (output), logits_dtype
    (the temporary per-partition output), and sink_dtype (sinks). A model whose
    sinks are fp32, say, hashes to a different folder and falls back to JIT.
  * context_partition_num {1..8} is complete *because* the gluon decode path caps
    it at 8 (get_recommended_splits); if that cap is raised, widen the recipe.
  * Zero-config payoff requires AITER_ROOT_DIR to be UNSET at runtime. Unset ->
    BUILD_DIR == the packaged jit dir, so these .so are found. If AITER_ROOT_DIR
    is set (e.g. to a persistent volume), BUILD_DIR becomes <root>/build, which
    has no prebuilt artifacts -> one JIT recompile into that volume. So setting
    AITER_ROOT_DIR *disables* the wheel prebuild (counterintuitive but expected).
  * Arch binding (same contract PREBUILD_KERNELS already imposes on module_*):
    compile_lib bakes the target arch into the .so via --offload-arch= (from
    GPU_ARCHS), but the folder name (md5 of the op signature) does NOT include the
    arch, and at runtime not_built() only checks file existence -- never arch match.
    So a .so prebuilt for e.g. gfx942 is loaded as-is on a gfx950 box without
    recompiling. Pure JIT avoids this because GPU_ARCHS defaults to the running
    GPU's native arch; a shipped .so is never re-validated. Safe ONLY when the
    wheel's GPU_ARCHS covers the deployment arch (or is a multi-arch fat binary) --
    keep this in mind for cross-arch wheel deployments.

Packaging: these .so are written under {BUILD_DIR}/pa_ps_<hash>/lib.so, i.e.
aiter/jit/build/..., which MANIFEST.in prunes. MANIFEST.in must re-include
aiter/jit/build/pa_ps_*/lib.so or the wheel ships without them.
"""

import os
import shutil

from csrc.cpp_itfs.pa.pa_ps import compile as compile_pa_ps
from csrc.cpp_itfs.utils import BUILD_DIR, get_default_func_name

# Each recipe is one model/dtype family. context_partition_num is the only axis that
# varies at runtime, and the gluon decode path caps it at 8 (see
# aiter/ops/triton/gluon/pa_decode_gluon.py::get_recommended_splits ->
# `return min(max_context_partition_num, 8)`). The cap is independent of context
# length / batch / concurrency, so {1..8} is the COMPLETE set for this kernel --
# it fully covers every run_bench.sh scenario (ISL 1000/5000/10000, all CONC), all
# of which serve gpt-oss-120b and decode through pa_ps. If that cap is ever raised,
# widen this range to match. Each .so is ~22K.
PA_PS_PREBUILD_RECIPES = [
    {
        # gpt-oss-120b: head_dim=64, GQA query group=8, bf16 out/logits/sink,
        # GPT-OSS-style attention sinks enabled.
        "head_size": 64,
        "query_group_size": 8,
        "out_dtype": "__hip_bfloat16",
        "logits_dtype": "__hip_bfloat16",
        "sink_dtype": "__hip_bfloat16",
        "use_sinks": True,
        "context_partition_nums": range(1, 9),
    },
]


def _iter_variants():
    for recipe in PA_PS_PREBUILD_RECIPES:
        for cpn in recipe["context_partition_nums"]:
            yield {
                "head_size": recipe["head_size"],
                "query_group_size": recipe["query_group_size"],
                "context_partition_num": cpn,
                "out_dtype": recipe["out_dtype"],
                "logits_dtype": recipe["logits_dtype"],
                "sink_dtype": recipe["sink_dtype"],
                "use_sinks": recipe["use_sinks"],
            }


def _clean_intermediates(folder):
    """Keep only lib.so in a built folder -- drop .cpp/.o/include copies (~23M -> 22K)."""
    d = os.path.join(BUILD_DIR, folder)
    if not os.path.isdir(d):
        return
    for entry in os.listdir(d):
        if entry == "lib.so":
            continue
        p = os.path.join(d, entry)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


def prebuild_pa_ps():
    """Compile every pa_ps variant in the recipes into BUILD_DIR (called from setup.py)."""
    variants = list(_iter_variants())
    print(f"[aiter] prebuild pa_ps: {len(variants)} variant(s) -> {BUILD_DIR}")
    ok, errs = 0, []
    for kw in variants:
        # func_name == folder name; matches compile_template_op's hashing.
        folder = get_default_func_name(
            "pa_ps",
            (
                kw["head_size"],
                kw["query_group_size"],
                kw["context_partition_num"],
                kw["out_dtype"],
                kw["logits_dtype"],
                kw["sink_dtype"],
                kw["use_sinks"],
            ),
        )
        try:
            compile_pa_ps(**kw)
            _clean_intermediates(folder)
            ok += 1
        except Exception as e:  # noqa: BLE001
            # One bad variant must not abort the whole wheel build.
            errs.append((kw, repr(e)))
            print(f"[aiter] prebuild pa_ps FAILED {folder}: {e}")
    print(f"[aiter] prebuild pa_ps: {ok}/{len(variants)} built, {len(errs)} failed")
    if ok == 0 and variants:
        raise RuntimeError("pa_ps prebuild built nothing; check GPU_ARCHS / hipcc")
    return ok


if __name__ == "__main__":
    prebuild_pa_ps()
