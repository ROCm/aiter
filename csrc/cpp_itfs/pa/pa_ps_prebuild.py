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
  * Found via a read-only fallback, independent of AITER_ROOT_DIR. The runtime
    looks up a kernel in two places (csrc/cpp_itfs/utils.py::_find_built): the
    writable BUILD_DIR (AITER_ROOT_DIR/build or ~/.aiter/build) first, then the
    packaged dir (get_user_jit_dir()/build) where these .so ship. So the prebuild
    is used with ZERO config whether or not AITER_ROOT_DIR is set -- pointing it at
    a persistent volume no longer disables the prebuild (the volume just becomes the
    preferred write/cache location, with the packaged .so as fallback).
  * Arch binding: compile_lib bakes GPU_ARCHS into the .so via --offload-arch=.
    The prebuild writes aiter_prebuild_meta.json next to every lib.so, and runtime
    packaged fallback uses it to verify the live GPU arch before loading the .so.
    If the wheel was built only for gfx942 and runs on gfx950, the packaged .so is
    skipped and cpp_itfs falls back to native JIT into the writable BUILD_DIR. For
    zero cold-start JIT across deployments, build a multi-arch/fat wheel whose
    GPU_ARCHS covers every target arch.

Packaging: these .so and their arch metadata are written under the PACKAGED build dir
get_user_jit_dir()/build/pa_ps_<hash>/, i.e. aiter/jit/build/... in the source tree
(NOT the runtime-writable utils.BUILD_DIR == ~/.aiter/build; see _packaged_build_dir).
MANIFEST.in prunes aiter/jit/build, so it must re-include both lib.so and
aiter_prebuild_meta.json, and package_data names them too, or the wheel ships without
usable pa_ps prebuilds.
"""

import json
import os
import shutil

import csrc.cpp_itfs.utils as cpp_utils
from csrc.cpp_itfs.pa.pa_ps import compile as compile_pa_ps
from csrc.cpp_itfs.utils import (
    PREBUILD_META_FILE,
    get_default_func_name,
    validate_and_update_archs,
)


def _packaged_build_dir():
    """The dir the wheel actually ships from: get_user_jit_dir()/build.

    IMPORTANT -- at RUNTIME utils.BUILD_DIR is the *writable* JIT cache
    (AITER_ROOT_DIR/build or ~/.aiter/build), and the wheel-shipped prebuilds live in
    a SEPARATE read-only dir (utils.PACKAGED_BUILD_DIR == get_user_jit_dir()/build).
    That split is exactly what fixed the review's BUILD_DIR-migration / AITER_REBUILD
    concerns. The consequence for the build: we must write the AOT artifacts straight
    into that packaged dir, because that is what MANIFEST.in / package_data /
    setup.py's `bd = get_user_jit_dir()/build` package. Writing to the runtime
    BUILD_DIR (~/.aiter/build) would leave the wheel EMPTY -- the runtime BUILD_DIR is
    not under the source tree and is never packaged.
    """
    target = cpp_utils._resolve_packaged_build_dir()
    if target is None:
        raise RuntimeError(
            "pa_ps prebuild: cannot resolve get_user_jit_dir()/build (aiter.jit.core "
            "not importable), so artifacts cannot be placed where the wheel packages "
            "them. Run inside the aiter source tree with jit on sys.path."
        )
    return target


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
    """Keep only runtime files in a built folder -- drop .cpp/.o/include copies."""
    d = os.path.join(cpp_utils.BUILD_DIR, folder)
    if not os.path.isdir(d):
        return
    for entry in os.listdir(d):
        if entry in {"lib.so", PREBUILD_META_FILE}:
            continue
        p = os.path.join(d, entry)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


def _write_prebuild_metadata(folder):
    d = os.path.join(cpp_utils.BUILD_DIR, folder)
    metadata = {
        "format_version": 1,
        "kind": "cpp_itfs_prebuild",
        "op": "pa_ps",
        "gpu_archs": validate_and_update_archs(),
    }
    with open(os.path.join(d, PREBUILD_META_FILE), "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True)
        f.write("\n")


def prebuild_pa_ps():
    """Compile every pa_ps variant in the recipes into the packaged dir (from setup.py).

    Writes into get_user_jit_dir()/build (the wheel-packaged dir), NOT the runtime
    writable BUILD_DIR -- see _packaged_build_dir() for why. The cpp_itfs compile
    machinery (compile_lib/run_lib/not_built/_find_built) all key off the module-global
    utils.BUILD_DIR, so we point it at the packaged dir for the duration of the build
    and restore it afterwards (so importing/calling this never corrupts a live
    process's runtime cache location).
    """
    variants = list(_iter_variants())
    target = _packaged_build_dir()
    os.makedirs(target, exist_ok=True)
    print(f"[aiter] prebuild pa_ps: {len(variants)} variant(s) -> {target}")

    prev_build_dir = cpp_utils.BUILD_DIR
    cpp_utils.BUILD_DIR = target
    # compile() and _find_built() are lru_cached and read BUILD_DIR on a cache miss;
    # clear so the redirect actually takes effect for already-touched folders.
    if hasattr(compile_pa_ps, "cache_clear"):
        compile_pa_ps.cache_clear()
    if hasattr(cpp_utils._find_built, "cache_clear"):
        cpp_utils._find_built.cache_clear()

    ok, errs = 0, []
    try:
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
                shutil.rmtree(os.path.join(target, folder), ignore_errors=True)
                compile_pa_ps(**kw)
                _write_prebuild_metadata(folder)
                _clean_intermediates(folder)
                ok += 1
            except Exception as e:  # noqa: BLE001
                # One bad variant must not abort the whole wheel build.
                errs.append((kw, repr(e)))
                print(f"[aiter] prebuild pa_ps FAILED {folder}: {e}")
    finally:
        cpp_utils.BUILD_DIR = prev_build_dir
        if hasattr(cpp_utils._find_built, "cache_clear"):
            cpp_utils._find_built.cache_clear()

    print(f"[aiter] prebuild pa_ps: {ok}/{len(variants)} built, {len(errs)} failed")
    if ok == 0 and variants:
        raise RuntimeError("pa_ps prebuild built nothing; check GPU_ARCHS / hipcc")
    return ok


if __name__ == "__main__":
    prebuild_pa_ps()
