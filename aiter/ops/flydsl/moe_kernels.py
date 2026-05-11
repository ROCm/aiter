# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import re

from typing import Dict, Optional

import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}


def _get_dtypes():
    from aiter.utility import dtypes

    return dtypes


_SUFFIX_RE = re.compile(r"(?P<fp4>_fp4)?(?P<fp8>_fp8)?(?:_sbm(?P<sbm>\d+))?$")


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
    sort_block_m: int = 0,
) -> str:
    """Construct kernel name: ``flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}][_sbm{S}]``."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def get_flydsl_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name.

    Strips ``_fp4`` / ``_fp8`` / ``_sbm{N}`` suffixes transparently.
    """
    params = _KERNEL_PARAMS.get(name)
    if params is not None:
        return params
    m = _SUFFIX_RE.search(name)
    if m and m.group(0):
        base_name = name[: m.start()]
        params = _KERNEL_PARAMS.get(base_name)
        if params is not None:
            extra: Dict = {}
            if m.group("fp4"):
                extra["out_dtype"] = "fp4"
            if m.group("fp8"):
                extra["out_dtype"] = "fp8"
                extra["a_scale_one"] = True
            if m.group("sbm") is not None:
                extra["sort_block_m"] = int(m.group("sbm"))
            return {**params, **extra}
    return None


def stage1_kernel_native_gate_mode(kernel_name1: str) -> str:
    """Infer the *native* caller_gate_mode implied by a stage1 kernel name.

    Backend-aware rules (used by both UT and AOT to pin every CSV row to
    its CSV-author-intended gate_mode before deciding whether to swap):

      - ``flydsl_*_gui*`` -> ``"interleave"``
      - ``flydsl_*``      -> ``"separated"``
      - ``cktile_*``      -> ``"interleave"``  (cktile uses GUI W layout)
      - ``ck_*``          -> ``"separated"``   (plain ck uses SEP layout)
      - anything else     -> ``"separated"``   (default fallback)
    """
    n = kernel_name1 or ""
    if n.startswith("flydsl_"):
        return "interleave" if "_gui" in n else "separated"
    if n.startswith("cktile_"):
        return "interleave"
    if n.startswith("ck_"):
        return "separated"
    return "separated"


def is_csv_fallback_row(row) -> bool:
    """A CSV row is a fallback backup (not the real deployment kernel) when
    its ``_tag`` column contains ``fallback``.  Such rows are still
    iterated (so we exercise / compile them) but do NOT participate in the
    CSV's deployment-mode decision."""
    if hasattr(row, "get"):
        tag = str(row.get("_tag", "") or "").lower()
    else:
        # pandas Series-like: support both .get and dict-style access
        try:
            tag = str(row["_tag"]).lower() if "_tag" in row else ""
        except Exception:
            tag = ""
    return "fallback" in tag


def csv_caller_gate_modes(rows) -> list:
    """Decide which caller_gate_mode(s) to sweep for a whole CSV.

    Same per-CSV pinning rules used by both the unit test
    (``op_tests/test_moe_2stage.py``) and the AOT pre-compiler
    (``aiter.aot.flydsl.moe``) so they always cover the SAME kernel set.

    Only **non-fallback** rows participate in the decision (rows tagged
    ``flydsl_fallback`` etc are backup kernels for the same shape and
    must not poison the deployment intent).

    Pinning rules (one decision per CSV, applied to every row):
      1. Any non-fallback row uses plain ``ck_*`` stage1 (not cktile)  -> SEP only.
      2. Any non-fallback row uses ``cktile_*`` stage1                 -> INTL only.
      3. Both ck_ AND cktile_ present                                  -> SEP (ck wins).
      4. Pure flydsl AND CSV is a4w4 (``afp4_wfp4`` in stage1 names)   -> [SEP, INTL].
      5. Otherwise (pure flydsl, non-a4w4)                             -> majority by
         ``_gui`` suffix among non-fallback flydsl rows.

    ``rows`` may be any iterable of dict-like / pandas Series rows
    (csv.DictReader, list of dicts, pandas DataFrame, etc).  Returns a
    list of caller_gate_mode strings (length 1 for cases 1/2/3/5,
    length 2 for case 4).
    """
    has_ck = False
    has_cktile = False
    has_a4w4 = False
    sep = intl = 0

    if hasattr(rows, "iterrows"):
        iterator = (row for _, row in rows.iterrows())
    else:
        iterator = iter(rows)

    for row in iterator:
        if is_csv_fallback_row(row):
            continue
        kn1 = str(row.get("kernelName1", "") or "") if hasattr(row, "get") else (
            str(row["kernelName1"]) if "kernelName1" in row else ""
        )
        if not kn1:
            continue
        if kn1.startswith("cktile_"):
            has_cktile = True
        elif kn1.startswith("ck_"):
            has_ck = True
        if "afp4_wfp4" in kn1:
            has_a4w4 = True
        if kn1.startswith("flydsl_"):
            if "_gui" in kn1:
                intl += 1
            else:
                sep += 1

    if has_ck and not has_cktile:
        return ["separated"]
    if has_cktile and not has_ck:
        return ["interleave"]
    if has_ck and has_cktile:
        # Mixed (rare, both real-deployment) -- prefer SEP for ck side;
        # cktile side dispatches its own kernel anyway.
        return ["separated"]
    if has_a4w4:
        return ["separated", "interleave"]
    return ["interleave" if intl > sep else "separated"]


def swap_flydsl_stage1_kernel_for_gate_mode(name: str, target_gate_mode) -> str:
    """Return the SEP/GUI sibling of a stage1 kernel name for a given gate_mode.

    Caller-facing ``tile_n`` is per-side N for both SEP and GUI (the kernel
    internally sets ``_acc_set_n = 2*tile_n`` for GUI so each CTA always
    covers ``2*tile_n`` columns of W regardless of layout).  The stage1
    registry is built so that **for every fp4-weight (tm, tile_n, tk) cfg
    BOTH a SEP and a GUI sibling exist at the IDENTICAL tile_n** (see
    ``get_flydsl_stage1_kernels``), so swapping the gate_mode never has to
    rescale ``tile_n`` -- we only toggle the ``_gui`` suffix and the
    swapped kernel does the same work-per-tile as the original
    (performance-neutral).

    Both ``str`` (``"separated"`` / ``"interleave"``) and ``GateMode`` enum
    values are accepted for ``target_gate_mode``.
    """
    if target_gate_mode is None:
        return name
    target = (
        target_gate_mode.value
        if hasattr(target_gate_mode, "value")
        else target_gate_mode
    )
    if target not in ("separated", "interleave"):
        return name
    params = get_flydsl_kernel_params(name)
    if params is None or params.get("stage") != 1:
        return name
    cur = params.get("gate_mode", "separated")
    if cur == target:
        return name
    if cur not in ("separated", "interleave"):
        return name

    # Strip any ``_fp4`` / ``_fp8`` / ``_sbm`` tail before pattern surgery,
    # remember it, and re-apply it to the swapped name so the suffix-aware
    # lookup keeps working.
    suffix_match = _SUFFIX_RE.search(name)
    if suffix_match and suffix_match.group(0):
        base = name[: suffix_match.start()]
        tail = suffix_match.group(0)
    else:
        base = name
        tail = ""

    if target == "interleave":
        # SEP -> GUI: insert ``_gui`` before the ``_xcd*`` suffix (if any),
        # otherwise append.  tile_n stays identical.
        if "_gui" in base:  # already GUI -- defensive
            new_base = base
        elif "_xcd" in base:
            new_base = base.replace("_xcd", "_gui_xcd")
        else:
            new_base = base + "_gui"
    else:
        # GUI -> SEP: drop ``_gui``, tile_n stays identical.
        if "_gui" not in base:
            return name
        new_base = base.replace("_gui", "")

    new_name = new_base + tail
    sibling = get_flydsl_kernel_params(new_name)
    if sibling is not None and sibling.get("gate_mode") == target:
        return new_name
    return name


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs.

    Registry invariants (relied on by ``swap_flydsl_stage1_kernel_for_gate_mode``):

      - ``tile_n`` in kernel names and registry params is ALWAYS the
        caller-facing per-side N.  The kernel internally derives
        ``_acc_set_n = 2*tile_n`` for GUI / mock_gate_only paths so each
        CTA always covers ``2*tile_n`` columns of W (see
        ``compile_mixed_moe_gemm1``).  No legacy halving / doubling
        happens here.
      - For every fp4-weight (tm, tile_n, tk, wpe, kb, bnt, xcd) cfg,
        BOTH the SEP and the GUI sibling are registered at the IDENTICAL
        ``tile_n``.  Swap is therefore a pure ``_gui`` suffix toggle and
        is performance neutral (work-per-tile preserved).
      - ``_go`` (mock_gate_only) and ``_kb`` (split-K) candidates are
        gated to the (tm=32, fp4_a, !gui, wpe=3) corner that historically
        used them; ``_gui`` cannot combine with ``_go`` since both fold
        gate+up.
    """
    kernels = {}
    is_fp4_a = a_dtype == "fp4"
    is_fp4_b = b_dtype == "fp4"

    tile_ks = [256]
    tile_ms = [32, 64, 128]
    waves_per_eus = [1, 2, 3, 4]
    k_batches = [1, 2, 4, 7, 14]
    b_nts = [0, 2]
    xcd_swizzles = [0, 4]

    # GUI/INTERLEAVE only meaningful for fp4-weight kernels (the W layout
    # interleaves gate/up at 4-bit packing granularity).  For non-fp4 W
    # only the SEP layout is registered.
    is_gui_choices = (False, True) if is_fp4_b else (False,)

    def _tile_ns(tm):
        """Caller-facing per-side ``tile_n`` candidates for this tm.

        SAME set is used for SEP and GUI siblings.  Values were chosen as
        the union of historical SEP and GUI tile_n's actually referenced
        by tuned CSVs (so no kernel name disappears) plus the same-tile_n
        siblings needed by ``swap_flydsl_stage1_kernel_for_gate_mode``."""
        if not is_fp4_b:
            return [128]
        if tm == 32:
            return [32, 64, 128]
        return [64, 128]  # tm in {64, 128}

    for is_gui in is_gui_choices:
        for tm in tile_ms:
            for tn in _tile_ns(tm):
                for tk in tile_ks:
                    for wpe in waves_per_eus:
                        kbs = (
                            k_batches
                            if (wpe == 3 and tm == 32 and is_fp4_a and not is_gui)
                            else [1]
                        )
                        for kb in kbs:
                            for bnt in b_nts:
                                gate_onlys = (
                                    [False, True]
                                    if (kb > 1 and is_fp4_a and not is_gui)
                                    else [False]
                                )
                                for go in gate_onlys:
                                    for xcd in xcd_swizzles:
                                        name = flydsl_kernel_name(
                                            1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                        )
                                        if wpe != 1:
                                            name += f"_w{wpe}"
                                        if kb != 1:
                                            name += f"_kb{kb}"
                                        if bnt != 2:
                                            name += f"_bnt{bnt}"
                                        if go:
                                            name += "_go"
                                        if is_gui:
                                            name += "_gui"
                                        if xcd > 0:
                                            name += f"_xcd{xcd}"
                                        kernels[name] = {
                                            "stage": 1,
                                            "a_dtype": a_dtype,
                                            "b_dtype": b_dtype,
                                            "out_dtype": out_dtype,
                                            "tile_m": tm,
                                            "tile_n": tn,
                                            "tile_k": tk,
                                            "MPerBlock": tm,
                                            "waves_per_eu": wpe,
                                            "k_batch": kb,
                                            "b_nt": bnt,
                                            "gate_mode": (
                                                "mock_gate_only"
                                                if go
                                                else (
                                                    "interleave"
                                                    if is_gui
                                                    else "separated"
                                                )
                                            ),
                                            "xcd_swizzle": xcd,
                                        }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128]
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    modes = ["atomic", "reduce"]

    b_nts = [0, 2]

    xcd_swizzles = [0, 4]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    for bnt in b_nts:
                        for xcd in xcd_swizzles:
                            base_name = flydsl_kernel_name(
                                2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                            )
                            if bnt != 0:
                                base_name += f"_bnt{bnt}"
                            if xcd > 0:
                                base_name += f"_xcd{xcd}"
                            base_params = {
                                "stage": 2,
                                "a_dtype": a_dtype,
                                "b_dtype": b_dtype,
                                "out_dtype": out_dtype,
                                "tile_m": tm,
                                "tile_n": tn,
                                "tile_k": tk,
                                "mode": mode,
                                "MPerBlock": tm,
                                "b_nt": bnt,
                                "xcd_swizzle": xcd,
                            }
                            kernels[base_name] = base_params
                            kernels[base_name + "_persist"] = {
                                **base_params,
                                "persist": True,
                            }
    return kernels


def get_flydsl_stage1_kernels_int4_bf16(out_dtype: str) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported int4_bf16 stage1 configs."""
    kernels = {}
    a_dtype = "bf16"
    b_dtype = "int4"
    tile_ks = [128, 256]
    tile_ms = [16, 32, 64, 128]
    tile_ns = [64, 128]
    k_batches = [1, 2, 4, 7, 14]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for kb in k_batches:
                    name = flydsl_kernel_name(
                        1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                    )
                    if kb != 1:
                        name += f"_kb{kb}"
                    kernels[name] = {
                        "stage": 1,
                        "a_dtype": a_dtype,
                        "b_dtype": b_dtype,
                        "out_dtype": out_dtype,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "MPerBlock": tm,
                        "in_dtype": "int4_bf16",
                        "k_batch": kb,
                    }
    return kernels


def get_flydsl_stage2_kernels_int4_bf16(out_dtype: str) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported int4_bf16 stage2 configs."""
    kernels = {}
    a_dtype = "bf16"
    b_dtype = "int4"
    tile_ks = [128, 256]
    tile_ms = [16, 32, 64, 128]
    tile_ns = [128]
    # modes = ["atomic", "reduce"]
    modes = ["atomic"]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    base_name = flydsl_kernel_name(
                        2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                    )
                    base_params = {
                        "stage": 2,
                        "a_dtype": a_dtype,
                        "b_dtype": b_dtype,
                        "out_dtype": out_dtype,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "mode": mode,
                        "MPerBlock": tm,
                        "in_dtype": "int4_bf16",
                    }
                    kernels[base_name] = base_params
                    kernels[base_name + "_persist"] = {
                        **base_params,
                        "persist": True,
                    }
    return kernels


def _register_all_configs():
    """Pre-populate _KERNEL_PARAMS with all supported configs at import time."""
    for a in ("fp8", "fp4", "fp16"):
        for b in ("fp4",):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))
    # int4_bf16 (a16wi4) configs
    for out in ("bf16", "f16"):
        _KERNEL_PARAMS.update(get_flydsl_stage1_kernels_int4_bf16(out))
        _KERNEL_PARAMS.update(get_flydsl_stage2_kernels_int4_bf16(out))


_register_all_configs()


def compile_flydsl_moe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str = "silu",
    persist_m: int = 1,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_mode: str = "separated",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    enable_bias: bool = False,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    swiglu_limit: float = 0.0,
):
    """Compile stage1 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1
        from .moe_common import GateMode

        return compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            act=act,
            persist_m=persist_m,
            use_async_copy=use_async_copy,
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            gate_mode=GateMode(gate_mode),
            model_dim_pad=model_dim_pad,
            inter_dim_pad=inter_dim_pad,
            enable_bias=enable_bias,
            a_scale_one=a_scale_one,
            xcd_swizzle=xcd_swizzle,
            swiglu_limit=swiglu_limit,
        )
    elif a_dtype == "bf16" and b_dtype == "int4":
        # a16wi4: bf16 activations, int4 weights with groupwise scale
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        # split-K needs cshuffle (None -> auto-enable); non-split-K uses direct epilog
        _use_cshuffle = None if k_batch > 1 else False

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype="int4_bf16",
            group_size=32,
            out_dtype=out_dtype,
            use_cshuffle_epilog=_use_cshuffle,
            scale_is_bf16=True,
            k_batch=k_batch,
        )
    else:
        raise ValueError(
            f"Unsupported stage1 dtype combination: a_dtype={a_dtype}, b_dtype={b_dtype}"
        )


def compile_flydsl_moe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
    persist_m: int = 1,
    sort_block_m: int = 0,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    enable_bias: bool = False,
):
    """Compile stage2 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        return compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
            persist_m=persist_m,
            sort_block_m=sort_block_m,
            b_nt=b_nt,
            model_dim_pad=model_dim_pad,
            inter_dim_pad=inter_dim_pad,
            xcd_swizzle=xcd_swizzle,
            enable_bias=enable_bias,
        )
    elif a_dtype == "bf16" and b_dtype == "int4":
        # a16wi4: bf16 activations, int4 weights with groupwise scale
        from .kernels.moe_gemm_2stage import compile_moe_gemm2

        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype="int4_bf16",
            group_size=32,
            out_dtype=out_dtype,
            accumulate=accumulate,
            scale_is_bf16=True,
        )
    else:
        raise ValueError(
            f"Unsupported stage2 dtype combination: a_dtype={a_dtype}, b_dtype={b_dtype}"
        )


# Private helpers


_DLPACK_SAFE = (torch.uint8, torch.float16, torch.bfloat16, torch.float32)


def _view_safe(t: torch.Tensor) -> torch.Tensor:
    """View as uint8 if dtype is not dlpack-safe, otherwise return as-is."""
    return (
        t.view(torch.uint8)
        if t is not None and t.numel() > 0 and t.dtype not in _DLPACK_SAFE
        else t
    )


def _s1_args_fp4(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out_scale_sorted,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    dev,
    bias=None,
    stream=None,
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    _bias = bias if bias is not None else empty_f32
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        _view_safe(out),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        _bias,
        out_scale_sorted,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        stream,
    )


def _s1_args_std(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    stream=None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        out,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        stream,
    )


def _s2_args_fp4(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    dev,
    bias=None,
    stream=None,
):
    _bias = (
        bias.view(-1)
        if bias is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        _view_safe(target),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        _bias,
        token_num,
        n_in,
        k_in,
        blocks,
        stream,
    )


def _s2_args_std(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    stream=None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        target,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        blocks,
        stream,
    )


def _run_compiled(exe, args):
    """Call the JitFunction with the given args.
    JitFunction.__call__ handles compilation caching internally.
    """
    try:
        exe(*args)
    except Exception:
        # JitFunction.__call__ leaks ir.Context on compilation failure,
        # causing all subsequent JitFunction calls to take a wrong code path
        # (self.func(*args) without CompilationContext → gpu_module_body error).
        # Clean up leaked contexts to isolate failures.
        try:
            from flydsl._mlir import ir

            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:
            pass
        raise


@functools.cache
def _get_compiled_silu_fused(
    inter_dim: int,
    topk: int,
    quant_mode: str = "fp4",
    gui_layout: bool = False,
    act: str = "silu",
    enable_bias: bool = False,
    swiglu_limit: float = 0.0,
):
    """Compile and cache the fused gate activation + quant + scale-sort kernel."""
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    return build_silu_and_mul_fq_module(
        inter_dim,
        topk,
        quant_mode,
        gui_layout,
        act=act,
        enable_bias=enable_bias,
        swiglu_limit=swiglu_limit,
    )


@functools.cache
def _get_compiled_swiglu(inter_dim: int):
    """Compile and cache the fused swiglu_and_mul kernel (interleaved input)."""
    from aiter.ops.flydsl.kernels.swiglu_and_mul import build_swiglu_and_mul_module

    return build_swiglu_and_mul_module(inter_dim)


# Public API


def flydsl_moe_stage1(
    a: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    act: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    persist_m: int = 0,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 0,
    gate_mode: str = "separated",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    bias: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    swiglu_limit: float = 0.0,
):
    """Fused gate+up GEMM (MOE stage1).

    a: (token_num, model_dim), w1: (E, 2*inter_dim, model_dim) pre-shuffled.
    model_dim and inter_dim INCLUDE padding (model_dim_pad, inter_dim_pad).
    bias: optional (E, 2*inter_dim) f32 bias added before activation.
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as
    `shuffle_weight_a16w4(w1, 16, True)` and `shuffle_scale_a16w4(w1_scale, E, True)`.

    When fuse_quant=True, the kernel fuses quantization (fp4/fp8, inferred from
    out_dtype) and writes e8m0 scales in sorted tiled layout directly.

    When k_batch>1 (split-K), the kernel outputs gate/up partials via atomic
    add into a zeroed buffer, then silu_and_mul fuses activation + reduction.

    gate_mode controls the gate/up computation strategy (see GateMode enum).

    Returns:
        Basic:                      out
        fuse_quant:                 (out, out_scale_sorted)
    """
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    _need_fp4 = out_dtype == "fp4"
    _need_fp8 = out_dtype == "fp8"
    _fuse_any_quant = _need_fp4 or _need_fp8
    _base_out_dtype = "bf16" if _fuse_any_quant else out_dtype
    dtypes = _get_dtypes()

    if _need_fp4:
        torch_out_dtype = dtypes.fp4x2
    elif _need_fp8:
        torch_out_dtype = dtypes.fp8
    else:
        torch_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    _is_splitk = k_batch > 1
    gate_up_interleave = gate_mode == "interleave"

    dev = a.device
    _splitk_fp4 = _is_splitk and _need_fp4
    _gui_sk = gate_up_interleave and _is_splitk
    _gui_sk_fused = _gui_sk and _fuse_any_quant

    if out is None:
        if _need_fp4 or (_gui_sk_fused and _need_fp4):
            out = torch.empty(
                (token_num, topk, inter_dim // 2), dtype=dtypes.fp4x2, device=dev
            )
        elif _need_fp8 or (_gui_sk_fused and _need_fp8):
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=dtypes.fp8, device=dev
            )
        else:
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
            )

    if _is_splitk:
        torch_tmp_out_dtype = dtypes.bf16 if _base_out_dtype == "bf16" else dtypes.fp16
        tmp_out = torch.zeros(
            (token_num, topk, inter_dim * 2), dtype=torch_tmp_out_dtype, device=dev
        )
    else:
        tmp_out = None

    flat_a_scale = (
        a1_scale.view(-1) if a1_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w1_scale.view(-1) if w1_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    _need_quant = _fuse_any_quant or _splitk_fp4 or _gui_sk_fused
    _need_sort = _need_quant

    _sort_block_m = tile_m
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    _persist_m = persist_m if persist_m > 0 else 1

    # Allocate sorted-scale buffer with padding for tiled layout
    scale_cols = inter_dim // 32
    sorted_size = max(
        sorted_token_ids.shape[0], sorted_expert_ids.shape[0] * _sort_block_m
    )
    padded_rows = (sorted_size + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted_flat = (
        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
        if _need_sort
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )

    # split-K GEMM kernel does not fuse quant; the fused silu_and_mul_fq kernel
    # handles activation + quant + scale-sort after the GEMM completes.
    _gemm_out_dtype = _base_out_dtype if _is_splitk else out_dtype

    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    _kernel_out = tmp_out if _is_splitk else out
    kernel_bias = None if _is_splitk else bias
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * 2 if is_fp4 else inter_dim
    _k_in = model_dim

    if is_fp4:
        args = _s1_args_fp4(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            out_scale_sorted_flat.view(-1),
            token_num,
            _n_in,
            _k_in,
            _grid_y,
            dev,
            bias=(
                kernel_bias.view(-1)
                if kernel_bias is not None
                else torch.empty(0, device=dev)
            ),
        )
    else:
        args = _s1_args_std(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            _grid_y,
        )

    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=_gemm_out_dtype,
        act=act,
        persist_m=_persist_m,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_mode=gate_mode,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        enable_bias=(kernel_bias is not None),
        a_scale_one=a_scale_one,
        xcd_swizzle=xcd_swizzle,
        swiglu_limit=swiglu_limit,
    )
    _run_compiled(exe, args)

    num_sorted_rows = sorted_token_ids.shape[0]
    use_splitk_bias = _is_splitk and bias is not None
    if use_splitk_bias and topk_ids is None:
        raise ValueError("topk_ids are required for split-K FlyDSL stage1 bias")
    # sorted_token_ids only gives (token_id, slot_id). Bias is stored per expert,
    # so the post-activation kernel needs topk_ids[token_id * topk + slot_id].
    topk_ids_arg = (
        topk_ids.to(torch.int32).contiguous().view(-1)
        if use_splitk_bias
        else sorted_token_ids.view(-1)
    )
    bias_arg = (
        bias.contiguous().view(-1)
        if use_splitk_bias
        else (
            bias.contiguous().view(-1)[:0]
            if bias is not None
            else torch.empty(0, device=sorted_token_ids.device, dtype=torch.float32)
        )
    )
    if _gui_sk_fused:
        _quant_mode = "fp4" if _need_fp4 else "fp8"
        _silu_fused_k = _get_compiled_silu_fused(
            inter_dim,
            topk,
            _quant_mode,
            gui_layout=True,
            act=act,
            enable_bias=use_splitk_bias,
            swiglu_limit=swiglu_limit,
        )
        _run_compiled(
            _silu_fused_k,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1).view(torch.uint8),
                out_scale_sorted_flat,
                sorted_token_ids,
                num_valid_ids,
                topk_ids_arg,
                bias_arg,
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _gui_sk:
        _silu_fused_k = _get_compiled_silu_fused(
            inter_dim,
            topk,
            "none",
            gui_layout=True,
            act=act,
            enable_bias=use_splitk_bias,
            swiglu_limit=swiglu_limit,
        )
        _run_compiled(
            _silu_fused_k,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1).view(torch.uint8),
                out_scale_sorted_flat,
                sorted_token_ids,
                num_valid_ids,
                topk_ids_arg,
                bias_arg,
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _splitk_fp4:
        _silu_fused_k = _get_compiled_silu_fused(
            inter_dim,
            topk,
            act=act,
            enable_bias=use_splitk_bias,
            swiglu_limit=swiglu_limit,
        )
        _run_compiled(
            _silu_fused_k,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1).view(torch.uint8),
                out_scale_sorted_flat,
                sorted_token_ids,
                num_valid_ids,
                topk_ids_arg,
                bias_arg,
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _is_splitk:
        from aiter.ops.activation import (
            silu_and_mul,
            silu_and_mul_bias,
            swiglu_and_mul,
            swiglu_and_mul_bias,
        )

        post_input = tmp_out.view(-1, inter_dim * 2)
        post_out = out.view(-1, inter_dim)
        post_bias = bias.contiguous() if bias is not None else None
        if bias is not None and act == "swiglu":
            swiglu_and_mul_bias(post_out, post_input, topk_ids_arg, post_bias)
        elif bias is not None and act == "silu":
            silu_and_mul_bias(post_out, post_input, topk_ids_arg, post_bias)
        elif act == "swiglu":
            swiglu_and_mul(post_out, post_input)
        else:
            if bias is not None:
                post_input = post_input + bias[topk_ids.to(torch.long)].view(
                    -1, inter_dim * 2
                )
            silu_and_mul(post_out, post_input)

    if _fuse_any_quant and _need_sort:
        from aiter.utility.dtypes import fp8_e8m0

        out_scale_sorted = out_scale_sorted_flat.view(fp8_e8m0).view(
            padded_rows, padded_cols
        )
        return out, out_scale_sorted

    return out


def flydsl_moe_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    sort_block_m: int = 0,
    persist: Optional[bool] = None,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).
    bias: optional (E, model_dim) f32 bias added after GEMM.

    sort_block_m: block_size used by moe_sorting / stage1. When 0 (default),
        assumed equal to tile_m. When set, stage2 can use a different tile_m
        from sorting/stage1.
    persist: if True, use persistent round-robin mode (grid_y=cu_num);
        if False, use legacy persist_m mode; if None, auto-select.
    """

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    accumulate = mode != "reduce"

    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    if out is None:
        alloc_fn = torch.zeros if accumulate else torch.empty
        out = alloc_fn(
            (token_num, model_dim), dtype=torch_out_dtype, device=inter_states.device
        )
    elif accumulate:
        out.fill_(0)

    dev = inter_states.device
    flat_a_scale = (
        a2_scale.view(-1) if a2_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w2_scale.view(-1) if w2_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(sorted_token_ids.shape, dtype=torch.float32, device=dev)
    )

    _sbm = sort_block_m if sort_block_m > 0 else tile_m
    if _sbm == tile_m:
        m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)
    else:
        total_sorted = sorted_expert_ids.shape[0] * _sbm
        m_blocks = (total_sorted + tile_m - 1) // tile_m
    if persist is True:
        _persist_m = -1
    elif persist is False:
        _persist_m = 4 if m_blocks > 256 else 1
    else:
        _persist_m = -1 if m_blocks > 256 else 1

    if a_dtype == "fp8":
        _persist_m = 1

    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    target = out
    if not accumulate:
        target = torch.empty(
            (token_num * topk * model_dim,),
            device=out.device,
            dtype=out.dtype,
        )

    if is_fp4:
        args = _s2_args_fp4(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
            dev,
            bias=bias,
        )
    else:
        args = _s2_args_std(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
        )

    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=_persist_m,
        sort_block_m=sort_block_m,
        b_nt=b_nt,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        xcd_swizzle=xcd_swizzle,
        enable_bias=(bias is not None),
    )
    _run_compiled(exe, args)

    if not accumulate:
        torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)

    return out
