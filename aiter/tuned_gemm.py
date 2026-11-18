"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2026, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import functools
import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from aiter import dtypes, gemm_a16w16_asm, hipb_create_extension, hipb_mm, logger
from aiter.jit.core import AITER_CONFIGS, AITER_LOG_MORE, AITER_LOG_TUNED_CONFIG
from aiter.jit.utils.build_targets import base_gfx_arch
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.gemm_op_common import get_padded_m
from aiter.ops.gradlib import _hipb_mm
from aiter.plugin import _config as _plugin_config
from aiter.plugin import route as _route

try:
    from aiter.ops.opus import opus_gemm as _opus_launch
except Exception:  # noqa: BLE001  blanket catch is intentional here
    _opus_launch = None


@functools.lru_cache(maxsize=1)
def _get_flydsl_gemm_kernels():
    from aiter.ops.flydsl import gemm_kernels

    return gemm_kernels


this_dir = os.path.dirname(os.path.abspath(__file__))

extensions_created = False
untune_path = f"{this_dir}/configs/bf16_untuned_gemm.csv"
tune_path = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
tuned_df = pd.DataFrame(
    columns=[
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ]
)


@functools.lru_cache(maxsize=1)
def get_GEMM_A16W16_config_():
    tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
    gemm_dict = {}
    if os.path.exists(tuned_file):
        gemm_dict = pd.read_csv(f"{tuned_file}").drop_duplicates()
        gemm_dict = gemm_dict.set_index(
            [
                "gfx",
                "cu_num",
                "M",
                "N",
                "K",
                "bias",
                "dtype",
                "outdtype",
                "scaleAB",
                "bpreshuffle",
            ]
        ).to_dict("index")
    return gemm_dict


def is_skinny_default_shape(
    M: int,
    N: int,
    K: int,
    dtype,
    cu_num: int | None = None,
):
    if isinstance(dtype, str):
        dtype = eval(dtype)
    cu_num = get_cu_num() if cu_num is None else cu_num
    return (
        dtype in [dtypes.fp16, dtypes.bf16]
        and K % 8 == 0
        and (
            (
                ((M == 1 and N <= 2 * cu_num) or (M > 1 and M <= 4 and N <= cu_num))
                and K <= 9216
            )
            or ((M > 4 and M <= 8 and N <= cu_num) and K <= 5120)
            or ((M > 8 and M <= 16 and N <= cu_num) and K <= 256)
        )
    )


def get_GEMM_A16W16_tuned_config(
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: str,
    otype: str,
    scaleAB: bool = False,
    bpreshuffle: bool = False,
    gfx: str | None = None,
    cu_num: int | None = None,
):
    """Return the tuned config for this GEMM, or None if no usable row exists.

    Rows naming a FlyDSL kernel missing from the catalog, or an OPUS candidate
    that no longer resolves, are skipped as if absent.
    """
    cfg = get_GEMM_A16W16_config_()
    cu_num = get_cu_num() if cu_num is None else cu_num
    padded_M = M
    config = None
    if gfx is None:
        gfx = get_gfx_runtime()
    # A0 bring-up: a gfx1250-strict build has no tuned rows of its own; reuse
    # the gfx1250 family rows for coverage (perf parity is a later follow-up).
    # An exact strict row still wins when one exists; the guard keeps the scan
    # off the normal (non-strict) runtime path entirely.
    if base_gfx_arch(gfx) != gfx and gfx not in {
        k[0] for k in cfg if isinstance(k, tuple) and k
    }:
        gfx = base_gfx_arch(gfx)
    warned_invalid_opus = set()
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = cfg.get(
            (
                gfx,
                cu_num,
                padded_M,
                N,
                K,
                bias,
                str(dtype),
                str(otype),
                scaleAB,
                bpreshuffle,
            ),
            None,
        )
        if config is not None:
            if config["libtype"] == "flydsl":
                flydsl_config = (
                    _get_flydsl_gemm_kernels().get_flydsl_hgemm_kernel_params(
                        config["kernelName"]
                    )
                )
                # None means the tuned CSV names a kernel absent from this
                # catalog version; it is unrelated to FlyDSL import availability.
                if flydsl_config is None or (
                    bpreshuffle
                    and (
                        flydsl_config["policy"] not in ("ft", "sk")
                        or (
                            flydsl_config["policy"] == "ft"
                            and (
                                flydsl_config["split_k"] != 1
                                or flydsl_config["k_waves"] != 1
                            )
                        )
                        or gfx != "gfx950"
                        or str(dtype) != str(dtypes.bf16)
                    )
                ):
                    logger.warning(
                        f"FlyDSL kernel '{config['kernelName']}' from tuned config is not "
                        "recognized by the current catalog; falling back to next candidate."
                    )
                    config = None
            if config is None:
                continue
            if config["libtype"] == "opus":
                resolved = None
                if _opus_launch is not None:
                    from aiter.ops.opus.policy import resolve_a16w16_tuned_candidate
                    from csrc.opus_gemm.opus_gemm_common import (
                        get_kernel_instance,
                        parse_integer,
                    )

                    try:
                        kid = parse_integer(config["solidx"], name="saved kernel id")
                        instance = get_kernel_instance(
                            gfx, "a16w16", kid, cu_num=cu_num
                        )
                        name = str(config.get("kernelName", ""))
                        if instance is not None and (
                            name in ("", "nan", "None") or name == instance.name
                        ):
                            resolved = resolve_a16w16_tuned_candidate(
                                arch=gfx,
                                M=M,
                                N=N,
                                K=K,
                                batch=1,
                                cu_num=cu_num,
                                has_bias=bias,
                                input_dtype=eval(dtype),
                                output_dtype=eval(otype),
                                requested_kid=kid,
                                requested_split_k=config.get("splitK"),
                            )
                    except (ValueError, KeyError, TypeError):
                        pass
                if resolved is None:
                    # Discard the whole stale (kid, split-K) pair before
                    # trying another padded row or the default fallback.
                    invalid_row = (
                        padded_M,
                        config.get("solidx"),
                        config.get("splitK"),
                    )
                    if invalid_row not in warned_invalid_opus:
                        logger.warning(
                            "Ignoring invalid OPUS tuned row for gfx=%s, "
                            "shape=(%d,%d,%d), kid=%r, splitK=%r; trying "
                            "the next padded row or default backend",
                            gfx,
                            padded_M,
                            N,
                            K,
                            config.get("solidx"),
                            config.get("splitK"),
                        )
                        warned_invalid_opus.add(invalid_row)
                    config = None
                    continue
                config = dict(config)
                config["solidx"] = int(resolved.resolved_kid)
            if AITER_LOG_TUNED_CONFIG:
                kernelName = (
                    config["kernelName"] if config["libtype"] != "hipblaslt" else ""
                )
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, {bpreshuffle=} found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE}, libtype is {config['libtype']}, kernel name is {kernelName}"
                )
            return config
    return None


@functools.lru_cache(maxsize=4096)
def get_GEMM_A16W16_config(
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: str,
    otype: str,
    scaleAB: bool = False,
    bpreshuffle: bool = False,
    gfx: str | None = None,
    cu_num: int | None = None,
):
    cu_num = get_cu_num() if cu_num is None else cu_num
    if gfx is None:
        gfx = get_gfx_runtime()
    config = get_GEMM_A16W16_tuned_config(
        M, N, K, bias, dtype, otype, scaleAB, bpreshuffle, gfx, cu_num
    )
    if config is None:
        default_config = {}
        if bpreshuffle:
            default_config["bpreshuffle"] = True
            if gfx == "gfx942":
                default_config["libtype"] = "hipblaslt"
                default_config["solidx"] = -1
                default_config["kernelName"] = ""
            elif (
                eval(dtype) == dtypes.bf16
                and N % 64 == 0
                and K % 64 == 0
                and (eval(otype) == dtypes.bf16 or eval(otype) == dtypes.fp32)
            ):
                default_config["libtype"] = "asm"
                default_config["solidx"] = 0
                default_config["splitK"] = None
                default_config["kernelName"] = None
            else:
                assert (
                    False
                ), f"no solution for {M=} {N=} {K=} {dtype=} {bias=}, {scaleAB=}, {bpreshuffle=}"
        # `otype == dtype` because the skinny kernels allocate their output at
        # the input dtype and write it there; picking one when the caller asked
        # for a wider output means the widening happens after the values have
        # already been rounded, which is not what asking for it meant. Those
        # shapes go to torch instead, which writes the accumulator.
        elif (
            gfx in ("gfx90a", "gfx942", "gfx950")
            and otype == dtype
            and is_skinny_default_shape(M, N, K, dtype, cu_num)
        ):
            # soltype, solution_idx = 3, 2
            default_config["libtype"] = "skinny"
            default_config["solidx"] = 2
            default_config["kernelName"] = ""
        if not default_config:
            # gfx1250 has no tuned ASM/skinny/hipblaslt bf16 kernels, so the
            # torch fallback lands on hipBLASLt, which is markedly slower than
            # the Triton (gluon) a16w16 kernel for these shapes. Prefer Triton
            # for unscaled bf16/fp16 GEMMs; explicit tuned CSV entries still win
            # since they are matched before this fallback is reached.
            if (
                gfx in ("gfx1250", "gfx1250-strict")
                and not scaleAB
                and eval(dtype) in (dtypes.bf16, dtypes.fp16)
            ):
                default_config["libtype"] = "triton"
                default_config["solidx"] = 0
            else:
                default_config["libtype"] = "torch"
                default_config["solidx"] = 0
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, {bpreshuffle=}, not found tuned config in {AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE}, will use default config! using {default_config['libtype']} solution:{default_config['solidx']}"
        )
        return default_config

    return config


@functools.cache
def _record_gemm_shapes() -> bool:
    """AITER_TUNE_GEMM, read once: shape recording is on or off for the run."""
    return bool(int(os.environ.get("AITER_TUNE_GEMM", "0")))


def save_shapes(
    M,
    N,
    K,
    bias,
    dtype,
    otype,
    scaleAB,
    bpreshuffle,
):
    global tuned_df
    if _record_gemm_shapes():
        tuned_df = pd.concat(
            [
                tuned_df,
                pd.DataFrame(
                    {
                        "M": [M],
                        "N": [N],
                        "K": [K],
                        "bias": [bias is not None],
                        "dtype": [dtype],
                        "outdtype": [otype],
                        "scaleAB": [scaleAB],
                        "bpreshuffle": [bpreshuffle],
                    }
                ),
            ]
        ).drop_duplicates()
        tuned_df.to_csv(untune_path, index=False)


def gen_gemm_a16w16_fake_tensor(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
) -> Tensor:
    return torch.empty(
        *A.shape[:-1],
        B.shape[0],
        dtype=otype or A.dtype,
        device=A.device,
    )


@torch_compile_guard(mutates_args=[], gen_fake=gen_gemm_a16w16_fake_tensor)
def gemm_a16w16(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
) -> Tensor:
    return _gemm_a16w16(A, B, bias, otype, scale_a, scale_b, scale_c)


@torch_compile_guard(mutates_args=["out"], gen_fake=lambda *a, **k: None)
def gemm_a16w16_out(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
) -> None:
    """`gemm_a16w16` written into ``out`` -- the [M, N] tensor of ``otype`` --
    for a caller whose consumer reads one fixed address."""
    result = _gemm_a16w16(A, B, bias, otype, scale_a, scale_b, scale_c, out)
    # The return value is dropped here, so a provider that answers in a tensor
    # of its own would leave `out` unwritten; the native path asserts the same.
    if result.data_ptr() != out.data_ptr():
        raise RuntimeError("gemm_a16w16_out: the result was not written into out=")


# The custom ops: what torch.compile records, and what runs every call the
# eager fast path below declines.
_gemm_a16w16_op = gemm_a16w16
_gemm_a16w16_out_op = gemm_a16w16_out
# `_eager_body` is set only when the guard registered the op itself; a
# registration made elsewhere may run a different body, which the fast path
# must not stand in for.
_eager_ops_are_ours = all(
    hasattr(op, "_eager_body") for op in (gemm_a16w16, gemm_a16w16_out, _hipb_mm)
)

# Eager fast path. Outside torch.compile a hipBLASLt GEMM spends far longer in
# Python than in C++: two torch.ops dispatches (aiter::gemm_a16w16, then
# aiter::_hipb_mm), the routing in `_gemm_a16w16`, the gradlib wrapper and the
# per-call argument conversion of the develop-mode op. `_hipblaslt_eager` makes
# the same C++ call directly -- the same config lookup, solution index and
# arguments, so the same algorithm and the same bits -- and declines (None)
# whenever that path may do something else.
_eager_hipb = None  # pybind entry points once resolved; False: off for the run
_eager_gfx_cu: dict[int, tuple[str, int]] = {}
_eager_tensor_types: dict[type, bool] = {torch.Tensor: True, torch.nn.Parameter: True}
_eager_weight_handles: OrderedDict = OrderedDict()
_EAGER_WEIGHT_HANDLES_MAX = 1024


def _eager_tensor_type(cls: type) -> bool:
    """Whether tensors of type ``cls`` may skip the dispatcher.

    A subclass that takes over ``__torch_dispatch__`` (FakeTensor, functional
    and wrapper tensors) has to see the op. One that only overrides
    ``__torch_function__`` is caught per call by ``torch._C._has_torch_function``.
    """
    ok = _eager_tensor_types.get(cls)
    if ok is None:
        ok = issubclass(cls, torch.Tensor) and (
            cls.__torch_dispatch__ is torch.Tensor.__torch_dispatch__
        )
        _eager_tensor_types[cls] = ok
    return ok


def _eager_weight_t_handle(B: Tensor, convert):
    """The aiter_tensor_t of ``B.t()``, cached per tensor descriptor.

    The handle is a function of (data_ptr, shape, stride, dtype, device) alone
    and holds no reference to the tensor, so a cached one is exactly what a
    fresh conversion would build, whether or not ``B`` is still alive. A bounded
    LRU: weights stay, activations passed as ``B`` rotate out.
    """
    key = (B.data_ptr(), B.shape, B.stride(), B.dtype, B.get_device())
    handles = _eager_weight_handles
    handle = handles.get(key)
    if handle is None:
        handle = handles[key] = convert(B.t())
        if len(handles) > _EAGER_WEIGHT_HANDLES_MAX:
            handles.popitem(last=False)
    else:
        try:
            handles.move_to_end(key)
        except KeyError:  # evicted by another thread meanwhile
            pass
    return handle


def _resolve_eager_hipb():
    """The pybind entry points `_hipblaslt_eager` calls, once they exist.

    None until the custom-op path has created the hipBLASLt extension: the first
    hipBLASLt GEMM of a process always runs there. False, for the rest of the
    run, when that path has per-call side effects someone asked for (shape
    recording, per-op argument logging) or when an op runs a body other than
    the one registered here.
    """
    global _eager_hipb
    if not extensions_created:
        return None
    from aiter.jit.core import _pybind_develop_hooks, get_module

    if not _eager_ops_are_ours or _record_gemm_shapes() or AITER_LOG_MORE >= 2:
        _eager_hipb = False
        return False
    convert, _, raw_stream, current_device = _pybind_develop_hooks()
    module = get_module("module_hipbsolgemm")
    _eager_hipb = (
        module._set_current_hip_stream,
        module.hipb_mm,
        convert,
        raw_stream,
        current_device,
    )
    return _eager_hipb


def _hipblaslt_eager(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None,
    otype: torch.dtype | None,
    out: Tensor | None,
) -> Tensor | None:
    """An unscaled `gemm_a16w16` (``out`` None) or `gemm_a16w16_out`, run
    without the dispatcher when its tuned config is hipBLASLt.

    Returns the result -- ``out`` itself when given -- or None to leave the call
    to the custom op: while a kernel plugin may serve `gemm_a16w16`, under
    torch.compile or jit tracing, inside any dispatch mode, torch-function mode
    or functorch transform, for tensor subclasses that intercept ops, inputs
    autograd would record, inputs `_gemm_a16w16` does not simply reshape, and
    every config that is not hipBLASLt.
    ``get_GEMM_A16W16_config`` is called through the module attribute on every
    call, exactly as `_gemm_a16w16` calls it, so a replacement installed at any
    time is honoured and its results are never cached here.
    """
    if torch.compiler.is_compiling():
        return None
    hipb = _eager_hipb
    if hipb is None:
        hipb = _resolve_eager_hipb()
    if not hipb:
        return None
    # `_gemm_a16w16` is routed: unless no provider can serve any call, which is
    # when its dispatch runs the native body and records nothing, the call must
    # go through the route.
    plugin_config = _plugin_config()
    if plugin_config.enabled and not plugin_config.kill_switch:
        return None
    set_stream, hipb_mm_op, convert, raw_stream, current_device = hipb
    known = _eager_tensor_types.get
    if (
        torch._C._len_torch_dispatch_stack()
        or torch._C._are_functorch_transforms_active()
        or torch._C._is_tracing()
        or torch._C._has_torch_function((A, B, bias, out))
        or not (known(type(A)) or _eager_tensor_type(type(A)))
        or not (known(type(B)) or _eager_tensor_type(type(B)))
        or not (bias is None or known(type(bias)) or _eager_tensor_type(type(bias)))
        or not (out is None or known(type(out)) or _eager_tensor_type(type(out)))
    ):
        return None
    if torch.is_grad_enabled() and (
        A.requires_grad
        or B.requires_grad
        or (bias is not None and bias.requires_grad)
        or (out is not None and out.requires_grad)
    ):
        return None
    if B.dim() != 2:
        return None
    ndim = A.dim()
    if ndim == 2:
        inp = A
    elif ndim > 2:
        try:
            inp = A.view(-1, A.size(-1))
        except RuntimeError:
            return None
    else:
        return None
    index = inp.get_device()
    if index < 0:
        return None
    m, k = inp.shape
    n = B.shape[0]
    if otype is None:
        otype = inp.dtype
    elif type(otype) is not torch.dtype:
        return None
    if out is not None and out.dtype != otype:
        return None
    bpreshuffle = getattr(B, "is_shuffled", None) is True
    gfx, cu_num = _eager_gfx_cu.get(index) or _eager_gfx_cu.setdefault(
        index, _gemm_gfx_cu(inp.device)
    )
    config = get_GEMM_A16W16_config(
        M=m,
        N=n,
        K=k,
        bias=bias is not None,
        dtype=str(inp.dtype),
        otype=str(otype),
        scaleAB=False,
        bpreshuffle=bpreshuffle,
        gfx=gfx,
        cu_num=cu_num,
    )
    if not isinstance(config, dict) or config.get("libtype") != "hipblaslt":
        return None
    if out is None:
        result = torch.empty((m, n), dtype=otype, device=inp.device)
    else:
        try:
            result = out.view(-1, n)
        except RuntimeError:
            return None
    # What hipb_gemm -> hipb_mm -> the develop-mode _hipb_mm wrapper hand over.
    set_stream(raw_stream(current_device()))
    hipb_mm_op(
        convert(inp),
        _eager_weight_t_handle(B, convert),
        config["solidx"],
        convert(result),
        None if bias is None else convert(bias),
        None,
        None,
        None,
        bpreshuffle,
        None,
        None,
        None,
    )
    if out is not None:
        return out
    return result if ndim == 2 else result.view(*A.shape[:-1], n)


def gemm_a16w16(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
) -> Tensor:
    """``A @ B.T`` (+ ``bias``) in ``otype``: the tuned BF16/FP16 GEMM."""
    if scale_a is None and scale_b is None and scale_c is None:
        result = _hipblaslt_eager(A, B, bias, otype, None)
        if result is not None:
            return result
    return _gemm_a16w16_op(A, B, bias, otype, scale_a, scale_b, scale_c)


def gemm_a16w16_out(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
) -> None:
    """`gemm_a16w16` written into ``out`` -- the [M, N] tensor of ``otype`` --
    for a caller whose consumer reads one fixed address."""
    if (
        scale_a is None
        and scale_b is None
        and scale_c is None
        and _hipblaslt_eager(A, B, bias, otype, out) is not None
    ):
        return
    _gemm_a16w16_out_op(A, B, out, bias, otype, scale_a, scale_b, scale_c)


def _gemm_gfx_cu(device: torch.device) -> tuple[str, int]:
    """The (gfx, cu_num) a GEMM on ``device`` looks its tuned config up with."""
    from aiter.ops.opus._arch import SUPPORTED_OPUS_ARCHES, _device_arch_and_cu

    gfx, cu_num = _device_arch_and_cu(device)
    if gfx not in SUPPORTED_OPUS_ARCHES:
        gfx = get_gfx_runtime()
        cu_num = get_cu_num()
    return gfx, cu_num


# Routed here, beneath both guarded spellings and `tgemm.mm`, for the reason
# `rmsnorm` is: every one of them arrives here, so this is where the op begins.
@_route(op="gemm_a16w16", version=1)
def _gemm_a16w16(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Dense GEMM ``A @ B.T + bias`` in ``otype``, by the tuned backend.

    Returns the ``[*A.shape[:-1], N]`` result. Without ``out`` it is a newly
    allocated tensor and nothing is written through the arguments. With
    ``out`` the result is written into ``out`` and ``out`` is returned, viewed
    to that shape: a provider serving this contract must write it there, since
    `gemm_a16w16_out` drops the return value and reads only the buffer.

    Args:
        A: ``[..., K]`` activations, bf16 or fp16 (fp8 with the scales below);
            the leading dims are the ``M`` rows.
        B: ``[N, K]`` weights. ``B.is_shuffled`` True says ``B`` holds the
            preshuffled layout the asm kernels read, not row-major bytes.
        bias: ``[N]``, added to every row, or None.
        otype: output dtype; None is ``A``'s dtype.
        scale_a, scale_b, scale_c: per-tensor scales of a scaled GEMM (``A``
            and ``B`` dequant, output); all None for a plain 16-bit GEMM.
        out: ``[M, N]`` buffer of ``otype``, or one shaped like the result;
            None allocates the result.
    """
    bpreshuffle = False
    if hasattr(B, "is_shuffled") and B.is_shuffled is True:
        bpreshuffle = True
    if A.dim() >= 3:
        try:
            inp_view = A.view(-1, A.size(-1))
            batched = True
        except RuntimeError:
            if not bpreshuffle:
                return _into(out, F.linear(A, B, bias))
            inp_view = A.reshape(-1, A.size(-1))
            batched = True
    else:
        inp_view = A
        batched = False
    m, k = inp_view.shape
    n = B.shape[0]
    use_bias = bias is not None
    otype = otype if otype is not None else inp_view.dtype
    if out is not None and out.dtype != otype:
        raise ValueError(f"gemm_a16w16: out is {out.dtype} but otype is {otype}")
    gfx, cu_num = _gemm_gfx_cu(inp_view.device)
    config = get_GEMM_A16W16_config(
        M=m,
        N=n,
        K=k,
        bias=use_bias,
        dtype=str(inp_view.dtype),
        otype=str(otype),
        scaleAB=scale_a is not None or scale_b is not None,
        bpreshuffle=bpreshuffle,
        gfx=gfx,
        cu_num=cu_num,
    )
    libtype = config["libtype"]
    solution_idx = config["solidx"]
    solfunc = solMap[libtype]
    out_2d = None if out is None else out.view(-1, n)
    result = solfunc(
        inp_view,
        B,
        solution_idx,
        bias,
        otype,
        scale_a,
        scale_b,
        scale_c,
        bpreshuffle,
        config=config,
        out=out_2d,
    )
    # Every backend must answer in the caller's buffer: one that hands back a
    # tensor of its own leaves `out` unwritten, and nothing downstream notices.
    assert (
        out_2d is None or result.data_ptr() == out_2d.data_ptr()
    ), f"the {libtype} gemm did not write into out="
    if batched:
        result = result.view(*A.shape[:-1], B.shape[0])
    if otype is not None and result.dtype != otype:
        result = result.to(otype)
    save_shapes(
        m,
        n,
        k,
        bias,
        inp_view.dtype,
        otype,
        scale_a is not None or scale_b is not None,
        bpreshuffle,
    )
    return result


def _result_buffer(
    out: Tensor | None, m: int, n: int, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """The [m, n] buffer a backend writes: the caller's ``out``, else a new one."""
    return torch.empty(m, n, dtype=dtype, device=device) if out is None else out


def _into(out: Tensor | None, result: Tensor) -> Tensor:
    """``result``, or the caller's ``out`` holding it.

    For the paths that cannot write in place: a copy, so only off the tuned ones.
    A ``result`` that already is ``out`` is returned as is.
    """
    if out is None or result is out:
        return result
    return out.copy_(result)


def skinny_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
    out: Tensor | None = None,
):
    import aiter as ops

    assert not bpreshuffle, "bpreshuffle is not supported in skinny_gemm!"
    # The skinny kernels write the input dtype, so a caller's buffer of another
    # dtype takes the result by copy rather than as the kernels' output.
    direct = out if out is not None and out.dtype == inp.dtype else None
    result = _result_buffer(
        direct, inp.shape[0], weights.shape[0], inp.dtype, inp.device
    )
    if solidx == 0:
        ops.wvSpltK(weights, inp, result, inp.shape[0], get_cu_num())
    elif solidx == 1:
        ops.LLMM1(weights, inp, result, 4)
    if solidx == 2:
        ops.wv_splitk_small_fp16_bf16(weights, inp, result, inp.shape[0], get_cu_num())
    if bias is not None:
        result += bias
    if otype is not None and result.dtype != otype:
        result = result.to(otype)
    return _into(out, result)


def hipb_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
    out: Tensor | None = None,
):
    if otype is None:
        otype = inp.dtype
    global extensions_created
    if not extensions_created:
        hipb_create_extension()
        extensions_created = True
    return hipb_mm(
        inp,
        weights.t(),
        solidx,
        bias,
        otype,
        scale_a,
        scale_b,
        scale_c,
        bpreshuffle,
        out=out,
    )


def _torch_scaled_gemm(
    inp: Tensor,
    weights: Tensor,
    bias: Tensor | None,
    otype: torch.dtype | None,
    scale_a: Tensor | None,
    scale_b: Tensor | None,
) -> Tensor:
    """FP8 through `torch._scaled_mm`, or an FP32 emulation where it is refused."""
    if scale_a is None:
        scale_a = torch.ones(1, dtype=dtypes.fp32, device=inp.device)
    if scale_b is None:
        scale_b = torch.ones(1, dtype=dtypes.fp32, device=inp.device)
    try:
        return torch._scaled_mm(
            inp,
            weights.t(),
            out_dtype=otype,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
        )
    except RuntimeError:
        result = (
            F.linear(inp.to(dtypes.fp32), weights.to(dtypes.fp32)) * scale_a * scale_b
        )
        return (result.to(otype) + bias) if bias is not None else result.to(otype)


def torch_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
    out: Tensor | None = None,
):
    assert not bpreshuffle, "bpreshuffle is not supported in torch_gemm!"
    if inp.dtype == dtypes.fp8:
        return _into(
            out, _torch_scaled_gemm(inp, weights, bias, otype, scale_a, scale_b)
        )
    if otype == dtypes.fp32 and inp.dtype in (dtypes.bf16, dtypes.fp16):
        # `F.linear` returns the *input* dtype, so widening its result
        # afterwards preserves nothing: the values are already on the BF16
        # grid and a caller that asked for FP32 silently gets BF16 precision.
        # `out_dtype` writes the accumulator instead, which is what the tuned
        # ASM (`bf16gemm_fp32bf16_*`) and OPUS kernels do -- an untuned shape
        # must not disagree with them about what `otype` means.
        result = torch.mm(inp, weights.t(), out_dtype=otype, out=out)
        return result if bias is None else result.add_(bias)
    if out is not None and out.dtype == inp.dtype:
        # F.linear's own form, bias added in the accumulator.
        if bias is None:
            return torch.mm(inp, weights.t(), out=out)
        return torch.addmm(bias, inp, weights.t(), out=out)
    result = F.linear(inp, weights, bias)
    if otype is not None and result.dtype != otype:
        result = result.to(otype)
    return _into(out, result)


def asm_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
    out: Tensor | None = None,
):
    kernelName = config.get("kernelName") if config else None
    splitK = config.get("splitK") if config else None
    result = _result_buffer(out, inp.shape[0], weights.shape[0], otype, inp.device)
    return gemm_a16w16_asm(inp, weights, result, bias, splitK, kernelName, bpreshuffle)


@functools.cache
def _flydsl_hgemm_kernel_params(kernel_name: str) -> dict | None:
    """Parsed once per tuned name; `flydsl_gemm` only reads the result."""
    return _get_flydsl_gemm_kernels().get_flydsl_hgemm_kernel_params(kernel_name)


def flydsl_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
    out: Tensor | None = None,
):
    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "FlyDSL hgemm does not support scaling yet."
    flydsl_gemm_kernels = _get_flydsl_gemm_kernels()
    flydsl_config = _flydsl_hgemm_kernel_params(config["kernelName"])
    skinny = flydsl_config["policy"] == flydsl_gemm_kernels.SKINNY_POLICY
    fused_bias = None
    if (
        not skinny
        and bias is not None
        and (otype is None or otype == inp.dtype)
        and bias.dtype == inp.dtype
    ):
        fused_bias = bias
    if skinny or flydsl_config["policy"] in ("bs", *flydsl_gemm_kernels.TDM_POLICIES):
        policy = flydsl_config["policy"]
    else:
        policy = "ht" if flydsl_config["use_half_tile_interleaved"] else "ft"
    result = flydsl_gemm_kernels.flydsl_hgemm(
        inp,
        weights,
        out=out,
        bias=fused_bias,
        block_m=flydsl_config["block_m"],
        block_n=flydsl_config["block_n"],
        block_k=flydsl_config["block_k"],
        split_k=flydsl_config["split_k"],
        m_waves=flydsl_config["m_waves"],
        n_waves=flydsl_config["n_waves"],
        k_waves=flydsl_config["k_waves"],
        stages=flydsl_config["stages"],
        group_m=flydsl_config["group_m"],
        k_stagger=flydsl_config["k_stagger"],
        b_non_temporal=flydsl_config["b_non_temporal"],
        policy=policy,
        out_dtype=None if skinny else otype,
        bpreshuffle=bpreshuffle,
    )

    if bias is not None and fused_bias is None:
        # Into the caller's buffer it can only be added in place, at its dtype.
        result = result.to(bias.dtype) + bias if out is None else result.add_(bias)
    if otype is not None and result.dtype != otype:
        result = result.to(otype)
    return result


def opus_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle: bool | None = False,
    config: dict | None = None,
    out: Tensor | None = None,
):
    """Run one tuned OPUS A16W16 row through the exact-kid interface."""
    if _opus_launch is None:
        logger.warning(
            "opus tuned config found but opus is not available; falling back to torch"
        )
        return torch_gemm(
            inp,
            weights,
            solidx,
            bias,
            otype,
            scale_a,
            scale_b,
            scale_c,
            bpreshuffle,
            config,
        )
    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "opus_gemm does not support scaling"
    assert not bpreshuffle, "opus_gemm does not support bpreshuffle"
    splitK = int(config.get("splitK", 0)) if config is not None else 0
    m, _k = inp.shape
    n = weights.shape[0]
    Y = _result_buffer(out, m, n, otype or inp.dtype, inp.device)
    _opus_launch(
        inp,
        weights,
        Y,
        kid=int(solidx),
        bias=bias,
        split_k=splitK,
        family="a16w16",
    )
    # The OPUS launcher already applies bias, including split-K reduction.
    return Y


def triton_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle: bool | None = False,
    config: dict | None = None,
    out: Tensor | None = None,
):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "Triton gemm_a16w16 does not support scaling yet"
    assert not bpreshuffle, "Triton gemm_a16w16 does not support bpreshuffle yet."
    return gemm_a16w16(inp, weights, bias=bias, dtype=otype, y=out)


solMap = {
    "torch": torch_gemm,
    "hipblaslt": hipb_gemm,
    "skinny": skinny_gemm,
    "asm": asm_gemm,
    "triton": triton_gemm,
    "flydsl": flydsl_gemm,
    "opus": opus_gemm,
}


class TunedGemm:
    """bf16/fp16 with per tensor fp8 quant"""

    def __init__(self):
        # self.extensions_created = False
        self.save_gemm = int(os.environ.get("AITER_TUNE_GEMM", "0"))
        self.untune_path = f"{this_dir}/configs/bf16_untuned_gemm.csv"
        self.tune_path = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
        if self.save_gemm == 1:
            self.tuned_df = pd.DataFrame(
                columns=[
                    "M",
                    "N",
                    "K",
                    "bias",
                    "dtype",
                    "outdtype",
                    "scaleAB",
                    "bpreshuffle",
                ]
            )
        else:
            self.tuned_df = None

    def mm(
        self,
        inp: Tensor,
        weights: Tensor,
        bias: Tensor | None = None,
        otype: torch.dtype | None = None,
        scale_a: Tensor | None = None,
        scale_b: Tensor | None = None,
        scale_c: Tensor | None = None,
        out: Tensor | None = None,
    ):
        """``out``: write the result there (`gemm_a16w16_out`) and return it."""
        if out is None:
            return gemm_a16w16(
                inp,
                weights,
                bias=bias,
                otype=otype,
                scale_a=scale_a,
                scale_b=scale_b,
                scale_c=scale_c,
            )
        gemm_a16w16_out(
            inp,
            weights,
            out,
            bias=bias,
            otype=otype,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_c=scale_c,
        )
        return out


tgemm = TunedGemm()
