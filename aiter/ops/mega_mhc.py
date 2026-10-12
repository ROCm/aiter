# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP delayed Mega-mHC with BF16 or fused MXFP8 output in a single launch."""

import os
from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

__all__ = ["MegaMHCWorkspace", "mega_mhc", "prepare_mega_mhc"]


_STREAM_HIDDEN_SIZES = (4096, 5120, 6144, 7168)

_STREAM_VARIANTS = (
    "decode4",
    "decode_small",
    "decode8",
    "small16",
    "small128",
    "small",
    "mid256",
    "mid384",
    "mid512",
    "mid1024",
    "mid2048",
    "large",
    "tail",
    "generic",
)

_DISTRIBUTED_VARIANTS = frozenset(
    (
        "decode_small",
        "decode4",
        "decode8",
        "small",
        "small16",
        "small128",
        "mid256",
        "mid384",
        "mid512",
        "mid1024",
        "mid2048",
    )
)


def _select_stream_variant(tokens, hidden=5120):
    if tokens == 0:
        return "generic", 1
    if tokens == 4:
        return "decode4", 32
    if tokens < 4:
        return "decode_small", 32
    if tokens <= 8:
        return "decode8", 32
    if tokens == 16:
        return "small16", 32
    if tokens == 128:
        return "small128", 32
    if tokens <= 128:
        return "small", 32
    for limit, split in ((256, 16), (384, 10), (512, 8), (1024, 4), (2048, 2)):
        if tokens <= limit:
            if limit == 384 and hidden != 5120:
                split = 8
            return f"mid{limit}", split
    if tokens == 4160:
        return "tail", 1
    return "large", 1


def _pack_stream_fn(fn):
    """[H/32, stream, operand, lane, 8] BF16; the last operand packs hi/lo.

    The first 16 projection rows use two operands; rows 16..23 use the two
    halves of one operand. This reduces four MFMA instructions to three.
    """
    with torch.no_grad():
        hidden = fn.shape[1] // 4
        hi = fn.to(torch.bfloat16)
        lo = (fn - hi.float()).to(torch.bfloat16)
        hi = hi.view(24, 4, hidden)
        lo = lo.view(24, 4, hidden)
        operands = torch.stack([hi[:16], lo[:16], torch.cat([hi[16:], lo[16:]])])
        return (
            operands.view(3, 16, 4, hidden // 32, 4, 8)
            .permute(3, 2, 0, 4, 1, 5)
            .contiguous()
            .view(torch.int32)
        )


@cache
def _native_launcher(hidden=5120, arch="gfx950"):
    # Compile all token/output specializations once for this H and architecture.
    if not 0 < hidden <= 65536 or hidden % 32:
        raise ValueError(f"unsupported Mega-MHC hidden size: {hidden}")
    if arch not in ("gfx950", "gfx942"):
        raise ValueError(f"unsupported Mega-MHC architecture: {arch}")

    def launch(
        residual_out: Tensor,
        post_out: Tensor,
        comb_out: Tensor,
        layer_input: Tensor,
        quant_out: Tensor,
        quant_scale: Tensor,
        next_pre: Tensor,
        residual: Tensor,
        fn_packed: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
        pre_mix: Tensor,
        sublayer_out: Tensor,
        post_mix: Tensor,
        comb_mix: Tensor,
        norm_weight: Tensor,
        partials: Tensor,
        counters: Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult: float,
        norm_eps: float,
        sinkhorn_repeat: int,
        split_k: int,
        has_post: bool,
        identity_pre: bool,
        variant: int,
        mxfp8: bool,
    ) -> None: ...

    base_module = "module_mega_mhc_stream"
    module = f"{base_module}_h{hidden}_{arch}"
    launch.__name__ = "_" + module.removeprefix("module_")
    launch.__qualname__ = launch.__name__

    def build_args(*args, **kwargs):
        flags = [
            "-gline-tables-only",
            f"-DMHC_HIDDEN={hidden}",
            f"-DMHC_ARCH_GFX950={int(arch == 'gfx950')}",
        ]
        return {
            "md_name": module,
            "flags_extra_hip": flags,
        }

    return compile_ops(
        base_module,
        fc_name="mega_mhc_bf16",
        gen_func=build_args,
        develop=True,
    )(launch)


@dataclass(frozen=True)
class MegaMHCWorkspace:
    """Owns the weight pack and scratch for one shape on one stream.

    Keep this object alive for the lifetime of any graph capturing it. Reuse on
    its preparation stream only; prepare another workspace for another stream.
    Updating fn requires preparing a new workspace. Inference tensors have no
    version counter, so callers must keep those weights immutable. No scratch is grown/freed
    behind already captured graphs, and no process-global scratch is shared.
    For automatic tuning, split_k records scratch capacity: post kernels may
    use fewer splits. The same workspace supports both post and no-post calls.
    """

    fn: Tensor
    fn_version: int | None
    fn_packed: Tensor
    partials: Tensor
    counters: Tensor
    tokens: int
    hidden: int
    split_k: int
    stream: int
    variant: str = "generic"
    generic_partials: Tensor | None = None
    generic_counters: Tensor | None = None


def _require_tensor(x, name, shape, dtype, device):
    if not isinstance(x, Tensor) or x.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if x.dtype != dtype or x.device != device or not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtype} on {device}")


def _fn_version(fn):
    return None if fn.is_inference() else fn._version


def prepare_mega_mhc(fn: Tensor, tokens: int, *, split_k: int | None = None):
    """Pack FP32 fn and initialize per-stream scratch outside graph capture.

    H=4096/5120/6144/7168 on gfx950 uses the tuned stream kernels. Explicit split_k,
    other H and gfx942 select the portable generic kernel with the same weight
    layout. The workspace also supports non-default Sinkhorn repeats via the
    generic path, sharing the weight pack without allocating in capture.
    """
    if not fn.is_cuda or fn.dtype != torch.float32 or fn.ndim != 2:
        raise ValueError("fn must be a GPU FP32 matrix")
    if fn.shape[0] != 24 or fn.shape[1] % 128 or not fn.is_contiguous():
        raise ValueError("fn must be contiguous (24, 4*H), H divisible by 32")
    hidden = fn.shape[1] // 4
    if (
        not isinstance(tokens, int)
        or tokens < 0
        or tokens > (2**31 - 1) // 32
        or not 0 < hidden <= 65536
    ):
        raise ValueError("unsupported T or H")
    with torch.cuda.device(fn.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_mega_mhc must run before graph capture")
        arch = get_gfx_runtime()
        if arch not in ("gfx950", "gfx942"):
            raise RuntimeError(f"mega_mhc requires gfx950 or gfx942, got {arch}")
        blocks = (tokens + 15) // 16
        cu = torch.cuda.get_device_properties(fn.device).multi_processor_count
        if arch == "gfx950" and hidden in _STREAM_HIDDEN_SIZES:
            variant, selected_split = _select_stream_variant(tokens, hidden)
        else:
            variant = "generic"
            selected_split = max(
                k
                for k in range(1, 21)
                if hidden % (32 * k) == 0 and (k == 1 or max(blocks, 1) * k <= cu)
            )
        if split_k is not None:
            variant, selected_split = "generic", split_k
        distributed = variant in _DISTRIBUTED_VARIANTS
        masked = any(
            os.environ.get(name) is not None
            for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")
        )
        if distributed and (cu < 256 or masked):
            variant, selected_split, distributed = "generic", 1, False
        split_k = selected_split
        if (
            not isinstance(split_k, int)
            or not 1 <= split_k <= 32
            or hidden % (32 * split_k)
        ):
            raise ValueError("split_k must be 1..32 and H divisible by 32*split_k")
        # The distributed kernels have a bounded wait and handoff path;
        # still keep the tuned grid at no more than one WG per CU.
        if distributed and ((blocks + 7) // 8) * split_k > 32:
            raise ValueError("distributed split grid exceeds one workgroup per CU")
        generic_partials = torch.empty(
            (split_k, blocks, 16, 32), device=fn.device, dtype=torch.float32
        )
        partials = (
            generic_partials.view(-1)[: split_k * tokens * 32]
            if distributed
            else generic_partials
        )
        counters = torch.zeros(blocks * 32, device=fn.device, dtype=torch.int32)
        # Generation-tagged counters cannot be reused by the generic
        # zero-based arrival protocol. Both sets belong to this workspace.
        generic_counters = torch.zeros_like(counters) if distributed else counters
        return MegaMHCWorkspace(
            fn,
            _fn_version(fn),
            _pack_stream_fn(fn),
            partials,
            counters,
            tokens,
            hidden,
            split_k,
            torch.cuda.current_stream(fn.device).cuda_stream,
            variant,
            generic_partials,
            generic_counters,
        )


def _counter_state_ok(workspace):
    """Synchronizing diagnostic for tests; never call during graph capture."""
    counters = workspace.counters
    if workspace.variant in _DISTRIBUTED_VARIANTS:
        counters = counters.view(-1, 32)
        generation = (counters[:, 0].to(torch.int64) & 0xFFFFFFFF) >> 8
        idle = (
            not bool((counters[:, 0] & 255).count_nonzero().item())
            and not bool(counters[:, 2].count_nonzero().item())
            and torch.equal(counters[:, 3].to(torch.int64), generation)
            and not bool(counters[:, 4:].count_nonzero().item())
        )
    else:
        idle = not bool(counters.count_nonzero().item())
    return idle and (
        workspace.generic_counters is None
        or not bool(workspace.generic_counters.count_nonzero().item())
    )


def mega_mhc(
    residual: Tensor,
    fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: Tensor | None,
    sublayer_out: Tensor | None = None,
    post_layer_mix: Tensor | None = None,
    comb_res_mix: Tensor | None = None,
    norm_weight: Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    residual_out: Tensor | None = None,
    workspace: MegaMHCWorkspace | None = None,
    out_dtype: str = "bf16",
):
    """Delayed post/pre/RMSNorm seam, returning (R, post, comb, norm, next_pre).

    Same BF16 semantics as mhc_fused_post_pre_delayed_rmsnorm. pre_mix=None
    specializes to stream 0; sublayer_out=None skips post-mix. RMS/gates and
    MFMA accumulation use FP32; activation/weight operands use BF16 hi/lo.

    With a prepared workspace this launches exactly one HIP kernel. Without
    one, this convenience wrapper also packs fn and initializes scratch on
    every call. Prepare once and warm up on the capture stream before capture.
    H=4096/5120/6144/7168 on gfx950 uses stream dispatch for Sinkhorn=20; other
    positive repeat counts use the generic stream kernel with the same pack.

    out_dtype="mxfp8" (gfx950 only) returns (norm, q, e8m0) in place of norm:
    BF16 norm (T, H), float8_e4m3fn q (T, H), and uint8 exponent bytes (T, H/32).
    Quantization uses the rounded BF16 norm and group32 power-of-two scales,
    ceil_pow2(max(amax, 1e-10) / 448), matching FlyDSL Mega-MHC. Reinterpret
    scale bytes with .view(torch.float8_e8m0fnu) for consumers requiring E8M0.
    The same workspace supports both output modes, including graph replay.
    """
    if not residual.is_cuda or residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("residual must be a GPU (T, 4, H) tensor")
    t, _, h = residual.shape
    device = residual.device
    if out_dtype not in ("bf16", "mxfp8"):
        raise ValueError("out_dtype must be 'bf16' or 'mxfp8'")
    _require_tensor(residual, "residual", (t, 4, h), torch.bfloat16, device)
    _require_tensor(fn, "fn", (24, 4 * h), torch.float32, device)
    _require_tensor(hc_scale, "hc_scale", (3,), torch.float32, device)
    _require_tensor(hc_base, "hc_base", (24,), torch.float32, device)
    _require_tensor(norm_weight, "norm_weight", (h,), torch.bfloat16, device)
    if h <= 0 or h % 32 or h > 65536:
        raise ValueError("H must be a positive multiple of 32, at most 65536")
    if pre_mix is not None:
        _require_tensor(pre_mix, "pre_mix", (t, 4), torch.float32, device)
    has_post = sublayer_out is not None
    if has_post:
        _require_tensor(sublayer_out, "sublayer_out", (t, h), torch.bfloat16, device)
        if post_layer_mix is None or post_layer_mix.shape not in ((t, 4), (t, 4, 1)):
            raise ValueError("post_layer_mix must be (T, 4) or (T, 4, 1)")
        _require_tensor(
            post_layer_mix,
            "post_layer_mix",
            post_layer_mix.shape,
            torch.float32,
            device,
        )
        _require_tensor(comb_res_mix, "comb_res_mix", (t, 4, 4), torch.float32, device)
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        _require_tensor(
            residual_out, "residual_out", residual.shape, torch.bfloat16, device
        )
        if (
            t
            and residual_out.untyped_storage().data_ptr()
            == residual.untyped_storage().data_ptr()
        ):
            raise ValueError("residual_out must not share storage with residual")
    else:
        if (
            residual_out is not None
            or post_layer_mix is not None
            or comb_res_mix is not None
        ):
            raise ValueError(
                "no_post requires residual_out/post_layer_mix/comb_res_mix=None"
            )
        residual_out = residual
    with torch.cuda.device(device):
        if out_dtype == "mxfp8" and get_gfx_runtime() != "gfx950":
            raise RuntimeError("Mega-MHC mxfp8 requires gfx950")
        if workspace is None:
            workspace = prepare_mega_mhc(fn, t)
        if (
            workspace.fn is not fn
            or workspace.fn_version != _fn_version(fn)
            or (workspace.tokens, workspace.hidden) != (t, h)
        ):
            raise ValueError("workspace must match fn, its version, and (T, H)")
        if workspace.stream != torch.cuda.current_stream(device).cuda_stream:
            raise ValueError("prepare a separate workspace on the current stream")
        post_out = torch.empty((t, 4, 1), device=device, dtype=torch.float32)
        comb_out = torch.empty((t, 4, 4), device=device, dtype=torch.float32)
        norm = torch.empty((t, h), device=device, dtype=torch.bfloat16)
        next_pre = torch.empty((t, 4), device=device, dtype=torch.float32)
        dummy = hc_base
        quant_out, quant_scale = dummy, dummy
        layer_input = norm
        if out_dtype == "mxfp8":
            quant_out = torch.empty((t, h), device=device, dtype=torch.float8_e4m3fn)
            quant_scale = torch.empty((t, h // 32), device=device, dtype=torch.uint8)
            layer_input = (norm, quant_out, quant_scale)
        partials, counters = workspace.partials, workspace.counters
        variant = workspace.variant
        if sinkhorn_repeat != 20:
            variant = "generic"
            partials, counters = (
                workspace.generic_partials,
                workspace.generic_counters,
            )
        launch = _native_launcher(h, get_gfx_runtime())
        launch(
            residual_out,
            post_out,
            comb_out,
            norm,
            quant_out,
            quant_scale,
            next_pre,
            residual,
            workspace.fn_packed,
            hc_scale,
            hc_base,
            pre_mix if pre_mix is not None else dummy,
            sublayer_out if has_post else residual,
            post_layer_mix if has_post else dummy,
            comb_res_mix if has_post else dummy,
            norm_weight,
            partials,
            counters,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            norm_eps,
            sinkhorn_repeat,
            workspace.split_k,
            has_post,
            pre_mix is None,
            _STREAM_VARIANTS.index(variant),
            out_dtype == "mxfp8",
        )
    return residual_out, post_out, comb_out, layer_input, next_pre
