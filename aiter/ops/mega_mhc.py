# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP delayed Mega-mHC with BF16 or fused MXFP8 output in a single launch."""

import os
import threading
from dataclasses import dataclass
from functools import cache

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

__all__ = [
    "MEGA_MHC_RES_KS",
    "MegaMHCWorkspace",
    "mega_mhc",
    "mega_mhc_post_shuffled",
    "mega_mhc_res_preshuffle_supported",
    "mega_mhc_res_repeat",
    "mega_mhc_res_shuffle",
    "mega_mhc_res_unshuffle",
    "mega_mhc_shuffle_fn",
    "precompile_mega_mhc_post_shuffled",
    "prepare_mega_mhc",
]


_STREAM_HIDDEN_SIZES = (5120, 7168)

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
    # gfx1250: IDs 14.. select the kernels in csrc/kernels/mega_mhc_gfx1250.cu.
    "gfx1250_split32",
    "gfx1250_split16",
    "gfx1250_split8",
    "gfx1250_split4",
    "gfx1250_split2",
    "gfx1250_large",
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
        "gfx1250_split32",
        "gfx1250_split16",
        "gfx1250_split8",
        "gfx1250_split4",
        "gfx1250_split2",
    )
)

# Counters carry generation tags and partials are compact [split, T, 32].
# gfx1250_large uses them for the split tiles of a partial final CU round.
_TAGGED_VARIANTS = _DISTRIBUTED_VARIANTS | {"gfx1250_large"}

# Largest split first; each token tile and split gets its own CU.
_GFX1250_SPLITS = (32, 16, 8, 4, 2)


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


def _select_gfx1250_variant(tokens, cu):
    if tokens == 0:
        return "generic", 1
    blocks = (tokens + 15) // 16
    for split in _GFX1250_SPLITS:
        if blocks * split <= cu:
            return f"gfx1250_split{split}", split
    return "gfx1250_large", 1


def mega_mhc_shuffle_fn(fn):
    """[H/32, stream, operand, lane, 8] BF16; the last operand packs hi/lo.

    The first 16 projection rows use two operands; rows 16..23 use the two
    halves of one operand. This reduces four MFMA instructions to three.
    """
    if not fn.is_cuda or fn.dtype != torch.float32 or fn.ndim != 2:
        raise ValueError("fn must be a GPU FP32 matrix")
    if fn.shape[0] != 24 or fn.shape[1] % 128 or not fn.is_contiguous():
        raise ValueError("fn must be contiguous (24, 4*H), H divisible by 32")
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
def _native_launcher(hidden):
    # Fixed names share the same JSON recipes between JIT and wheel prebuilds.
    suffix = f"h{hidden}" if hidden else "generic"

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
        res_preshuffle: bool,
        scale_row_major: bool,
    ) -> None: ...

    launch.__name__ = f"_mega_mhc_stream_{suffix}"
    launch.__qualname__ = launch.__name__
    return compile_ops(
        f"module_mega_mhc_stream_{suffix}",
        fc_name="mega_mhc_bf16",
        develop=True,
    )(launch)


@dataclass(frozen=True)
class MegaMHCWorkspace:
    """Scratch and dispatch configuration for one shape on one device/stream.

    Contains no weights; serial calls with different packs can share it.
    Explicitly prepared workspaces own private buffers. Keep them alive for any
    graph capturing them and reuse only on their preparation device/stream.
    For automatic tuning, split_k records scratch capacity: post kernels may
    use fewer splits. The same workspace supports both post and no-post calls.
    """

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


class _Scratch:
    """Weight-independent scratch shared by serial calls on one device/stream.

    Kernels re-arm counters themselves. Only allocation/growth clears them.
    Keep replaced buffers alive for captured graphs; geometric growth bounds
    retained storage. Graphs using the same scratch must also replay serially.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.buffers = {}
        self.retired = []

    def get(self, device, n_partials, n_counters):
        key = (device, torch.cuda.current_stream(device).cuda_stream)
        with self.lock:
            part, cnt = self.buffers.get(key, (None, None))
            grow_part = part is None or part.numel() < n_partials
            grow_cnt = cnt is None or cnt.shape[1] < n_counters
            if (grow_part or grow_cnt) and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up Mega-MHC scratch on the capture stream before capture"
                )
            if grow_part:
                capacity = max(n_partials, 32 if part is None else 2 * part.numel())
                if part is not None:
                    self.retired.append(part)
                part = torch.empty(capacity, device=device, dtype=torch.float32)
            if grow_cnt:
                capacity = max(n_counters, 32 if cnt is None else 2 * cnt.shape[1])
                if cnt is not None:
                    self.retired.append(cnt)
                # Generic and generation-tagged protocols need separate counters.
                cnt = torch.zeros((2, capacity), device=device, dtype=torch.int32)
            self.buffers[key] = (part, cnt)
            return part[:n_partials], cnt[0, :n_counters], cnt[1, :n_counters]


_SCRATCH = _Scratch()


def prepare_mega_mhc(
    device: torch.device,
    tokens: int,
    hidden: int,
    *,
    split_k: int | None = None,
):
    """Prepare private scratch from dimensions on the current stream of device.

    H=5120/7168 on gfx950 and gfx1250 uses the tuned stream kernels.
    Explicit split_k, other H and gfx942 select the portable generic kernel
    with the same weight layout. The workspace also supports non-default
    Sinkhorn repeats via the generic path. It never receives or stores
    weights. Prepare before graph capture; pass the packed weight separately
    to each mega_mhc call.
    """
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_mega_mhc must run before graph capture")
        return _prepare_scratch(device, tokens, hidden, split_k, reuse_scratch=False)


def _prepare_scratch(device, tokens, hidden, split_k=None, *, reuse_scratch=True):
    """Select the kernel and obtain its buffers from dimensions only."""
    if (
        not isinstance(tokens, int)
        or tokens < 0
        or tokens > (2**31 - 1) // 32
        or not 0 < hidden <= 65536
        or hidden % 32
    ):
        raise ValueError("unsupported T or H")
    with torch.cuda.device(device):
        arch = get_gfx_runtime()
        if arch not in ("gfx950", "gfx942", "gfx1250"):
            raise RuntimeError(
                f"mega_mhc requires gfx950, gfx942 or gfx1250, got {arch}"
            )
        blocks = (tokens + 15) // 16
        cu = torch.cuda.get_device_properties(device).multi_processor_count
        if arch == "gfx950" and hidden in _STREAM_HIDDEN_SIZES:
            variant, selected_split = _select_stream_variant(tokens, hidden)
        elif arch == "gfx1250" and hidden in _STREAM_HIDDEN_SIZES:
            variant, selected_split = _select_gfx1250_variant(tokens, cu)
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
        if distributed and (masked or (arch != "gfx1250" and cu < 256)):
            variant, selected_split, distributed = "generic", 1, False
        tagged = variant in _TAGGED_VARIANTS
        split_k = selected_split
        if (
            not isinstance(split_k, int)
            or not 1 <= split_k <= 32
            or hidden % (32 * split_k)
        ):
            raise ValueError("split_k must be 1..32 and H divisible by 32*split_k")
        # The distributed kernels have a bounded wait and handoff path;
        # still keep the tuned grid at no more than one WG per CU.
        if distributed and (
            blocks * split_k > cu
            if arch == "gfx1250"
            else ((blocks + 7) // 8) * split_k > 32
        ):
            raise ValueError("distributed split grid exceeds one workgroup per CU")
        if reuse_scratch:
            storage, generic_counters, distributed_counters = _SCRATCH.get(
                device, split_k * blocks * 16 * 32, blocks * 32
            )
            generic_partials = storage.view(split_k, blocks, 16, 32)
            counters = distributed_counters if tagged else generic_counters
        else:
            generic_partials = torch.empty(
                (split_k, blocks, 16, 32), device=device, dtype=torch.float32
            )
            # Initialize both protocols in one operation for a private workspace.
            counter_storage = torch.zeros(
                (2 if tagged else 1, blocks * 32),
                device=device,
                dtype=torch.int32,
            )
            counters = counter_storage[0]
            generic_counters = counter_storage[1] if tagged else counters
        partials = (
            generic_partials.view(-1)[: split_k * tokens * 32]
            if tagged
            else generic_partials
        )
        return MegaMHCWorkspace(
            partials,
            counters,
            tokens,
            hidden,
            split_k,
            torch.cuda.current_stream(device).cuda_stream,
            variant,
            generic_partials,
            generic_counters,
        )


def _counter_state_ok(workspace):
    """Synchronizing diagnostic for tests; never call during graph capture."""
    counters = workspace.counters
    if workspace.variant in _TAGGED_VARIANTS:
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
    w_preshuffle_bf16: bool = False,
    out_dtype: str = "bf16",
    res_preshuffle: bool = False,
    scale_row_major: bool = False,
):
    """Delayed post/pre/RMSNorm seam, returning (R, post, comb, norm, next_pre).

    Same BF16 semantics as mhc_fused_post_pre_delayed_rmsnorm. pre_mix=None
    specializes to stream 0; sublayer_out=None skips post-mix. RMS/gates and
    MFMA accumulation use FP32; activation/weight operands use BF16 hi/lo.

    Pack fn once after loading with mega_mhc_shuffle_fn and pass
    w_preshuffle_bf16=True. The wrapper reuses weight-independent scratch per
    device/stream and launches one HIP kernel after warmup. Warm up on the
    capture stream with sufficient scratch capacity before graph capture.
    Calls and graphs sharing that scratch must execute serially. The read-only
    pack can be shared across streams after the loading stream has finished.
    Pass an explicit workspace to own private scratch instead. Raw FP32 fn is
    not accepted; w_preshuffle_bf16 must be True, with or without a workspace.
    H=5120/7168 on gfx950 and gfx1250 uses stream dispatch for Sinkhorn=20;
    other positive repeat counts use the generic stream kernel with the same
    pack.

    out_dtype="mxfp8" (gfx950 and gfx1250) returns (norm, q, e8m0) in place of
    norm: the BF16 norm (T, H), float8_e4m3fn q (T, H) and uint8 E8M0 exponent
    bytes, one per group of 32 columns. The call is the same on both
    architectures. The exponent bytes use the layout of each architecture's
    MXFP8 GEMM:

    * gfx950: (T, H/32), row-major.
    * gfx1250: (pad32(T), H/32) in the gfx1250 MXFP8 GEMM A-scale layout, that
      is the bytes of aiter.ops.shuffle.shuffle_mxfp8fp4_scale. Padding rows
      hold 0x7F. H must be divisible by 128.

    scale_row_major=True makes gfx1250 write the gfx950 layout, (T, H/32)
    row-major with no padding rows, for consumers that read row-major A-scales,
    such as the group32 GEMM with 32x32 weight scales. gfx950 already writes it.

    Quantization uses the rounded BF16 norm and group32 power-of-two scales,
    ceil_pow2(max(amax, 1e-10) / 448), matching FlyDSL Mega-MHC. Reinterpret
    the scale bytes with .view(torch.float8_e8m0fnu) for consumers that require
    E8M0. The same workspace supports both output modes, including graph replay.

    res_preshuffle=True (gfx1250 only): residual and the returned residual are
    in the aiter.ops.mhc.mhc_res_shuffle layout with ks=32, that is
    resS[k // 32][stream][token][k % 32] in the same (T, 4, H) shape.
    sublayer_out and the layer input stay plain.
    """
    assert w_preshuffle_bf16, "mega_mhc requires w_preshuffle_bf16=True"
    if not residual.is_cuda or residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("residual must be a GPU (T, 4, H) tensor")
    t, _, h = residual.shape
    device = residual.device
    if out_dtype not in ("bf16", "mxfp8"):
        raise ValueError("out_dtype must be 'bf16' or 'mxfp8'")
    _require_tensor(residual, "residual", (t, 4, h), torch.bfloat16, device)
    _require_tensor(fn, "fn", (h // 32, 4, 3, 4, 16, 4), torch.int32, device)
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
        arch = get_gfx_runtime()
        if out_dtype == "mxfp8" and arch not in ("gfx950", "gfx1250"):
            raise RuntimeError("Mega-MHC mxfp8 requires gfx950 or gfx1250")
        if out_dtype == "mxfp8" and arch == "gfx1250" and h % 128:
            raise ValueError("Mega-MHC mxfp8 on gfx1250 requires H divisible by 128")
        if res_preshuffle and arch != "gfx1250":
            raise RuntimeError("Mega-MHC res_preshuffle requires gfx1250")
        if workspace is None:
            scratch = _prepare_scratch(device, t, h)
        else:
            if workspace.partials.device != device or (
                workspace.tokens,
                workspace.hidden,
            ) != (t, h):
                raise ValueError("workspace must match residual device and (T, H)")
            if workspace.stream != torch.cuda.current_stream(device).cuda_stream:
                raise ValueError("prepare a separate workspace on the current stream")
            scratch = workspace
        post_out = torch.empty((t, 4, 1), device=device, dtype=torch.float32)
        comb_out = torch.empty((t, 4, 4), device=device, dtype=torch.float32)
        norm = torch.empty((t, h), device=device, dtype=torch.bfloat16)
        next_pre = torch.empty((t, 4), device=device, dtype=torch.float32)
        dummy = hc_base
        quant_out, quant_scale = dummy, dummy
        layer_input = norm
        if out_dtype == "mxfp8":
            quant_out = torch.empty((t, h), device=device, dtype=torch.float8_e4m3fn)
            # gfx1250 pads rows to 32 for the m32k4 GEMM A-scale layout.
            m32k4 = arch == "gfx1250" and not scale_row_major
            scale_rows = (t + 31) // 32 * 32 if m32k4 else t
            quant_scale = torch.empty(
                (scale_rows, h // 32), device=device, dtype=torch.uint8
            )
            layer_input = (norm, quant_out, quant_scale)
        partials, counters = scratch.partials, scratch.counters
        variant = scratch.variant
        if sinkhorn_repeat != 20:
            variant = "generic"
            partials, counters = (
                scratch.generic_partials,
                scratch.generic_counters,
            )
        launch = _native_launcher(h if h in _STREAM_HIDDEN_SIZES else 0)
        launch(
            residual_out,
            post_out,
            comb_out,
            norm,
            quant_out,
            quant_scale,
            next_pre,
            residual,
            fn,
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
            scratch.split_k,
            has_post,
            pre_mix is None,
            _STREAM_VARIANTS.index(variant),
            out_dtype == "mxfp8",
            res_preshuffle,
            scale_row_major,
        )
    return residual_out, post_out, comb_out, layer_input, next_pre


# ---------------------------------------------------------------------------
# Carrying the residual in the res_preshuffle layout between Mega-MHC calls.
#
# mega_mhc(res_preshuffle=True) reads and returns the residual as
# resS[k // MEGA_MHC_RES_KS][stream][token][k % MEGA_MHC_RES_KS], same
# (T, hc, H) shape. A model that keeps its residual in that layout from the
# embedding to the end of the stack needs to enter it (repeat), leave it
# (unshuffle), and, where a sublayer's post cannot fold into the next
# mega_mhc call, apply the post in place of mhc_post (mega_mhc_post_shuffled).
# Unlike aiter.ops.mhc's mhc_res_* helpers these never pick the block width
# or the layout from the row count: the caller decides, and every helper
# here takes and returns the layout mega_mhc reads, for any rows.
# ---------------------------------------------------------------------------

#: Block width of the residual layout mega_mhc(res_preshuffle=True) reads.
MEGA_MHC_RES_KS = 32


@cache
def _res_layout_ops():
    try:
        from .mhc import (
            mhc_res_repeat_flydsl,
            mhc_res_shuffle_flydsl,
            mhc_res_unshuffle_flydsl,
        )
    except ImportError:
        return None
    return mhc_res_repeat_flydsl, mhc_res_shuffle_flydsl, mhc_res_unshuffle_flydsl


@cache
def mega_mhc_res_preshuffle_supported() -> bool:
    """Whether mega_mhc(res_preshuffle=True) and the helpers below run here."""
    return get_gfx_runtime() == "gfx1250" and _res_layout_ops() is not None


def _res_rows(residual: Tensor) -> Tensor:
    hc, hidden = residual.shape[-2:]
    return residual.reshape(-1, hc, hidden).contiguous()


def mega_mhc_res_shuffle(residual: Tensor) -> Tensor:
    """Plain (..., hc, H) residual -> the res_preshuffle layout, same shape."""
    return _res_layout_ops()[1](_res_rows(residual), MEGA_MHC_RES_KS).view_as(residual)


def mega_mhc_res_unshuffle(residual: Tensor) -> Tensor:
    """res_preshuffle-layout (..., hc, H) residual -> the plain layout."""
    return _res_layout_ops()[2](_res_rows(residual), MEGA_MHC_RES_KS).view_as(residual)


def mega_mhc_res_repeat(hidden_states: Tensor, hc_mult: int) -> Tensor:
    """(..., H) hidden states -> the (..., hc_mult, H) residual of hc_mult
    copies, written straight in the res_preshuffle layout."""
    dim = hidden_states.shape[-1]
    rows = hidden_states.reshape(-1, dim).contiguous()
    return _res_layout_ops()[0](rows, hc_mult, MEGA_MHC_RES_KS).view(
        *hidden_states.shape[:-1], hc_mult, dim
    )


# `num_tokens` only scales strides that are multiples of KS, so specializing on
# its value buys nothing: one compile per launch config, whatever the rows.
@triton.jit(do_not_specialize=["num_tokens"])
def _mega_mhc_post_shuffled_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    num_tokens,
    DIM: tl.constexpr,
    TB: tl.constexpr,
    CH: tl.constexpr,
    KS: tl.constexpr,
):
    # hc = 4. One program: TB tokens x CH columns, all four streams, so each
    # stream's rows are TB * KS contiguous elements per KS block.
    tok = (tl.program_id(0) * TB + tl.arange(0, TB)).to(tl.int64)
    col = tl.program_id(1) * CH + tl.arange(0, CH)
    tlive = tok < num_tokens
    live = tlive[:, None] & (col < DIM)[None, :]
    x = tl.load(x_ptr + tok[:, None] * DIM + col[None, :], live, 0).to(tl.float32)
    base = (
        (col // KS)[None, :] * (4 * num_tokens * KS)
        + tok[:, None] * KS
        + (col % KS)[None, :]
    )
    stream = num_tokens * KS
    r0 = tl.load(res_ptr + base, live, 0).to(tl.float32)
    r1 = tl.load(res_ptr + base + stream, live, 0).to(tl.float32)
    r2 = tl.load(res_ptr + base + 2 * stream, live, 0).to(tl.float32)
    r3 = tl.load(res_ptr + base + 3 * stream, live, 0).to(tl.float32)
    for b in tl.static_range(4):
        # comb_res_mix[token, input_stream, output_stream]
        c0 = tl.load(comb_ptr + tok * 16 + 0 * 4 + b, tlive, 0)[:, None]
        c1 = tl.load(comb_ptr + tok * 16 + 1 * 4 + b, tlive, 0)[:, None]
        c2 = tl.load(comb_ptr + tok * 16 + 2 * 4 + b, tlive, 0)[:, None]
        c3 = tl.load(comb_ptr + tok * 16 + 3 * 4 + b, tlive, 0)[:, None]
        p = tl.load(post_ptr + tok * 4 + b, tlive, 0)[:, None]
        mixed = c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3
        out = mixed + p * x
        tl.store(out_ptr + base + b * stream, out, live)


# (rows upper bound, (TB, CH, num_warps)), from a gfx1250 H=5120 sweep: small
# tiles while the grid still fills the device, then taller token tiles, whose
# TB * KS contiguous elements per KS block keep the loads and stores long.
_POST_SHUFFLED_LAUNCH = (
    (256, (8, 128, 4)),
    (512, (8, 64, 4)),
    (None, (32, 64, 2)),
)


def _post_shuffled_launch(rows: int):
    for bound, config in _POST_SHUFFLED_LAUNCH:
        if bound is None or rows <= bound:
            return config


def _launch_post_shuffled(x, residual, post_mix, combination, out, rows, config):
    tb, ch, warps = config
    hidden = residual.shape[-1]
    _mega_mhc_post_shuffled_kernel[(triton.cdiv(rows, tb), triton.cdiv(hidden, ch))](
        x,
        residual,
        post_mix,
        combination,
        out,
        rows,
        DIM=hidden,
        TB=tb,
        CH=ch,
        KS=MEGA_MHC_RES_KS,
        num_warps=warps,
        enable_fp_fusion=False,
    )


def mega_mhc_post_shuffled(
    sublayer_out: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """mhc_post on a res_preshuffle-layout residual, returning that layout.

    new[t, b] = sum_a comb_res_mix[t, a, b] * residual[t, a]
                + post_layer_mix[t, b] * sublayer_out[t], FP32 then BF16,
    with residual and the result as (..., 4, H) in the res_preshuffle layout
    and sublayer_out (..., H) plain. For a post that cannot fold into the
    next mega_mhc call (e.g. a module that reads the residual in between).
    """
    hc, hidden = residual.shape[-2:]
    if hc != 4 or hidden % MEGA_MHC_RES_KS:
        raise ValueError(
            f"mega_mhc_post_shuffled needs hc=4 and H % {MEGA_MHC_RES_KS} == 0"
        )
    rows = residual.numel() // (hc * hidden)
    out = torch.empty_like(residual)
    if rows:
        _launch_post_shuffled(
            sublayer_out.reshape(rows, hidden).contiguous(),
            residual.contiguous(),
            post_layer_mix.reshape(rows, hc).float().contiguous(),
            comb_res_mix.reshape(rows, hc, hc).float().contiguous(),
            out,
            rows,
            _post_shuffled_launch(rows),
        )
    return out


def precompile_mega_mhc_post_shuffled(hidden: int, device) -> None:
    """Compile every launch config of mega_mhc_post_shuffled on one dummy row,
    so no step pays a Triton compile (`num_tokens` is not specialized: one
    compile per config, whatever the rows). Call before graph capture."""
    x = torch.zeros(1, hidden, dtype=torch.bfloat16, device=device)
    residual = torch.zeros(1, 4, hidden, dtype=torch.bfloat16, device=device)
    post_mix = torch.zeros(1, 4, dtype=torch.float32, device=device)
    combination = torch.zeros(1, 4, 4, dtype=torch.float32, device=device)
    out = torch.empty_like(residual)
    for _, config in _POST_SHUFFLED_LAUNCH:
        _launch_post_shuffled(x, residual, post_mix, combination, out, 1, config)
