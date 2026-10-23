# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import contextlib
import importlib
import inspect
import itertools
import os
import tempfile
import zlib
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.aot.flydsl.common import override_env
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.int4_utils import convert_int8_to_uint32_int4, rearrange_4bit_elements
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_i4_quant
from aiter.ops.shuffle import (
    pack_int8_to_packed_int4,
    shuffle_scale_a16w4,
    shuffle_scale_for_int4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
_FUSED_MOE = importlib.import_module("aiter.fused_moe")
# --summed-only also runs against an AITER without per_slot_output.
HAS_PER_SLOT = "per_slot_output" in inspect.signature(fused_moe).parameters

# Muse Spark Open MoE (moe_hidden_size 4096, 256 experts, top-8) at TP8 (inter_dim
# 512 per rank) and at EP8 (inter_dim 4096, 32 local experts):
# model_dim, inter_dim, E, topk, ep.
TP8 = (4096, 512, 256, 8, 1)
EP8 = (4096, 4096, 256, 8, 8)


@dataclass(frozen=True)
class Family:
    q_type: QuantType
    a_dtype: torch.dtype | None
    w_dtype: torch.dtype | None
    activation: ActivationType = ActivationType.Silu
    gate_mode: GateMode = GateMode.SEPARATED
    env: tuple = ()
    gfx: tuple = ("gfx942", "gfx950")
    expert_mask: bool = True
    shapes: tuple = (TP8,)
    tokens: tuple = (16, 256)
    # Token counts of the extra variants: num_local_tokens (half the rows are
    # padding), prequantized activations, and a per-slot call split in chunks.
    num_local_tokens: tuple = ()
    prequant: tuple = ()
    chunked: tuple = ()
    # Summed selection pinned to a FlyDSL v2 GEMM2 with this epilogue.
    v2_epilog: str = ""


_PER_1X32 = dict(q_type=QuantType.per_1x32, gfx=("gfx950",))
_V2 = dict(
    a_dtype=dtypes.fp4x2,
    w_dtype=dtypes.fp4x2,
    shapes=(TP8, EP8),
    num_local_tokens=(256,),
    **_PER_1X32,
)
FAMILIES = {
    "bf16": Family(
        QuantType.No,
        None,
        None,
        shapes=(TP8, EP8),
        tokens=(1, 16, 128, 1024, 4096),
        chunked=(256,),
    ),
    "a4w4": Family(
        a_dtype=dtypes.fp4x2,
        w_dtype=dtypes.fp4x2,
        shapes=(TP8, EP8),
        # One count per tier of the per-slot default.
        tokens=(1, 16, 128, 1024, 4096, 8192),
        num_local_tokens=(16, 1024),
        prequant=(16, 256),
        chunked=(256,),
        **_PER_1X32,
    ),
    # model_dim below what the per-slot default's K-split stage 1 needs.
    "a4w4_small": Family(
        a_dtype=dtypes.fp4x2,
        w_dtype=dtypes.fp4x2,
        shapes=((256, 256, 16, 4, 1), (256, 256, 16, 4, 2)),
        tokens=(64,),
        **_PER_1X32,
    ),
    "a4w4_v2_atomic": Family(v2_epilog="atomic", **_V2),
    "a4w4_v2_reduce": Family(v2_epilog="reduce", **_V2),
    "a4w4_v2_scatter": Family(v2_epilog="scatter", **_V2),
    # fused_moe reaches a16w4 (bf16 x MXFP4) through SiTUv2, which rejects
    # expert_mask, so there is no EP case.
    "a16w4": Family(
        a_dtype=dtypes.bf16,
        w_dtype=dtypes.fp4x2,
        activation=ActivationType.Situv2,
        expert_mask=False,
        tokens=(1, 16, 128, 1024, 4096),
        num_local_tokens=(16, 1024),
        **_PER_1X32,
    ),
    # fp8 activations need the gate/up-interleaved layout; bound 0 keeps them
    # fp8 at every token count, as test_moe_2stage does.
    "a8w4": Family(
        a_dtype=dtypes.fp8,
        w_dtype=dtypes.fp4x2,
        gate_mode=GateMode.INTERLEAVE,
        env=(("AITER_BF16_FP8_MOE_BOUND", "0"),),
        shapes=(TP8, EP8),
        tokens=(16, 1024),
        **_PER_1X32,
    ),
    "a16wi4": Family(
        a_dtype=dtypes.bf16,
        w_dtype=dtypes.i4x2,
        shapes=(TP8, EP8),
        tokens=(1, 16, 128, 1024),
        num_local_tokens=(16, 1024),
        **_PER_1X32,
    ),
    "mxfp8": Family(
        a_dtype=dtypes.fp8,
        w_dtype=dtypes.fp8,
        gate_mode=GateMode.INTERLEAVE,
        prequant=(16, 256),
        **_PER_1X32,
    ),
    # bf16 activations below 256 tokens and fp4/fp8 above, so the expanded
    # path (topk times the rows) must keep the summed call's dtype.
    "a16w4_swiglu": Family(
        a_dtype=dtypes.bf16,
        w_dtype=dtypes.fp4x2,
        activation=ActivationType.Swiglu,
        expert_mask=False,
        tokens=(16, 64),
        **_PER_1X32,
    ),
    "a16w4_interleave": Family(
        a_dtype=dtypes.bf16,
        w_dtype=dtypes.fp4x2,
        activation=ActivationType.Swiglu,
        gate_mode=GateMode.INTERLEAVE,
        expert_mask=False,
        tokens=(16, 64),
        **_PER_1X32,
    ),
    "fp8_tensor": Family(QuantType.per_Tensor, dtypes.fp8, dtypes.fp8, prequant=(16,)),
    "fp8_token": Family(
        QuantType.per_Token, dtypes.fp8, dtypes.fp8, prequant=(16, 256)
    ),
    "fp8_block": Family(QuantType.per_128x128, dtypes.fp8, dtypes.fp8, prequant=(16,)),
    # CK int4 weights (packed into uint32) with fp8 per-token activations.
    "a8w4_ck": Family(QuantType.per_Token, dtypes.fp8, torch.uint32),
}


def quant_per_128x128(w, quant_dtype):
    E, dim1, dim2 = w.shape
    blocks = w.view(E, dim1 // 128, 128, dim2 // 128, 128).permute(0, 1, 3, 2, 4)
    w_q, scale = aiter.pertoken_quant(
        blocks.contiguous().view(E, -1, 128 * 128), quant_dtype=quant_dtype
    )
    w_q = w_q.view(E, dim1 // 128, dim2 // 128, 128, 128).permute(0, 1, 3, 2, 4)
    return w_q.contiguous().view(E, dim1, dim2), scale.view(E, dim1 // 128, dim2 // 128)


def quantize_weights(w, fam, is_w1):
    """(reference weight, reference scale, aiter weight, aiter scale)."""
    E = w.shape[0]
    if fam.q_type == QuantType.No:
        return w, None, shuffle_weight(w, layout=(16, 16)), None
    if fam.q_type == QuantType.per_Tensor:
        w_q, scale = aiter.pertoken_quant(w.view(E, -1), quant_dtype=fam.w_dtype)
        w_q = w_q.view(w.shape)
        return w_q, scale, shuffle_weight(w_q, layout=(16, 16)), scale
    if fam.q_type == QuantType.per_128x128:
        w_q, scale = quant_per_128x128(w, fam.w_dtype)
        return w_q, scale, shuffle_weight(w_q, layout=(16, 16)), scale
    if fam.w_dtype == torch.uint32:
        w_q, scale = aiter.pertoken_quant(w, quant_dtype=dtypes.i8, dtypeMax=7)
        w_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(shuffle_weight(w_q, (16, 16), use_int4=True))
        )
        return w_q, scale, w_aiter, scale
    if fam.q_type == QuantType.per_Token:
        w_q, scale = aiter.get_torch_quant(fam.q_type)(w, quant_dtype=fam.w_dtype)
        w_q = w_q.view(w.shape)
        return w_q, scale, shuffle_weight(w_q, layout=(16, 16)), scale
    if fam.w_dtype == dtypes.i4x2:
        w_q, scale = per_1x32_i4_quant(w)
        w_q = w_q.view(dtypes.i4x2).view(w.shape)
        w_aiter = pack_int8_to_packed_int4(
            shuffle_weight(w_q.view(dtypes.i8), (16, 16))
        )
        w_aiter = w_aiter.view(*w.shape[:-1], w.shape[-1] // 2).view(dtypes.i4x2)
        scale_aiter = shuffle_scale_for_int4(scale, group_size=32)
        return w_q, scale, w_aiter, scale_aiter.view(-1).contiguous()
    if fam.w_dtype == dtypes.fp8:
        w_q, scale = per_1x32_f8_scale_f8_quant(
            w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        w_q = w_q.view(w.shape)
        if is_w1:
            scale_aiter = shuffle_scale_a16w4(scale, E, True)
        else:
            scale_aiter = fp4_utils.e8m0_shuffle(scale)
        return w_q, scale, shuffle_weight_a16w4(w_q, 16, is_w1), scale_aiter
    w_q, scale = aiter.get_torch_quant(fam.q_type)(w, quant_dtype=dtypes.fp4x2)
    w_q = w_q.view(*w.shape[:-1], w.shape[-1] // 2)
    if fam.a_dtype == dtypes.fp4x2:
        return (
            w_q,
            scale,
            shuffle_weight(w_q, layout=(16, 16)),
            fp4_utils.e8m0_shuffle(scale),
        )
    gate_up = is_w1 and (
        fam.a_dtype == dtypes.fp8 or fam.gate_mode == GateMode.INTERLEAVE
    )
    return (
        w_q,
        scale,
        shuffle_weight_a16w4(w_q, 16, gate_up),
        shuffle_scale_a16w4(scale, E, gate_up),
    )


def quantize_activation(x, fam, stage1):
    """What the kernels consume, for the reference: (tensor, scale)."""
    if (
        fam.q_type == QuantType.No
        or fam.a_dtype == dtypes.bf16
        or (fam.q_type == QuantType.per_1x32 and fam.w_dtype == dtypes.fp8)
    ):
        return x, None
    if fam.q_type == QuantType.per_128x128:
        x_q, scale = aiter.pertoken_quant(
            x.view(x.shape[0], -1, 128), quant_dtype=fam.a_dtype
        )
        x_q = x_q.view(x.shape)
        if stage1:
            return x_q, scale.squeeze(-1)
        return x_q, scale.view(*x.shape[:2], -1)
    if fam.q_type == QuantType.per_1x32 and fam.a_dtype == dtypes.fp8:
        return per_1x32_f8_scale_f8_quant(
            x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    return aiter.get_torch_quant(fam.q_type)(x, quant_dtype=fam.a_dtype)


def situ_kwargs(fam):
    if fam.activation != ActivationType.Situv2:
        return {}
    return {"beta": DEFAULT_SITUV2_BETA, "linear_beta": DEFAULT_SITUV2_LINEAR_BETA}


def run_torch(
    x, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, fam, dtype, reduce
):
    """The torch MoE reference; [token, topk, model_dim] summands unless reduce."""
    a1, a1_scale = quantize_activation(x, fam, stage1=True)
    out1 = torch_moe_stage1(
        a1,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=fam.activation,
        quant_type=fam.q_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    )
    a2, a2_scale = quantize_activation(out1, fam, stage1=False)
    return torch_moe_stage2(
        a2.view(*topk_ids.shape, -1),
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=fam.q_type,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        **({} if reduce else {"reduce": False}),
    )


def logits_diff(ref, out, mask=None):
    """1 - cosine similarity, the fp4 MoE accuracy measure of test_moe_2stage."""
    x, y = ref.double(), out.double()
    if mask is not None:
        x, y = torch.where(mask, x, 0), torch.where(mask, y, 0)
    den = (x * x + y * y).sum()
    return 0.0 if den == 0 else float(1 - 2 * (x * y).sum() / den)


def inaccurate(err, diff, threshold=0.01):
    """test_moe_2stage's accuracy rule."""
    return err != 0 and diff > threshold


def replayed(fn, out):
    """out after one CUDA-graph replay of fn, with out poisoned before it."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    return out.clone()


# (model_dim, inter_dim, local experts) -> (GEMM1, v2 GEMM2 with "{}" for the
# epilogue): kernel pairs from AITER's Muse Spark Open MXFP4 tuned rows.
V2_KERNELS = {
    (4096, 512, 256): (
        "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_{}_nt_sbm32",
    ),
    (4096, 4096, 32): (
        "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_{}_nt_sbm32",
    ),
}
# The scatter epilogue supports only 128-row blocks.
V2_SCATTER_KERNELS = (
    "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_fp4",
    "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_scatter_sbm128",
    128,
)
_CONFIG_DIR = None


def v2_kernels(fam, model_dim, inter_dim, local_E):
    """(GEMM1, GEMM2, block_m) of the row pinned for fam.v2_epilog."""
    if fam.v2_epilog == "scatter":
        return V2_SCATTER_KERNELS
    kn1, kn2 = V2_KERNELS[(model_dim, inter_dim, local_E)]
    return kn1, kn2.format(fam.v2_epilog), 32


def v2_config(fam, dtype, token, model_dim, inter_dim, local_E, topk):
    """A tuned-config file whose one row pins the summed selection to a FlyDSL
    v2 GEMM2 with fam.v2_epilog."""
    global _CONFIG_DIR
    if _CONFIG_DIR is None:
        _CONFIG_DIR = tempfile.TemporaryDirectory(prefix="aiter_per_slot_")
    kn1, kn2, block_m = v2_kernels(fam, model_dim, inter_dim, local_E)
    row = dict(
        gfx=get_gfx(),
        cu_num=get_cu_num(),
        token=_FUSED_MOE.get_padded_M(token),
        model_dim=model_dim,
        inter_dim=inter_dim,
        expert=local_E,
        topk=topk,
        act_type=str(fam.activation),
        dtype=str(dtype),
        q_dtype_a=str(fam.a_dtype),
        q_dtype_w=str(fam.w_dtype),
        q_type=str(fam.q_type),
        use_g1u1=1,
        doweight_stage1=0,
        block_m=block_m,
        ksplit=0,
        us1=0,
        kernelName1=kn1,
        err1="",
        us2=0,
        kernelName2=kn2,
        err2="",
        us=0,
        run_1stage=0,
        xbf16=0,
        flat=0,
        tflops=0,
        bw=0,
        _tag="",
    )
    name = f"{fam.v2_epilog}-{token}-{model_dim}-{inter_dim}-{local_E}"
    path = os.path.join(_CONFIG_DIR.name, f"{name}.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


@contextlib.contextmanager
def tuned_rows(path):
    """AITER's tuned fused_moe rows replaced by the file at path."""

    def reload():
        # The resolved config path is cached by variable name, not value.
        type(AITER_CONFIGS).get_config_file.cache_clear()
        _FUSED_MOE.cfg_2stages = None
        _FUSED_MOE.get_2stage_cfgs.cache_clear()

    with override_env("AITER_CONFIG_FMOE", path):
        reload()
        try:
            yield
        finally:
            reload()


def prequantize(x, fam):
    """Activations quantized by the caller, as a quantizing dispatch hands them."""
    if fam.q_type == QuantType.per_1x32 and fam.a_dtype == dtypes.fp4x2:
        x_q, scale = aiter.get_torch_quant(QuantType.per_1x32)(
            x, quant_dtype=dtypes.fp4x2
        )
        return x_q.view(dtypes.fp4x2), scale
    if fam.q_type == QuantType.per_1x32:
        return per_1x32_f8_scale_f8_quant(
            x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    if fam.q_type == QuantType.per_128x128:
        x_q, scale = aiter.pertoken_quant(
            x.view(x.shape[0], -1, 128), quant_dtype=fam.a_dtype
        )
        return x_q.view(x.shape), scale.squeeze(-1)
    return aiter.get_torch_quant(fam.q_type)(x, quant_dtype=fam.a_dtype)


def rejects_prequant(fam):
    """The per-slot expanded path cannot split these scales by token and raises."""
    return fam.q_type == QuantType.per_128x128


@contextlib.contextmanager
def recorded_dispatch():
    """The activation dtype of every kernel lookup of a call, and for a per-slot
    call which stage 2 the summed selection picked and whether it ran per-slot."""
    seen = {"paths": [], "q_dtype_a": [], "default": []}
    per_slot_metadata = _FUSED_MOE._per_slot_metadata
    get_2stage_cfgs = _FUSED_MOE.get_2stage_cfgs
    default_cfg = getattr(_FUSED_MOE, "_per_slot_default_cfg", None)

    def record_path(metadata, *args, **kwargs):
        adapted = per_slot_metadata(metadata, *args, **kwargs)
        stage2 = metadata.stage2
        if stage2 is None:
            name = "whole-graph" if metadata.full_impl is not None else "1-stage"
        else:
            name = getattr(stage2, "keywords", {}).get("kernelName") or getattr(
                getattr(stage2, "func", stage2), "__name__", str(stage2)
            )
        path = "native" if adapted is not None else "expanded"
        seen["paths"].append(f"{path} {name}")
        return adapted

    def record_cfgs(*args, **kwargs):
        seen["q_dtype_a"].append(
            str(args[6] if len(args) > 6 else kwargs.get("q_dtype_a"))
        )
        return get_2stage_cfgs(*args, **kwargs)

    def record_default(*args, **kwargs):
        cfg = default_cfg(*args, **kwargs)
        if cfg is not None:
            seen["default"].append(cfg["kernelName2"])
        return cfg

    # Cached selections would skip the per-slot default's call.
    get_2stage_cfgs.cache_clear()
    _FUSED_MOE._per_slot_metadata = record_path
    _FUSED_MOE.get_2stage_cfgs = record_cfgs
    if default_cfg is not None:
        _FUSED_MOE._per_slot_default_cfg = record_default
    try:
        yield seen
    finally:
        _FUSED_MOE._per_slot_metadata = per_slot_metadata
        _FUSED_MOE.get_2stage_cfgs = get_2stage_cfgs
        if default_cfg is not None:
            _FUSED_MOE._per_slot_default_cfg = default_cfg


@contextlib.contextmanager
def per_slot_chunks(tokens):
    """Per-slot calls split every `tokens` rows, as above the 32-bit limit."""
    original = _FUSED_MOE._per_slot_max_tokens
    _FUSED_MOE._per_slot_max_tokens = lambda *args: tokens
    try:
        yield
    finally:
        _FUSED_MOE._per_slot_max_tokens = original


@benchmark()
def test_fmoe_per_slot(
    family,
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    ep,
    variant,
    summed_only,
):
    fam = FAMILIES[family]
    case = (family, str(dtype), token, model_dim, inter_dim, E, topk, ep, variant)
    seed = zlib.crc32(repr(case).encode())
    torch.manual_seed(seed)
    local_E = E // ep
    x = torch.randn((token, model_dim), dtype=dtype)
    w1 = torch.randn((local_E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((local_E, model_dim, inter_dim), dtype=dtype) / 10
    topk_weights, topk_ids = fused_topk(
        x, torch.randn((token, E), dtype=dtype), topk, True
    )
    w1_ref, w1_scale_ref, w1_aiter, w1_scale = quantize_weights(w1, fam, True)
    w2_ref, w2_scale_ref, w2_aiter, w2_scale = quantize_weights(w2, fam, False)

    # This rank owns experts [0, local_E): local ids equal global ids. Remote
    # slots are masked out of every check: the native paths leave them
    # unwritten and the expanded path zero-fills them. So are the padding rows
    # past num_local_tokens.
    expert_mask = None
    local = torch.ones_like(topk_ids, dtype=torch.bool)
    if ep > 1:
        expert_mask = torch.zeros((E,), dtype=dtypes.i32)
        expert_mask[:local_E] = 1
        local = topk_ids < local_E
    valid_tokens = max(1, token // 2) if variant == "num_local_tokens" else token
    valid = torch.arange(token).unsqueeze(-1) < valid_tokens
    slot_mask = (local & valid).unsqueeze(-1)
    ref = run_torch(
        x,
        w1_ref,
        w2_ref,
        w1_scale_ref,
        w2_scale_ref,
        topk_weights,
        torch.where(local, topk_ids, -1),
        fam,
        dtype,
        reduce=summed_only,
    )
    ref_sum = ref.to(dtypes.fp32) if summed_only else ref.to(dtypes.fp32).sum(1)
    ref_sum = torch.where(valid, ref_sum, 0)

    hidden, a1_scale = prequantize(x, fam) if variant == "prequant" else (x, None)
    nlt = None
    if variant == "num_local_tokens":
        nlt = torch.tensor([valid_tokens], dtype=dtypes.i32)
    kwargs = dict(
        expert_mask=expert_mask,
        activation=fam.activation,
        quant_type=fam.q_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        dtype=dtype,
        gate_mode=fam.gate_mode.value,
        **situ_kwargs(fam),
    )
    summed = torch.empty((token, model_dim), dtype=dtype)
    # A canary past the per-slot rows catches stores beyond the buffer.
    slab = torch.full((token * topk + 2 * topk, model_dim), -7.0, dtype=dtype)
    per_slot = slab[: token * topk].view(token, topk, model_dim)
    canary = slab[token * topk :]
    expanded = torch.empty((token * topk, model_dim), dtype=dtype)
    chunk = max(1, token // 3) if variant == "chunked" else 0

    def run_summed():
        return fused_moe(
            hidden,
            w1_aiter,
            w2_aiter,
            topk_weights,
            topk_ids,
            output=summed,
            num_local_tokens=nlt,
            **kwargs,
        )

    def run_per_slot():
        with per_slot_chunks(chunk) if chunk else contextlib.nullcontext():
            return fused_moe(
                hidden,
                w1_aiter,
                w2_aiter,
                topk_weights,
                topk_ids,
                per_slot_output=per_slot,
                num_local_tokens=nlt,
                **kwargs,
            )

    # What callers do without per_slot_output: top-1 over a per-route copy.
    def run_expanded():
        fused_moe(
            hidden.repeat_interleave(topk, dim=0),
            w1_aiter,
            w2_aiter,
            topk_weights.reshape(-1, 1),
            topk_ids.reshape(-1, 1),
            output=expanded,
            num_local_tokens=None if nlt is None else nlt * topk,
            **{
                **kwargs,
                "a1_scale": (
                    None if a1_scale is None else a1_scale.repeat_interleave(topk, 0)
                ),
            },
        )
        return expanded.view(token, topk, model_dim)

    flops = 2 * token * topk * model_dim * inter_dim * 3
    nbytes = sum(t.numel() * t.element_size() for t in (w1_aiter, w2_aiter, hidden))
    ret = {"gfx": get_gfx()}

    def timed(name, fn, **opts):
        out, us = run_perftest(fn, **opts)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = (nbytes + out.numel() * out.element_size()) / us / 1e6
        return out.clone()

    try:
        summed_seen = {"q_dtype_a": []}
        with recorded_dispatch() if HAS_PER_SLOT else contextlib.nullcontext() as seen:
            run_summed()
            summed_seen = seen or summed_seen
        out = timed("summed", run_summed)
    except Exception as e:  # noqa: BLE001
        ret["status"] = f"unsupported: summed fused_moe raised {type(e).__name__}: {e}"
        return ret
    out = torch.where(valid, out, 0)
    summed_err = checkAllclose(
        ref_sum, out.to(dtypes.fp32), msg="summed: fused_moe vs reference"
    )
    ret["summed logits_diff"] = logits_diff(ref_sum, out)
    ret["summed err"] = summed_err
    notes = []
    summed_inaccurate = inaccurate(summed_err, ret["summed logits_diff"])
    if summed_inaccurate:
        notes.append("summed fused_moe fails the reference check")
    if summed_only:
        ret["status"] = "; ".join(notes) or "ok"
        return ret
    out_summed = out

    failures = []
    if variant == "prequant" and rejects_prequant(fam):
        try:
            run_per_slot()
        except NotImplementedError:
            ret["status"] = "ok: per-slot rejects this a1_scale layout"
        else:
            ret["status"] = "FAIL: per-slot accepted an a1_scale it cannot split"
        return ret
    try:
        with recorded_dispatch() as seen:
            rows = run_per_slot().clone()
        ret["path"] = seen["paths"][0] if seen["paths"] else "expanded"
        timed("per_slot", run_per_slot)
    except Exception as e:  # noqa: BLE001
        ret["status"] = f"FAIL: per-slot fused_moe raised {type(e).__name__}: {e}"
        return ret
    ret["q_dtype_a"] = ",".join(sorted(set(seen["q_dtype_a"])))
    ret["per_slot default"] = ",".join(sorted(set(seen["default"])))
    if fam.v2_epilog:
        pinned = v2_kernels(fam, model_dim, inter_dim, local_E)[1]
        if not ret["path"].endswith(pinned):
            failures.append(f"summed selection is not the pinned {pinned}")
    is_a4w4 = (
        fam.q_type == QuantType.per_1x32
        and fam.a_dtype == dtypes.fp4x2
        and fam.w_dtype == dtypes.fp4x2
    )
    if seen["default"] and not is_a4w4:
        failures.append("per-slot default chosen outside MXFP4 a4w4")
    if set(seen["q_dtype_a"]) != set(summed_seen["q_dtype_a"]):
        failures.append(
            f"per-slot activations {ret['q_dtype_a']}, summed "
            + ",".join(sorted(set(summed_seen["q_dtype_a"])))
        )
    try:
        rows_graph = replayed(run_per_slot, per_slot)
        timed("per_slot graph", run_per_slot, testGraph=True)
    except Exception as e:  # noqa: BLE001
        rows_graph = None
        try:
            replayed(run_summed, summed)
            failures.append(f"per-slot graph capture raised {type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001
            notes.append("summed fused_moe is not graph-capturable either")
    if not (canary == -7.0).all():
        failures.append("per-slot call wrote past its buffer")
    out_expanded = timed("expanded", run_expanded)

    checked = [("per_slot", rows)]
    if rows_graph is not None:
        checked.append(("per_slot graph", rows_graph))
    for name, out in checked:
        if not torch.isfinite(torch.where(slot_mask, out, 0)).all():
            failures.append(f"{name} rows are not finite")
    ret["per_slot logits_diff"] = logits_diff(ref, rows, slot_mask)
    ret["per_slot err"] = checkAllclose(
        ref.to(dtypes.fp32),
        rows.to(dtypes.fp32),
        mask=slot_mask,
        msg="per_slot: fused_moe per-slot vs reference",
    )
    if inaccurate(ret["per_slot err"], ret["per_slot logits_diff"]):
        if summed_inaccurate:
            notes.append("per-slot rows fail it too")
        else:
            failures.append("per-slot rows vs reference")
    if rows_graph is not None:
        ret["per_slot graph vs eager err"] = checkAllclose(
            rows.to(dtypes.fp32),
            rows_graph.to(dtypes.fp32),
            mask=slot_mask,
            msg="per_slot graph: replay vs eager rows",
        )
        if ret["per_slot graph vs eager err"] != 0:
            failures.append("graph replay rows differ from eager rows")
    rows_sum = torch.where(slot_mask, rows, 0).to(dtypes.fp32).sum(1)
    ret["per_slot sum vs summed logits_diff"] = logits_diff(out_summed, rows_sum)
    ret["per_slot sum vs summed err"] = checkAllclose(
        out_summed.to(dtypes.fp32),
        rows_sum,
        msg="per_slot: local rows sum to the summed output",
    )
    if inaccurate(
        ret["per_slot sum vs summed err"],
        ret["per_slot sum vs summed logits_diff"],
        threshold=1e-3,
    ):
        failures.append("per-slot rows do not sum to the summed output")
    ret["expanded logits_diff"] = logits_diff(ref, out_expanded, slot_mask)
    ret["status"] = (
        "FAIL: " + "; ".join(failures) if failures else "; ".join(notes) or "ok"
    )
    return ret


def main():
    gfx = get_gfx()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="fused_moe(per_slot_output=...) against a per-slot torch reference",
    )
    parser.add_argument(
        "-f",
        "--family",
        type=str,
        nargs="*",
        default=[name for name, fam in FAMILIES.items() if gfx in fam.gfx],
        choices=list(FAMILIES),
        help="Quant families, e.g. -f a4w4 a16w4 (default: all on this arch)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="Output dtype, e.g. -d bf16",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=None,
        help="Token counts for every family, e.g. -t 16 1024 (default: per family)",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="model_dim,inter_dim,E,topk,ep, e.g. -s 4096,512,256,8,1 "
        "(default: per family)",
    )
    parser.add_argument(
        "--summed-only",
        action="store_true",
        help="Run only the summed fused_moe call (also on an AITER without "
        "per_slot_output), for summed-path comparisons",
    )
    parser.add_argument("--csv", type=str, default=None, help="Also write the table")
    args = parser.parse_args()
    if not args.summed_only and not HAS_PER_SLOT:
        raise SystemExit("this AITER has no fused_moe(per_slot_output=)")

    df = []
    for dtype, family in itertools.product(args.dtype, args.family):
        fam = FAMILIES[family]
        if gfx not in fam.gfx:
            aiter.logger.warning("skip %s: unsupported on %s", family, gfx)
            continue
        variants = (
            ("", fam.tokens),
            ("num_local_tokens", fam.num_local_tokens),
            ("prequant", fam.prequant),
            ("chunked", fam.chunked),
        )
        cases = [
            (shape, token, variant)
            for variant, tokens in variants
            if tokens
            for shape, token in itertools.product(
                args.shape or fam.shapes, args.token or tokens
            )
        ]
        for (model_dim, inter_dim, E, topk, ep), token, variant in cases:
            if ep > 1 and not fam.expert_mask:
                continue
            with contextlib.ExitStack() as stack:
                for name, value in fam.env:
                    stack.enter_context(override_env(name, value))
                if fam.v2_epilog:
                    stack.enter_context(
                        tuned_rows(
                            v2_config(
                                fam, dtype, token, model_dim, inter_dim, E // ep, topk
                            )
                        )
                    )
                df.append(
                    test_fmoe_per_slot(
                        family,
                        dtype,
                        token,
                        model_dim,
                        inter_dim,
                        E,
                        topk,
                        ep,
                        variant,
                        args.summed_only,
                    )
                )
            torch.cuda.empty_cache()
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fused_moe per-slot summary (markdown):\n%s", df.to_markdown(index=False)
    )
    if args.csv:
        df.to_csv(args.csv, index=False)
    failed = df[df["status"].astype(str).str.startswith("FAIL")]
    if len(failed):
        raise AssertionError(
            f"{len(failed)} of {len(df)} per-slot cases failed:\n"
            + failed[["family", "token", "ep", "variant", "status"]].to_string()
        )


if __name__ == "__main__":
    main()
