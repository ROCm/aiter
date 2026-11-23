# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA sparse prefill: the operand rules that ``qsa_sparse_prefill_problem`` reports and
the op raises, in their order.

These run on CPU tensors: the architecture and the ROCm device rule are injected, and a
rejected call raises before any kernel or GPU API is reached. FP8 K/V are gfx942's
E4M3FNUZ (torch.float8_e4m3fnuz); other FP8 formats are rejected. Each tensor names its
device, since a module imported in the same session may set a default device. Message
texts are written out here, not taken from the code under test.

The support query (``qsa_sparse_prefill_widths``, ``qsa_sparse_prefill_supports_selection``)
has to answer exactly what the entry accepts: gfx942 or gfx950, the widths 4T + 3,
T = 256/512/768/1024, and an int32 selection under 2**31 bytes.
op_tests/test_flydsl_qsa_sparse_prefill.py exercises the kernels themselves.
"""

import contextlib
import itertools
from collections.abc import Callable
from typing import NamedTuple

import pytest
from conftest import requires_built_aiter

requires_built_aiter()
pytest.importorskip("flydsl")

import torch

import aiter.ops.flydsl.qsa_sparse_prefill as op
from aiter.ops import qsa
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_common as common

ROWS = 8
GROUP = 12
DIM = 256
TOPK = 256
WIDTH = 4 * TOPK + 3

SHAPE = "Q/K/V must be BF16 [tokens, heads, 256] (K/V may be FP8 E4M3FNUZ on gfx942)"
COMPUTE = 'compute must be "fp8" or "bf16"'
SCALES_ON_BF16 = "k_scale/v_scale apply only to FP8 K/V"
FP8_SCALES = (
    "FP8 K/V require contiguous FP32 k_scale/v_scale on their device, of shape [1] or "
    "[8, 1] (K's [N, HK], or [P, HK] for a pool)"
)
SCALE_LAYOUTS = "k_scale and v_scale must both be per tensor or per token"
INVARIANT = "batch_invariant must be a bool"
SCALE_TYPE = "softmax_scale must be a host scalar or None"
SCALE_RANGE = "softmax_scale * log2(e) must be finite and positive in FP32"
SLOTS = "kv_slots must be a contiguous device int32 [N] tensor"
SLOTS_SPAN = "kv_slots requires a byte span <= 2**32 - 2**20"
LENS = "Host query/prefix lengths must describe the packed Q/K/V"
ROCM = "QSA sparse attention requires ROCm gfx942 or gfx950"
ALIGN = "Q/K/V require 16-byte alignment"
Q_CONTIGUOUS = "Q must be contiguous"
KV_HEADS = "K/V must match and Q heads must be a positive multiple of KV heads"
KV_STRIDE = (
    "K/V rows must hold dense [heads, 256] values one 16-byte aligned stride "
    ">= heads x 256 and < 2**28 elements apart, the same for K and V"
)
GROUP_LIMIT = "At most 16 Q heads per KV head fit the direct fallback wave"
SPAN = "Q and each request's K/V require byte spans <= 2**32 - 2**20"
GFX = "QSA sparse attention requires gfx942 or gfx950"
STRIDED_GFX = "Strided K/V rows are supported on gfx950 only"
FP8_GFX = "FP8 K/V are supported on gfx942 only"
INDICES = (
    "indices must be contiguous device int32 [M,4T+3], T = 256/512/768/1024, "
    "with byte span < 2**31"
)
OUT = "out must be aligned, contiguous and match Q"
OVERLAP = "out must not overlap Q/K/V, indices, kv_slots or scales"


def zeros(*shape, dtype=torch.bfloat16, device="cpu"):
    return torch.zeros(*shape, dtype=dtype, device=device)


def never(message):
    """A stand-in that fails the test where a call goes on to something it must not
    reach: on CPU tensors that would abort the process instead of failing a test."""

    def reached(*args, **kwargs):
        raise AssertionError(message)

    return reached


def misaligned(*shape, dtype=torch.bfloat16):
    """A tensor whose data pointer is 2 bytes past a 16-byte boundary."""
    size = 1
    for n in shape:
        size *= n
    return zeros(size + 8, dtype=dtype)[1 : 1 + size].view(*shape)


def packed():
    """Operands of one request of ROWS rows with no prefix: every rule holds."""
    return {
        "q": zeros(ROWS, GROUP, DIM),
        "k": zeros(ROWS, 1, DIM),
        "v": zeros(ROWS, 1, DIM),
        "indices": zeros(ROWS, WIDTH, dtype=torch.int32),
    }


def fp8(per_token=True):
    """FP8 K/V of packed() with FP32 scales per token and KV head [ROWS, 1], or per
    tensor."""
    return {
        **packed(),
        "k": zeros(ROWS, 1, DIM, dtype=torch.float8_e4m3fnuz),
        "v": zeros(ROWS, 1, DIM, dtype=torch.float8_e4m3fnuz),
        "k_scale": zeros(ROWS, 1, dtype=torch.float32) if per_token else ones(),
        "v_scale": zeros(ROWS, 1, dtype=torch.float32) if per_token else ones(),
    }


def e4m3fn():
    """K/V of fp8() in OCP E4M3FN, gfx950's FP8 (aiter's dtypes.fp8 there)."""
    return {name: zeros(ROWS, 1, DIM, dtype=torch.float8_e4m3fn) for name in ("k", "v")}


def ones():
    return torch.ones(1, dtype=torch.float32, device="cpu")


def interleaved():
    """K and V as the two halves of a [ROWS, 2, DIM] record per token (row stride 2 DIM)."""
    kv = zeros(ROWS, 2, DIM)
    return {**packed(), "k": kv[:, 0:1], "v": kv[:, 1:2]}


def rows_of(stride, dtype=torch.bfloat16):
    """K or V [ROWS, 1, DIM] over rows `stride` elements apart."""
    return zeros(ROWS * max(stride, DIM), dtype=dtype).as_strided(
        (ROWS, 1, DIM), (stride, DIM, 1)
    )


def out_is_q():
    args = packed()
    args["out"] = args["q"]
    return args


def inside_out(name):
    """An out [ROWS, GROUP, DIM] and operands with `name` stored inside it."""
    out = zeros(ROWS, GROUP, DIM)
    ints = out.view(torch.int32).reshape(-1)
    inside = {
        "k": out.reshape(-1)[: ROWS * DIM].view(ROWS, 1, DIM),
        "v": out.reshape(-1)[: ROWS * DIM].view(ROWS, 1, DIM),
        "indices": ints[: ROWS * WIDTH].view(ROWS, WIDTH),
        "kv_slots": ints[:ROWS],
        "k_scale": out.view(torch.float32).reshape(-1)[:ROWS].view(ROWS, 1),
    }
    return {"out": out, name: inside[name]}


def fp8_k_and_out(start):
    """FP8 K [ROWS, 1, DIM] (one byte per value) and an out starting `start` bytes after
    K's first byte, in one buffer."""
    out_bytes = ROWS * GROUP * DIM * 2
    buffer = zeros(max(ROWS * DIM, start + out_bytes), dtype=torch.uint8)
    return {
        "k": buffer[: ROWS * DIM].view(torch.float8_e4m3fnuz).view(ROWS, 1, DIM),
        "out": buffer[start : start + out_bytes]
        .view(torch.bfloat16)
        .view(ROWS, GROUP, DIM),
    }


def fp8_pool_over_the_span_limit(compute):
    """FP8 K/V of a 208-row slot pool for one request of ROWS rows over a 200-token
    prefix: its K span is past a _MAX_SPAN of Q's bytes (see SPAN_OF_Q)."""
    return {
        "k": zeros(208, 1, DIM, dtype=torch.float8_e4m3fnuz),
        "v": zeros(208, 1, DIM, dtype=torch.float8_e4m3fnuz),
        "k_scale": ones(),
        "v_scale": ones(),
        "kv_slots": torch.arange(208, dtype=torch.int32, device="cpu"),
        "query_lens": (ROWS,),
        "prefix_lens": (200,),
        "compute": compute,
    }


SPAN_OF_Q = (("_MAX_SPAN", ROWS * GROUP * DIM * 2),)


class Case(NamedTuple):
    changes: Callable[[], dict]
    message: str
    error: type = ValueError
    gfx: str = "gfx950"
    rocm: bool = True
    constants: tuple = ()
    base: Callable[[], dict] = packed


def case(name, changes, message, **kwargs):
    return pytest.param(Case(changes, message, **kwargs), id=name)


def sparse_bf16():
    return torch.sparse_coo_tensor(
        zeros(3, 0, dtype=torch.int64), zeros(0), (ROWS, GROUP, DIM), device="cpu"
    )


TENSOR_RULES = {
    "dtype": (lambda: zeros(ROWS, GROUP, DIM, dtype=torch.float16), SHAPE),
    "layout": (sparse_bf16, SHAPE),
    "last_dim": (lambda: zeros(ROWS, GROUP, DIM // 2), SHAPE),
    "alignment": (lambda: misaligned(ROWS, GROUP, DIM), ALIGN),
}

REJECTED = [
    # Shapes, compute, FP8 scales, scale, batch mode, slots, lengths: the rules before
    # the device.
    case("q_2d", lambda: {"q": zeros(ROWS, GROUP * DIM)}, SHAPE),
    case("k_2d", lambda: {"k": zeros(ROWS, DIM)}, SHAPE),
    case("v_2d", lambda: {"v": zeros(ROWS, DIM)}, SHAPE),
    case("compute_fp16", lambda: {"compute": "fp16"}, COMPUTE),
    case(
        "scales_on_bf16", lambda: {"k_scale": ones(), "v_scale": ones()}, SCALES_ON_BF16
    ),
    case(
        "fp8_without_scales",
        lambda: {"k_scale": None, "v_scale": None},
        FP8_SCALES,
        base=fp8,
    ),
    case(
        "fp8_scale_bf16",
        lambda: {"k_scale": zeros(ROWS, 1)},
        FP8_SCALES,
        base=fp8,
    ),
    case(
        "fp8_scale_rows",
        lambda: {"k_scale": zeros(ROWS - 1, 1, dtype=torch.float32)},
        FP8_SCALES,
        base=fp8,
    ),
    case(
        "fp8_scale_layouts_differ", lambda: {"k_scale": ones()}, SCALE_LAYOUTS, base=fp8
    ),
    case("invariant_int", lambda: {"batch_invariant": 1}, INVARIANT, error=TypeError),
    case(
        "scale_tensor",
        lambda: {"softmax_scale": torch.tensor(0.0625, device="cpu")},
        SCALE_TYPE,
        error=TypeError,
    ),
    case("scale_bool", lambda: {"softmax_scale": True}, SCALE_TYPE, error=TypeError),
    case("scale_zero", lambda: {"softmax_scale": 0.0}, SCALE_RANGE),
    case("scale_negative", lambda: {"softmax_scale": -1.0}, SCALE_RANGE),
    case("scale_nan", lambda: {"softmax_scale": float("nan")}, SCALE_RANGE),
    case("scale_inf", lambda: {"softmax_scale": float("inf")}, SCALE_RANGE),
    # 1e-50 x log2(e) < 2**-149, 1e40 x log2(e) > the largest FP32.
    case("scale_underflow", lambda: {"softmax_scale": 1e-50}, SCALE_RANGE),
    case("scale_overflow", lambda: {"softmax_scale": 1e40}, SCALE_RANGE),
    case("scale_int_overflow", lambda: {"softmax_scale": 10**400}, SCALE_RANGE),
    case("slots_list", lambda: {"kv_slots": list(range(ROWS))}, SLOTS),
    case("slots_int64", lambda: {"kv_slots": zeros(ROWS, dtype=torch.int64)}, SLOTS),
    case("slots_2d", lambda: {"kv_slots": zeros(ROWS, 1, dtype=torch.int32)}, SLOTS),
    case(
        "slots_other_device",
        lambda: {"kv_slots": zeros(ROWS, dtype=torch.int32, device="meta")},
        SLOTS,
    ),
    case(
        "slots_strided",
        lambda: {"kv_slots": zeros(2 * ROWS, dtype=torch.int32)[::2]},
        SLOTS,
    ),
    # 8 slots are 32 bytes.
    case(
        "slots_span",
        lambda: {"kv_slots": zeros(ROWS, dtype=torch.int32)},
        SLOTS_SPAN,
        constants=(("_MAX_SPAN", 16),),
    ),
    case("lens_count", lambda: {"query_lens": (ROWS,), "prefix_lens": (0, 0)}, LENS),
    case("lens_float", lambda: {"query_lens": (float(ROWS),)}, LENS),
    case("lens_negative", lambda: {"query_lens": (ROWS + 1, -1)}, LENS),
    case("lens_2_31", lambda: {"prefix_lens": (2**31,)}, LENS),
    # Q rows 7 of 8, but the keys still add up (7 + 1 = 8): only the Q-rows rule fires.
    case("lens_rows", lambda: {"query_lens": (ROWS - 1,), "prefix_lens": (1,)}, LENS),
    case("lens_keys", lambda: {"query_lens": (ROWS,), "prefix_lens": (1,)}, LENS),
    # The device rule runs the real predicate: a CPU tensor is not on a ROCm device.
    case("device_cpu", dict, ROCM, rocm=False),
    # FP8 K/V of any format off gfx942.
    case("fp8_on_gfx950", dict, FP8_GFX, base=fp8),
    case("fp8_per_tensor_on_gfx950", dict, FP8_GFX, base=lambda: fp8(False)),
    case("e4m3fn_on_gfx950", e4m3fn, FP8_GFX, base=fp8),
    # Q, K, V in turn: dtype, layout, last dimension, then alignment.
    *[
        case(f"{name}_{rule}", lambda name=name, make=make: {name: make()}, message)
        for name in ("q", "k", "v")
        for rule, (make, message) in TENSOR_RULES.items()
    ],
    *[
        case(
            f"{name}_other_device",
            lambda name=name: {name: zeros(ROWS, 1, DIM, device="meta")},
            SHAPE,
        )
        for name in ("k", "v")
    ],
    # On gfx942 FP8 K/V are E4M3FNUZ, both of them.
    case("e4m3fn_on_gfx942", e4m3fn, SHAPE, gfx="gfx942", base=fp8),
    case(
        "fp8_k_bf16_v_on_gfx942",
        lambda: {"v": zeros(ROWS, 1, DIM)},
        SHAPE,
        gfx="gfx942",
        base=fp8,
    ),
    case(
        "q_not_contiguous",
        lambda: {"q": zeros(GROUP, ROWS, DIM).transpose(0, 1)},
        Q_CONTIGUOUS,
    ),
    case("kv_shapes_differ", lambda: {"v": zeros(ROWS, 2, DIM)}, KV_HEADS),
    case("q_no_heads", lambda: {"q": zeros(ROWS, 0, DIM)}, KV_HEADS),
    case(
        "kv_no_heads",
        lambda: {"k": zeros(ROWS, 0, DIM), "v": zeros(ROWS, 0, DIM)},
        KV_HEADS,
    ),
    case(
        "heads_not_a_multiple",
        lambda: {
            "q": zeros(ROWS, 5, DIM),
            "k": zeros(ROWS, 2, DIM),
            "v": zeros(ROWS, 2, DIM),
        },
        KV_HEADS,
    ),
    # Row strides: a multiple of 8 elements, at least a dense row, dense heads and
    # elements, the same for K and V, below the limit.
    case(
        "stride_not_a_multiple_of_8",
        lambda: {"k": rows_of(260), "v": rows_of(260)},
        KV_STRIDE,
    ),
    case(
        "stride_below_a_dense_row",
        lambda: {"k": rows_of(128), "v": rows_of(128)},
        KV_STRIDE,
    ),
    case(
        "stride_sparse_heads",
        lambda: {
            "k": zeros(ROWS * 1024).as_strided((ROWS, 2, DIM), (1024, 512, 1)),
            "v": zeros(ROWS * 1024).as_strided((ROWS, 2, DIM), (1024, 512, 1)),
        },
        KV_STRIDE,
    ),
    case(
        "stride_sparse_elements",
        lambda: {
            "k": zeros(ROWS * 512).as_strided((ROWS, 1, DIM), (512, DIM, 2)),
            "v": zeros(ROWS * 512).as_strided((ROWS, 1, DIM), (512, DIM, 2)),
        },
        KV_STRIDE,
    ),
    case("stride_k_differs_from_v", lambda: {"v": rows_of(512)}, KV_STRIDE),
    case(
        "stride_at_the_limit",
        interleaved,
        KV_STRIDE,
        constants=(("_MAX_KV_STRIDE", 2 * DIM),),
    ),
    case("group_17", lambda: {"q": zeros(ROWS, 17, DIM)}, GROUP_LIMIT),
    # Spans: Q alone (49152 bytes against 1024), then a request's K/V (208 rows).
    case("span_q", dict, SPAN, constants=(("_MAX_SPAN", 1024),)),
    case(
        "span_request_kv",
        lambda: {
            "k": zeros(208, 1, DIM),
            "v": zeros(208, 1, DIM),
            "query_lens": (ROWS,),
            "prefix_lens": (200,),
        },
        SPAN,
        constants=SPAN_OF_Q,
    ),
    case("arch_unsupported", dict, GFX, gfx="gfx90a"),
    case("strided_on_gfx942", dict, STRIDED_GFX, gfx="gfx942", base=interleaved),
    # indices [M, 4T + 3]: T = 256, 512, 768 or 1024.
    case(
        "indices_extra_column",
        lambda: {"indices": zeros(ROWS, WIDTH + 1, dtype=torch.int32)},
        INDICES,
    ),
    case(
        "indices_extra_row",
        lambda: {"indices": zeros(ROWS + 1, WIDTH, dtype=torch.int32)},
        INDICES,
    ),
    case("indices_0d", lambda: {"indices": zeros((), dtype=torch.int32)}, INDICES),
    case(
        "indices_topk_128",
        lambda: {"indices": zeros(ROWS, 4 * 128 + 3, dtype=torch.int32)},
        INDICES,
    ),
    case(
        "indices_topk_300",
        lambda: {"indices": zeros(ROWS, 4 * 300 + 3, dtype=torch.int32)},
        INDICES,
    ),
    # 530,000 rows of 4 x 256 + 3 int32 are 2.18e9 bytes, past 2**31; Q (3.25e9 bytes)
    # stays within the span limit. Nothing reads the tensors: they are never written.
    case(
        "indices_over_2_gib",
        lambda: {
            "q": torch.empty(530_000, GROUP, DIM, dtype=torch.bfloat16, device="cpu"),
            "k": torch.empty(530_000, 1, DIM, dtype=torch.bfloat16, device="cpu"),
            "v": torch.empty(530_000, 1, DIM, dtype=torch.bfloat16, device="cpu"),
            "indices": torch.empty(530_000, WIDTH, dtype=torch.int32, device="cpu"),
        },
        INDICES,
    ),
    case(
        "indices_topk_1280",
        lambda: {"indices": zeros(ROWS, 4 * 1280 + 3, dtype=torch.int32)},
        INDICES,
    ),
    case(
        "indices_int64",
        lambda: {"indices": zeros(ROWS, WIDTH, dtype=torch.int64)},
        INDICES,
    ),
    case(
        "indices_other_device",
        lambda: {"indices": zeros(ROWS, WIDTH, dtype=torch.int32, device="meta")},
        INDICES,
    ),
    case(
        "indices_strided",
        lambda: {"indices": zeros(ROWS, 2 * WIDTH, dtype=torch.int32)[:, ::2]},
        INDICES,
    ),
    # out, then its overlap with each input it may not alias.
    case("out_shape", lambda: {"out": zeros(ROWS, GROUP + 1, DIM)}, OUT),
    case(
        "out_dtype", lambda: {"out": zeros(ROWS, GROUP, DIM, dtype=torch.float16)}, OUT
    ),
    case(
        "out_other_device", lambda: {"out": zeros(ROWS, GROUP, DIM, device="meta")}, OUT
    ),
    case(
        "out_strided",
        lambda: {"out": zeros(GROUP, ROWS, DIM).transpose(0, 1)},
        OUT,
    ),
    case("out_alignment", lambda: {"out": misaligned(ROWS, GROUP, DIM)}, OUT),
    case("out_is_q", dict, OVERLAP, base=out_is_q),
    *[
        case(f"out_overlaps_{name}", lambda name=name: inside_out(name), OVERLAP)
        for name in ("k", "v", "indices", "kv_slots")
    ],
    case(
        "out_overlaps_k_scale",
        lambda: inside_out("k_scale"),
        OVERLAP,
        gfx="gfx942",
        base=fp8,
    ),
    # FP8 K rows hold one byte per value: an out 16 bytes before K's end overlaps it.
    case(
        "out_overlaps_fp8_k",
        lambda: fp8_k_and_out(ROWS * DIM - 16),
        OVERLAP,
        gfx="gfx942",
        base=fp8,
    ),
    # compute="bf16" runs FP8 K/V as packed BF16 copies, addressed per request: a slot
    # pool does not lift the request's K span rule then.
    case(
        "fp8_compute_bf16_pool_over_the_span_limit",
        lambda: fp8_pool_over_the_span_limit("bf16"),
        SPAN,
        gfx="gfx942",
        base=fp8,
        constants=SPAN_OF_Q,
    ),
    # The first broken rule wins: the span rule precedes the strided-row rule, the
    # host lengths the device rule, the ranks the scale's type, the device the FP8
    # architecture rule, which precedes the dtypes.
    case(
        "order_span_before_arch",
        dict,
        SPAN,
        gfx="gfx942",
        base=interleaved,
        constants=(("_MAX_SPAN", 1024),),
    ),
    case("order_lens_before_device", lambda: {"prefix_lens": (1,)}, LENS, rocm=False),
    case(
        "order_ndim_before_scale",
        lambda: {"q": zeros(ROWS, GROUP * DIM), "softmax_scale": True},
        SHAPE,
    ),
    case(
        "order_fp8_scales_before_scale",
        lambda: {"k_scale": ones(), "v_scale": ones(), "softmax_scale": True},
        SCALES_ON_BF16,
    ),
    case(
        "order_scale_before_invariant",
        lambda: {"softmax_scale": 0.0, "batch_invariant": 1},
        SCALE_RANGE,
    ),
    case("order_device_before_fp8_arch", dict, ROCM, rocm=False, base=fp8),
    # E4M3FN K/V without scales off gfx942: the scale rules come first.
    case("order_fp8_scales_before_fp8_arch", e4m3fn, FP8_SCALES),
    case(
        "order_fp8_arch_before_dtype",
        lambda: {"v": zeros(ROWS, 1, DIM)},
        FP8_GFX,
        base=fp8,
    ),
    case(
        "order_device_before_dtype",
        lambda: {"q": zeros(ROWS, GROUP, DIM, dtype=torch.float16)},
        ROCM,
        rocm=False,
    ),
    case(
        "order_scale_before_slots",
        lambda: {"softmax_scale": 0.0, "kv_slots": list(range(ROWS))},
        SCALE_RANGE,
    ),
    case(
        "order_q_contiguity_before_kv_heads",
        lambda: {
            "q": zeros(GROUP, ROWS, DIM).transpose(0, 1),
            "v": zeros(ROWS, 2, DIM),
        },
        Q_CONTIGUOUS,
    ),
    case(
        "order_stride_before_group",
        lambda: {"q": zeros(ROWS, 17, DIM), "k": rows_of(260), "v": rows_of(260)},
        KV_STRIDE,
    ),
    case("order_arch_before_strided", dict, GFX, gfx="gfx90a", base=interleaved),
    # K and V that differ in shape, over rows whose stride is not a multiple of 8.
    case(
        "order_kv_heads_before_stride",
        lambda: {"k": rows_of(260), "v": zeros(ROWS, 2, DIM)},
        KV_HEADS,
    ),
]


def arrange(monkeypatch, case):
    # Operand rules read the operands and the architecture, not the process's state.
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        never("the call reached a query of the process's state"),
    )
    monkeypatch.setattr(op, "get_gfx_runtime", lambda: case.gfx)
    if case.rocm:
        monkeypatch.setattr(op, "_on_rocm_device", lambda q: True)
    for name, value in case.constants:
        monkeypatch.setattr(op, name, value)
    return {**case.base(), **case.changes()}


@pytest.mark.parametrize("rejected", REJECTED)
def test_the_first_broken_rule_is_reported_and_raised(monkeypatch, rejected):
    args = arrange(monkeypatch, rejected)
    assert qsa.qsa_sparse_prefill_problem(**args) == rejected.message
    with pytest.raises(rejected.error) as raised:
        qsa.qsa_sparse_prefill(**args)
    assert str(raised.value) == rejected.message


def out_in_kv_gap():
    """K/V rows 32768 elements apart (HK 1) and an out in the padding after row 0:
    byte offset 2048, 16-byte aligned, inside K's bounding span but off every row."""
    stride = 32768
    k = zeros(ROWS * stride)
    out = k[1024 : 1024 + ROWS * GROUP * DIM].view(ROWS, GROUP, DIM)
    return {
        "k": k.as_strided((ROWS, 1, DIM), (stride, DIM, 1)),
        "v": rows_of(stride),
        "out": out,
    }


def accepted(name, changes, **kwargs):
    return pytest.param(Case(changes, "", **kwargs), id=name)


ACCEPTED = [
    accepted("packed", dict),
    accepted("packed_on_gfx942", dict, gfx="gfx942"),
    accepted("scale_given", lambda: {"softmax_scale": 1}),
    accepted("explicit_lengths", lambda: {"query_lens": (ROWS,), "prefix_lens": (0,)}),
    accepted("two_requests", lambda: {"query_lens": (3, 5)}),
    accepted(
        "slot_pool",
        lambda: {
            "k": zeros(64, 1, DIM),
            "v": zeros(64, 1, DIM),
            "kv_slots": torch.arange(ROWS, dtype=torch.int32, device="cpu"),
        },
    ),
    # A pool is addressed from its base: only a request's K/V span packed K/V.
    accepted(
        "slot_pool_over_the_span_limit",
        lambda: {
            "k": zeros(208, 1, DIM),
            "v": zeros(208, 1, DIM),
            "kv_slots": torch.arange(208, dtype=torch.int32, device="cpu"),
            "query_lens": (ROWS,),
            "prefix_lens": (200,),
        },
        constants=SPAN_OF_Q,
    ),
    accepted("out_apart", lambda: {"out": zeros(ROWS, GROUP, DIM)}),
    accepted("interleaved_on_gfx950", dict, base=interleaved),
    accepted("out_in_the_gap_between_kv_rows", out_in_kv_gap),
    accepted("group_16", lambda: {"q": zeros(ROWS, 16, DIM)}),
    accepted("batch_invariant", lambda: {"batch_invariant": True}),
    accepted("fp8_on_gfx942", dict, gfx="gfx942", base=fp8),
    accepted("fp8_per_tensor_on_gfx942", dict, gfx="gfx942", base=lambda: fp8(False)),
    accepted(
        "fp8_compute_bf16_on_gfx942",
        lambda: {"compute": "bf16"},
        gfx="gfx942",
        base=fp8,
    ),
    accepted(
        "fp8_pool_over_the_span_limit",
        lambda: fp8_pool_over_the_span_limit("fp8"),
        gfx="gfx942",
        base=fp8,
        constants=SPAN_OF_Q,
    ),
    accepted(
        "out_right_after_fp8_k",
        lambda: fp8_k_and_out(ROWS * DIM),
        gfx="gfx942",
        base=fp8,
    ),
    accepted(
        "topk_1024", lambda: {"indices": zeros(ROWS, 4 * 1024 + 3, dtype=torch.int32)}
    ),
]


@pytest.mark.parametrize("accepted_call", ACCEPTED)
def test_a_call_that_breaks_no_rule_has_no_problem(monkeypatch, accepted_call):
    args = arrange(monkeypatch, accepted_call)
    assert qsa.qsa_sparse_prefill_problem(**args) is None


def test_states_of_the_process_stay_with_the_op(monkeypatch):
    # A call that breaks no operand rule has no problem, and the op still raises when the
    # process is in a state it cannot serve the call in: one GPU per process, or a graph
    # capture of a layout not warmed on the stream. The device context and the stream's id
    # are stood in for, since the operands are CPU tensors.
    args = arrange(monkeypatch, Case(dict, ""))
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        torch._C, "_cuda_getCurrentRawStream", lambda i: 0, raising=False
    )
    # Past the state checks the op would build a workspace and launch kernels.
    monkeypatch.setattr(
        op,
        "_Workspace",
        never("a state check did not stop the call: it built a workspace"),
    )

    monkeypatch.setattr(op, "_device", torch.device("cuda", 1))
    assert qsa.qsa_sparse_prefill_problem(**args) is None
    with pytest.raises(ValueError, match="one GPU per process"):
        qsa.qsa_sparse_prefill(**args)

    monkeypatch.setattr(op, "_device", None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert qsa.qsa_sparse_prefill_problem(**args) is None
    with pytest.raises(RuntimeError, match="before graph capture"):
        qsa.qsa_sparse_prefill(**args)


def test_without_flydsl_the_op_is_unavailable_and_every_call_has_a_problem(monkeypatch):
    args = arrange(monkeypatch, Case(dict, ""))
    assert qsa.qsa_sparse_prefill_available()
    assert qsa.qsa_sparse_prefill_problem(**args) is None
    assert qsa.can_use_flydsl_qsa_sparse_prefill(**args)
    monkeypatch.setattr(qsa, "_HAS_FLYDSL_PREFILL", False)
    assert not qsa.qsa_sparse_prefill_available()
    assert (
        qsa.qsa_sparse_prefill_problem(**args) == "QSA sparse prefill requires FlyDSL"
    )
    assert not qsa.can_use_flydsl_qsa_sparse_prefill(**args)
    with pytest.raises(RuntimeError, match="requires FlyDSL"):
        qsa.qsa_sparse_prefill(**args)


@pytest.mark.parametrize(
    ("value", "switched"), [("1", True), ("0", False), ("true", None), ("2", None)]
)
def test_a_switch_of_the_platform_entry_is_written_0_or_1(monkeypatch, value, switched):
    monkeypatch.setattr(common, "_calibrated", lambda: ("gfx950", 256))
    monkeypatch.setenv(common.OVERRIDE, f"route_all_or_none={value}")
    common.platform.cache_clear()
    try:
        if switched is None:
            with pytest.raises(ValueError, match="written 0 or 1"):
                common.platform()
        else:
            assert common.platform().route_all_or_none is switched
    finally:
        common.platform.cache_clear()


# The support query of qsa_sparse_prefill.

WIDTHS = (1027, 2051, 3075, 4099)


def _set_gfx(monkeypatch, gfx):
    monkeypatch.setattr(op, "get_gfx_runtime", lambda: gfx)


@pytest.fixture(params=["gfx942", "gfx950"])
def supported_gfx(monkeypatch, request):
    _set_gfx(monkeypatch, request.param)


def _entry_rule(rows, width):
    """What the entry accepted inline before the query existed."""
    topk = (width - 3) // 4
    return (
        width == 4 * topk + 3
        and topk % 256 == 0
        and 256 <= topk <= 1024
        and rows * width * 4 < 2**31
    )


def test_widths_are_the_kernel_budgets_in_increasing_order(supported_gfx):
    assert op.SELECTION_WIDTHS == WIDTHS
    assert qsa.qsa_sparse_prefill_widths() == WIDTHS


@pytest.mark.parametrize("width", WIDTHS)
def test_byte_limit_is_the_last_row_under_2_31_bytes(supported_gfx, width):
    rows = (2**31 - 1) // (4 * width)
    assert qsa.qsa_sparse_prefill_supports_selection(rows, width)
    assert not qsa.qsa_sparse_prefill_supports_selection(rows + 1, width)


def test_query_matches_the_entry_rule(supported_gfx):
    widths = sorted(
        {w + d for w in WIDTHS for d in (-4, -1, 0, 1, 4)} | {0, 3, 7, 4103, 5123}
    )
    rows = [0, 1, 8192, 131072, 131073, 2**20]
    for r, w in itertools.product(rows, widths):
        assert qsa.qsa_sparse_prefill_supports_selection(r, w) == _entry_rule(r, w), (
            r,
            w,
        )


@pytest.mark.parametrize("gfx", ["gfx90a", "gfx1250"])
def test_other_gpus_support_nothing(monkeypatch, gfx):
    _set_gfx(monkeypatch, gfx)
    assert not op.flydsl_qsa_sparse_prefill_available()
    assert qsa.qsa_sparse_prefill_widths() == ()
    assert not qsa.qsa_sparse_prefill_supports_selection(8, 2051)


def test_without_flydsl_nothing_is_supported(supported_gfx, monkeypatch):
    monkeypatch.setattr(qsa, "_HAS_FLYDSL_PREFILL", False)
    assert qsa.qsa_sparse_prefill_widths() == ()
    assert not qsa.qsa_sparse_prefill_supports_selection(8, 2051)
