# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA sparse prefill: the operand rules that ``qsa_sparse_prefill_problem`` reports and
the op raises, in their order.

These run on CPU tensors: the architecture and the ROCm device rule are injected, and a
rejected call raises before any kernel or GPU API is reached. Message texts are written
out here, not taken from the code under test.
"""

import contextlib
from collections.abc import Callable
from typing import NamedTuple

import pytest
from conftest import requires_built_aiter

requires_built_aiter()
pytest.importorskip("flydsl")

import torch

import aiter.ops.flydsl.qsa_sparse_prefill as op
from aiter.ops import qsa

ROWS = 8
GROUP = 12
DIM = 256
TOPK = 256
WIDTH = 4 * TOPK + 3

SHAPE = "Q/K/V must be BF16 [tokens, heads, 256]"
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
INDICES = (
    "indices must be contiguous device int32 [M,4T+3], T = 256/512/768/1024, "
    "with byte span < 2**31"
)
OUT = "out must be aligned, contiguous and match Q"
OVERLAP = "out must not overlap Q/K/V, indices or kv_slots"


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
    }
    return {"out": out, name: inside[name]}


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
        torch.zeros(3, 0, dtype=torch.int64), zeros(0), (ROWS, GROUP, DIM)
    )


TENSOR_RULES = {
    "dtype": (lambda: zeros(ROWS, GROUP, DIM, dtype=torch.float16), SHAPE),
    "layout": (sparse_bf16, SHAPE),
    "last_dim": (lambda: zeros(ROWS, GROUP, DIM // 2), SHAPE),
    "alignment": (lambda: misaligned(ROWS, GROUP, DIM), ALIGN),
}

REJECTED = [
    # Shapes, scale, slots, lengths: the rules before the device.
    case("q_2d", lambda: {"q": zeros(ROWS, GROUP * DIM)}, SHAPE),
    case("k_2d", lambda: {"k": zeros(ROWS, DIM)}, SHAPE),
    case("v_2d", lambda: {"v": zeros(ROWS, DIM)}, SHAPE),
    case(
        "scale_tensor",
        lambda: {"softmax_scale": torch.tensor(0.0625)},
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
        constants=(("_MAX_SPAN", ROWS * GROUP * DIM * 2),),
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
            "q": torch.empty(530_000, GROUP, DIM, dtype=torch.bfloat16),
            "k": torch.empty(530_000, 1, DIM, dtype=torch.bfloat16),
            "v": torch.empty(530_000, 1, DIM, dtype=torch.bfloat16),
            "indices": torch.empty(530_000, WIDTH, dtype=torch.int32),
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
    # The first broken rule wins: the span rule precedes the architecture rules, the
    # host lengths the device rule, the ranks the scale's type, the device the dtypes.
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
            "kv_slots": torch.arange(ROWS, dtype=torch.int32),
        },
    ),
    # A pool is addressed from its base: only a request's K/V span packed K/V.
    accepted(
        "slot_pool_over_the_span_limit",
        lambda: {
            "k": zeros(208, 1, DIM),
            "v": zeros(208, 1, DIM),
            "kv_slots": torch.arange(208, dtype=torch.int32),
            "query_lens": (ROWS,),
            "prefix_lens": (200,),
        },
        constants=(("_MAX_SPAN", ROWS * GROUP * DIM * 2),),
    ),
    accepted("out_apart", lambda: {"out": zeros(ROWS, GROUP, DIM)}),
    accepted("interleaved_on_gfx950", dict, base=interleaved),
    accepted("group_16", lambda: {"q": zeros(ROWS, 16, DIM)}),
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
