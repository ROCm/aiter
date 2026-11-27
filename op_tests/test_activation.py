import argparse
import functools
import math

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import aiter
from aiter import dtypes
from aiter.test_common import (
    assertAllclose,
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.utility import fp4_utils


def torch_silu_and_mul(input: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    if limit > 0:
        x = torch.clamp(x, max=limit)
        y = torch.clamp(y, min=-limit, max=limit)
    out = F.silu(x) * y
    return out


@benchmark()
def test_scaled_silu_and_mul(m, n, dtype, output_dtype=None):
    """
    Test scaled_silu_and_mul with flexible input/output types.
    If output_dtype is None, defaults to fp8 for quantization.
    """
    ret = {}
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out_dtype = output_dtype if output_dtype is not None else dtypes.fp8
    out = torch.empty((m, n // 2), dtype=out_dtype, device="cuda")

    # Reference: compute, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(out_dtype)

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
    )

    # Check if the results are close
    err = checkAllclose(ref.to(torch.float), out.to(torch.float))

    # Record input/output types for clarity
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
    }
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


def test_scaled_silu_and_mul_edge_cases():
    """Saturation, d % vec_size != 0, strided rows, and a scale updated in place
    between graph replays."""
    fp8_info = torch.finfo(dtypes.fp8)
    for d, pad in [(3, 0), (4098, 0), (14336, 8), (14338, 3)]:
        # One extra row and `pad` extra columns that the kernel must not write.
        x_buf = torch.randn(65, 2 * d + pad, dtype=torch.bfloat16, device="cuda") * 8
        out_buf = torch.full((65, d + pad), 0x5A, dtype=torch.uint8, device="cuda")
        x = x_buf[:64, : 2 * d]
        out = out_buf[:64, :d].view(dtypes.fp8)
        scale = torch.tensor([0.01], device="cuda")
        aiter.scaled_silu_and_mul(out, x, scale)
        assert (out_buf[:64, d:] == 0x5A).all() and (
            out_buf[64:] == 0x5A
        ).all(), f"d={d}: write outside the output rows"
        ref = torch_silu_and_mul(x.float()) / scale
        ref = ref.clamp(fp8_info.min, fp8_info.max).to(dtypes.fp8).float()
        err = (out.float() - ref).abs()
        assert not out.float().isnan().any(), f"d={d}: NaN instead of saturation"
        assert (err <= ref.abs() * 0.125 + 2**-9).all(), f"d={d}: wrong values"

    x = torch.randn(64, 8192, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(64, 4096, dtype=dtypes.fp8, device="cuda")
    expected = torch.empty_like(out)
    scale = torch.tensor([1.0], device="cuda")
    aiter.scaled_silu_and_mul(out, x, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        aiter.scaled_silu_and_mul(out, x, scale)
    scale.fill_(0.5)
    graph.replay()
    aiter.scaled_silu_and_mul(expected, x, scale)
    assert torch.equal(
        out.view(torch.uint8), expected.view(torch.uint8)
    ), "graph replay did not pick up the updated scale"


@benchmark()
def test_silu_and_mul(m, n, dtype, output_dtype=None, limit=0.0):
    """
    Test silu_and_mul with flexible input/output types.
    If output_dtype is None, output matches input dtype.
    """
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out_dtype = output_dtype if output_dtype is not None else dtype
    out = torch.empty((m, n // 2), dtype=out_dtype, device="cuda")

    # Reference: compute in input dtype, convert to output dtype if needed
    ref = torch_silu_and_mul(input, limit=limit)
    if output_dtype is not None:
        ref = ref.to(output_dtype)

    _, us_aiter = run_perftest(
        aiter.silu_and_mul,
        out,
        input,
        limit,
    )

    # Check if the results are close
    err = checkAllclose(ref, out)

    # Record input/output types for clarity
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    ret = {}
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["limit"] = limit
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


class GELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input: Tensor) -> Tensor:
        return (
            input
            * 0.5
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


def torch_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    out = GELUTanh()(x)
    return out


def gelu_fast_wrapper(input: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(input)
    aiter.gelu_fast(out, input)
    return out


@benchmark()
def test_gelu_fast(m, n, dtype, output_dtype=None):
    ret = {}
    input = torch.randn(m, 1, n, dtype=dtype, device="cuda")
    out_dtype = output_dtype if output_dtype is not None else dtype

    out, us_aiter = run_perftest(gelu_fast_wrapper, input)
    ref, us_torch = run_perftest(torch_gelu_ref, input)

    if output_dtype is not None:
        ref = ref.to(output_dtype)

    # Check if the results are close
    err = checkAllclose(ref, out)

    # Record input/output types for clarity
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    ret = {}
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["torch_us"] = us_torch
    ret["speedup_vs_torch"] = us_torch / us_aiter
    ret["perf_gain_vs_torch_pct"] = (us_torch - us_aiter) / us_torch * 100.0
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


def torch_relu2_ref(x: torch.Tensor) -> torch.Tensor:
    r = torch.nn.functional.relu(x.float())
    return r * r


def relu2_wrapper(input: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(input)
    aiter.relu2(out, input)
    return out


@benchmark()
def test_relu2(m, n, dtype, output_dtype=None):
    ret = {}
    input = torch.randn(m, 1, n, dtype=dtype, device="cuda")
    out_dtype = output_dtype if output_dtype is not None else dtype
    if out_dtype != dtype:
        raise ValueError(
            "test_relu2 does not support output_dtype != dtype: relu2_wrapper "
            "always allocates `out` with the same dtype as `input` "
            "(torch.empty_like(input)), so a differing output_dtype would "
            "silently compare a mis-cast reference against a same-dtype "
            "kernel output."
        )

    out, us_aiter = run_perftest(relu2_wrapper, input)
    ref, us_torch = run_perftest(torch_relu2_ref, input)

    # Always cast the fp32 reference down to the kernel's output dtype
    # before comparing (torch_relu2_ref computes in fp32 via x.float()).
    ref = ref.to(out_dtype)

    err = checkAllclose(ref, out)

    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["torch_us"] = us_torch
    ret["speedup_vs_torch"] = us_torch / us_aiter
    ret["perf_gain_vs_torch_pct"] = (us_torch - us_aiter) / us_torch * 100.0
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


@benchmark()
def test_relu2_bounds(numel, dtype):
    ret = {
        "numel": numel,
        "dtype": {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, str(dtype)),
    }
    input = torch.randn(1, numel, dtype=dtype, device="cuda")
    out = torch.empty_like(input)
    aiter.relu2(out, input)
    if numel == 0:
        assert out.numel() == 0
        ret["err"] = 0.0
        return ret
    ref = torch_relu2_ref(input).to(dtype)
    ret["err"] = checkAllclose(ref, out)
    return ret


_GUARD = 64  # elements around a guarded tensor
_SENTINEL = 12288.0  # exact in bf16 and fp16
_DTYPE_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def torch_swiglu_and_mul(gate: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
    # swiglu_and_mul has alpha 1.702 and clamps both halves at 7.
    gate = gate.clamp(max=7.0)
    return gate * torch.sigmoid(1.702 * gate) * (linear.clamp(-7.0, 7.0) + 1.0)


# fp32 references of the gate ops; a *_bias op adds its bias and uses the same one.
_GATE_REFS = {
    "silu_and_mul": lambda a, b: F.silu(a) * b,
    "gelu_and_mul": lambda a, b: F.gelu(a) * b,
    "gelu_tanh_and_mul": lambda a, b: F.gelu(a, approximate="tanh") * b,
    "swiglu_and_mul": torch_swiglu_and_mul,
}
_GATE_OPS = [
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "swiglu_and_mul",
    "silu_and_mul_bias",
    "swiglu_and_mul_bias",
    "gelu_and_mul_bias",
]


def _guarded(shape, dtype, fill):
    """A tensor of ``shape`` that is a view inside a larger buffer filled with
    ``fill``. Returns the buffer and the view."""
    numel = math.prod(shape)
    buf = torch.full((numel + 2 * _GUARD,), fill, dtype=dtype, device="cuda")
    return buf, buf[_GUARD : _GUARD + numel].view(*shape)


def _guard_changed(buf, numel, fill):
    """How many elements around a guarded tensor no longer hold ``fill``."""
    pad = torch.cat([buf[:_GUARD], buf[_GUARD + numel :]])
    return int((pad != fill).sum())


def _assert_close_16bit(ref, out, atol, msg):
    """Fails unless every element of ``out`` is within one unit in the last
    place of its 16-bit type, plus ``atol``, of the fp32 reference."""
    return assertAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=torch.finfo(out.dtype).eps,
        atol=atol,
        tol_err_ratio=0.0,
        msg=msg,
    )


def _perf_columns(us, read_bytes, write_bytes):
    return {
        "us": us,
        "TB/s": (read_bytes + write_bytes) / us / 1e6,
        "RD TB/s": read_bytes / us / 1e6,
        "WR TB/s": write_bytes / us / 1e6,
    }


@benchmark()
def test_gelu_fast_bounds(numel, dtype):
    """gelu_fast on ``numel`` elements whose output sits inside a guarded buffer:
    the result matches the reference and nothing around the output changes,
    whether or not the tensor ends on a whole vector."""
    _, x = _guarded((1, numel), dtype, 100.0)
    x.copy_((3.0 * torch.randn(1, numel, device="cuda")).to(dtype))
    buf, out = _guarded((1, numel), dtype, _SENTINEL)
    _, us = run_perftest(aiter.gelu_fast, out, x)
    changed = _guard_changed(buf, numel, _SENTINEL)
    assert changed == 0, f"gelu_fast wrote {changed} elements outside its output"
    # The kernel's tanh is within 1e-3 of tanh: at most 1.7e-3 in the result.
    ref = torch_gelu_ref(x.to(dtypes.fp32))
    err = _assert_close_16bit(ref, out, 2e-3, f"gelu_fast {numel=}")
    ret = {"dtype": _DTYPE_NAMES.get(dtype, str(dtype))}
    ret.update(_perf_columns(us, x.nbytes, out.nbytes))
    ret["err"] = err
    return ret


def _gate_case(op, dtype, m, d):
    """A gate op on [m, 2 * d] with its output inside a guarded buffer: the
    buffer, the output, the op's arguments, the fp32 reference and the bytes
    the op reads. Three sigma inputs reach past the SwiGLU clamp."""
    _, x = _guarded((m, 2 * d), dtype, 100.0)
    x.copy_((3.0 * torch.randn(m, 2 * d, device="cuda")).to(dtype))
    buf, out = _guarded((m, d), dtype, _SENTINEL)
    call_args = [out, x]
    x32 = x.to(dtypes.fp32)
    read_bytes = x.nbytes
    if op.endswith("_bias"):
        expert_ids = (torch.arange(m, device="cuda") % 3).to(torch.int32)
        bias = 0.1 * torch.randn(3, 2 * d, dtype=dtypes.fp32, device="cuda")
        call_args += [expert_ids, bias]
        x32 = x32 + bias[expert_ids.long()]
        read_bytes += expert_ids.nbytes + m * bias[0].nbytes
    ref = _GATE_REFS[op.replace("_bias", "")](*x32.split([d, d], dim=-1))
    return buf, out, call_args, ref, read_bytes


@benchmark()
def test_act_and_mul_row_bound(op, dtype, m, d):
    """A gate op on half rows of whole 4-byte words: the result matches the
    fp32 reference and nothing around the output changes."""
    buf, out, call_args, ref, read_bytes = _gate_case(op, dtype, m, d)
    _, us = run_perftest(getattr(aiter, op), *call_args)
    changed = _guard_changed(buf, m * d, _SENTINEL)
    assert changed == 0, f"{op} wrote {changed} elements outside its output"
    err = _assert_close_16bit(ref, out, 1e-5, f"{op} {m=} {d=}")
    ret = {"dtype": _DTYPE_NAMES.get(dtype, str(dtype))}
    ret.update(_perf_columns(us, read_bytes, out.nbytes))
    ret["err"] = err
    return ret


@benchmark()
def test_act_and_mul_split_word_refused(op, dtype, m, d):
    """Half rows that are not whole 4-byte words (an odd d with 16-bit elements)
    make the kernel reach one element past every row, so the op has to refuse
    the call and leave the output alone. No kernel runs: nothing to time."""
    buf, out, call_args, _, _ = _gate_case(op, dtype, m, d)
    try:
        getattr(aiter, op)(*call_args)
    except RuntimeError as e:
        assert "whole 4-byte words" in str(e), str(e)
    else:
        raise AssertionError(f"{op} accepted half rows of {d} {dtype} elements")
    torch.cuda.synchronize()
    assert bool((out == _SENTINEL).all()), f"{op} wrote into a refused output"
    assert _guard_changed(buf, m * d, _SENTINEL) == 0
    return {"dtype": _DTYPE_NAMES.get(dtype, str(dtype)), "refused": True}


def _dequant_fp8_group(q, s, group_size):
    m, n = q.shape
    return (
        q.float().view(m, n // group_size, group_size) * s.float().view(m, -1, 1)
    ).view(m, n)


def _dequant_fp4_group(q, s, group_size):
    from aiter.utility import fp4_utils

    m = q.shape[0]
    n = s.shape[1] * group_size
    packed_cols = n // 2
    q_u8 = q.view(torch.uint8) if q.dtype == dtypes.fp4x2 else q
    if q_u8.shape[1] > packed_cols:
        q_u8 = q_u8[:, :packed_cols]
    q_f32 = fp4_utils.mxfp4_to_f32(q_u8).view(m, n // group_size, group_size)
    s_f32 = fp4_utils.e8m0_to_f32(s).float().view(m, -1, 1)
    return (q_f32 * s_f32).view(m, n)


def _ref_group_scales_fp8(x: torch.Tensor, group_size: int, out_dtype) -> torch.Tensor:
    m, n = x.shape
    xg = x.view(m, n // group_size, group_size).float()
    dmax = torch.finfo(out_dtype).max
    x_max = torch.amax(torch.abs(xg), dim=-1)
    x_max = torch.maximum(x_max, torch.full_like(x_max, 1e-10))
    return x_max / dmax


def _ref_group_scales_fp4(x: torch.Tensor, group_size: int) -> torch.Tensor:
    m, n = x.shape
    xg = x.view(m, n // group_size, group_size).float()
    x_max = torch.amax(torch.abs(xg), dim=-1)
    x_max = torch.maximum(x_max, torch.full_like(x_max, 1e-10))
    # NV ROUND_UP / DSv4 / FlashInfer default: scale = ceil_pow2(amax / 6)
    # (matches HIP kernel ``aiter::fp4_f32_to_e8m0_scale`` and
    # silu_and_mul_quant FP4 path).
    scale_e8m0 = fp4_utils.fp4_f32_to_e8m0_scale(x_max)
    return scale_e8m0.view(torch.uint8)


@benchmark()
def test_silu_and_mul_quant(m, n, dtype, group_size, output_dtype=None, limit=0.0):
    """
    Test silu_and_mul_quant with per-group quantization to fp8 or fp4.
    Benchmarks HIP kernel and validates against PyTorch reference.
    """
    ret = {}
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    d = n // 2
    out_dtype = output_dtype if output_dtype is not None else dtypes.fp8
    num_groups = d // group_size

    is_fp4 = out_dtype == dtypes.fp4x2
    if is_fp4:
        out = torch.empty((m, d // 2), dtype=out_dtype, device="cuda")
        scale = torch.empty((m, num_groups), dtype=torch.uint8, device="cuda")
    else:
        out = torch.empty((m, d), dtype=out_dtype, device="cuda")
        scale = torch.empty((m, num_groups), dtype=torch.float32, device="cuda")

    _, us_aiter = run_perftest(
        aiter.silu_and_mul_quant,
        out,
        input,
        scale,
        group_size,
        limit,
    )

    # Accuracy validation
    ref = torch_silu_and_mul(input, limit=limit).float()

    if is_fp4:
        q_atol, q_rtol = 0.5, 0.5
        hip_deq = _dequant_fp4_group(out, scale, group_size)
        ref_scale = _ref_group_scales_fp4(ref, group_size)
        scale_diff = (scale.to(torch.int16) - ref_scale.to(torch.int16)).abs()
        scale_max_abs_diff = scale_diff.max().item()
        scale_mismatch_ratio = (scale != ref_scale).float().mean().item()
        err_scale = checkAllclose(
            scale.float(),
            ref_scale.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"HIP scale vs ref (M={m}, N={n}, gs={group_size}): ",
        )
    else:
        q_atol, q_rtol = 0.05, 0.05
        hip_deq = _dequant_fp8_group(out, scale, group_size)
        ref_scale = _ref_group_scales_fp8(ref, group_size, out_dtype)
        scale_diff = (scale.float() - ref_scale.float()).abs()
        scale_max_abs_diff = scale_diff.max().item()
        scale_mismatch_ratio = (
            (~torch.isclose(scale.float(), ref_scale.float(), rtol=1e-3, atol=1e-3))
            .float()
            .mean()
            .item()
        )
        err_scale = checkAllclose(
            scale.float(),
            ref_scale.float(),
            rtol=1e-3,
            atol=1e-3,
            msg=f"HIP scale vs ref (M={m}, N={n}, gs={group_size}): ",
        )

    err_hip = checkAllclose(
        ref,
        hip_deq,
        rtol=q_rtol,
        atol=q_atol,
        msg=f"HIP vs ref (M={m}, N={n}, gs={group_size}): ",
    )

    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
        dtypes.fp4x2: "fp4",
    }
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["limit"] = limit
    ret["group_size"] = group_size
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes + scale.nbytes) / us_aiter / 1e6
    ret["err_hip"] = err_hip
    ret["err_scale"] = err_scale
    ret["scale_max_abs_diff"] = scale_max_abs_diff
    ret["scale_mismatch_ratio"] = scale_mismatch_ratio
    return ret


@benchmark()
def test_scaled_silu_and_mul_mixed_dtype(m, n, input_dtype, output_dtype):
    """Test fp32 input with fp16/bf16 output for scaled activation"""
    input = torch.randn(m, n, dtype=input_dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out = torch.empty((m, n // 2), dtype=output_dtype, device="cuda")

    # Reference: compute in fp32, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(output_dtype)

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
    )

    err = checkAllclose(ref.to(torch.float), out.to(torch.float))
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
    }
    ret = {}
    ret["input_dtype"] = dtype_map.get(input_dtype, str(input_dtype))
    ret["output_dtype"] = dtype_map.get(output_dtype, str(output_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
    nargs="*",
    metavar="{fp16, bf16}",
    default="fp16, bf16",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    choices=[1, 32, 64, 128, 256, 512, 1024, 4096, 8192, 163840],
    default=[1, 32, 64, 128, 256, 512, 1024, 4096, 8192, 163840],
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    type=int,
    nargs="*",
    choices=[1024, 4096, 6400, 8192],
    default=[1024, 4096, 6400, 8192],
    help="""N of mnk.
    e.g.: -n 1024""",
)

args = parser.parse_args()

test_scaled_silu_and_mul_edge_cases()

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_scaled_silu_and_mul(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    ["M", "N", "input_dtype", "output_dtype", "us", "TB/s", "RD TB/s", "WR TB/s", "err"]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("scaled_silu_and_mul summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_silu_and_mul(m, n, dtype)
            df.append(ret)
# Add fp32 input with fp16/bf16 output (bandwidth optimization)
for output_dtype in [torch.float16, torch.bfloat16]:
    for m in args.m:
        for n in args.n:
            ret = test_silu_and_mul(m, n, torch.float32, output_dtype=output_dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    ["M", "N", "input_dtype", "output_dtype", "us", "TB/s", "RD TB/s", "WR TB/s", "err"]
]

df_md = df.to_markdown(index=False)
aiter.logger.info("silu_and_mul summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_silu_and_mul(m, n, dtype, limit=10.0)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    [
        "M",
        "N",
        "input_dtype",
        "output_dtype",
        "limit",
        "us",
        "TB/s",
        "RD TB/s",
        "WR TB/s",
        "err",
    ]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("silu_and_mul with limit=10.0 summary (markdown):\n%s", df_md)

quant_cols = [
    "M",
    "N",
    "input_dtype",
    "output_dtype",
    "group_size",
    "us",
    "TB/s",
    "RD TB/s",
    "WR TB/s",
    "err_hip",
    "err_scale",
    "scale_max_abs_diff",
    "scale_mismatch_ratio",
]

# silu_and_mul_quant with fp8 (group_size=64, 128)
df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            for gs in [64, 128]:
                d = n // 2
                if d >= gs and d % gs == 0:
                    ret = test_silu_and_mul_quant(m, n, dtype, group_size=gs)
                    df.append(ret)
if df:
    df = pd.DataFrame(df)
    df = df[quant_cols]
    df_md = df.to_markdown(index=False)
    aiter.logger.info("silu_and_mul_quant (fp8) summary (markdown):\n%s", df_md)

# silu_and_mul_quant with fp4 (group_size=32)
# FP4 (e2m1) output uses CDNA4-only MFMA and is compiled in only for
# gfx950/gfx1250. On other archs (e.g. gfx942/MI300A) the kernel is not
# built, so skip instead of aborting inside the C++ dispatch.
_fp4_gfx = aiter.get_gfx()
if _fp4_gfx not in ("gfx950", "gfx1250"):
    aiter.logger.info(
        "skip silu_and_mul_quant (fp4): fp4 output requires gfx950/gfx1250, got %s",
        _fp4_gfx,
    )
else:
    df = []
    for dtype in args.dtype:
        for m in args.m:
            for n in args.n:
                d = n // 2
                gs = 32
                if d >= gs and d % gs == 0:
                    ret = test_silu_and_mul_quant(
                        m, n, dtype, group_size=gs, output_dtype=dtypes.fp4x2
                    )
                    df.append(ret)
    if df:
        df = pd.DataFrame(df)
        df = df[quant_cols]
        df_md = df.to_markdown(index=False)
        aiter.logger.info("silu_and_mul_quant (fp4) summary (markdown):\n%s", df_md)

# silu_and_mul_quant with fp8 + limit=10
df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            d = n // 2
            gs = 128
            if d >= gs and d % gs == 0:
                ret = test_silu_and_mul_quant(m, n, dtype, group_size=gs, limit=10.0)
                df.append(ret)
if df:
    df = pd.DataFrame(df)
    df = df[quant_cols + ["limit"]]
    df_md = df.to_markdown(index=False)
    aiter.logger.info(
        "silu_and_mul_quant (fp8, limit=10) summary (markdown):\n%s", df_md
    )

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_gelu_fast(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    [
        "M",
        "N",
        "input_dtype",
        "output_dtype",
        "us",
        "torch_us",
        "speedup_vs_torch",
        "perf_gain_vs_torch_pct",
        "TB/s",
        "RD TB/s",
        "WR TB/s",
        "err",
    ]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("gelu_fast summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_relu2(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    [
        "M",
        "N",
        "input_dtype",
        "output_dtype",
        "us",
        "torch_us",
        "speedup_vs_torch",
        "perf_gain_vs_torch_pct",
        "TB/s",
        "RD TB/s",
        "WR TB/s",
        "err",
    ]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("relu2 summary (markdown):\n%s", df_md)
df = []
for dtype in args.dtype:
    for numel in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63]:
        ret = test_relu2_bounds(numel, dtype)
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("relu2_bounds summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for numel in [1, 2, 7, 8, 9, 15, 16, 17, 255, 1001, 4095, 4101]:
        ret = test_gelu_fast_bounds(numel, dtype)
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("gelu_fast_bounds summary (markdown):\n%s", df_md)

df = []
df_refused = []
for op in _GATE_OPS:
    for dtype in args.dtype:
        for m in [4, 257]:
            for d in [255, 256, 1001, 1002]:
                if d * dtype.itemsize % 4 == 0:
                    df.append(test_act_and_mul_row_bound(op, dtype, m, d))
                else:
                    ret = test_act_and_mul_split_word_refused(op, dtype, m, d)
                    df_refused.append(ret)
df_md = pd.DataFrame(df).to_markdown(index=False)
aiter.logger.info("act_and_mul row bound summary (markdown):\n%s", df_md)
df_md = pd.DataFrame(df_refused).to_markdown(index=False)
aiter.logger.info("act_and_mul split word refusal summary (markdown):\n%s", df_md)
