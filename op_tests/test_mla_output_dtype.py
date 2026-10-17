# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Output-width guards and an MLA accuracy/performance sweep.

The default BF16 sweep matches the reported SGLang caller. Use ``-d bf16 fp16``
to audit both 16-bit formats. Existing single-split direct-output paths can
write BF16 bytes into an FP16 output; the full sweep reports those failures.
The guards prevent unsafe output widths, not that separate numerical defect.
"""

import argparse
import ctypes
import hashlib
import itertools
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import JIT_LIB_EXT, get_user_jit_dir
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import mla_decode_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.dtypes import aiter_tensor_t, torch_to_aiter

SUPPORTED_GFX = ("gfx942", "gfx950")
ERROR = "output must use a 16-bit floating dtype"


@pytest.mark.parametrize("query_dtype", [dtypes.bf16, dtypes.fp8])
@pytest.mark.parametrize("output_dtype", [dtypes.fp8, dtypes.fp32])
def test_mla_decode_rejects_unsupported_output_dtype(query_dtype, output_dtype):
    # This must fail before any CUDA allocation or metadata access.
    q = torch.empty((1, 16, 576), dtype=query_dtype, device="cpu")
    kv = torch.empty((1, 1, 1, 576), dtype=query_dtype, device="cpu")
    output = torch.empty((1, 16, 512), dtype=output_dtype, device="cpu")
    with pytest.raises(ValueError, match=ERROR):
        mla_decode_fwd(q, kv, output, None, None, None, None, 1)


def _stage1_arguments(output_dtype):
    device = "cuda"
    return {
        "Q": torch.zeros((1, 16, 576), dtype=dtypes.bf16, device=device),
        "KV": torch.zeros((1, 1, 1, 576), dtype=dtypes.bf16, device=device),
        "qo_indptr": torch.tensor([0, 1], dtype=dtypes.i32, device=device),
        "kv_indptr": torch.tensor([0, 1], dtype=dtypes.i32, device=device),
        "kv_page_indices": torch.tensor([0], dtype=dtypes.i32, device=device),
        "kv_last_page_lens": torch.ones(1, dtype=dtypes.i32, device=device),
        "num_kv_splits_indptr": torch.tensor([0, 1], dtype=dtypes.i32, device=device),
        "work_meta_data": None,
        "work_indptr": None,
        "work_info_set": None,
        "max_seqlen_q": 1,
        "page_size": 1,
        "nhead_kv": 1,
        "softmax_scale": 576**-0.5,
        "splitData": torch.full((1, 1, 16, 512), -7, dtype=dtypes.fp32, device=device),
        "splitLse": torch.full((1, 1, 16, 1), -7, dtype=dtypes.fp32, device=device),
        "output": torch.full((1, 16, 512), 1, dtype=output_dtype, device=device),
    }


def _raw_stage1_call(arguments):
    # The public stage1 call below builds this exact DSO first. Bypass the
    # Python ctypes error translator here and check the C status/TLS ABI.
    lib = ctypes.CDLL(str(Path(get_user_jit_dir()) / f"module_mla_asm{JIT_LIB_EXT}"))
    lib.aiter_ctypes_abi_version.restype = ctypes.c_int
    assert lib.aiter_ctypes_abi_version() >= 2
    lib.aiter_get_last_error.restype = ctypes.c_char_p
    lib.aiter_clear_last_error.restype = None
    # Explicit ABI order: unlike the public API this call must not depend on
    # compile_ops preserving Python signatures after lazy loading.
    parameters = dict(
        arguments,
        lse=None,
        q_scale=None,
        kv_scale=None,
        g_kv_indptr=None,
        cp_world_size=1,
        cp_rank=0,
        valid_split_count=None,
        use_valid_split_count_reduce=0,
        causal=True,
        kv_shuffled=0,
        kv_indptr_is_token_level=False,
    )
    argtypes, values, keepalive = [], [], []
    tensor_pointer = ctypes.POINTER(aiter_tensor_t)
    for name, value in parameters.items():
        if isinstance(value, torch.Tensor):
            tensor = torch_to_aiter(value)
            keepalive.append(tensor)
            argtypes.append(tensor_pointer)
            values.append(ctypes.pointer(tensor))
        elif value is None:
            argtypes.append(tensor_pointer)
            values.append(None)
        elif isinstance(value, bool):
            argtypes.append(ctypes.c_int)
            values.append(value)
        elif isinstance(value, float):
            argtypes.append(ctypes.c_float)
            values.append(value)
        else:
            assert isinstance(value, int), (name, type(value))
            argtypes.append(ctypes.c_int64)
            values.append(value)
    function = lib.mla_decode_stage1_asm_fwd
    function.argtypes = argtypes + [ctypes.c_void_p]
    function.restype = ctypes.c_int
    lib.aiter_clear_last_error()
    status = function(*values, torch.cuda.current_stream().cuda_stream)
    assert status == -1, status
    error = lib.aiter_get_last_error()
    assert error is not None and ERROR in error.decode(), error
    lib.aiter_clear_last_error()
    assert not lib.aiter_get_last_error()


def _check_direct_rejection(dtype_name, entrypoint):
    arguments = _stage1_arguments(getattr(dtypes, dtype_name))
    before = {
        name: arguments[name].clone() for name in ("output", "splitData", "splitLse")
    }
    # Also builds module_mla_asm before the raw-CDLL check. This must raise,
    # not abort the interpreter or launch a kernel with the invalid buffer.
    with pytest.raises(RuntimeError, match=ERROR):
        aiter.mla_decode_stage1_asm_fwd(**arguments)
    if entrypoint == "raw":
        _raw_stage1_call(arguments)
    torch.cuda.synchronize()
    for name, expected in before.items():
        torch.testing.assert_close(
            arguments[name].float(), expected.float(), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype_name", ["fp8", "fp32"])
@pytest.mark.parametrize("entrypoint", ["stage1", "raw"])
def test_direct_stage1_rejects_unsupported_output_dtype(dtype_name, entrypoint):
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        pytest.skip("requires a gfx942/gfx950 ROCm GPU")
    # A broken C ABI must fail this test, not terminate the entire test suite.
    script = (
        "import runpy; "
        f"m=runpy.run_path({str(Path(__file__).resolve())!r}); "
        f"m['_check_direct_rejection']({dtype_name!r}, {entrypoint!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (
        f"{entrypoint}/{dtype_name}: exit {result.returncode}\n"
        f"{result.stdout}\n{result.stderr}"
    )


def _persistent_metadata(qo_indptr, kv_indptr, last_page, query_dtype, splits):
    names = (
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    )
    info = aiter.get_mla_metadata_info_v1(
        qo_indptr.numel() - 1,
        1,
        16,
        query_dtype,
        query_dtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=splits,
    )
    metadata = {
        name: torch.empty(shape, dtype=dtype, device="cuda")
        for name, (shape, dtype) in zip(names, info)
    }
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        last_page,
        16,
        1,
        True,
        metadata["work_meta_data"],
        metadata["work_info_set"],
        metadata["work_indptr"],
        metadata["reduce_indptr"],
        metadata["reduce_final_map"],
        metadata["reduce_partial_map"],
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=splits,
        dtype_q_nope=query_dtype,
        dtype_kv_nope=query_dtype,
    )
    return metadata


def run_torch(q, kv, batch_size, kv_len):
    # Page size 1, one query/request, shared MLA K/V (512 NoPE + 64 RoPE).
    keys = kv.reshape(batch_size, kv_len, 576).float()
    scores = torch.einsum("bhd,bkd->bhk", q.float(), keys) * 576**-0.5
    return torch.einsum("bhk,bkd->bhd", scores.softmax(-1), keys[..., :512])


@benchmark()
def benchmark_valid_output(
    batch_size, kv_len, query_dtype, output_dtype, persistent, splits
):
    torch.manual_seed(5120)
    q = torch.randn((batch_size, 16, 576), dtype=dtypes.bf16, device="cuda").to(
        query_dtype
    )
    kv = torch.randn(
        (batch_size * kv_len, 1, 1, 576), dtype=dtypes.bf16, device="cuda"
    ).to(query_dtype)
    output = torch.full(
        (batch_size, 16, 512), float("nan"), dtype=output_dtype, device="cuda"
    )
    qo_indptr = torch.arange(batch_size + 1, dtype=dtypes.i32, device="cuda")
    kv_indptr = qo_indptr * kv_len
    indices = torch.arange(batch_size * kv_len, dtype=dtypes.i32, device="cuda")
    last_page = torch.ones(batch_size, dtype=dtypes.i32, device="cuda")
    kwargs = {"num_kv_splits": splits}
    if query_dtype == dtypes.fp8:
        kwargs.update(
            q_scale=torch.ones(1, device="cuda"), kv_scale=torch.ones(1, device="cuda")
        )
    if persistent:
        kwargs.update(
            _persistent_metadata(qo_indptr, kv_indptr, last_page, query_dtype, splits)
        )
    reference = run_torch(q, kv, batch_size, kv_len)
    candidates = {
        "asm": lambda: mla_decode_fwd(
            q, kv, output, qo_indptr, kv_indptr, indices, last_page, 1, **kwargs
        )
    }
    flops = 2 * batch_size * 16 * kv_len * (576 + 512)
    nbytes = (
        q.numel() * q.element_size()
        + kv.numel() * kv.element_size()
        + output.numel() * output.element_size()
    )
    ret = {"gfx": get_gfx()}
    for name, call in candidates.items():
        _, us = run_perftest(call, num_iters=101, use_cuda_event=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(25):
            call()
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - started) * 1e6 / 25
        actual = output.float()
        finite = bool(torch.isfinite(actual).all())
        delta = actual - reference
        ret[f"{name} us"] = us
        ret[f"{name} wall us"] = wall_us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} max_abs"] = delta.abs().max().item() if finite else None
        ret[f"{name} rmse"] = delta.square().mean().sqrt().item() if finite else None
        ret[f"{name} err"] = (
            checkAllclose(reference, actual, rtol=1e-2, atol=1e-2, printLog=False)
            if finite
            else 1.0
        )
        ret[f"{name} finite"] = finite
        ret[f"{name} accuracy_pass"] = finite and ret[f"{name} err"] <= 0.05
        ret[f"{name} output_sha256"] = hashlib.sha256(
            output.view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "MLA output dtype sweep unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        default=[dtypes.bf16],
    )
    parser.add_argument(
        "--query-dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        default=[dtypes.bf16, dtypes.fp8],
    )
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--kv-len", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument(
        "--persistent", type=int, choices=[0, 1], nargs="+", default=[0, 1]
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    code = pytest.main([__file__, "-v"])
    if code:
        raise SystemExit(code)
    rows = [
        benchmark_valid_output(
            batch, length, query_dtype, output_dtype, persistent, splits
        )
        for batch, length, query_dtype, output_dtype, persistent, splits in itertools.product(
            args.batch,
            args.kv_len,
            args.query_dtype,
            args.dtype,
            args.persistent,
            args.splits,
        )
    ]
    aiter.logger.info(
        "MLA output dtype summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    if any(not row["asm accuracy_pass"] for row in rows):
        raise AssertionError("MLA output dtype sweep has accuracy failures; see table")


if __name__ == "__main__":
    main()
