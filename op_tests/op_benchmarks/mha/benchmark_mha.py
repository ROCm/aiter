# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import aiter
import math
import torch
import torch.utils.benchmark as benchmark
import argparse


def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_fwd_bwd(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


def perf_test(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    dropout_p,
    causal,
    deterministic,
    dtype
):
    repeats = 30

    q = torch.zeros(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.zeros(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.zeros(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    window_size = (-1, -1)
    attn_bias = None
    alibi_slopes = None

    f, b = time_fwd_bwd(
        aiter.flash_attn_func,
        q,
        k,
        v,
        dropout_p,
        causal=causal,
        window_size=window_size,
        bias=attn_bias,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=True,
        return_attn_probs=False,
        repeats=repeats,
        verbose=False
        )

    print(f"fwd: {f * 1000} ms" f", bwd: {b * 1000} ms")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    # formatter_class=argparse.RawTextHelpFormatter,
    # description="config input of test",
    # )
    # parser.add_argument(
    #     "-b",
    #     "--batch_size",
    #     type=int,
    #     default=2,
    #     help="""Batch size. Default is 2.
    #     e.g.: -b 16""",
    # )

    batch_size = 1280
    nheads = 2
    nheads_k = 2
    seqlen_q = 8
    seqlen_k = 232
    d = 128
    d_v = 128
    dropout_p = 0.0
    causal = False
    deterministic = True
    dtype = torch.bfloat16

    perf_test(
        batch_size,
        nheads,
        nheads_k,
        seqlen_q,
        seqlen_k,
        d,
        d_v,
        dropout_p,
        causal,
        deterministic,
        dtype
    )

