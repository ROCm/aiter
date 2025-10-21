# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_common import (
    perftest,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)
import itertools
import pytest
import sys
from dataclasses import dataclass
from typing import Tuple
from enum import IntEnum

REF_BY_TRITON = False

if REF_BY_TRITON:
    from aiter.ops.triton.mha import flash_attn_func
else:
    from aiter import flash_attn_func, flash_attn_varlen_func


def run_torch(
    q,
    k,
    v,
    causal,
    upcast=True,
    reorder_ops=False,
):
    out, _ = attention_ref(
        q,
        k,
        v,
        causal=causal,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )

    return out


@perftest()
def profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time_in_us):
    return flop / time_in_us / 10**6


@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("nheads", [6])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("seed", [None])
def test_fmha_v3_fwd_ck(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    causal,
    mha_type,
    dtype,
    seed,
    profile=False,
):
    if seed is not None:
        torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0

    if causal and seqlen_k < seqlen_q:
        pytest.skip("Causal attention not supported for seqlen_k < seqlen_q")

    def print_tensor(tensor, tensor_name):
        tensor_list = tensor.tolist()

        for i, row in enumerate(tensor_list):
            formatted_row = ", ".join("{:5.2f}".format(x) for x in row)
            print("[HOST] {0}[{1:3}] = {2}".format(tensor_name, i, formatted_row))
        sys.stdout.flush()

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )
    # print(f'{q.shape=}')
    # print(f'{k.shape=}')
    # print(f'{v.shape=}')

    def save_tensor(tensor, fname):
        tensor_np = tensor.cpu().numpy()
        tensor_np.tofile(fname)

    # save_tensor(q.squeeze(0).squeeze(1), f"q_{q.size(1)}x{q.size(3)}.bin")
    # save_tensor(k.squeeze(0).squeeze(1), f"k_{k.size(1)}x{k.size(3)}.bin")
    # save_tensor(v.squeeze(0).squeeze(1), f"v_{v.size(1)}x{v.size(3)}.bin")

    # print_tensor(q.squeeze(0).squeeze(1), 'Q')
    # print_tensor(k.squeeze(0).squeeze(1), 'K')
    # print_tensor(v.squeeze(0).squeeze(1), 'V')

    attention = aiter.fmha_v3_fwd_ck_func
    if profile:
        out, time = profile_func(attention, q, k, v, causal=causal)
        tflops = efficiency(flops(batch_size, seqlen_q, d, nheads, causal), time)
        print(f"time: {time:.2f} us, {tflops:.2f} TFlops")
    else:
        out = attention(q, k, v, causal=causal)

    # print_tensor(out.squeeze(0).squeeze(1), 'O')

    if profile:
        return

    _strict = False

    if _strict:
        out_ref = run_torch(q, k, v, causal=causal)

        # print_tensor(out_ref.squeeze(0).squeeze(1), 'out_ref')

        out_pt = run_torch(
            q,
            k,
            v,
            causal=causal,
            upcast=False,
            reorder_ops=True,
        )

        # print_tensor(out_pt.squeeze(0).squeeze(1), 'out_pt')

        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        assert (out - out_ref).abs().max().item() <= 2 * (
            out_pt - out_ref
        ).abs().max().item()
    else:
        out_ref, _ = flash_attn_func(q, k, v, causal=causal, return_lse=True)
        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("nheads", [6])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("seed", [None])
def test_fmha_v3_varlen_fwd_ck(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    causal,
    mha_type,
    dtype,
    seed,
    profile=False,
):
    if seed is not None:
        torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0

    if causal and seqlen_k < seqlen_q:
        pytest.skip("Causal attention not supported for seqlen_k < seqlen_q")

    def print_tensor(tensor, tensor_name):
        tensor_list = tensor.tolist()

        for i, row in enumerate(tensor_list):
            formatted_row = ", ".join("{:5.2f}".format(x) for x in row)
            print("[HOST] {0}[{1:3}] = {2}".format(tensor_name, i, formatted_row))
        sys.stdout.flush()

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d_v, device="cuda", dtype=dtype, requires_grad=False)
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, "cuda", mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, "cuda", mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    # print(f'{q_unpad.shape=}')
    # print(f'{k_unpad.shape=}')
    # print(f'{v_unpad.shape=}')

    def save_tensor(tensor, fname):
        tensor_np = tensor.cpu().numpy()
        tensor_np.tofile(fname)

    # save_tensor(q.squeeze(0).squeeze(1), f"q_{q.size(1)}x{q.size(3)}.bin")
    # save_tensor(k.squeeze(0).squeeze(1), f"k_{k.size(1)}x{k.size(3)}.bin")
    # save_tensor(v.squeeze(0).squeeze(1), f"v_{v.size(1)}x{v.size(3)}.bin")

    # print_tensor(q.squeeze(0).squeeze(1), 'Q')
    # print_tensor(k.squeeze(0).squeeze(1), 'K')
    # print_tensor(v.squeeze(0).squeeze(1), 'V')

    attention = aiter.fmha_v3_varlen_fwd_ck_func
    if profile:
        out, time = profile_func(
                attention,
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal)
        tflops = efficiency(flops(batch_size, seqlen_q, d, nheads, causal), time)
        print(f"time: {time:.2f} us, {tflops:.2f} TFlops")
    else:
        out = attention(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal)

    # print_tensor(out.squeeze(0).squeeze(1), 'O')

    if profile:
        return

    _strict = False

    if _strict:
        out_ref = run_torch(q, k, v, causal=causal)

        # print_tensor(out_ref.squeeze(0).squeeze(1), 'out_ref')

        out_pt = run_torch(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            upcast=False,
            reorder_ops=True,
        )

        # print_tensor(out_pt.squeeze(0).squeeze(1), 'out_pt')

        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        assert (out - out_ref).abs().max().item() <= 2 * (
            out_pt - out_ref
        ).abs().max().item()
    else:
        out_ref, _ = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                causal=causal, return_lse=True)
        out = output_pad_fn(out)
        out_ref = output_pad_fn(out_ref)
        print(f'out {out.shape}')
        print(f'out_ref {out_ref.shape}')
        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-2)

if __name__ == "__main__":

    class MaskType(IntEnum):
        CAUSAL = 1
        NOT_CAUSAL = 2
        BOTH = 3

    @dataclass
    class ProblemSize:
        batch_size: int
        nheads_qk: Tuple[int, ...]
        seqlens: Tuple[int, ...]
        head_sizes: Tuple[int, ...]
        causal: MaskType

    profile = False
    seed = 0

    problem_sizes = [
        # batch_size, (nheads, nheads_k), (seqlen_q, seqlen_k), (d, d_v)
#        ProblemSize(32, (16,), (512,), (128,), MaskType.BOTH),
#        ProblemSize(16, (16,), (1024,), (128,), MaskType.BOTH),
#        ProblemSize(8, (16,), (2048,), (128,), MaskType.BOTH),
#        ProblemSize(4, (16,), (4096,), (128,), MaskType.BOTH),
#        ProblemSize(2, (16,), (8192,), (128,), MaskType.BOTH),
#        ProblemSize(1, (16,), (16384,), (128,), MaskType.BOTH),
#        ProblemSize(1, (64,), (16384,), (128,), MaskType.BOTH),
#        ProblemSize(1, (16, 1), (65536,), (128,), MaskType.BOTH),
#        ProblemSize(1, (40,), (37200,), (128,), MaskType.BOTH),
        ProblemSize(1, (6, 1), (1024, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (6, 1), (2048, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (6, 1), (4096, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (6, 1), (8192, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (6, 1), (16384, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (6, 1), (32768, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (6, 1), (65536, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (6, 1), (131072, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (16, 1), (65536, ), (128, ), MaskType.CAUSAL),
        ProblemSize(1, (40,), (37200,), (128,), MaskType.NOT_CAUSAL),
    ]

    l_dtypes = [dtypes.bf16]

    for dtype, problem_size in itertools.product(l_dtypes, problem_sizes):
        batch_size = problem_size.batch_size
        nheads, nheads_k = (
            problem_size.nheads_qk
            if 1 < len(problem_size.nheads_qk)
            else problem_size.nheads_qk * 2
        )
        seqlen_q, seqlen_k = (
            problem_size.seqlens
            if 1 < len(problem_size.seqlens)
            else problem_size.seqlens * 2
        )
        d, d_v = (
            problem_size.head_sizes
            if 1 < len(problem_size.head_sizes)
            else problem_size.head_sizes * 2
        )

        if problem_size.causal == MaskType.BOTH:
            l_causal = [False, True]
        else:
            l_causal = [True] if problem_size.causal == MaskType.CAUSAL else [False]

        assert nheads == nheads_k or nheads_k == 1
        mha_type = "mha" if nheads == nheads_k else "mqa"

        for causal in l_causal:
            print(
                f"b:{batch_size}, h:{nheads}/{nheads_k}, s={seqlen_q}/{seqlen_k}, causal={causal}, dtype={dtype}"
            )
            test_fmha_v3_varlen_fwd_ck(
                batch_size,
                nheads,
                seqlen_q,
                seqlen_k,
                d,
                d_v,
                causal,
                mha_type,
                dtype,
                seed,
                profile=profile,
            )
