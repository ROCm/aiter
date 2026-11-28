# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random

import pytest
import torch

from aiter.ops.triton.attention.mla import (
    mla_decode_fwd,
    mla_prefill_fwd,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_batched, shuffle_weight
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.test_common import checkAllclose
from op_tests.triton_tests.quant.test_quant_mxfp4 import (
    torch_dynamic_mxfp4_quant,
)

DEVICE_ARCH = arch_info.get_arch()


def uniform_random(shape, start=0, end=1, dtype=None, device=None):
    return (end - start) * torch.rand(shape, dtype=dtype, device=device) + start


def input_helper(
    batch_size: int,
    ctx_lens: int,
    decode_qlen: int,
    num_query_heads: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int | None,
    varlen: bool,
    device: str = "cuda",
):
    """Generates the paged-KV shapes/tensors shared by the decode and prefill
    tests. ``decode_qlen > 0`` fixes the query length (decode); ``0`` lets it
    vary with ``varlen`` like the KV length (prefill).

    ``num_blocks`` is the size of the block pool to draw (possibly repeated)
    block indices from. Pass ``None`` to instead allocate exactly as many
    blocks as needed and assign each sequence a unique set of blocks (e.g. for
    benchmarking realistic KV cache footprint/bandwidth)."""
    qk_head_dim = kv_lora_rank + qk_rope_head_dim

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int, device=device)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int, device=device)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
    else:
        seq_lens_kv.fill_(ctx_lens)
    if decode_qlen > 0:
        seq_lens_qo.fill_(decode_qlen)
    else:
        if varlen:
            for i in range(batch_size):
                seq_lens_qo[i] = max(
                    min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
                )
        else:
            seq_lens_qo.fill_(ctx_lens)

    cu_seqlens_q[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_num_query_tokens = cu_seqlens_q[-1].item()

    max_seqlen_kv = seq_lens_kv.max().item()
    max_num_blocks_per_seq = (max_seqlen_kv + block_size - 1) // block_size
    if num_blocks is None:
        num_blocks = batch_size * max_num_blocks_per_seq
        block_tables = torch.randperm(
            num_blocks, dtype=torch.int32, device=device
        ).reshape(batch_size, max_num_blocks_per_seq)
    else:
        block_tables = torch.randint(
            0,
            num_blocks,
            (batch_size, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )
    kv_buffer = torch.randn(
        (num_blocks, block_size, num_kv_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        (total_num_query_tokens, num_query_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    sm_scale = 1.0 / (qk_head_dim**0.5)

    return (
        cu_seqlens_q,
        seq_lens_kv,
        max_seqlen_kv,
        block_tables,
        kv_buffer,
        query,
        sm_scale,
        qk_head_dim,
    )


def shuffle_kv_buffer(
    kv_buffer: torch.Tensor,
    kv_lora_rank: int,
):
    """
    Shuffle key and value cache layout for optimized memory access.

        layout: (num_lanes, num_elements_per_thread)
            gfx1250: (16, 8) for BF16 and FP8.
            gfx950: (16, 8) for BF16 and (16, 16) for FP8.

        WMMA/MFMA instruction shape:
            BF16: 16x16x32
            FP8: 16x16x64
    """

    dtype = kv_buffer.dtype
    assert dtype in (torch.bfloat16, e4m3_dtype)

    if dtype == torch.bfloat16:
        layout = (16, 8)
    else:
        # Caution: in gfx1250, the 16-bit and 8-bit layout should both be (16, 8), however, in order to enable ds_load_b128 for 8-bit WMMA,
        # we use (16, 16) here, noted that you must set k_width to 16 in the corresponding DotOperandLayout, the math will be equivalent.
        layout = (16, 16)

    _num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape

    assert block_size >= 16

    num_lanes, num_elements_per_thread = layout

    def shuffle(kv_lora_or_rope_buffer):
        d = kv_lora_or_rope_buffer.shape[-1]
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.view(
            -1,
            num_kv_heads,
            block_size // num_lanes,
            num_lanes,
            d // (2 * num_elements_per_thread),
            2,  # there are 2 groups of threads, t0 ~ t15 and t16 ~ t31
            num_elements_per_thread,
        )
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.permute(
            0, 1, 2, 4, 5, 3, 6
        ).contiguous()
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.view(
            -1, num_kv_heads, block_size // 16, d * 16
        )
        return kv_lora_or_rope_buffer

    kv_buffer_shuffled = kv_buffer.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    kv_buffer_shuffled_lora = shuffle(kv_buffer_shuffled[..., :kv_lora_rank])
    kv_buffer_shuffled_rope = shuffle(kv_buffer_shuffled[..., kv_lora_rank:])
    kv_buffer_shuffled_lora = kv_buffer_shuffled_lora.view(
        -1, num_kv_heads, block_size * kv_lora_rank
    )
    kv_buffer_shuffled_rope = kv_buffer_shuffled_rope.view(
        -1, num_kv_heads, block_size * (head_size - kv_lora_rank)
    )
    kv_buffer_shuffled = torch.cat(
        [kv_buffer_shuffled_lora, kv_buffer_shuffled_rope], dim=-1
    ).contiguous()
    kv_buffer_shuffled = kv_buffer_shuffled.view(
        -1, num_kv_heads, block_size, head_size
    )

    return kv_buffer_shuffled


def dynamic_nvfp4_quant_kv_buffer(
    kv_buffer: torch.Tensor,
    kv_lora_rank: int,
):
    dtype = kv_buffer.dtype
    assert dtype == torch.bfloat16

    _num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape

    assert block_size >= 128

    def quant_and_shuffle(kv_lora_or_rope_buffer):
        d = kv_lora_or_rope_buffer.shape[-1]
        quant_head_size = d // 2
        scale_width = d // 16
        cache_shuffled, cache_shuffled_scale = torch_dynamic_mxfp4_quant(
            kv_lora_or_rope_buffer, is_nvfp4=True
        )
        cache_shuffled_scale = cache_shuffled_scale.view(
            -1, num_kv_heads, block_size, scale_width
        )
        cache_shuffled = shuffle_weight(cache_shuffled, arch="gfx950").view(
            -1, num_kv_heads, block_size * quant_head_size
        )
        cache_shuffled_scale = shuffle_scale_batched(cache_shuffled_scale).view(
            -1, num_kv_heads, block_size * scale_width
        )
        cache_shuffled = torch.cat(
            [
                cache_shuffled.view(torch.uint8),
                cache_shuffled_scale.view(torch.uint8),
            ],
            dim=-1,
        ).contiguous()
        cache_shuffled = cache_shuffled.view(
            -1, num_kv_heads, block_size, quant_head_size + scale_width
        )
        return cache_shuffled, quant_head_size, scale_width

    kv_buffer_shuffled = kv_buffer.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    kv_buffer_quant_and_shuffled_lora, quant_head_size_lora, scale_width_lora = (
        quant_and_shuffle(kv_buffer_shuffled[..., :kv_lora_rank])
    )
    kv_buffer_quant_and_shuffled_rope, quant_head_size_rope, scale_width_rope = (
        quant_and_shuffle(kv_buffer_shuffled[..., kv_lora_rank:])
    )
    kv_buffer_quant_and_shuffled_lora = kv_buffer_quant_and_shuffled_lora.view(
        -1, num_kv_heads, block_size * (quant_head_size_lora + scale_width_lora)
    )
    kv_buffer_quant_and_shuffled_rope = kv_buffer_quant_and_shuffled_rope.view(
        -1, num_kv_heads, block_size * (quant_head_size_rope + scale_width_rope)
    )
    kv_buffer_quant_and_shuffled = torch.cat(
        [kv_buffer_quant_and_shuffled_lora, kv_buffer_quant_and_shuffled_rope], dim=-1
    ).contiguous()
    kv_buffer_quant_and_shuffled = kv_buffer_quant_and_shuffled.view(
        -1,
        num_kv_heads,
        block_size,
        quant_head_size_lora
        + quant_head_size_rope
        + scale_width_lora
        + scale_width_rope,
    )

    return kv_buffer_quant_and_shuffled


def ref_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    q_descale: torch.Tensor | None = None,
    kv_descale: torch.Tensor | None = None,
) -> torch.Tensor:
    query_len = q.shape[0]
    kv_len = k.shape[0]
    if q.shape[1] != k.shape[1]:
        k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
        v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    k = k.to(q.dtype)
    attn = torch.einsum("qhd,khd->hqk", q, k).float()  # GEMM at q.dtype precision
    attn *= scale
    if q_descale is not None:
        attn *= q_descale
    if kv_descale is not None:
        attn *= kv_descale
    empty_mask = torch.ones(query_len, kv_len, device=q.device, dtype=torch.float32)
    mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
    attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    attn = attn.to(q.dtype)
    v = v.to(q.dtype)
    out = torch.einsum("hqk,khd->qhd", attn, v)  # GEMM at q.dtype precision
    if kv_descale is not None:
        out *= kv_descale

    return out


def torch_mla_extend(
    query,  # [total_q, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_block, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    cu_seqlens_q,
    seq_lens_kv,
    block_tables,
    qk_lora_rank,
    scale: float,
    q_descale: torch.Tensor | None = None,
    kv_descale: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    o_dtype: torch.dtype | None = torch.bfloat16,
):
    _, block_size, num_kv_heads, qk_head_dim = kv_buffer.shape
    num_seqs = cu_seqlens_q.shape[0] - 1

    outputs: list[torch.Tensor] = []
    for i in range(num_seqs):
        q = query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]

        kv_len = seq_lens_kv[i]
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = kv_buffer[block_indices].view(-1, num_kv_heads, qk_head_dim)
        k = k[:kv_len]
        v = k[..., :qk_lora_rank]

        out = ref_masked_attention(q, k, v, scale, q_descale, kv_descale)

        outputs.append(out)

    out = torch.cat(outputs, dim=0)
    if out_scale is not None:
        out = out / out_scale
    return out.to(o_dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 32])
@pytest.mark.parametrize("decode_qlen", [1, 3])
@pytest.mark.parametrize("ctx_lens", [1328, 4371])
@pytest.mark.parametrize("num_heads", [(16, 1), (128, 1)])
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("num_blocks", [32768])
@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, block_size, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, 64, False),
        (torch.bfloat16, e4m3_dtype, torch.bfloat16, 64, False),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, 64, False),
        (e4m3_dtype, e4m3_dtype, e4m3_dtype, 64, True),
        (e4m3_dtype, torch.uint8, torch.bfloat16, 128, False),
        (torch.uint8, torch.uint8, torch.bfloat16, 128, False),
    ],
)
@pytest.mark.parametrize("shuffled_kv_cache", [True, False])
@torch.inference_mode()
def test_mla_decode_fwd(
    batch_size: int,
    decode_qlen: int,
    ctx_lens: int,
    num_heads: tuple[int, int],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    varlen: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_out_scale: bool,
    shuffled_kv_cache: bool,
):
    if DEVICE_ARCH not in (
        "gfx950",
        "gfx1250",
        "gfx1250-strict",
    ):
        # gfx1250 -> Gluon
        # gfx950 -> Triton
        pytest.skip(f"skip {DEVICE_ARCH}")

    if kv_dtype == torch.uint8:
        if DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
            pytest.skip(f"NVFP4 requires {DEVICE_ARCH}")
        if not shuffled_kv_cache:
            pytest.skip("NVFP4 requires shuffled KV cache")

    torch.cuda.empty_cache()
    random.seed(0)
    torch.manual_seed(0)
    num_query_heads, num_kv_heads = num_heads

    (
        cu_seqlens_q,
        seq_lens_kv,
        max_seqlen_kv,
        block_tables,
        kv_buffer,
        query,
        sm_scale,
        qk_head_dim,
    ) = input_helper(
        batch_size,
        ctx_lens,
        decode_qlen,
        num_query_heads,
        num_kv_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        block_size,
        num_blocks,
        varlen,
    )
    total_num_query_tokens = cu_seqlens_q[-1].item()

    query_scales = None
    if q_dtype == torch.uint8:
        query = query / 10
        maybe_quant_query = query.view(-1, qk_head_dim)
        maybe_quant_query, query_scales = torch_dynamic_mxfp4_quant(
            maybe_quant_query, is_nvfp4=True
        )
        maybe_quant_query = maybe_quant_query.view(
            -1, num_query_heads, qk_head_dim // 2
        )
        query_scales = query_scales.view(-1, num_query_heads, qk_head_dim // 16)
        query = query.to(e4m3_dtype)
    else:
        query = query.to(q_dtype)
        maybe_quant_query = query

    q_descale = None
    kv_descale = None
    output_scale = None
    if q_dtype != torch.bfloat16:
        q_descale = uniform_random(
            1, start=1e-4, end=1.0, dtype=torch.float32, device="cuda"
        )

    if kv_dtype != torch.bfloat16:
        kv_descale = uniform_random(
            1, start=1e-4, end=1.0, dtype=torch.float32, device="cuda"
        )

    if use_out_scale:
        output_scale = 1 / uniform_random(
            1, start=1e-4, end=1.0, dtype=torch.float32, device="cuda"
        )

    output = torch.empty(
        (total_num_query_tokens, num_query_heads, kv_lora_rank),
        dtype=out_dtype,
        device="cuda",
    )

    if shuffled_kv_cache:
        if kv_dtype == torch.uint8:
            maybe_shuffled_kv_buffer = dynamic_nvfp4_quant_kv_buffer(
                kv_buffer, kv_lora_rank
            )
            kv_buffer = kv_buffer.to(e4m3_dtype)
        else:
            kv_buffer = kv_buffer.to(kv_dtype)
            maybe_shuffled_kv_buffer = shuffle_kv_buffer(kv_buffer, kv_lora_rank)
    else:
        kv_buffer = kv_buffer.to(kv_dtype)
        maybe_shuffled_kv_buffer = kv_buffer

    mla_decode_fwd(
        q=maybe_quant_query,
        kv_buffer=maybe_shuffled_kv_buffer,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_lens_kv,
        max_seqlen_kv=max_seqlen_kv,
        block_tables=block_tables,
        softmax_scale=sm_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        causal=True,
        q_descale=q_descale,
        kv_descale=kv_descale,
        q_scales=query_scales,
        out_scale=output_scale,
        shuffled_kv_cache=shuffled_kv_cache,
    )

    ref_output = torch_mla_extend(
        query,
        kv_buffer,
        cu_seqlens_q,
        seq_lens_kv,
        block_tables,
        kv_lora_rank,
        sm_scale,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=output_scale,
        o_dtype=out_dtype,
    )

    atol, rtol = 1.5e-2, 1e-2
    if q_dtype != torch.bfloat16 or kv_dtype != torch.bfloat16:
        atol, rtol = 1.5e-1, 1.5e-1
    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            output.to(torch.bfloat16),
            ref_output.to(torch.bfloat16),
            atol=atol,
            rtol=rtol,
            tol_err_ratio=tol_err_ratio,
            msg="mla_decode_fwd output",
        )
        <= tol_err_ratio
    )


@pytest.mark.parametrize(
    "output_kind,num_kv_heads",
    [("padded", 2), ("strided", 1), ("offset", 1), ("fp8", 1)],
)
@torch.inference_mode()
def test_mla_decode_fwd_lds_pipeline_ring_wraparound(
    monkeypatch, output_kind: str, num_kv_heads: int
):
    if DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("LDS pipeline requires gfx1250")

    import aiter.ops.triton.attention.mla as mla_module

    kernel = mla_module.gluon_mla_decode_fwd_kernel
    launches = []

    class CaptureLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)
                assert kwargs["USE_LDS_PIPELINE"]
                assert kwargs["num_stages"] == 4
                assert kwargs["num_warps"] == 4
                return kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(mla_module, "gluon_mla_decode_fwd_kernel", CaptureLaunch())

    # Exercise the single-segment specialization with a small batch that also
    # contains long sequences. The normal occupancy heuristic may split it.
    select_config = mla_module.select_3d_config

    def single_segment(*args, **kwargs):
        attn, reduce = select_config(*args, **kwargs)
        attn["NUM_SEGMENTS_PER_SEQ"] = 1
        return attn, reduce

    monkeypatch.setattr(mla_module, "select_3d_config", single_segment)
    torch.manual_seed(0)
    # Short paths, first buffer reuse, and repeated wraparound with a partial page.
    lengths = [1, 64, 65, 128, 129, 192, 193, 256, 257, 4097]
    batch_size, num_query_heads = len(lengths), 128 * num_kv_heads
    seq_lens_kv = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
    block_tables = torch.randint(
        0, 128, (batch_size, 65), dtype=torch.int32, device="cuda"
    )
    kv_buffer = torch.randn(
        (128, 64, num_kv_heads, 576), device="cuda", dtype=torch.bfloat16
    ).to(e4m3_dtype)
    query = torch.randn(
        (batch_size, num_query_heads, 576), device="cuda", dtype=torch.bfloat16
    ).to(e4m3_dtype)
    if output_kind == "fp8":
        # The one-token sequence exercises both saturation limits without
        # amplifying rounding error near zero throughout the other sequences.
        block_tables[0, 0] = 0
        kv_buffer[0, 0, :, 0] = torch.finfo(e4m3_dtype).max
        kv_buffer[0, 0, :, 1] = torch.finfo(e4m3_dtype).min
    shuffled_kv_buffer = shuffle_kv_buffer(kv_buffer, 512)
    q_descale = (
        None
        if output_kind == "strided"
        else torch.tensor([0.3], device="cuda", dtype=torch.float32)
    )
    kv_descale = (
        None
        if output_kind == "strided"
        else torch.tensor([0.4], device="cuda", dtype=torch.float32)
    )
    output_scale = torch.tensor(
        [0.25 if output_kind == "fp8" else 0.7], device="cuda", dtype=torch.float32
    )
    out_dtype = e4m3_dtype if output_kind == "fp8" else torch.bfloat16
    width = 513 if output_kind == "strided" else 520
    storage = torch.full(
        (batch_size, num_query_heads, width), 16.0, device="cuda", dtype=out_dtype
    )
    offset = 1 if output_kind == "offset" else 0
    output = storage[..., offset : offset + 512]

    def run(skip_reduce=False):
        return mla_decode_fwd(
            q=query,
            kv_buffer=shuffled_kv_buffer,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seq_lens_kv,
            max_seqlen_kv=max(lengths),
            block_tables=block_tables,
            softmax_scale=576**-0.5,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            causal=True,
            q_descale=q_descale,
            kv_descale=kv_descale,
            out_scale=output_scale,
            shuffled_kv_cache=True,
            skip_reduce=skip_reduce,
        )

    run()
    actual = output.float().clone()
    ref_output = torch_mla_extend(
        query,
        kv_buffer,
        cu_seqlens_q,
        seq_lens_kv,
        block_tables,
        512,
        576**-0.5,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=output_scale,
        o_dtype=torch.float32,
    )
    if output_kind == "fp8":
        ref_output = ref_output.clamp(
            torch.finfo(e4m3_dtype).min, torch.finfo(e4m3_dtype).max
        )
    ref_output = ref_output.to(out_dtype).float()
    assert torch.isfinite(actual).all()
    if output_kind == "fp8":
        assert (actual[0, :, 0] == torch.finfo(e4m3_dtype).max).all()
        assert (actual[0, :, 1] == torch.finfo(e4m3_dtype).min).all()
    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            output.to(torch.bfloat16),
            ref_output.to(torch.bfloat16),
            atol=1.5e-1,
            rtol=1.5e-1,
            tol_err_ratio=tol_err_ratio,
            msg="mla_decode_fwd LDS pipeline output",
        )
        <= tol_err_ratio
    )
    assert run(skip_reduce=True) is output
    assert len(launches) == 2
    torch.testing.assert_close(output.float(), actual, rtol=0, atol=0)
    # TDM and direct stores must respect the output view, including its padding.
    assert (storage[..., offset + 512 :].float() == 16).all()
    if offset:
        assert (storage[..., :offset].float() == 16).all()


@pytest.mark.parametrize(
    "kv_lora_rank, block_size, dtype",
    [
        (64, 64, e4m3_dtype),
        (128, 64, e4m3_dtype),
        (512, 64, e4m3_dtype),
        (128, 128, torch.bfloat16),
        (512, 128, e4m3_dtype),
    ],
)
@torch.inference_mode()
def test_mla_decode_fwd_unshuffled_pipeline_ring_wraparound(
    monkeypatch, kv_lora_rank, block_size, dtype
):
    if DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("unshuffled LDS pipeline requires gfx1250")

    import aiter.ops.triton.attention.mla as mla_module

    kernel = mla_module.gluon_mla_decode_fwd_kernel
    launches = []
    heuristic_configs = []
    select_config = mla_module.select_3d_config

    def capture_config(*args, **kwargs):
        attn, reduce = select_config(*args, **kwargs)
        heuristic_configs.append(attn.copy())
        return attn, reduce

    monkeypatch.setattr(mla_module, "select_3d_config", capture_config)

    class CaptureLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)
                assert not kwargs["SHUFFLED_KV_CACHE"]
                if block_size == 128:
                    for key, value in heuristic_configs[-1].items():
                        assert kwargs[key] == value
                    assert not kwargs.get("Q_FROM_LDS", False)
                    assert not kwargs.get("TDM_PARTIAL_STORE", False)
                else:
                    assert kwargs["num_stages"] == 3
                    assert kwargs["Q_FROM_LDS"]
                    assert kwargs["TDM_PARTIAL_STORE"]
                return kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(mla_module, "gluon_mla_decode_fwd_kernel", CaptureLaunch())
    torch.manual_seed(0)
    batch_size, num_query_heads, num_blocks = 32, 64, 128
    qk_head_dim = kv_lora_rank + 64
    max_seqlen_kv = 16385
    # Mixed lengths cover page boundaries, ring reuse and partial final pages.
    # Rank 64 uses direct stores despite the tuned TDM store flag.
    if block_size == 128 and dtype == e4m3_dtype:
        tuned = mla_module._tuned_decode_config(
            "pipelined_unshuffled",
            "fp8",
            "fp8",
            num_query_heads,
            batch_size,
            max_seqlen_kv,
        )
        assert tuned
        assert (
            mla_module._pipelined_unshuffled_lds_bytes(
                block_size,
                tuned["BLOCK_M"],
                kv_lora_rank,
                64,
                1,
                1,
                tuned["num_stages"],
                tuned["Q_FROM_LDS"],
            )
            > 256 * 1024
        )
        assert (
            mla_module._pipelined_unshuffled_lds_bytes(
                block_size, num_query_heads, kv_lora_rank, 64, 1, 1
            )
            <= 256 * 1024
        )
    lengths = [
        1,
        block_size,
        block_size + 1,
        2 * block_size + 1,
        16 * block_size + 1,
        max_seqlen_kv,
    ]
    lengths.extend([block_size + 1] * (batch_size - len(lengths)))
    seq_lens_kv = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, (max_seqlen_kv + block_size - 1) // block_size),
        dtype=torch.int32,
        device="cuda",
    )
    # Logical pages may reuse physical pages; the oracle follows the same table.
    kv_buffer = torch.randn(
        (num_blocks, block_size, 1, qk_head_dim), dtype=torch.bfloat16, device="cuda"
    ).to(dtype)
    query = torch.randn(
        (batch_size, num_query_heads, qk_head_dim), dtype=torch.bfloat16, device="cuda"
    ).to(dtype)
    q_descale = torch.tensor([0.3], dtype=torch.float32, device="cuda")
    kv_descale = torch.tensor([0.4], dtype=torch.float32, device="cuda")
    output_scale = torch.tensor([0.7], dtype=torch.float32, device="cuda")
    output = torch.empty(
        (batch_size, num_query_heads, kv_lora_rank), dtype=torch.bfloat16, device="cuda"
    )
    mla_decode_fwd(
        q=query,
        kv_buffer=kv_buffer,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_lens_kv,
        max_seqlen_kv=max_seqlen_kv,
        block_tables=block_tables,
        softmax_scale=qk_head_dim**-0.5,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=64,
        causal=True,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=output_scale,
        shuffled_kv_cache=False,
    )
    assert len(launches) == 1
    ref_output = torch_mla_extend(
        query,
        kv_buffer,
        cu_seqlens_q,
        seq_lens_kv,
        block_tables,
        kv_lora_rank,
        qk_head_dim**-0.5,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=output_scale,
        o_dtype=torch.float32,
    )
    assert torch.isfinite(output).all()
    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            output.float(),
            ref_output,
            atol=1.5e-1,
            rtol=1.5e-1,
            tol_err_ratio=tol_err_ratio,
            msg="mla_decode_fwd unshuffled pipeline output",
        )
        <= tol_err_ratio
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("ctx_lens", [200])
@pytest.mark.parametrize("num_heads", [(16, 1), (128, 1)])
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_blocks", [16384])
@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, e4m3_dtype, torch.bfloat16, True),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, True),
    ],
)
def test_mla_prefill_fwd(
    batch_size: int,
    ctx_lens: int,
    num_heads: tuple[int, int],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    varlen: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_out_scale: bool,
):
    torch.cuda.empty_cache()
    random.seed(0)
    torch.manual_seed(0)

    num_query_heads, num_kv_heads = num_heads

    (
        cu_seqlens_q,
        seq_lens_kv,
        max_seqlen_kv,
        block_tables,
        kv_buffer,
        q,
        sm_scale,
        _,
    ) = input_helper(
        batch_size,
        ctx_lens,
        0,  # decode_qlen=0 -> query length varies with varlen, like prefill
        num_query_heads,
        num_kv_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        block_size,
        num_blocks,
        varlen,
    )
    total_num_query_tokens = cu_seqlens_q[-1].item()
    kv_buffer = kv_buffer.to(kv_dtype)
    q = q.to(q_dtype)

    q_descale = None
    kv_descale = None
    if q_dtype != torch.bfloat16:
        q_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

    if kv_dtype != torch.bfloat16:
        kv_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

    out_scale = None
    if use_out_scale:
        out_scale = 1 / torch.rand(1, dtype=torch.float32, device="cuda")

    out = torch.empty(
        (total_num_query_tokens, num_query_heads, kv_lora_rank),
        dtype=out_dtype,
        device="cuda",
    )

    mla_prefill_fwd(
        q,
        kv_buffer,
        out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_lens_kv,
        max_seqlen_kv=max_seqlen_kv,
        block_tables=block_tables,
        softmax_scale=sm_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        causal=True,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=out_scale,
        shuffled_kv_cache=False,
    )

    out_ref = torch_mla_extend(
        q,
        kv_buffer,
        cu_seqlens_q,
        seq_lens_kv,
        block_tables,
        kv_lora_rank,
        sm_scale,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=out_scale,
        o_dtype=out_dtype,
    )

    atol, rtol = 1.5e-2, 1e-2
    if q_dtype == e4m3_dtype or kv_dtype == e4m3_dtype:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        out, out_ref, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(out - out_ref))}"


def _mla_gluon_decode(
    mla_gluon,
    kv_c,
    kv_indices,
    kv_indptr,
    q_nope,
    q_pe,
    batch,
    nhead,
    min_kv,
    kv_split_info=None,
):
    kv_lora_rank = q_nope.shape[-1]
    o = torch.empty(
        (batch, nhead, kv_lora_rank), dtype=q_nope.dtype, device=q_nope.device
    )
    mla_gluon(
        q_nope=q_nope,
        q_pe=q_pe,
        kv_c=kv_c,
        o=o,
        page_table=kv_indices,  # 1-D kv_indices (as vLLM passes for decode)
        seq_info=kv_indptr,  # 1-D kv_indptr
        sm_scale=1.0 / (kv_c.shape[-1] ** 0.5),
        k_pe=None,  # shared kv_c layout
        kv_pe_offset=kv_lora_rank,
        use_2d_view=False,
        kv_scale=1.0,
        min_kv_seq_len=min_kv,
        kv_split_info=kv_split_info,
    )
    torch.cuda.synchronize()
    return o


# Review: a num_iter==1 config reaches only the prologue loads; also cover the
# loop-body and epilogue loads (num_iter = cdiv(per_split, BLOCK_N=64)):
#   (8, 777)   -> num_iter 1 (prologue; last split is a partial block)
#   (4, 33000) -> num_iter 9 (loop-body + epilogue partial block)
@pytest.mark.parametrize("batch, ctx", [(8, 777), (4, 33000)])
@pytest.mark.parametrize("nhead", [8])  # small-head Gluon decode (Kimi K2.6/K3 @ TP8)
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
def test_mla_gluon_decode_over_2gb(batch, ctx, nhead, kv_lora_rank, qk_rope_head_dim):
    """Gluon MLA decode with a >2 GB paged KV cache (gfx950).

    A KV cache larger than 2 GB sets ``within_2gb=False`` so the kernel loads KV
    through ``global_load_to_shared`` (64-bit offsets) instead of
    ``buffer_load_to_shared``.  That path must carry the same bounds mask and
    ``other=0.0`` zero-fill as the buffer path, or the last partial split block
    reads out of bounds -> illegal memory access (garbage -> NaN).  The two paths
    load identical KV, so their outputs must be bit-identical; compared here with
    the reference KV placed at high row indices in the >2 GB cache.
    """
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Gluon MLA decode is gfx950-only")
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < int(3.5 * 2**30):
        pytest.skip("needs >3.5 GiB free for the >2 GB cache")

    from aiter.ops.triton.gluon.mla_gluon import mla_gluon

    head_dim = kv_lora_rank + qk_rope_head_dim
    gb2 = 0x80000000
    n_big = (gb2 // (head_dim * 2)) + 200_000  # ~2.2 GB cache -> within_2gb=False
    dtype = torch.bfloat16

    random.seed(0)
    torch.manual_seed(0)
    used = batch * ctx
    kv_data = torch.randn((used, head_dim), dtype=dtype, device="cuda")
    q_nope = torch.randn((batch, nhead, kv_lora_rank), dtype=dtype, device="cuda")
    q_pe = torch.randn((batch, nhead, qk_rope_head_dim), dtype=dtype, device="cuda")
    kv_indptr = torch.arange(
        0, (batch + 1) * ctx, ctx, dtype=torch.int32, device="cuda"
    )

    # reference: small cache (rows 0..used) -> within_2gb=True -> buffer_load path
    idx_ref = torch.arange(used, dtype=torch.int32, device="cuda")
    assert kv_data.shape[0] * kv_data.stride(0) * 2 <= gb2
    o_ref = _mla_gluon_decode(
        mla_gluon, kv_data, idx_ref, kv_indptr, q_nope, q_pe, batch, nhead, ctx
    )

    # under test: >2 GB cache, identical KV at the top -> global_load path
    kv_big = torch.zeros((n_big, head_dim), dtype=dtype, device="cuda")
    base = n_big - used
    kv_big[base : base + used].copy_(kv_data)
    idx_big = torch.arange(used, dtype=torch.int32, device="cuda") + base
    assert kv_big.shape[0] * kv_big.stride(0) * 2 > gb2
    o_big = _mla_gluon_decode(
        mla_gluon, kv_big, idx_big, kv_indptr, q_nope, q_pe, batch, nhead, ctx
    )

    assert torch.isfinite(o_big.float()).all(), "NaN/Inf in the >2 GB global_load path"
    # the two load paths read identical KV and reduce identically -> exact parity
    torch.testing.assert_close(o_big, o_ref, atol=0, rtol=0)

    # the same parity with the split rebalance (kv_split_info)
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon_split_metadata

    info = mla_gluon_split_metadata(kv_indptr, nhead, dtype, use_2d_view=False)
    assert info is not None
    o_ref = _mla_gluon_decode(
        mla_gluon, kv_data, idx_ref, kv_indptr, q_nope, q_pe, batch, nhead, ctx, info
    )
    o_big = _mla_gluon_decode(
        mla_gluon, kv_big, idx_big, kv_indptr, q_nope, q_pe, batch, nhead, ctx, info
    )
    torch.testing.assert_close(o_big, o_ref, atol=0, rtol=0)


def _mla_gluon_ref(
    q_nope, q_pe, kv_c, seq_lens, token_idx, sm_scale, kv_scale, attn_sink=None
):
    """fp32 MLA decode; query position i of qlen attends KV [0, len - qlen + i].
    attn_sink [nhead] adds one logit with no value to each softmax."""
    batch, qlen, nhead, kv_lora_rank = q_nope.shape
    out = torch.zeros(
        (batch, qlen, nhead, kv_lora_rank), dtype=torch.float32, device=q_nope.device
    )
    lse = torch.full(
        (batch, qlen, nhead), float("-inf"), dtype=torch.float32, device=q_nope.device
    )
    q = torch.cat([q_nope, q_pe], dim=-1).float()
    q_off = torch.arange(qlen, device=q_nope.device)[:, None]
    for b, seq_len in enumerate(seq_lens):
        if seq_len == 0:
            if attn_sink is not None:
                lse[b] = attn_sink.float()
            continue
        kv = kv_c[token_idx[b][:seq_len].long()].float() * kv_scale
        s = torch.einsum("qhd,ld->qhl", q[b], kv) * sm_scale
        pos = torch.arange(seq_len, device=q_nope.device)[None, :]
        s = s.masked_fill(~(pos <= seq_len - qlen + q_off)[:, None, :], float("-inf"))
        if attn_sink is None:
            lse[b] = torch.logsumexp(s, dim=-1)
        else:
            sink = attn_sink.float()[None, :, None].expand(qlen, nhead, 1)
            lse[b] = torch.logsumexp(torch.cat([s, sink], dim=-1), dim=-1)
        out[b] = torch.exp(s - lse[b][..., None]) @ kv[:, :kv_lora_rank]
    return out, lse


def _split_rebalance_inputs(batch, qlen, nhead, kv_dtype, max_len, kv_lora_rank=512):
    """Q and a paged KV pool where each (batch, position) has its own random row."""
    head_dim = kv_lora_rank + 64
    kv_c = torch.randn(
        (batch * max_len, head_dim), dtype=torch.bfloat16, device="cuda"
    ).to(kv_dtype)
    block_table = (
        torch.randperm(batch * max_len, device="cuda")
        .to(torch.int32)
        .view(batch, max_len)
    )
    q_nope = torch.randn(
        (batch, qlen, nhead, kv_lora_rank), dtype=torch.bfloat16, device="cuda"
    )
    q_pe = torch.randn((batch, qlen, nhead, 64), dtype=torch.bfloat16, device="cuda")
    return kv_c, block_table, q_nope, q_pe


def _paged_seq_info(block_table, seq_lens, use_2d_view):
    """(page_table, seq_info): the block table and cache_seqlens (2-D view), or
    kv_indices and kv_indptr (1-D view)."""
    if use_2d_view:
        return block_table, seq_lens
    lens = seq_lens.tolist()
    page_table = torch.cat([block_table[b, :n] for b, n in enumerate(lens)])
    seq_info = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    seq_info[1:] = torch.cumsum(seq_lens, 0)
    return page_table, seq_info


def _mla_gluon_paged(
    mla_gluon, q_nope, q_pe, kv_c, page_table, seq_info, use_2d_view, kv_scale, info
):
    """mla_gluon with q [batch, qlen, nhead, d]; returns (o, lse) in that shape."""
    batch, qlen, nhead, kv_lora_rank = q_nope.shape
    if qlen == 1:
        q_nope, q_pe = q_nope[:, 0], q_pe[:, 0]
    o = torch.empty(q_nope.shape, dtype=torch.bfloat16, device="cuda")
    o, lse = mla_gluon(
        q_nope=q_nope,
        q_pe=q_pe,
        kv_c=kv_c,
        o=o,
        page_table=page_table,
        seq_info=seq_info,
        sm_scale=1.0 / (kv_c.shape[-1] ** 0.5),
        k_pe=None,
        kv_pe_offset=kv_lora_rank,
        use_2d_view=use_2d_view,
        kv_scale=kv_scale,
        return_lse=True,
        kv_split_info=info,
    )
    return o.view(batch, qlen, nhead, kv_lora_rank), lse.view(batch, qlen, nhead)


def _check_split_partition(info, lens, block_n, indptr=None):
    """The work items cut each sequence's KV once, in BLOCK_N-aligned segments of
    one length (the last one of a sequence can be shorter), and split_meta lists
    each sequence's work items."""
    work_info, split_meta = info
    rows = work_info[:, :5].tolist()
    meta = split_meta.tolist()
    live = [(w, r) for w, r in enumerate(rows) if r[1] < r[2]]
    assert meta[len(lens)] == len(live) <= len(rows), "work item count"
    seg_lens, last_lens = set(), []
    for b, seq_len in enumerate(lens):
        mine = [(w, r) for w, r in live if r[0] == b]
        assert [w for w, _ in mine] == list(range(meta[b], meta[b + 1])), b
        if seq_len == 0:
            assert not mine, b
            continue
        starts = [r[1] for _, r in mine]
        ends = [r[2] for _, r in mine]
        assert starts == [0] + ends[:-1] and ends[-1] == seq_len, b
        assert all(s % block_n == 0 for s in starts), b
        assert all(r[3] == seq_len for _, r in mine), b
        if indptr is not None:
            assert all(r[4] == indptr[b] for _, r in mine), b
        seg_lens.update(e - s for s, e in zip(starts[:-1], ends[:-1]))
        last_lens.append(ends[-1] - starts[-1])
    assert len(seg_lens) <= 1, f"segments of different lengths: {sorted(seg_lens)}"
    if seg_lens:
        assert max(last_lens) <= seg_lens.pop(), "a last segment is the longest"


def _assert_mla_close(o, o_ref, o_base):
    """o is within 2**-6 of the largest output of the fp32 reference, and not
    more than twice as far from it as o_base (today's split)."""
    scale = o_ref.abs().max().item()
    err = (o.float() - o_ref).abs().max().item()
    err_base = (o_base.float() - o_ref).abs().max().item()
    torch.testing.assert_close(o.float(), o_ref, atol=2**-6 * scale, rtol=2**-6)
    assert (
        err <= 2 * err_base + 2**-9 * scale
    ), f"max err {err:.3e}, today's split {err_base:.3e} (scale {scale:.3e})"


# Uneven lengths: one sequence far longer than the rest (it gets most of the
# work items) and, for qlen 1, an empty one (no work item, zero output).
@pytest.mark.parametrize("batch", [1, 60])
@pytest.mark.parametrize("qlen", [1, 2])
@pytest.mark.parametrize(
    "nhead, kv_dtype",
    [
        (8, torch.float8_e4m3fn),
        (16, torch.float8_e4m3fn),
        (8, torch.bfloat16),
        (32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("use_2d_view", [True, False])
def test_mla_gluon_decode_split_rebalance(batch, qlen, nhead, kv_dtype, use_2d_view):
    """kv_split_info against an fp32 reference and today's split."""
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Gluon MLA decode is gfx950-only")
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon, mla_gluon_split_metadata

    random.seed(0)
    torch.manual_seed(0)
    if batch == 1:
        lens = [7000]
    else:
        lens = [random.randint(qlen, 3000) for _ in range(batch)]
        lens[1] = 12000
        if qlen == 1:
            lens[2] = 0
    kv_c, block_table, q_nope, q_pe = _split_rebalance_inputs(
        batch, qlen, nhead, kv_dtype, max(lens)
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    page_table, seq_info = _paged_seq_info(block_table, seq_lens, use_2d_view)
    kv_scale = 0.5 if kv_dtype == torch.float8_e4m3fn else 1.0

    info = mla_gluon_split_metadata(
        seq_info, nhead, kv_dtype, qlen=qlen, use_2d_view=use_2d_view
    )
    assert info is not None
    _check_split_partition(
        info,
        lens,
        128 if kv_dtype == torch.float8_e4m3fn else 64,
        indptr=None if use_2d_view else seq_info.tolist(),
    )
    args = (mla_gluon, q_nope, q_pe, kv_c, page_table, seq_info, use_2d_view, kv_scale)
    o, lse = _mla_gluon_paged(*args, info)
    o_base, _ = _mla_gluon_paged(*args, None)
    o_ref, lse_ref = _mla_gluon_ref(
        q_nope, q_pe, kv_c, lens, block_table, 1.0 / (kv_c.shape[-1] ** 0.5), kv_scale
    )
    _assert_mla_close(o, o_ref, o_base)
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=0)


@pytest.mark.parametrize("num_queries", [64, 256])
def test_mla_gluon_prefill_attn_sink(num_queries):
    """Prefill mode with 64 heads (bh64): 64 queries get 4 KV splits, 256 one. The
    sink is in the softmax once for each query, not once for each split, and an
    empty query gives o = 0 and lse = sink."""
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Gluon MLA decode is gfx950-only")
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon

    random.seed(0)
    torch.manual_seed(0)
    nhead = 64
    # bh64 needs more than 3 * BLOCK_N keys in each split
    lens = [random.randint(1000, 3000) for _ in range(num_queries)]
    lens[2] = 0
    kv_c, block_table, q_nope, _ = _split_rebalance_inputs(
        len(lens), 1, nhead, torch.bfloat16, max(lens)
    )
    # one 512-wide row is the key and the value
    kv_c = kv_c[:, :512].contiguous()
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    page_table, seq_info = _paged_seq_info(block_table, seq_lens, use_2d_view=False)
    sm_scale = 1.0 / (kv_c.shape[-1] ** 0.5)
    attn_sink = torch.randn(nhead, dtype=torch.float32, device="cuda") + 4.0
    o, lse = mla_gluon(
        q_nope=q_nope[:, 0],
        q_pe=None,
        kv_c=kv_c,
        # NaN, so that a row the kernel does not write fails
        o=torch.full(
            q_nope[:, 0].shape, float("nan"), dtype=torch.bfloat16, device="cuda"
        ),
        page_table=page_table,
        seq_info=seq_info,
        sm_scale=sm_scale,
        # the bh64 bound is for non-empty splits; an empty query returns before
        # the main loop
        min_kv_seq_len=min(n for n in lens if n),
        return_lse=True,
        has_pe=False,
        attn_sink=attn_sink,
    )
    o_ref, lse_ref = _mla_gluon_ref(
        q_nope, q_nope[..., :0], kv_c, lens, block_table, sm_scale, 1.0, attn_sink
    )
    scale = o_ref.abs().max().item()
    torch.testing.assert_close(
        o.view(o_ref.shape).float(), o_ref, atol=2**-6 * scale, rtol=2**-6
    )
    torch.testing.assert_close(lse.view(lse_ref.shape), lse_ref, atol=1e-3, rtol=0)


# 200 sequences: one KV split each by default; the shorter long sequence keeps
# kv_c below 2 GB
@pytest.mark.parametrize("batch, long_len", [(60, 12000), (200, 6000)])
@pytest.mark.parametrize("split_rebalance", [True, False])
@pytest.mark.parametrize("qlen", [1, 2])
@pytest.mark.parametrize(
    "nhead, kv_dtype", [(8, torch.float8_e4m3fn), (16, torch.bfloat16)]
)
def test_mla_gluon_decode_attn_sink(
    batch, long_len, split_rebalance, qlen, nhead, kv_dtype
):
    """With several KV splits (60 sequences: 2 or 4 each by default, more for the
    long ones with kv_split_info) the sink is in the softmax once for each
    sequence, not once for each split; with one split (200 sequences) an empty
    sequence gives o = 0 and lse = sink."""
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Gluon MLA decode is gfx950-only")
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon, mla_gluon_split_metadata

    random.seed(0)
    torch.manual_seed(0)
    lens = [random.randint(qlen, 3000) for _ in range(batch)]
    lens[1] = long_len
    if qlen == 1:
        lens[2] = 0
    kv_c, block_table, q_nope, q_pe = _split_rebalance_inputs(
        len(lens), qlen, nhead, kv_dtype, max(lens)
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    kv_scale = 0.5 if kv_dtype == torch.float8_e4m3fn else 1.0
    sm_scale = 1.0 / (kv_c.shape[-1] ** 0.5)
    # sinks near the logits, so that they change the softmax
    attn_sink = torch.randn(nhead, dtype=torch.float32, device="cuda") + 4.0
    info = (
        mla_gluon_split_metadata(seq_lens, nhead, kv_dtype, qlen=qlen)
        if split_rebalance
        else None
    )
    q_n, q_p = (q_nope[:, 0], q_pe[:, 0]) if qlen == 1 else (q_nope, q_pe)
    o, lse = mla_gluon(
        q_nope=q_n,
        q_pe=q_p,
        kv_c=kv_c,
        # NaN, so that a row the kernel does not write fails
        o=torch.full(q_n.shape, float("nan"), dtype=torch.bfloat16, device="cuda"),
        page_table=block_table,
        seq_info=seq_lens,
        sm_scale=sm_scale,
        kv_pe_offset=512,
        kv_scale=kv_scale,
        return_lse=True,
        attn_sink=attn_sink,
        kv_split_info=info,
    )
    o_ref, lse_ref = _mla_gluon_ref(
        q_nope, q_pe, kv_c, lens, block_table, sm_scale, kv_scale, attn_sink
    )
    scale = o_ref.abs().max().item()
    torch.testing.assert_close(
        o.view(o_ref.shape).float(), o_ref, atol=2**-6 * scale, rtol=2**-6
    )
    torch.testing.assert_close(lse.view(lse_ref.shape), lse_ref, atol=1e-3, rtol=0)


def test_mla_gluon_split_metadata_not_applicable():
    """None where the split rebalance does not apply."""
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Gluon MLA decode is gfx950-only")
    from aiter.ops.triton.gluon.mla_gluon import (
        mla_gluon_split_metadata,
        mla_gluon_split_metadata_info,
    )
    from aiter.ops.triton.utils.device_info import get_num_sms

    lens = torch.full((64,), 1000, dtype=torch.int32, device="cuda")
    assert mla_gluon_split_metadata(lens, 128, torch.bfloat16) is None  # bh64
    assert mla_gluon_split_metadata_info(64, 128, torch.bfloat16) is None
    # one work item for each CU: no more work items than sequences
    full = torch.full((get_num_sms(),), 1000, dtype=torch.int32, device="cuda")
    assert mla_gluon_split_metadata(full, 8, torch.float8_e4m3fn) is None
    assert mla_gluon_split_metadata(lens, 8, torch.float8_e4m3fn, qlen=4) is None


@pytest.mark.parametrize(
    "nhead, kv_dtype, qlen",
    [
        (8, torch.float8_e4m3fn, 1),
        (16, torch.float8_e4m3fn, 2),
        (32, torch.bfloat16, 1),
        (96, torch.bfloat16, 1),
    ],
)
def test_mla_gluon_split_metadata_info(nhead, kv_dtype, qlen):
    """Buffers from mla_gluon_split_metadata_info, for a larger batch, give the
    partition of the allocating call; wrong buffers and sizes raise."""
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Gluon MLA decode is gfx950-only")
    from aiter.ops.triton.gluon.mla_gluon import (
        mla_gluon_split_metadata,
        mla_gluon_split_metadata_info,
    )

    random.seed(0)
    batch, max_batch = 8, 32
    lens = [random.randint(1, 5000) for _ in range(batch)]
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    (work_shape, work_dtype), (meta_shape, meta_dtype) = mla_gluon_split_metadata_info(
        max_batch, nhead, kv_dtype, qlen=qlen
    )
    assert work_dtype == meta_dtype == torch.int32
    assert meta_shape == (max_batch + 1,)

    ref_work, ref_meta = mla_gluon_split_metadata(seq_lens, nhead, kv_dtype, qlen=qlen)
    assert tuple(ref_work.shape) == work_shape
    assert ref_meta.numel() == batch + 1

    work = torch.empty(work_shape, dtype=work_dtype, device="cuda")
    meta = torch.empty(meta_shape, dtype=meta_dtype, device="cuda")
    got_work, got_meta = mla_gluon_split_metadata(
        seq_lens, nhead, kv_dtype, qlen=qlen, out=(work, meta)
    )
    assert got_work is work and got_meta.numel() == batch + 1
    assert got_meta.data_ptr() == meta.data_ptr()
    # columns 5..7 of work_info are not written
    torch.testing.assert_close(got_work[:, :5], ref_work[:, :5], atol=0, rtol=0)
    torch.testing.assert_close(got_meta, ref_meta, atol=0, rtol=0)

    with pytest.raises(AssertionError):
        mla_gluon_split_metadata_info(0, nhead, kv_dtype, qlen=qlen)
    with pytest.raises(AssertionError):  # dtype
        mla_gluon_split_metadata(
            seq_lens, nhead, kv_dtype, qlen=qlen, out=(work.long(), meta)
        )
    with pytest.raises(AssertionError):  # split_meta smaller than batch + 1
        mla_gluon_split_metadata(
            seq_lens, nhead, kv_dtype, qlen=qlen, out=(work, meta[:batch])
        )
    with pytest.raises(AssertionError):  # work_info shape
        mla_gluon_split_metadata(
            seq_lens, nhead, kv_dtype, qlen=qlen, out=(work[1:], meta)
        )


def test_mla_gluon_split_metadata_cuda_graph():
    """The metadata launch follows the live lengths when a CUDA graph replays,
    into preallocated buffers from mla_gluon_split_metadata_info for a larger
    batch."""
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Gluon MLA decode is gfx950-only")
    from aiter.ops.triton.gluon.mla_gluon import (
        mla_gluon,
        mla_gluon_split_metadata,
        mla_gluon_split_metadata_info,
    )

    random.seed(0)
    torch.manual_seed(0)
    batch, nhead, kv_dtype, kv_scale = 64, 8, torch.float8_e4m3fn, 0.5
    lens_a = [2048] * batch
    lens_b = [random.randint(64, 3000) for _ in range(batch)]
    lens_b[5] = 12000
    kv_c, block_table, q_nope, q_pe = _split_rebalance_inputs(
        batch, 1, nhead, kv_dtype, max(lens_a + lens_b)
    )
    seq_lens = torch.tensor(lens_a, dtype=torch.int32, device="cuda")
    q_nope, q_pe = q_nope[:, 0].contiguous(), q_pe[:, 0].contiguous()
    o = torch.empty((batch, nhead, 512), dtype=torch.bfloat16, device="cuda")
    work_spec, meta_spec = mla_gluon_split_metadata_info(2 * batch, nhead, kv_dtype)
    work_info = torch.empty(work_spec[0], dtype=work_spec[1], device="cuda")
    split_meta = torch.empty(meta_spec[0], dtype=meta_spec[1], device="cuda")
    # buffers for 2 * batch; the metadata call writes and returns batch + 1 of split_meta
    bufs = (work_info, split_meta)

    def run():
        info = mla_gluon_split_metadata(seq_lens, nhead, kv_dtype, out=bufs)
        mla_gluon(
            q_nope=q_nope,
            q_pe=q_pe,
            kv_c=kv_c,
            o=o,
            page_table=block_table,
            seq_info=seq_lens,
            sm_scale=1.0 / (kv_c.shape[-1] ** 0.5),
            kv_pe_offset=512,
            use_2d_view=True,
            kv_scale=kv_scale,
            kv_split_info=info,
        )
        return info

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        info = run()
    torch.cuda.current_stream().wait_stream(stream)
    assert info[0] is work_info and info[1].numel() == batch + 1
    assert info[1].data_ptr() == split_meta.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for lens in (lens_a, lens_b):
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        _check_split_partition(info, lens, 128)
        o_base, _ = _mla_gluon_paged(
            mla_gluon,
            q_nope[:, None],
            q_pe[:, None],
            kv_c,
            block_table,
            seq_lens,
            True,
            kv_scale,
            None,
        )
        o_ref, _ = _mla_gluon_ref(
            q_nope[:, None],
            q_pe[:, None],
            kv_c,
            lens,
            block_table,
            1.0 / (kv_c.shape[-1] ** 0.5),
            kv_scale,
        )
        _assert_mla_close(o, o_ref[:, 0], o_base[:, 0])


@pytest.mark.parametrize(
    "batch_size,ctx_lens,skip_reduce,wide_view,extra_tokens,expected_pipeline",
    [
        (128, 2048, False, None, 0, False),
        (129, 2048, False, None, 0, True),
        (4, 0, False, None, 0, False),
        (4, 192, False, None, 0, True),
        (4, 193, False, None, 0, False),
        (4, 192, True, None, 0, False),
        (512, 2048, False, "query", 0, False),
        (512, 2048, False, "output", 0, False),
        (512, 2048, False, "block_tables", 0, False),
        (512, 2048, False, None, 1, False),
    ],
)
def test_mla_decode_fwd_lds_pipeline_eligibility(
    monkeypatch,
    batch_size: int,
    ctx_lens: int,
    skip_reduce: bool,
    wide_view: str | None,
    extra_tokens: int,
    expected_pipeline: bool,
):
    import aiter.ops.triton.attention.mla as mla_module

    monkeypatch.setattr(mla_module, "DEVICE_ARCH", "gfx1250")
    monkeypatch.setattr(mla_module, "IS_DEVICE_ARCH_GFX12", True)
    monkeypatch.setattr(mla_module, "get_num_sms", lambda: 256)
    launches = []

    class CaptureLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)

            return launch

    class SkipLaunch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(mla_module, "gluon_mla_decode_fwd_kernel", CaptureLaunch())
    monkeypatch.setattr(mla_module, "triton_mla_decode_fwd_reduce_kernel", SkipLaunch())
    # Metadata-only dispatch checks: no large allocation and no GPU launch.
    query = torch.empty(
        (batch_size + extra_tokens, 128, 576), dtype=e4m3_dtype, device="meta"
    )
    kv_buffer = torch.empty((64, 1, 64, 576), dtype=e4m3_dtype, device="meta")
    output = torch.empty(
        (batch_size + extra_tokens, 128, 512), dtype=torch.float32, device="meta"
    )
    block_tables = torch.empty((batch_size, 32), dtype=torch.int32, device="meta")
    if wide_view:
        tensors = {"query": query, "output": output, "block_tables": block_tables}
        view = tensors[wide_view]
        strides = list(view.stride())
        strides[0] = 2**30  # fits i32 itself, but the sequence offset does not
        tensors[wide_view] = torch.empty_strided(
            view.shape, strides, dtype=view.dtype, device="meta"
        )
        query, output, block_tables = (
            tensors["query"],
            tensors["output"],
            tensors["block_tables"],
        )
    cu_seqlens_q = torch.empty((batch_size + 1,), dtype=torch.int32, device="meta")
    seq_lens_kv = torch.empty((batch_size,), dtype=torch.int32, device="meta")
    result = mla_module.mla_decode_fwd(
        q=query,
        kv_buffer=kv_buffer,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_lens_kv,
        max_seqlen_kv=ctx_lens,
        block_tables=block_tables,
        softmax_scale=576**-0.5,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        causal=True,
        q_descale=None,
        kv_descale=None,
        shuffled_kv_cache=True,
        skip_reduce=skip_reduce,
    )
    assert len(launches) == 1
    assert launches[0].get("USE_LDS_PIPELINE", False) == expected_pipeline
    if expected_pipeline:
        assert launches[0]["num_stages"] == 4
    else:
        assert launches[0]["num_stages"] == 2
    if skip_reduce and launches[0]["NUM_SEGMENTS_PER_SEQ"] > 1:
        assert isinstance(result, tuple) and len(result) == 3
    else:
        assert result is output


@pytest.mark.parametrize("query_stride", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, e4m3_dtype])
def test_mla_decode_fwd_unshuffled_query_stride_eligibility(
    monkeypatch, query_stride, dtype
):
    import aiter.ops.triton.attention.mla as mla_module

    monkeypatch.setattr(mla_module, "DEVICE_ARCH", "gfx1250")
    monkeypatch.setattr(mla_module, "IS_DEVICE_ARCH_GFX12", True)
    monkeypatch.setattr(mla_module, "get_num_sms", lambda: 96)
    launches = []

    class CaptureLaunch:
        def __init__(self, pipelined):
            self.pipelined = pipelined

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(self.pipelined)

            return launch

    monkeypatch.setattr(mla_module, "gluon_mla_decode_fwd_kernel", CaptureLaunch(True))
    monkeypatch.setattr(
        mla_module, "gluon_mla_decode_fwd_kernel_non_pipelined", CaptureLaunch(False)
    )
    # Metadata-only dispatch check. No GPU allocation or kernel launch.
    batch_size, heads, head_dim = 128, 64, 576
    query = torch.empty(
        (batch_size, heads, head_dim * query_stride), dtype=dtype, device="meta"
    )[..., ::query_stride]
    output = torch.empty((batch_size, heads, 512), dtype=torch.bfloat16, device="meta")
    result = mla_module.mla_decode_fwd(
        q=query,
        kv_buffer=torch.empty((1, 64, 1, head_dim), dtype=dtype, device="meta"),
        out=output,
        cu_seqlens_q=torch.empty((batch_size + 1,), dtype=torch.int32, device="meta"),
        seqused_k=torch.empty((batch_size,), dtype=torch.int32, device="meta"),
        max_seqlen_kv=1,
        block_tables=torch.empty((batch_size, 1), dtype=torch.int32, device="meta"),
        softmax_scale=head_dim**-0.5,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        causal=True,
        q_descale=None,
        kv_descale=None,
        shuffled_kv_cache=False,
    )
    assert query.stride(2) == query_stride
    assert launches == [query_stride == 1]
    assert result is output
