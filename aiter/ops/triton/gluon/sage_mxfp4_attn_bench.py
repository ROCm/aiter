"""
Benchmark wrapper for the Gluon Sage MXFP4 kernel.

Usage:
    python aiter/ops/triton/gluon/sage_mxfp4_attn_bench.py \
        --b 1 --hq 5 --sq 5600 --d 128 [--causal] [--qsmooth] [--compare-to-ref] \
        [--block-m 128] [--block-n 128] [--warps 4] [--waves 2] [--stages 2]
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
import triton

import aiter
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import map_dims
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    get_sage_fwd_configs_mxfp4,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    create_hadamard_matrix,
    sage_quant_mxfp4,
)
from aiter.ops.triton.gluon.sage_mxfp4_attn import gluon_sage_mxfp4_fwd


# ---------------------------------------------------------------------------
# Wrapper function
# ---------------------------------------------------------------------------

def gluon_sage_mxfp4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    bias: torch.Tensor = None,
    causal: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
):
    """Launch the Gluon sage mxfp4 kernel with pre-quantized inputs."""
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_size_qk = map_dims(q.shape, bshd_map)

    head_size_qk *= 2  # MXFP4 packed
    _, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd_map)
    _, _, _, head_size_v = map_dims(v.shape, bshd_map)

    assert q.dtype == torch.uint8 and k.dtype == torch.uint8
    assert nheads_q % nheads_k == 0
    assert layout in ["bhsd", "bshd"]

    if config is None:
        config = get_sage_fwd_configs_mxfp4()

    out = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
        dtype=torch.bfloat16,
        device=q.device,
    )

    stride_qb, stride_qm, stride_qh, _ = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, _ = map_dims(k.stride(), bshd_map)
    stride_vb, stride_vn, stride_vh, _ = map_dims(v.stride(), bshd_map)
    stride_ob, stride_om, stride_oh, _ = map_dims(out.stride(), bshd_map)

    if bias is not None:
        USE_BIAS = True
        stride_bz, stride_bh, stride_bm, stride_bn = bias.stride()
    else:
        USE_BIAS = False
        stride_bz = stride_bh = stride_bm = stride_bn = 0

    stride_qsz, stride_qsm, stride_qsh, stride_qsk = map_dims(q_descale.stride(), bshd_map)
    stride_ksz, stride_ksn, stride_ksh, stride_ksk = map_dims(k_descale.stride(), bshd_map)
    stride_vsz, stride_vsh, _ = v_descale.stride()

    padded_d_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_v = max(16, 1 << (head_size_v - 1).bit_length())

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)

    gluon_sage_mxfp4_fwd[grid](
        Q=q, K=k, V=v,
        bias=bias if bias is not None else torch.empty(0, device=q.device),
        Q_Descale=q_descale,
        K_Descale=k_descale,
        V_Descale=v_descale,
        Out=out,
        stride_qz=stride_qb, stride_qh=stride_qh, stride_qm=stride_qm,
        stride_kz=stride_kb, stride_kh=stride_kh,
        stride_kn=stride_kn, stride_kk=1,
        stride_vz=stride_vb, stride_vh=stride_vh,
        stride_vk=stride_vn, stride_vn=1,
        stride_oz=stride_ob, stride_oh=stride_oh, stride_om=stride_om,
        stride_qsz=stride_qsz, stride_qsh=stride_qsh,
        stride_qsm=stride_qsm, stride_qsk=stride_qsk,
        stride_ksz=stride_ksz, stride_ksh=stride_ksh,
        stride_ksn=stride_ksn, stride_ksk=stride_ksk,
        stride_vsz=stride_vsz, stride_vsh=stride_vsh,
        stride_bz=stride_bz, stride_bh=stride_bh,
        stride_bm=stride_bm, stride_bn=stride_bn,
        HQ=nheads_q, HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        IS_CAUSAL=causal,
        BLOCK_DMODEL_QK=padded_d_qk,
        BLOCK_DMODEL_V=padded_d_v,
        USE_BIAS=USE_BIAS,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        PRE_LOAD_V=config.get("PRE_LOAD_V", False),
        NUM_STAGES=config.get("NUM_STAGES", 2),
        num_warps=config.get("num_warps", 8),
        num_stages=config.get("num_stages", 2),
        waves_per_eu=config.get("waves_per_eu", 2),
    )

    return out


def gluon_sage_mxfp4_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    layout: str = "bshd",
    q_smooth: bool = False,
    config: Optional[dict] = None,
    R: torch.Tensor = None,
    BLOCK_R: int = 128,
):
    """High-precision entry point: quantize + run Gluon kernel."""
    if config is None:
        config = get_sage_fwd_configs_mxfp4()

    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(FP8_TYPE).max

    q_quantized, q_descale, k_quantized, k_descale, v_quantized, v_descale, delta_s = (
        sage_quant_mxfp4(
            q, k, v, FP8_TYPE, FP8_MAX,
            BLKQ=config["BLOCK_M"], BLKK=64,
            layout=layout, R=R, BLOCK_R=BLOCK_R,
            q_smoothing=q_smooth,
        )
    )

    return gluon_sage_mxfp4_func(
        q=q_quantized, k=k_quantized, v=v_quantized,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        bias=delta_s, causal=causal, layout=layout, config=config,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def attention_ref(q, k, v, causal=False, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    q_ = q.permute(0, 2, 1, 3).float()
    k_ = k.permute(0, 2, 1, 3).float()
    v_ = v.permute(0, 2, 1, 3).float()
    attn = torch.matmul(q_, k_.transpose(-2, -1)) * sm_scale
    if causal:
        sq, sk = attn.shape[-2], attn.shape[-1]
        mask = torch.triu(torch.ones(sq, sk, device=attn.device), diagonal=sk - sq + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_)
    return out.permute(0, 2, 1, 3)


def benchmark(args):
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    b, hq, sq, d = args.b, args.hq, args.sq, args.d
    hk = args.hk if args.hk else hq
    sk = args.sk if args.sk else sq
    layout = "bshd"

    q = torch.randn(b, sq, hq, d, device=device, dtype=dtype)
    k = torch.randn(b, sk, hk, d, device=device, dtype=dtype)
    v = torch.randn(b, sk, hk, d, device=device, dtype=dtype)

    R = create_hadamard_matrix(d, device=device)

    triton_config = get_sage_fwd_configs_mxfp4()

    config = dict(triton_config)
    if args.block_m is not None:
        config["BLOCK_M"] = args.block_m
    if args.block_n is not None:
        config["BLOCK_N"] = args.block_n
    if args.warps is not None:
        config["num_warps"] = args.warps
    if args.waves is not None:
        config["waves_per_eu"] = args.waves
    if args.stages is not None:
        config["NUM_STAGES"] = args.stages
    else:
        config["NUM_STAGES"] = 2

    print(f"Shape: b={b}, hq={hq}, hk={hk}, sq={sq}, sk={sk}, d={d}")
    print(f"Config: {config}")
    print(f"Causal: {args.causal}, QSmooth: {args.qsmooth}")
    print()

    out_gluon = gluon_sage_mxfp4_wrapper(
        q, k, v, causal=args.causal, layout=layout,
        q_smooth=args.qsmooth, config=config, R=R,
    )

    if args.compare_to_ref:
        from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
            fav3_sage_mxfp4_wrapper,
        )
        out_triton = fav3_sage_mxfp4_wrapper(
            q, k, v, causal=args.causal, layout=layout,
            q_smooth=args.qsmooth, hadamard_rotation=True, config=triton_config, R=R,
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            out_gluon.reshape(-1).float(),
            out_triton.reshape(-1).float(),
            dim=0,
        )
        max_diff = (out_gluon.float() - out_triton.float()).abs().max().item()
        print(f"vs Triton sage_mxfp4: cosine={cos_sim.item():.6f}  max_diff={max_diff:.6f}")
        print()

    # Benchmark
    warmup_iters = 5
    bench_iters = 20

    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(FP8_TYPE).max
    q_quant, q_ds, k_quant, k_ds, v_quant, v_ds, delta_s = sage_quant_mxfp4(
        q, k, v, FP8_TYPE, FP8_MAX,
        BLKQ=config["BLOCK_M"], BLKK=64,
        layout=layout, R=R, BLOCK_R=128,
        q_smoothing=args.qsmooth,
    )

    for _ in range(warmup_iters):
        gluon_sage_mxfp4_func(
            q_quant, k_quant, v_quant, q_ds, k_ds, v_ds,
            bias=delta_s, causal=args.causal, layout=layout, config=config,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        gluon_sage_mxfp4_func(
            q_quant, k_quant, v_quant, q_ds, k_ds, v_ds,
            bias=delta_s, causal=args.causal, layout=layout, config=config,
        )
    torch.cuda.synchronize()
    gluon_ms = (time.perf_counter() - start) / bench_iters * 1000

    print(f"Gluon sage_mxfp4:  {gluon_ms:.4f} ms")

    # Compare to original Triton
    from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
        fav3_sage_mxfp4_func,
    )

    q_quant_t, q_ds_t, k_quant_t, k_ds_t, v_quant_t, v_ds_t, delta_s_t = sage_quant_mxfp4(
        q, k, v, FP8_TYPE, FP8_MAX,
        BLKQ=triton_config["BLOCK_M"], BLKK=64,
        layout=layout, R=R, BLOCK_R=128,
        q_smoothing=args.qsmooth,
    )

    for _ in range(warmup_iters):
        fav3_sage_mxfp4_func(
            q_quant_t, k_quant_t, v_quant_t, q_ds_t, k_ds_t, v_ds_t,
            bias=delta_s_t, causal=args.causal, layout=layout, config=triton_config,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        fav3_sage_mxfp4_func(
            q_quant_t, k_quant_t, v_quant_t, q_ds_t, k_ds_t, v_ds_t,
            bias=delta_s_t, causal=args.causal, layout=layout, config=triton_config,
        )
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) / bench_iters * 1000

    print(f"Triton sage_mxfp4: {triton_ms:.4f} ms")
    if triton_ms > 0:
        print(f"Speedup: {triton_ms / gluon_ms:.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gluon Sage MXFP4")
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--hq", type=int, default=5)
    parser.add_argument("--hk", type=int, default=None)
    parser.add_argument("--sq", type=int, default=5600)
    parser.add_argument("--sk", type=int, default=None)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--qsmooth", action="store_true")
    parser.add_argument("--compare-to-ref", action="store_true")
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--warps", type=int, default=None)
    parser.add_argument("--waves", type=int, default=None)
    parser.add_argument("--stages", type=int, default=None, help="NUM_STAGES for async pipelining (default 2)")
    args = parser.parse_args()
    benchmark(args)
