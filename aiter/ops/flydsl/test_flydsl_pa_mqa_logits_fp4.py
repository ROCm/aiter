# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test for the Q FP4 / KV FP4 MQA logits kernel (gfx950)."""

import random

import pytest
import torch

pytest.importorskip("flydsl")
from aiter.ops.flydsl import is_flydsl_available  # noqa: E402

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return arch.lower().split(":")[0].startswith("gfx950")


pytestmark = pytest.mark.skipif(
    not _is_gfx950(),
    reason="pa_mqa_logits_fp4 (qfp4/kvfp4) is gfx950 only",
)

from aiter.ops.flydsl.kernels.pa_mqa_logits_fp4 import (  # noqa: E402
    DEFAULT_HEAD_DIM,
    DEFAULT_HEADS,
)
from aiter.ops.flydsl.pa_mqa_logits_kernels import (  # noqa: E402
    flydsl_pa_mqa_logits_fp4,
)
from aiter.test_common import checkAllclose, run_perftest  # noqa: E402

print(
    "[test] using pa_mqa_logits_fp4_qfp4_kvfp4 kernel (Q FP4, KV FP4, MFMA(Q_fp4, KV_fp4))"
)

dev = "cuda"
SEED = 42

SCALE_BLOCK = 32  # fp4 elements per scale block


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from aiter.utility import dtypes  # noqa: E402
from aiter.utility.fp4_utils import (  # noqa: E402
    dynamic_mxfp4_quant,
    e8m0_to_f32,
    mxfp4_to_f32,
)


def fp4_quant_e2m1_with_e8m0(x: torch.Tensor, block_size: int = 32):
    """Quantize ND bf16/fp32 → (fp4-packed uint8 [..., D/2], e8m0 uint8 [..., D/32])."""
    assert (
        block_size == SCALE_BLOCK
    ), f"MXFP4 spec fixes block_size=32, got {block_size}"
    *prefix, d = x.shape
    fp4_x2, scales_e8m0 = dynamic_mxfp4_quant(
        x.reshape(-1, d).to(torch.bfloat16), scaling_mode="even", shuffle=False
    )
    fp4_u8 = fp4_x2.view(torch.uint8).reshape(*prefix, d // 2).contiguous()
    scales_u8 = (
        scales_e8m0.view(torch.uint8).reshape(*prefix, d // block_size).contiguous()
    )
    return fp4_u8, scales_u8


def fp4_dequant_e2m1_with_e8m0(packed, e8m0_scales, block_size=32):
    """Dequantize (fp4-packed uint8 [..., D/2], e8m0 uint8 [..., D/32]) → fp32 [..., D]."""
    *prefix, d_half = packed.shape
    d = d_half * 2
    fp4_x2 = packed.view(dtypes.fp4x2) if packed.dtype == torch.uint8 else packed
    x_vals = mxfp4_to_f32(fp4_x2)
    scales_typed = (
        e8m0_scales.view(dtypes.fp8_e8m0)
        if e8m0_scales.dtype == torch.uint8
        else e8m0_scales
    )
    scale_f32 = e8m0_to_f32(scales_typed)
    x_blk = x_vals.reshape(*prefix, d // block_size, block_size)
    return (x_blk * scale_f32.unsqueeze(-1)).reshape(*prefix, d)


def create_paged_preshuffle_kv_fp4(kv_bf16, kv_block_size, num_blocks, block_tables):
    batch, t_max, d = kv_bf16.shape
    assert d % 128 == 0, f"head_dim must be multiple of 128, got {d}"
    assert t_max % kv_block_size == 0
    t_blocks = t_max // kv_block_size
    k_tiles = d // 128
    d_packed = d // 2
    d_scales = d // 32

    kv_flat = kv_bf16.reshape(-1, d)
    kv_fp4, kv_e8m0 = fp4_quant_e2m1_with_e8m0(kv_flat, block_size=SCALE_BLOCK)
    kv_fp4 = kv_fp4.reshape(batch, t_max, d_packed)
    kv_e8m0 = kv_e8m0.reshape(batch, t_max, d_scales)

    kv_chunks_perm = (
        kv_fp4.view(batch, t_blocks, kv_block_size, k_tiles, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(batch * t_blocks, k_tiles, 4, kv_block_size, 16)
    )
    kv_e8m0_perm = (
        kv_e8m0.view(batch, t_blocks, kv_block_size, k_tiles, 4)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(batch * t_blocks, k_tiles, 4, kv_block_size)
    )

    phys_flat = block_tables.reshape(-1).long()
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    kv_cache[phys_flat] = kv_chunks_perm
    kv_scale[phys_flat] = kv_e8m0_perm

    return kv_cache, kv_scale, kv_fp4, kv_e8m0


def ref_mqa_logits_mixed(
    q_packed, q_scale, kv_fp4, kv_scale, weights, context_lens, next_n=1
):
    batch = q_packed.shape[0]
    t_max = kv_fp4.shape[1]

    heads = q_packed.shape[2]
    head_dim_packed = q_packed.shape[3]
    head_dim_scales = q_scale.shape[3]
    head_dim_local = head_dim_packed * 2
    q_dq = fp4_dequant_e2m1_with_e8m0(
        q_packed.reshape(batch * next_n, heads, head_dim_packed),
        q_scale.reshape(batch * next_n, heads, head_dim_scales),
    ).reshape(batch, next_n, heads, head_dim_local)
    kv_dq = fp4_dequant_e2m1_with_e8m0(kv_fp4, kv_scale)

    ref_logits = torch.full(
        (batch * next_n, t_max), float("-inf"), device=dev, dtype=torch.float32
    )

    for b in range(batch):
        ctx = context_lens[b].item()
        if ctx == 0:
            continue
        kvi = kv_dq[b, :ctx]  # [ctx, D]
        for n in range(next_n):
            qi = q_dq[b, n]  # [H, D]
            wi = weights[b * next_n + n]  # [H]
            qk = qi @ kvi.T  # [H, ctx]
            qk = torch.relu(qk) * wi[:, None]
            logits_i = qk.sum(dim=0)  # [ctx]
            valid_max = ctx - next_n + n
            if valid_max + 1 < ctx:
                logits_i[valid_max + 1 :] = float("-inf")
            ref_logits[b * next_n + n, :ctx] = logits_i

    return ref_logits


def _torch_ref_step(q_dq_bn, kv_dq, w_bn, next_n=1):
    if next_n != 1:
        b_kv, t_kv, d_kv = kv_dq.shape
        kv_dq = (
            kv_dq.unsqueeze(1)
            .expand(-1, next_n, -1, -1)
            .reshape(b_kv * next_n, t_kv, d_kv)
        )
    qk = torch.bmm(q_dq_bn, kv_dq.transpose(1, 2))
    qk = torch.relu(qk) * w_bn[:, :, None]
    return qk.sum(dim=1)


def _make_varctx(batch, max_ctx, kv_block_size):
    base = [max_ctx * (i + 1) // batch for i in range(batch)]
    return [
        min(((c + kv_block_size - 1) // kv_block_size) * kv_block_size, max_ctx)
        for c in base
    ]


@pytest.mark.parametrize(
    "batch, max_ctx, kv_block_size, block_k, next_n, heads",
    [
        pytest.param(4, 16384, 16, 64, 1, 64, id="4x16k_n1_h64"),
        pytest.param(4, 32768, 16, 64, 1, 64, id="4x32k_n1_h64"),
        pytest.param(8, 65536, 16, 64, 1, 64, id="8x65k_n1_h64"),
        pytest.param(4, 16384, 16, 64, 2, 64, id="4x16k_n2_h64"),
        pytest.param(8, 65536, 16, 64, 2, 64, id="8x65k_n2_h64"),
        pytest.param(4, 16384, 16, 64, 1, 128, id="4x16k_n1_h128"),
    ],
)
def test_pa_mqa_logits_fp4_qfp4_kvfp4(
    batch,
    max_ctx,
    kv_block_size,
    block_k,
    next_n,
    heads,
    num_iters=20,
    num_warmup=3,
    num_warps=4,
    parallel_unit_num=512,
    head_dim=DEFAULT_HEAD_DIM,
):
    """End-to-end varctx test for the Q FP4 / KV FP4 kernel."""
    setup_seed(SEED)
    batch_size = batch
    assert (
        heads % 16 == 0 and heads <= 128
    ), f"heads={heads}: kernel requires multiple of 16, <= 128"
    assert head_dim % 128 == 0, f"head_dim={head_dim}: kernel requires multiple of 128"
    m_tiles = heads // 16
    k_tiles = head_dim // 128
    head_dim_packed = head_dim // 2
    head_dim_scales = head_dim // 32

    ctx_list = _make_varctx(batch_size, max_ctx, kv_block_size)
    context_lens = torch.tensor(ctx_list, dtype=torch.int32, device=dev)
    total_tokens = int(context_lens.sum().item())

    print("=" * 96)
    print(
        f"MQA Logits (Q FP4, KV FP4) varctx: batch={batch_size}, heads={heads}, "
        f"head_dim={head_dim}, max_ctx={max_ctx}, kv_block={kv_block_size}, "
        f"block_k={block_k}, next_n={next_n}"
    )
    print(
        f"  ctx_lens = {ctx_list}  (sum={total_tokens}, "
        f"avg={total_tokens // batch_size}, util={total_tokens/(batch_size*max_ctx):.1%})"
    )
    print("=" * 96)

    max_blocks_per_seq = (max_ctx + kv_block_size - 1) // kv_block_size
    num_blocks = max_blocks_per_seq * batch_size
    t_max = max_blocks_per_seq * kv_block_size

    q_bf16 = torch.randn(
        batch_size, next_n, heads, head_dim, dtype=torch.bfloat16, device=dev
    )
    kv_bf16 = torch.randn(batch_size, t_max, head_dim, dtype=torch.bfloat16, device=dev)
    weights = (
        torch.randn(batch_size * next_n, heads, dtype=torch.float32, device=dev) * 0.1
    )

    q_packed, q_e8m0 = fp4_quant_e2m1_with_e8m0(
        q_bf16.reshape(batch_size * next_n * heads, head_dim), block_size=SCALE_BLOCK
    )
    q_packed = q_packed.reshape(batch_size, next_n, heads, head_dim_packed)
    q_e8m0 = q_e8m0.reshape(batch_size, next_n, heads, head_dim_scales)

    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        batch_size, max_blocks_per_seq
    )
    kv_cache, kv_scale, kv_fp4_dense, kv_e8m0_dense = create_paged_preshuffle_kv_fp4(
        kv_bf16, kv_block_size, num_blocks, block_tables
    )

    ref_logits = ref_mqa_logits_mixed(
        q_packed,
        q_e8m0,
        kv_fp4_dense,
        kv_e8m0_dense,
        weights,
        context_lens,
        next_n=next_n,
    )

    out_logits = torch.full(
        (batch_size * next_n, t_max), float("-inf"), dtype=torch.float32, device=dev
    )

    qs_pad = ((m_tiles + 3) // 4) * 4
    qe_real = (
        q_e8m0.view(torch.uint8)
        .reshape(batch_size, next_n, m_tiles, 16, k_tiles, 4)
        .permute(0, 1, 4, 5, 3, 2)
        .contiguous()
    )
    qe = torch.nn.functional.pad(qe_real, (0, qs_pad - m_tiles)).contiguous()

    def launch_flydsl():
        flydsl_pa_mqa_logits_fp4(
            q_packed,
            qe,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            context_lens,
            out_logits,
            block_k=block_k,
            num_warps=num_warps,
            parallel_unit_num=parallel_unit_num,
        )

    out_logits.fill_(float("-inf"))
    launch_flydsl()
    torch.cuda.synchronize()

    mask = ~torch.isneginf(ref_logits)
    valid_out = out_logits[mask].double()
    valid_ref = ref_logits[mask].double()
    cos = (valid_out * valid_ref).sum() / (valid_out.norm() * valid_ref.norm() + 1e-12)
    max_abs_err = (valid_out - valid_ref).abs().max().item()
    mean_abs_err = (valid_out - valid_ref).abs().mean().item()
    err_ratio = checkAllclose(
        valid_ref.float(),
        valid_out.float(),
        rtol=0.05,
        atol=0.05,
        msg="flydsl-qfp4-kvfp4 vs ref",
        printLog=False,
    )
    out_past_ctx = out_logits.masked_select(~mask)
    neg_inf_ok = (
        bool(torch.isneginf(out_past_ctx).all().item())
        if out_past_ctx.numel()
        else True
    )
    print(
        f"  correctness: cosine_sim={cos.item():.6f}  "
        f"max_abs_err={max_abs_err:.6f}  mean_abs_err={mean_abs_err:.6f}  "
        f"err_ratio={err_ratio:.4f}  past_ctx_neginf={neg_inf_ok}"
    )
    assert (
        cos.item() > 0.99
    ), f"FlyDSL qfp4/kvfp4 vs ref cosine_sim={cos.item():.4f} < 0.99"
    assert neg_inf_ok, "OOB tokens were not NEG_INF — early-exit / pre-init broken"

    _, us_fly = run_perftest(launch_flydsl, num_iters=num_iters, num_warmup=num_warmup)
    torch.cuda.synchronize()

    q_dq_bf16 = (
        fp4_dequant_e2m1_with_e8m0(
            q_packed.reshape(-1, head_dim_packed),
            q_e8m0.reshape(-1, head_dim_scales),
        )
        .reshape(batch_size * next_n, heads, head_dim)
        .to(torch.bfloat16)
    )
    kv_dq_bf16 = fp4_dequant_e2m1_with_e8m0(kv_fp4_dense, kv_e8m0_dense).to(
        torch.bfloat16
    )
    w_bf16 = weights.to(torch.bfloat16)

    _, us_bf16 = run_perftest(
        _torch_ref_step,
        q_dq_bf16,
        kv_dq_bf16,
        w_bf16,
        next_n,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )

    flops = total_tokens * next_n * heads * (2 * head_dim + 3)
    bytes_q = batch_size * next_n * heads * (head_dim_packed + head_dim_scales)
    bytes_kv = total_tokens * (head_dim_packed + head_dim_scales)
    bytes_w = batch_size * next_n * heads * 4
    bytes_bt = batch_size * max_blocks_per_seq * 4
    bytes_out = total_tokens * next_n * 4
    bytes_total = bytes_q + bytes_kv + bytes_w + bytes_bt + bytes_out

    def metrics(us):
        if us <= 0:
            return 0.0, 0.0
        sec = us * 1e-6
        return flops / sec / 1e12, bytes_total / sec / 1e9

    tflops_fly, gbps_fly = metrics(us_fly)
    tflops_bf16, _ = metrics(us_bf16)

    print(
        f"\n  {'':>16} | {'us':>10} | {'TFLOPS':>8} | {'GB/s':>8} | {'vs flydsl':>10}"
    )
    print(
        f"  {'flydsl-qfp4/kvfp4':>16} | {us_fly:>10.2f} | {tflops_fly:>8.2f} | {gbps_fly:>8.1f} |"
    )
    print(
        f"  {'torch-bf16':>16} | {us_bf16:>10.2f} | {tflops_bf16:>8.2f} | {'-':>8} | "
        f"{us_bf16/us_fly:>9.2f}x"
    )
    print()

    _PERF_SUMMARY.append(
        (
            batch_size,
            heads,
            head_dim,
            max_ctx,
            next_n,
            kv_block_size,
            block_k,
            cos.item(),
            us_fly,
            tflops_fly,
            gbps_fly,
        )
    )


_PERF_SUMMARY = []


@pytest.fixture(scope="session", autouse=True)
def _perf_summary_at_end():
    yield
    if _PERF_SUMMARY:
        _print_perf_summary()


def _print_perf_summary():
    print("\n" + "=" * 96)
    print("Perf summary (flydsl-qfp4/kvfp4 across shapes)")
    print("=" * 96)
    print(
        f"  {'batch':>5} | {'heads':>5} | {'h_dim':>5} | {'ctx_len':>7} | {'next_n':>6} | "
        f"{'kv_blk':>6} | {'block_k':>7} | {'cos_sim':>8} | {'us':>9} | {'TFLOPS':>7} | {'GB/s':>7}"
    )
    print("  " + "-" * 103)
    for b, h, hd, ctx, nn, kvb, blk, cos_v, us, tflops, gbps in _PERF_SUMMARY:
        print(
            f"  {b:>5} | {h:>5} | {hd:>5} | {ctx:>7} | {nn:>6} | {kvb:>6} | {blk:>7} | "
            f"{cos_v:>8.4f} | {us:>9.2f} | {tflops:>7.2f} | {gbps:>7.1f}"
        )
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MQA Logits (Q FP4, KV FP4) Test + Benchmark (gfx950)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch", type=int, default=0, help="Batch size (0 = run default sweep)"
    )
    parser.add_argument(
        "--ctx", type=int, default=0, help="Context length (0 = run default sweep)"
    )
    parser.add_argument("--kv_block_size", type=int, default=64)
    parser.add_argument(
        "--block_k",
        type=int,
        default=256,
        help="Tokens per chunk (multiple of MFMA_N=16, divisible by num_warps)",
    )
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument(
        "--num_warps",
        type=int,
        default=4,
        help="warps per CTA (pipelined kernel only); BLOCK=num_warps*64",
    )
    parser.add_argument(
        "--parallel_unit_num",
        type=int,
        default=512,
        help="target CTA count for host schedule (default 512)",
    )
    parser.add_argument(
        "--next_n",
        type=int,
        default=1,
        help="MTP queries per batch (1 = standard MQA, 2 = MTP-1)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=DEFAULT_HEADS,
        help=f"Number of Q heads (multiple of 16, <= 128). Default {DEFAULT_HEADS}.",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=DEFAULT_HEAD_DIM,
        help=f"Per-head dim (multiple of 128). Default {DEFAULT_HEAD_DIM}.",
    )
    args = parser.parse_args()

    if args.batch > 0 and args.ctx > 0 and args.next_n > 0:
        configs = [(args.batch, args.ctx, args.next_n)]
    else:
        configs = [
            (1, 2 * 65536, 1),
            (2, 2 * 65536, 1),
            (4, 2 * 65536, 1),
            (8, 2 * 65536, 1),
            (1, 2 * 16384, 2),
            (1, 2 * 32768, 2),
            (1, 2 * 65536, 2),
            (2, 2 * 16384, 2),
            (2, 2 * 32768, 2),
            (2, 2 * 65536, 2),
            (4, 2 * 16384, 2),
            (4, 2 * 32768, 2),
            (4, 2 * 65536, 2),
        ]

    for b, c, nn in configs:
        try:
            test_pa_mqa_logits_fp4_qfp4_kvfp4(
                batch=b,
                max_ctx=c,
                next_n=nn,
                kv_block_size=args.kv_block_size,
                block_k=args.block_k,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
                num_warps=args.num_warps,
                parallel_unit_num=args.parallel_unit_num,
                heads=args.heads,
                head_dim=args.head_dim,
            )
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
        except Exception:
            import traceback

            traceback.print_exc()

    if _PERF_SUMMARY:
        _print_perf_summary()
