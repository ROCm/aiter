# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS A16W8 (no sink) against Gluon.

OPUS A16W8 is bf16 Q / fp8 KV, page 16 or 128, including MTP when
``qlen * gqa <= 16`` or GQA 16 with ``qlen`` 2..4 (token loop). Per-token KV
scales and transposed V use the same layouts as Gluon.

Both operators read the same KV tensors and are timed from a HIP graph replay.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch

from aiter import dtypes
from aiter.ops.pa_decode_opus import (
    pa_decode_opus_a16w8,
    pa_decode_opus_a16w8_ps,
    pa_decode_opus_ps_plan,
)
from aiter.ops.triton.gluon.pa_decode_gluon import (
    get_recommended_splits,
    pa_decode_gluon,
)
from op_tests.bench_pa_decode_opus_gptoss_vs_gluon import graph_wrap, timeit_pair

HEAD_DIM = 128
MAX_GQA_ROWS = 16
X = 16 // dtypes.fp8.itemsize
KV_SCALE = 0.03
FILL_CHUNK = 1 << 24

DEFAULT_BATCH_SIZES = [3, 81]
DEFAULT_SHAPES = [
    (8, 1, 128, 1027),
    (16, 1, 128, 8192),
]


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def fill_fp8(shape, device, generator):
    out = torch.empty(shape, dtype=dtypes.fp8, device=device)
    flat = out.view(-1)
    for start in range(0, flat.numel(), FILL_CHUNK):
        piece = flat[start : start + FILL_CHUNK]
        piece.copy_(
            torch.randn(
                piece.numel(), device=device, generator=generator, dtype=torch.float32
            ).div_(4)
        )
    return out


def make_token_scales(num_blocks, num_kv_heads, page_size, device, generator):
    scales = torch.rand(
        (num_blocks, num_kv_heads, page_size, 1),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    scales.mul_(KV_SCALE).add_(KV_SCALE * 0.5)
    return scales


def transpose_v(v_cache):
    num_blocks, num_kv_heads, head_dim, page_size = v_cache.shape
    return (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_kv_heads, page_size // X, X, head_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )


def build_inputs(
    batch,
    num_q_heads,
    num_kv_heads,
    ctx,
    page_size,
    query_length,
    trans_v=False,
    per_token=False,
    device="cuda",
    seed=0,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    max_blocks = (ctx + page_size - 1) // page_size
    num_blocks = batch * max_blocks
    q_shape = (
        (batch, num_q_heads, HEAD_DIM)
        if query_length == 1
        else (batch, query_length, num_q_heads, HEAD_DIM)
    )
    q = torch.randn(q_shape, device=device, generator=generator, dtype=torch.float32)
    q.div_(4)
    k_cache = fill_fp8(
        (num_blocks, num_kv_heads, HEAD_DIM // X, page_size, X), device, generator
    )
    v_cache = fill_fp8(
        (num_blocks, num_kv_heads, HEAD_DIM, page_size), device, generator
    )
    if trans_v:
        v_cache = transpose_v(v_cache)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(
        batch, max_blocks
    )
    context_lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    if per_token:
        k_scale = make_token_scales(
            num_blocks, num_kv_heads, page_size, device, generator
        )
        v_scale = make_token_scales(
            num_blocks, num_kv_heads, page_size, device, generator
        )
    else:
        k_scale = v_scale = None
    return (
        q.to(torch.bfloat16),
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        k_scale,
        v_scale,
    )


def _unpack_k(k_cache, pages, k_scale):
    k = k_cache[pages].float()
    if k_scale is None:
        k = k * KV_SCALE
    else:
        k = k * k_scale[pages][:, :, None, :, :]
    num_kv_heads = k_cache.shape[1]
    k = k.permute(1, 0, 3, 2, 4).reshape(num_kv_heads, -1, HEAD_DIM)
    return k


def _unpack_v(v_cache, pages, v_scale):
    v = v_cache[pages].float()
    num_kv_heads = v_cache.shape[1]
    if v.dim() == 5:
        _, _, page_over_x, _, x = v.shape
        page_size = page_over_x * x
        v = v.permute(0, 1, 2, 4, 3).reshape(
            pages.numel(), num_kv_heads, page_size, HEAD_DIM
        )
        if v_scale is None:
            v = v * KV_SCALE
        else:
            v = v * v_scale[pages]
        return v.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)
    if v_scale is None:
        v = v * KV_SCALE
    else:
        v = v * v_scale[pages].permute(0, 1, 3, 2)
    return v.permute(1, 0, 3, 2).reshape(num_kv_heads, -1, HEAD_DIM)


def reference(
    q, k_cache, v_cache, block_tables, context_lens, scale, rows, k_scale, v_scale
):
    q4 = q.unsqueeze(1) if q.dim() == 3 else q
    _, qlen, num_heads, _ = q4.shape
    num_kv_heads = k_cache.shape[1]
    page_size = (
        v_cache.shape[2] * v_cache.shape[4] if v_cache.dim() == 5 else v_cache.shape[3]
    )
    gqa = num_heads // num_kv_heads
    out = {}
    for s in rows:
        ctx = int(context_lens[s].item())
        pages = block_tables[s, : (ctx + page_size - 1) // page_size]
        k = _unpack_k(k_cache, pages, k_scale)[:, :ctx]
        v = _unpack_v(v_cache, pages, v_scale)[:, :ctx]
        k_t = k.transpose(1, 2)
        seq = []
        for t in range(qlen):
            valid = ctx - qlen + 1 + t
            qs = q4[s, t].float().reshape(num_kv_heads, gqa, HEAD_DIM)
            score = torch.bmm(qs, k_t) * scale
            if valid < ctx:
                score[..., valid:] = float("-inf")
            seq.append(
                torch.bmm(torch.softmax(score, dim=-1), v).reshape(num_heads, HEAD_DIM)
            )
        stacked = torch.stack(seq, dim=0)
        out[s] = stacked[0] if q.dim() == 3 else stacked
    return out


def rms_rel(got, ref):
    return float((got - ref).pow(2).mean().sqrt() / ref.abs().mean().clamp_min(1e-9))


def make_gluon_call(
    q, k_cache, v_cache, block_tables, context_lens, scale, k_scale, v_scale
):
    if q.dim() == 4:
        batch, query_length, num_heads, _ = q.shape
        q_gluon = q.reshape(batch * query_length, num_heads, HEAD_DIM)
    else:
        batch, num_heads, _ = q.shape
        query_length = 1
        q_gluon = q
    num_kv_heads = k_cache.shape[1]
    query_group_size = query_length * (num_heads // num_kv_heads)
    max_parts = get_recommended_splits(batch, num_kv_heads)
    shape = (batch, num_kv_heads, max_parts, query_group_size)
    exp_sums = torch.empty(shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty(shape, dtype=torch.float32, device=q.device)
    temporary_output = torch.empty(
        *shape, HEAD_DIM, dtype=torch.bfloat16, device=q.device
    )
    out = torch.empty(
        (batch * query_length, num_heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=q.device,
    )
    if k_scale is None:
        scale_k = torch.tensor([KV_SCALE], dtype=torch.float32, device=q.device)
        scale_v = scale_k
    else:
        scale_k, scale_v = k_scale, v_scale

    def call():
        pa_decode_gluon(
            out,
            q_gluon,
            k_cache,
            v_cache,
            context_lens,
            block_tables,
            scale,
            query_length,
            max_parts,
            256,
            dtypes.fp8,
            None,
            scale_k,
            scale_v,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=None,
            sliding_window=0,
            ps=True,
        )

    return call, out, max_parts


def skip_reason(
    num_q_heads, num_kv_heads, head_dim, query_length, page_size, trans_v, per_token
):
    if head_dim != HEAD_DIM:
        return f"OPUS A16W8 is compiled for D={HEAD_DIM}, got {head_dim}"
    if page_size not in (16, 128):
        return f"OPUS A16W8 page sizes are 16 and 128, got {page_size}"
    gqa = num_q_heads // num_kv_heads
    if query_length * gqa > MAX_GQA_ROWS and (gqa != MAX_GQA_ROWS or query_length > 4):
        return (
            f"OPUS MTP packs qlen*gqa into one 16-row tile, or GQA==16 "
            f"with qlen<=4 via a token loop "
            f"({query_length}*{gqa}={query_length * gqa} > {MAX_GQA_ROWS})"
        )
    return None


def run_case(
    batch,
    num_q_heads,
    num_kv_heads,
    ctx,
    page_size,
    query_length,
    persistent,
    verify_rows,
    rounds,
    iters,
    trans_v,
    per_token,
):
    q, k_cache, v_cache, block_tables, context_lens, k_scale, v_scale = build_inputs(
        batch,
        num_q_heads,
        num_kv_heads,
        ctx,
        page_size,
        query_length,
        trans_v=trans_v,
        per_token=per_token,
    )
    scale = float(HEAD_DIM**-0.5)
    gluon_call, gluon_out, gluon_parts = make_gluon_call(
        q, k_cache, v_cache, block_tables, context_lens, scale, k_scale, v_scale
    )
    opus_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    opus_k_scale = KV_SCALE if k_scale is None else k_scale
    opus_v_scale = KV_SCALE if v_scale is None else v_scale
    if persistent:
        plan = pa_decode_opus_ps_plan(
            block_tables,
            context_lens,
            num_q_heads,
            num_kv_heads,
            head_dim=HEAD_DIM,
            page_size=page_size,
        )

        def opus_call():
            pa_decode_opus_a16w8_ps(
                q,
                k_cache,
                v_cache,
                plan,
                scale,
                k_scale=opus_k_scale,
                v_scale=opus_v_scale,
                out=opus_out,
            )

    else:

        def opus_call():
            pa_decode_opus_a16w8(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                scale,
                k_scale=opus_k_scale,
                v_scale=opus_v_scale,
                out=opus_out,
            )

    opus_call()
    gluon_call()
    torch.cuda.synchronize()

    errors = {}
    rows = list(range(min(verify_rows, batch)))
    if rows:
        ref = reference(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            scale,
            rows,
            k_scale,
            v_scale,
        )
        opus_err = []
        gluon_err = []
        for s in rows:
            opus_view = opus_out[s].float()
            gluon_view = gluon_out.view_as(opus_out)[s].float()
            opus_err.append(rms_rel(opus_view, ref[s]))
            gluon_err.append(rms_rel(gluon_view, ref[s]))
        errors["opus_vs_ref"] = max(opus_err)
        errors["gluon_vs_ref"] = max(gluon_err)
    errors["opus_vs_gluon"] = rms_rel(
        opus_out.float(), gluon_out.view_as(opus_out).float()
    )

    t_opus, t_gluon = timeit_pair(
        graph_wrap(opus_call), graph_wrap(gluon_call), iters=iters, rounds=rounds
    )
    kv_bytes = (
        2
        * num_kv_heads
        * HEAD_DIM
        * dtypes.fp8.itemsize
        * int(context_lens.sum().item())
    )
    record = dict(
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qlen=query_length,
        ctx=ctx,
        page=page_size,
        trans_v=int(trans_v),
        per_token=int(per_token),
        opus_mode="ps" if persistent else "plain",
        gluon_splits=gluon_parts,
        opus_us=t_opus,
        gluon_us=t_gluon,
        speedup=t_gluon / t_opus,
        opus_tbs=kv_bytes / (t_opus * 1e-6) / 1e12,
        gluon_tbs=kv_bytes / (t_gluon * 1e-6) / 1e12,
        **errors,
    )
    print(
        f"b{batch:<4} hq{num_q_heads:<3} q{query_length} p{page_size:<3} "
        f"tv{int(trans_v)} pt{int(per_token)} ctx{ctx:<7} {record['opus_mode']:>5} "
        f"{t_opus:>9.2f}us {t_gluon:>9.2f}us {record['speedup']:>6.2f}x  "
        f"{record['opus_tbs']:.2f}/{record['gluon_tbs']:.2f} TB/s  "
        f"rms o/g/mutual="
        f"{errors.get('opus_vs_ref', float('nan')):.3f}/"
        f"{errors.get('gluon_vs_ref', float('nan')):.3f}/{errors['opus_vs_gluon']:.3f}",
        flush=True,
    )
    return record


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="Query/output dtype. OPUS A16W8 is bf16 only.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=_positive_int,
        nargs="*",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes. e.g.: -b 3 81 200",
    )
    parser.add_argument(
        "-q",
        "--query-length",
        type=_positive_int,
        nargs="+",
        default=[1],
        help="Query tokens per sequence: 1 decode, >1 MTP. e.g.: -q 1 2 4",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_SHAPES,
        help="(num_query_heads,num_kv_heads,head_dim,context_length). e.g.: -s 16,1,128,200000",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="*",
        choices=[16, 128],
        default=[16, 128],
        help="KV page sizes. OPUS A16W8 runs 16 and 128.",
    )
    parser.add_argument(
        "--trans-v",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0],
        help="V layout: 0 plain 4-D, 1 transposed 5-D [blocks, kvh, PAGE/x, D, x].",
    )
    parser.add_argument(
        "--per-token",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0],
        help="KV scale: 0 per-tensor, 1 per-token [blocks, kvh, PAGE, 1].",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["plain", "ps"], default=["plain"]
    )
    parser.add_argument("--verify-rows", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    print(
        f"gfx={torch.cuda.get_device_properties(0).gcnArchName} torch={torch.__version__}",
        flush=True,
    )
    records = []
    skipped = []
    for (
        dtype,
        batch,
        shape,
        page_size,
        trans_v,
        per_token,
        query_length,
        mode,
    ) in itertools.product(
        args.dtype,
        args.batch,
        args.shapes,
        args.block_size,
        args.trans_v,
        args.per_token,
        args.query_length,
        args.modes,
    ):
        num_q_heads, num_kv_heads, head_dim, ctx = shape
        why = skip_reason(
            num_q_heads,
            num_kv_heads,
            head_dim,
            query_length,
            page_size,
            trans_v,
            per_token,
        )
        if why is None and mode == "ps" and query_length > 1:
            why = "OPUS persistent A16W8 has no MTP"
        if why is None and mode == "ps" and (trans_v or per_token):
            why = "OPUS persistent A16W8 is per-tensor plain-V only"
        if dtype != dtypes.bf16:
            why = f"OPUS A16W8 Q is bf16, got {dtype}"
        if why:
            skipped.append(
                f"skip b{batch} hq{num_q_heads} q{query_length} p{page_size} "
                f"tv{trans_v} pt{per_token} ctx{ctx}: {why}"
            )
            continue
        records.append(
            run_case(
                batch,
                num_q_heads,
                num_kv_heads,
                ctx,
                page_size,
                query_length,
                mode == "ps",
                args.verify_rows,
                args.rounds,
                args.iters,
                bool(trans_v),
                bool(per_token),
            )
        )
        # run_case's tensors are unreachable once it returns; reclaim the
        # caching allocator blocks before building the next case.
        torch.cuda.empty_cache()
    for line in skipped:
        print(line, flush=True)
    if args.output:
        args.output.write_text(
            json.dumps({"records": records, "skipped": skipped}, indent=2)
        )


if __name__ == "__main__":
    main()
