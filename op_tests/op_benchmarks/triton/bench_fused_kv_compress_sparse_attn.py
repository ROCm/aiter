#!/usr/bin/env python3
"""Benchmark the DeepSeek-V4 fused KV compressor.

Drives the public launcher, aiter.ops.triton.quant.fused_mxfp8_quant
.fused_deepseek_v4_compress_norm_rope_store, which picks the kernel from head_dim and
dispatches it with the launcher's own num_warps -- so what is measured here is
what production runs, including any later retune of that launcher.

Every shape knob is an argument; the defaults are DeepSeek-V4-Pro at tp4 as
captured from a live vLLM run with VLLM_DSV4_AITER_SPARSE_MLA=1 (the aligned
640-byte record). config.json alternates compress_ratios 128/4 across the 61
layers, so --variant selects one of the two live layer groups and any
individual flag overrides it.

Timing uses triton.testing.do_bench_cudagraph, matching
bench_moe_gemm_a4w4_cudagraph.py; --rep is the measurement target in ms.

  python bench_fused_kv_compress_sparse_attn.py
  python bench_fused_kv_compress_sparse_attn.py --variant cr4
  python bench_fused_kv_compress_sparse_attn.py --tokens 64 256 1024
  python bench_fused_kv_compress_sparse_attn.py --head-dim 512 --token-stride 576

For register pressure / ISA rather than timing, compile the kernel under a
dedicated TRITON_CACHE_DIR and read the .amdgcn instead.
"""

# The timing closures below capture the shape loop's variables by
# reference. Every one is built and consumed inside its own iteration
# (do_bench runs before the loop advances), so late binding cannot bite.
# File-scoped rather than per-line: a trailing suppression comment sits
# on a line black then rewraps, which moves it off the line ruff reports.
# ruff: noqa: B023

import argparse
import sys
from types import SimpleNamespace

import torch
import triton

from aiter.ops.triton.quant.fused_mxfp8_quant import (
    fused_deepseek_v4_compress_norm_rope_store,
    fused_deepseek_v4_compress_norm_rope_store_two_stage,
)


def emitting_positions(a, device="cuda"):
    """Positions that actually produce a record, and stay in bounds doing it.

    The kernel returns immediately unless ``(position + 1) % compress_ratio
    == 0``, so an all-zero positions vector measures a grid of early exits and
    nothing else -- which is what this benchmark used to do.

    An emitting token then gathers ``position - (1 + overlap) * cr + 1 ..
    position`` and indexes ``block_table[req, pos // block_size]``, so the
    first usable position is ``(1 + overlap) * cr - 1`` and the last is bounded
    by the block table's width. block_table is zero here, so every gather
    lands in state_cache[0]; block_offsets stays under state_mid by
    construction.
    """
    cr = a.compress_ratio
    lo = (1 + bool(a.overlap)) * cr - 1
    hi = a.block_table_width * a.block_size
    valid = list(range(lo, hi, cr))
    assert valid, "no emitting position fits the block table"
    return torch.tensor(
        [valid[i % len(valid)] for i in range(a.tokens)],
        dtype=torch.int64,
        device=device,
    )


# DeepSeek-V4-Pro tp4, captured from a live run. --variant picks the layer
# group; everything here is overridable from the command line.
VARIANTS = {
    "cr128": {
        "state_width": 512,
        "compress_ratio": 128,
        "overlap": False,
        "block_size": 8,
        "state_mid": 8,
        "kv_page": 2,
        "block_table_width": 4096,
    },
    "cr4": {
        "state_width": 1024,
        "compress_ratio": 4,
        "overlap": True,
        "block_size": 4,
        "state_mid": 4,
        "kv_page": 64,
        "block_table_width": 8192,
    },
}


def build(a, device="cuda"):
    """Allocate the launcher's arguments at the requested shapes.

    state_cache and kv_cache are windows into larger allocations in the live
    run, so they are built with as_strided / slicing to carry the same
    strides; the kernel takes those strides as runtime arguments.
    """
    state_cache = torch.zeros(
        a.tokens * a.state_stride0, dtype=torch.float32, device=device
    ).as_strided(
        (a.tokens, a.state_mid, 2 * a.state_width),
        (a.state_stride0, 2 * a.state_width, 1),
    )
    state_cache[..., : a.state_width] = 1.0

    kv_backing = torch.zeros(
        a.tokens,
        a.rows_per_block,
        a.token_stride,
        dtype=torch.uint8,
        device=device,
    )
    kv_cache = kv_backing[:, : a.kv_page, :]

    cos_sin_cache = torch.zeros(
        a.cos_sin_rows, a.rope_dim, dtype=torch.float32, device=device
    )
    cos_sin_cache[:, : a.rope_dim // 2] = 1.0

    # Distinct records: slot s lands in block s // kv_page, row s % kv_page,
    # and s < tokens <= tokens * kv_page keeps every one inside kv_cache.
    # Pointing every token at slot 0, as this used to, makes the stores
    # collide on one line and understates the write cost.
    slot_mapping = torch.arange(a.tokens, dtype=torch.int64, device=device)
    return {
        "state_cache": state_cache,
        "num_actual": a.tokens,
        "token_to_req_indices": torch.zeros(a.tokens, dtype=torch.int32, device=device),
        "positions": emitting_positions(a, device),
        "slot_mapping": slot_mapping,
        # Spread over the state cache rather than pointing every gather at
        # row 0: an emitting token reads (1 + overlap) * compress_ratio rows,
        # and with one shared row they all come from cache, which flatters
        # the bandwidth. Entries stay below state_cache.shape[0], which is
        # what bounds the index.
        "block_table": torch.randint(
            0,
            a.tokens,
            (a.tokens, a.block_table_width),
            dtype=torch.int32,
            device=device,
        ),
        "block_size": a.block_size,
        "state_width": a.state_width,
        "cos_sin_cache": cos_sin_cache,
        "kv_cache": kv_cache,
        "k_cache_metadata": SimpleNamespace(slot_mapping=slot_mapping),
        "pdl_kwargs": {},
        "head_dim": a.head_dim,
        "rope_head_dim": a.rope_dim,
        "compress_ratio": a.compress_ratio,
        "overlap": a.overlap,
        "use_fp4_cache": False,
        "rms_norm_weight": torch.ones(a.head_dim, dtype=torch.bfloat16, device=device),
        "rms_norm_eps": a.rms_eps,
        "quant_block": a.quant_block,
        "token_stride": a.token_stride,
        "scale_dim": a.scale_dim,
    }


def counted_bytes(a):
    """Traffic the kernel must move, per emitting token:

      read   (1 + overlap) * compress_ratio state rows, twice over -- once
             for the scores and once for the values, head_dim fp32 each
      write  nope fp8 + rope bf16 + the uint8 scale group into the KV record

    The gather dominates by two orders of magnitude, so counting a single
    state row per token -- as this did while every token was still early
    exiting, and no traffic was moved at all -- understates the read side by
    roughly 2 * (1 + overlap) * compress_ratio.

    cos/sin are a small cached re-read and are not counted, so the reported
    bandwidth is a floor on what was achieved rather than an exact figure.
    """
    rows = (1 + bool(a.overlap)) * a.compress_ratio
    nope = a.head_dim - a.rope_dim
    read = 2 * rows * a.head_dim * 4
    write = nope + a.rope_dim * 2 + a.scale_dim
    return a.tokens * (read + write)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark the DeepSeek-V4 fused KV compressor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="cr128",
        help="which layer group's specialization to measure",
    )
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[256],
        help="grid size, i.e. tokens per launch (live run: 256)",
    )
    p.add_argument(
        "--rep",
        type=int,
        default=20,
        help="do_bench_cudagraph measurement target, in ms",
    )

    g = p.add_argument_group("shape (defaults are DeepSeek-V4-Pro tp4)")
    g.add_argument("--head-dim", type=int, default=512)
    g.add_argument("--rope-dim", type=int, default=64)
    g.add_argument("--quant-block", type=int, default=64)
    g.add_argument(
        "--token-stride",
        type=int,
        default=640,
        help="640 = aligned record, 576 = packed",
    )
    g.add_argument("--scale-dim", type=int, default=8)
    g.add_argument(
        "--rows-per-block",
        type=int,
        default=2387,
        help="KV rows per block in the backing allocation",
    )
    g.add_argument(
        "--state-stride0",
        type=int,
        default=381920,
        help="state_cache stride(0) in the live run",
    )
    g.add_argument("--cos-sin-rows", type=int, default=1048576)
    g.add_argument("--rms-eps", type=float, default=1e-6)

    v = p.add_argument_group("layer group (override --variant)")
    v.add_argument("--state-width", type=int)
    v.add_argument("--compress-ratio", type=int)
    v.add_argument("--block-size", type=int)
    v.add_argument("--state-mid", type=int)
    v.add_argument("--kv-page", type=int)
    v.add_argument("--block-table-width", type=int)
    ov = v.add_mutually_exclusive_group()
    ov.add_argument("--overlap", action="store_true", default=None)
    ov.add_argument("--no-overlap", dest="overlap", action="store_false")

    a = p.parse_args()
    for k, dflt in VARIANTS[a.variant].items():
        if getattr(a, k, None) is None:
            setattr(a, k, dflt)

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")

    print(
        f"compressor  variant={a.variant} head_dim={a.head_dim} "
        f"rope_dim={a.rope_dim} token_stride={a.token_stride} "
        f"state_width={a.state_width} compress_ratio={a.compress_ratio} "
        f"overlap={a.overlap}"
    )
    print(
        f"{'tokens':<10} {'single(us)':<12} {'2stage(us)':<12} "
        f"{'single GB/s':<12} {'us/token':<10} 2stage speedup"
    )

    all_tokens = a.tokens
    for n in all_tokens:
        a.tokens = n
        kwargs = build(a)

        def run():
            fused_deepseek_v4_compress_norm_rope_store(**kwargs)

        # The split exists to fan the compression across CUs when one program
        # per token cannot fill them, so it is only interesting against the
        # single-pass launcher at the same shape. num_decode_tokens=0 sends
        # every token down the split rather than falling back.
        scratch = torch.zeros(a.tokens, a.head_dim, dtype=torch.float32, device="cuda")

        def run_two_stage():
            fused_deepseek_v4_compress_norm_rope_store_two_stage(
                **kwargs, num_decode_tokens=0, compress_scratch=scratch
            )

        run()  # compile outside the timed region
        run_two_stage()
        torch.cuda.synchronize()
        ms = triton.testing.do_bench_cudagraph(run, rep=a.rep)
        ms2 = triton.testing.do_bench_cudagraph(run_two_stage, rep=a.rep)

        us, us2 = ms * 1e3, ms2 * 1e3
        gbs = counted_bytes(a) / (ms * 1e-3) / 1e9
        print(
            f"{n:<10d} {us:<12.2f} {us2:<12.2f} {gbs:<12.1f} "
            f"{us / n:<10.4f} {us / us2:.2f}x"
        )
    a.tokens = all_tokens


if __name__ == "__main__":
    main()
