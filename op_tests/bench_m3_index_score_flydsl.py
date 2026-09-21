# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cold-cache score timing versus a gather/max/write baseline.

The baseline preserves a unique [128,128] FP32 slab per (batch, chunk).
It is not a pure-read physical floor: its output traffic is substantial.
Historical ratios from the racy shared-output version are not comparable.
Scorer correctness lives in test_flydsl_minimax_m3_index_score.py; the small
independent check_floor regression checks every baseline slab before timing.
Run only on an idle GPU. Times explicitly flush L2 before each sample.
"""

import argparse
import itertools

import torch
import triton
import triton.language as tl

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
    IndexScoreConfig,
    alloc_score,
    make_work_map,
    resolve_config,
    score_flydsl,
    selection_filter,
    shuffle_cache,
    work_chunk,
)

P = 128
D = 128

# (name, batch, max_q, num_heads, seq_len[, skew]). The middle two
# are the user-confirmed production point: max_q=8, num_index_heads=4, so F=32.
#
# `skew` makes the batch ragged while holding sum(seq_lens) fixed, so the row is
# directly comparable in TB/s to its uniform twin. Real decode batches are
# always ragged; every case here was uniform until 2026-09-15, which hid a 2.3x
# loss. See lens_for() for why raggedness costs anything at all.
CASES = [
    ("decode_b16_s8k", 16, 1, 4, 8192),
    ("decode_b64_s8k", 64, 1, 4, 8192),
    ("spec_b16_q4_s8k", 16, 4, 4, 8192),
    ("spec_b50_q4_s100k", 50, 4, 4, 100000),
    ("serve_b32_q1_s128k", 32, 1, 4, 128000),
    ("serve_b32_q8_s128k", 32, 8, 4, 128000),
    ("serve_b50_q8_s100k", 50, 8, 4, 100000),
    ("ragged2x_b32_q8_s128k", 32, 8, 4, 128000, "2x"),
    ("ragged8x_b32_q8_s128k", 32, 8, 4, 128000, "8x"),
]


def lens_for(batch, seq_len, skew):
    """Per-request lengths for one case. sum() is `batch * seq_len` regardless.

    The grid is a rectangle sized by the LONGEST request, so a short request
    leaves holes in it. Holes are individually cheap (~0.2 us per 1000 when they
    sit together at the end) but ruinous when interleaved with real work along
    the dispatch axis: they consume launch slots and retire immediately, so the
    machine cannot keep enough K loads in flight. Measured at b32_q8_s128k fp8:
    uniform 159 us, "8x" 371 us, on identical bytes.
    """
    if skew is None:
        return [seq_len] * batch
    if skew == "2x":  # half short, half long, 3:1 ratio
        return [seq_len // 2] * (batch // 2) + [3 * seq_len // 2] * (batch - batch // 2)
    if skew == "8x":  # one outlier drags max_block up for all 32 rows
        rest = (batch * seq_len - 4 * seq_len) // (batch - 1)
        out = [4 * seq_len] + [rest] * (batch - 1)
        out[-1] += batch * seq_len - sum(out)  # absorb the rounding remainder
        return out
    raise ValueError(f"unknown skew {skew!r}")


def k_bytes(batch, seq_len, skew, dt):
    """K-cache bytes one launch must read -- the denominator for TB/s.

    Only the pages a request actually owns, so a ragged row reads exactly what
    its uniform twin reads and the two are directly comparable. Q is left out:
    at the widest case it is 262 KB against 524 MB of K, i.e. 0.05%.
    """
    nblk = sum(triton.cdiv(L, P) for L in lens_for(batch, seq_len, skew))
    return nblk * P * D * torch.tensor([], dtype=dt).element_size()


# Transcribed from ATOM's deleted `_decode_score_chunks` (TARGET_GRID 1<<14,
# MIN_BLOCKS 3, plus the count->size->count round trip that keeps every chunk
# non-empty). Copied rather than dropped for a simpler split: every floor number
# this kernel has ever been judged against was measured with exactly this grid,
# so re-tuning it would silently move the bar rather than raise it.
FLOOR_NUM_STAGES = 3
FLOOR_TARGET_GRID = 1 << 14
FLOOR_MIN_BLOCKS = 3


def floor_chunks(batch: int, max_block: int) -> int:
    if max_block <= 0:
        return 1  # a grid dim still has to be positive
    target = max(1, FLOOR_TARGET_GRID // max(1, batch))
    chunks = min(1 << (target.bit_length() - 1), max_block)
    chunks = min(chunks, max(1, triton.cdiv(max_block, FLOOR_MIN_BLOCKS)))
    return triton.cdiv(max_block, triton.cdiv(max_block, chunks))


@triton.jit
def _floor_kernel(
    k_ptr,
    out_ptr,
    bt_ptr,
    lens_ptr,
    chunk_blocks,
    sk_blk,
    sk_pos,
    sbt_b,
    BLOCK_SIZE_K: tl.constexpr,
    D: tl.constexpr,
):
    """Read every K page this request owns; reduce with elementwise max only.

    No dot or axis reduction; each CTA still computes and writes a full slab.
    """
    b = tl.program_id(0)
    c = tl.program_id(1)
    seq_len = tl.load(lens_ptr + b)
    nblk = tl.cdiv(seq_len, BLOCK_SIZE_K)

    pos = tl.arange(0, BLOCK_SIZE_K)
    d = tl.arange(0, D)
    acc = tl.full((BLOCK_SIZE_K, D), float("-inf"), tl.float32)

    for p in range(c * chunk_blocks, min((c + 1) * chunk_blocks, nblk)):
        page = tl.load(bt_ptr + b * sbt_b + p)
        # Keep the historical Triton .cg hint; do not assume it has the same
        # ISA bit semantics as FlyDSL's NT without inspecting generated ISA.
        k = tl.load(
            k_ptr
            + page.to(tl.int64) * sk_blk.to(tl.int64)
            + pos[:, None] * sk_pos
            + d[None, :],
            cache_modifier=".cg",
        )
        acc = tl.maximum(acc, k.to(tl.float32))

    tl.store(
        out_ptr
        + (b.to(tl.int64) * tl.num_programs(1) + c) * BLOCK_SIZE_K * D
        + pos[:, None] * D
        + d[None, :],
        acc,
    )


def check_floor():
    """Every CTA must preserve its full max slab, including empty chunks."""
    batch, mb, chunks = 2, 7, 3
    k = (
        torch.arange(batch * mb, device="cuda", dtype=torch.float32)[:, None, None]
        .expand(-1, P, D)
        .contiguous()
    )
    bt = torch.arange(batch * mb - 1, -1, -1, device="cuda", dtype=torch.int32).view(
        batch, mb
    )
    lens = torch.tensor([7 * P, 2 * P - 1], device="cuda", dtype=torch.int32)
    out = torch.full((batch, chunks, P, D), float("nan"), device="cuda")
    blocks = triton.cdiv(mb, chunks)
    _floor_kernel[(batch, chunks)](
        k,
        out,
        bt,
        lens,
        blocks,
        k.stride(0),
        k.stride(1),
        bt.stride(0),
        BLOCK_SIZE_K=P,
        D=D,
    )
    for b in range(batch):
        for c in range(chunks):
            ids = bt[
                b, c * blocks : min((c + 1) * blocks, triton.cdiv(int(lens[b]), P))
            ].long()
            ref = (
                k[ids].amax(0)
                if ids.numel()
                else torch.full((P, D), float("-inf"), device="cuda")
            )
            torch.testing.assert_close(out[b, c], ref, rtol=0, atol=0)


def make_inputs(batch, max_q, num_heads, seq_len, cache_dtype, seed=0, skew=None):
    torch.manual_seed(seed)
    total_q = batch * max_q
    lens = lens_for(batch, seq_len, skew)
    # Sized by the longest request, exactly as a serving runtime would.
    max_block = triton.cdiv(max(lens), P)
    idx_q = torch.randn(total_q, num_heads, D, dtype=torch.bfloat16, device="cuda")
    # One distinct physical page per (request, block): the real serving
    # footprint. Sharing pages would let L2 serve the reads and flatter every
    # number here equally. A ragged batch allocates only the pages its requests
    # actually own -- a paged pool has no page for a block that does not exist --
    # so the bytes read stay equal to the uniform twin's.
    nblk = [triton.cdiv(L, P) for L in lens]
    num_pages = sum(nblk)
    cache = torch.randn(num_pages, P, D, dtype=torch.bfloat16, device="cuda")
    if cache_dtype != torch.bfloat16:
        cache = cache.to(cache_dtype)
    perm = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    if skew is None:
        block_table = perm.view(batch, max_block).contiguous()
    else:
        # Row b owns nblk[b] pages; the rest of the row is never read (the
        # kernel clamps past num_pages) but must hold a legal page id.
        block_table = torch.zeros(batch, max_block, dtype=torch.int32, device="cuda")
        off = 0
        for b, n in enumerate(nblk):
            block_table[b, :n] = perm[off : off + n]
            off += n
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    score = torch.empty(
        (num_heads, total_q, max_block), dtype=torch.float32, device="cuda"
    )
    return idx_q, cache, block_table, seq_lens, score, max_block


def time_us(fn, flush, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        flush.add_(1)  # evict L2 so page reads come from HBM
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[0], times[len(times) // 2]


def cfg_label(cfg):
    """Short probe name for one config; defaults collapse to plain 'fly'."""
    s = "fly"
    if cfg.shuffled:
        s += "S"
    if cfg.pages_per_wave:  # 0 is auto, which is the default
        s += f"L{cfg.pages_per_wave}"
    if cfg.feat_waves > 1:
        s += f"F{cfg.feat_waves}"
    if cfg.token_waves > 1:
        s += f"T{cfg.token_waves}"
    if cfg.q_to_lds:
        s += "Q"
    if cfg.waves_per_eu:
        s += f"E{cfg.waves_per_eu}"
    if cfg.nt_k != 2:  # 2 (non-temporal) is the default K cache policy
        s += f"K{cfg.nt_k}"
    if cfg.sched > 0:  # 0 = no hint, -1 = auto (the default)
        s += f"C{cfg.sched}"
    return s


def run_case(name, batch, max_q, heads, seq_len, dt, cfgs, flush, args, skew=None):
    """Time every probe for one (shape, dtype). Returns {probe: median us}.

    A function rather than a loop body so each closure captures its own case:
    inside a loop they would all share the last iteration's tensors.
    """
    idx_q, cache, bt, lens, _score, max_block = make_inputs(
        batch, max_q, heads, seq_len, dt, skew=skew
    )
    nchunk = floor_chunks(batch, max_block)
    fout = torch.empty(batch * nchunk * P * D, dtype=torch.float32, device="cuda")
    print(f"# {name}: gather/max/write output footprint {fout.numel()*4/1e6:.3f} MB")
    # Shuffling is a one-off cost paid when the cache is written, so it is
    # hoisted out of the timed region -- and only paid if some config wants it.
    shuf = shuffle_cache(cache) if any(c.shuffled for c in cfgs) else None
    # FlyDSL picks its own score layout (alloc_score), which is why it does not
    # write into make_inputs' plain [H, total_q, max_block] tensor.
    fly_score = alloc_score(batch, max_q, heads, max_block, idx_q.device)

    def run_floor():
        _floor_kernel[(batch, nchunk)](
            cache, fout, bt, lens,
            triton.cdiv(max_block, nchunk),
            cache.stride(0), cache.stride(1), bt.stride(0),
            BLOCK_SIZE_K=P, D=D, num_stages=FLOOR_NUM_STAGES,
        )  # fmt: skip

    def make_flydsl(cfg):
        k = shuf if cfg.shuffled else cache
        # Resolve here too, not just inside score_flydsl: the work map is sized
        # by the chunk, so it has to be built against the same concrete config
        # the kernel will run.
        cfg = resolve_config(batch, max_block, cfg, max_q, heads)
        # The dispatch map depends only on seq_lens, which a serving runtime
        # computes once per decode step and reuses for every layer. Building it
        # inside the timed region would charge one kernel's worth of work with a
        # whole step's worth of setup.
        wm = make_work_map(lens, max_block, work_chunk(cfg))

        def run():
            score_flydsl(
                idx_q, k, bt, lens, max_q, heads, D**-0.5, max_block,
                out=fly_score, cfg=cfg, work_map=wm,
            )  # fmt: skip

        return run

    probes = [("floor", run_floor)]
    probes += [(cfg_label(c), make_flydsl(c)) for c in cfgs]

    res = {}
    for key, fn in probes:
        try:
            res[key] = time_us(fn, flush, args.warmup, args.repeat)[1]
        except Exception as exc:  # noqa: BLE001
            res[key] = None
            print(f"# {name}/{key}: {type(exc).__name__}: {exc}"[:160])

    return res


def main():
    if not torch.cuda.is_available() or get_gfx() not in ["gfx950"]:
        print("SKIP: score benchmark requires gfx950")
        return
    check_floor()
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--dtype", choices=["bf16", "fp8", "both"], default="both")
    ap.add_argument("--filter", default="")
    ap.add_argument("--ppw", default="0", help="comma list of pages_per_wave, 0=auto")
    ap.add_argument("--feat-waves", default="1", help="comma list of feat_waves")
    ap.add_argument("--token-waves", default="1", help="comma list of token_waves")
    ap.add_argument("--q-lds", default="0", help="comma list of 0/1 for q_to_lds")
    ap.add_argument("--shuffled", default="0", help="comma list of 0/1 for shuffled")
    ap.add_argument("--wpe", default="0", help="comma list of waves_per_eu (0=unset)")
    # Raw CDNA cache-policy values, not a boolean: 0 = default, 2 = nt,
    # 3 = sc0|nt. 2 is the kernel default; 1 exists only to complete the field.
    ap.add_argument("--nt", default="2", help="comma list of K cache policies")
    # -1 = auto, 0 = backend default, 1/2/3 = iglp_opt(0/1/2),
    # 4 = sched_group_barrier.
    ap.add_argument("--sched", default="-1", help="comma list of scheduling hints")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="ignore the knob flags and enumerate the whole tunable space",
    )
    args = ap.parse_args()

    arch = torch.cuda.get_device_properties(0).gcnArchName
    fp8_dtype = torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz
    dtypes = {"bf16": [torch.bfloat16], "fp8": [fp8_dtype]}.get(
        args.dtype, [torch.bfloat16, fp8_dtype]
    )
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

    print(f"# arch: {arch}")
    print(
        "# floor = gather/max/write baseline with unique slabs, NOT a pure-read lower bound\n"
    )
    hdr = (
        f"| {'case':<22} | {'dt':<4} | {'GB':>6} | {'flydsl':>9} | "
        f"{'floor':>9} | {'vs floor':>8} | {'TB/s':>5} |"
    )
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")

    def ints(v):
        return [int(x) for x in v.split(",")]

    if args.sweep:
        space = {
            "pages_per_wave": [1, 2], "feat_waves": [1, 2, 4], "token_waves": [1, 2, 4],
            "q_to_lds": [0, 1], "shuffled": [0, 1], "waves_per_eu": [0],
        }  # fmt: skip
    else:
        space = {
            "pages_per_wave": ints(args.ppw), "feat_waves": ints(args.feat_waves),
            "token_waves": ints(args.token_waves),
            "q_to_lds": ints(args.q_lds), "shuffled": ints(args.shuffled),
            "waves_per_eu": ints(args.wpe), "nt_k": ints(args.nt),
            "sched": ints(args.sched),
        }  # fmt: skip
    keys = list(space)
    all_cfgs = [
        IndexScoreConfig(**dict(zip(keys, v)))
        for v in itertools.product(*space.values())
    ]

    for dt in dtypes:
        tag = "bf16" if dt == torch.bfloat16 else "fp8"
        for name, batch, max_q, heads, seq_len, *rest in CASES:
            if args.filter and args.filter not in name:
                continue
            skew = rest[0] if rest else None
            # Legality depends on the shape (feat_waves must not exceed the
            # feature-tile count), so it is filtered per case, not once.
            cfgs = [c for c in all_cfgs if selection_filter(max_q, heads, c)]
            res = run_case(
                f"{name}/{tag}", batch, max_q, heads, seq_len, dt, cfgs, flush, args,
                skew=skew,
            )  # fmt: skip
            # The case's tensors die with run_case's frame; release the blocks
            # so the next (larger) case is not fighting the caching allocator.
            torch.cuda.empty_cache()

            best = min(
                (v for k, v in res.items() if k.startswith("fly") and v), default=None
            )
            f, fl = best, res["floor"]
            if len(cfgs) > 1:
                order = sorted(
                    (v, k) for k, v in res.items() if k.startswith("fly") and v
                )
                print(
                    f"#   {name}/{tag} -> "
                    + "  ".join(f"{k}={v:.1f}" for v, k in order)
                )

            def fmt(v):
                return f"{v:9.1f}" if v else f"{'ERR':>9}"

            vf = f"{f / fl:.2f}x" if f and fl else "-"
            nb = k_bytes(batch, seq_len, skew, dt)
            bw = f"{nb / (f * 1e-6) / 1e12:5.2f}" if f else f"{'-':>5}"
            print(
                f"| {name:<22} | {tag:<4} | {nb / 1e9:6.2f} | {fmt(f)} "
                f"| {fmt(fl)} | {vf:>8} | {bw} |"
            )


if __name__ == "__main__":
    main()
