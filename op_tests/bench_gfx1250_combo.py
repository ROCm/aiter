# SPDX-License-Identifier: MIT
"""Combined gfx1250 asm-kernel perf bench.

Imports the top-level @benchmark sweep fns from the aiter op_tests (which the
aiter-op-test skill keeps importable for exactly this kind of combination
testing) and runs each over its own shape axes.

Output discipline: this script prints ONLY the per-op summary tables. All the
underlying noise (per-config "calling ..." logs, JIT build output, aiter import
banners, pandas/torch/ROCTracer warnings, including C-level fd writes) is
silenced via os-level fd redirection while the kernels run; the markdown tables
are then printed to real stdout.

Run from the aiter repo root so `op_tests/` siblings import cleanly:

    cd /app/aiter
    python op_tests/bench_gfx1250_combo.py            # all ops, curated defaults
    python op_tests/bench_gfx1250_combo.py --ops sink  # just one op
    python op_tests/bench_gfx1250_combo.py --ops mla  # just MLA decode
    python op_tests/bench_gfx1250_combo.py --ops pa   # just PA decode
    python op_tests/bench_gfx1250_combo.py --ops moe  # just FlyDSL MoE
    python op_tests/bench_gfx1250_combo.py \
        --mxfp8-hk 8,2 64,2 --mxfp8-seqlen 1024 8192 --sink-seqlen 16384 32768

gfx1250's bundled CK does not compile, so the asm JIT modules must be built with
ENABLE_CK=0. The script sets it (before importing aiter) so a plain run just
works; an explicit env override still wins.
"""

import os

# Must be set BEFORE `import aiter` so the JIT build picks it up. setdefault =>
# an explicitly-exported ENABLE_CK from the caller is respected.
os.environ.setdefault("ENABLE_CK", "0")

# FlyDSL MoE env vars — must be set before importing the moe test module,
# which reads them at import time into module-level constants.
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_FORCE_GFX1250", "1")
os.environ.setdefault("AITER_GROUPED_GEMM_WAVE_SPECIALIZED", "1")

import argparse
import contextlib
import itertools
import warnings

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _silence():
    """Discard everything written to stdout/stderr — including native (C/C++)
    fd writes (ROCTracer, hipcc, aiter logger) — for the duration of the block.
    Redirects at the OS fd level so it catches more than sys.stdout swapping."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old1, old2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        os.close(devnull)
        os.close(old1)
        os.close(old2)


# Import aiter + the op-test modules quietly (import-time banners suppressed).
with _silence():
    import pandas as pd
    import aiter  # noqa: F401  (kept for parity / future logger use)
    from aiter.jit.utils.chip_info import get_gfx
    import test_fmha_fwd_mxfp8_asm as mxfp8_mod
    import test_fmha_fwd_with_sink_asm as sink_mod  # has __main__ guard
    import test_mla_decode_pagesize64 as mla_mod
    import test_pa_decode_bf16_asm as pa_mod  # sys.exit(0) on non-gfx1250
    import test_flydsl_grouped_gemm_gfx1250 as moe_mod
    import test_gemm_a4w4_mi400 as gemm_mod
    import test_mxfp8fp4gemm_mi400 as gemm84_mod
    from aiter import dtypes

SUPPORTED_GFX = ["gfx1250"]


def _hk_pair(s):
    """Parse 'nheads,nheads_k' -> (int, int)."""
    a, b = s.split(",")
    return int(a), int(b)


def _int_pair(s):
    """Parse 'a,b' -> (int, int) — used for MLA (GQA,decode_qlen) pairs."""
    a, b = s.split(",")
    return int(a), int(b)


def _str2split(value):
    """Parse split_kv: 'auto' -> None (auto), else int."""
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return int(value)


def _print_table(name, rows):
    df = pd.DataFrame([r for r in rows if r is not None])
    print(f"\n===== {name} =====")
    print(df.to_markdown(index=False))


# --- per-op runners: sweep axes silently, then print one table ---


def run_mxfp8(args):
    rows = []
    with _silence():
        for (nh, nhk), b, s, causal in itertools.product(
            args.mxfp8_hk, args.batch, args.mxfp8_seqlen, args.causal
        ):
            rows.append(
                mxfp8_mod.test_fmha_fwd_mxfp8(b, nh, nhk, s, args.d, bool(causal))
            )
    _print_table("fmha_fwd_mxfp8", rows)


def run_sink(args):
    # perf-only fn (no torch ref): sq==sk, hq=64, hk=8(d64)/4(d128), batch=1.
    # init is the OUTERMOST axis so same-init rows group together.
    rows = []
    with _silence():
        for init, head_dim, seqlen, causal in itertools.product(
            args.sink_init, args.sink_headdim, args.sink_seqlen, args.sink_causal
        ):
            hk = 8 if head_dim == 64 else 4
            rows.append(
                sink_mod.test_fmha_fwd_with_sink_asm_perf(
                    head_dim, 64, hk, seqlen, seqlen, 1, bool(causal), init
                )
            )
    _print_table("fmha_fwd_with_sink_asm", rows)


def run_mla(args):
    rows = []
    with _silence():
        for (nhead, decode_qlen), batch, ctx_len, split_kv, mask, init in (
            itertools.product(
                args.mla_nhead,
                args.mla_batch,
                args.mla_ctxlen,
                args.mla_split_kv,
                args.mla_mask,
                args.mla_init,
            )
        ):
            rows.append(
                mla_mod.test_mla(
                    batch,
                    ctx_len,
                    nhead,
                    decode_qlen,
                    split_kv,
                    mask,
                    dtypes.fp8,
                    dtypes.fp8,
                    init,
                )
            )
    _print_table("mla_decode_pagesize64", rows)


def run_pa(args):
    rows = []
    scales = tuple(args.pa_scales) if args.pa_scales else None
    with _silence():
        for batch, kv_head_num, ctx_len, mtp in itertools.product(
            args.pa_batch, args.pa_kvh, args.pa_ctxlen, args.pa_mtp
        ):
            rows.append(
                pa_mod.test_pa_decode(
                    batch, kv_head_num, ctx_len, mtp, scales,
                    varlen=False, use_sink=args.pa_sink,
                )
            )
    _print_table("pa_decode_bf16_asm", rows)


def run_moe(args):
    rows = []
    for fmt in args.moe_format:
        moe_mod.set_data_format(fmt)
        with _silence():
            metrics = moe_mod.run_moe(
                fmt,
                experts=args.moe_experts,
                tokens=args.moe_tokens,
                topk=args.moe_topk,
                model_dim=args.moe_model_dim,
                inter_dim=args.moe_inter_dim,
                layout="gugu",
                activation=moe_mod.ActivationType.Swiglu,
                kernel_bench=True,
                warmup=5,
                iters=128,
                const_init=0.5,
                check_aot_cache=False,
                raise_on_fail=False,
            )
        rows.append({
            "data_format": fmt,
            "logits_diff": metrics["logits_diff"],
            "rel_l2": metrics["rel_l2"],
            "pass": metrics["passed"],
            "gemm1_us": metrics.get("gemm1_us"),
            "gemm2_us": metrics.get("gemm2_us"),
        })
    _print_table("flydsl_grouped_gemm (kernel)", rows)


def run_gemm(args):
    rows = []
    with _silence():
        for intype, apre, init, (M, N, K) in itertools.product(
            ["mxfp4", "nvfp4"], [1], ["constant", "random"],
            [(16384, 16384, 16384)],
        ):
            rows.append(gemm_mod.test_gemm(intype, M, N, K, apre, init))
    _print_table("gemm_a4w4_mi400", rows)


def run_gemm84(args):
    rows = []
    with _silence():
        for intype, (M, N, K), apre, init in itertools.product(
            ["a8w8", "a8w4"],
            [(32768, 16384, 8192), (16384, 16384, 16384),
             (2, 1048576, 16384), (2, 1048576, 8192)],
            [1], ["constant", "random"],
        ):
            rows.append(gemm84_mod.test_gemm(intype, M, N, K, apre, init))
    _print_table("mxfp8fp4gemm_mi400", rows)


OPS = {
    "mxfp8": run_mxfp8,
    "sink": run_sink,
    "mla": run_mla,
    "pa": run_pa,
    "moe": run_moe,
    "gemm": run_gemm,
    "gemm84": run_gemm84,
}


def main():
    if get_gfx() not in SUPPORTED_GFX:
        print(f"combo bench targets {SUPPORTED_GFX} only; current {get_gfx()} — skipping")
        return

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="combined gfx1250 asm-kernel perf bench (prints only summaries)",
    )
    p.add_argument(
        "--ops",
        nargs="*",
        choices=list(OPS),
        default=list(OPS),
        help="which ops to bench (default: all)",
    )
    # shared axis
    p.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    # mxfp8 axes
    p.add_argument(
        "--mxfp8-hk",
        type=_hk_pair,
        nargs="*",
        default=[(8, 2)],
        help="mxfp8 (nheads,nheads_k) pairs, e.g. 8,2 64,2 (default: 8,2)",
    )
    p.add_argument("--mxfp8-seqlen", type=int, nargs="*", default=[1024, 8192])
    p.add_argument("-d", type=int, default=128, help="mxfp8 head dim (default 128)")
    p.add_argument("--causal", type=int, nargs="*", default=[0], choices=[0, 1])
    # sink (SWA) perf axes — defaults reproduce the `-s 16384 32768` subset
    p.add_argument("--sink-seqlen", type=int, nargs="*", default=[16384, 32768])
    p.add_argument(
        "--sink-headdim", type=int, nargs="*", default=[64, 128], choices=[64, 128]
    )
    p.add_argument("--sink-causal", type=int, nargs="*", default=[0, 1], choices=[0, 1])
    p.add_argument(
        "--sink-init",
        type=str,
        nargs="*",
        default=["randn", "const0.25"],
        choices=["randn", "const0.25"],
    )
    # mla decode axes
    p.add_argument(
        "--mla-nhead",
        type=_int_pair,
        nargs="*",
        default=[(8, 1), (8, 2), (16, 1), (32, 1), (128, 1)],
        help="MLA (GQA,decode_qlen) pairs (default: 8,1 8,2 16,1 32,1 128,1)",
    )
    p.add_argument("--mla-batch", type=int, nargs="*", default=[1, 4])
    p.add_argument("--mla-ctxlen", type=int, nargs="*", default=[128, 1024])
    p.add_argument(
        "--mla-split-kv",
        type=_str2split,
        nargs="*",
        default=[1, 2],
        help="MLA KV split count, or auto (default: 1 2)",
    )
    p.add_argument("--mla-mask", type=int, nargs="*", default=[1], choices=[0, 1])
    p.add_argument(
        "--mla-init",
        type=str,
        nargs="*",
        default=["randn"],
        choices=["randn", "const0.25"],
    )
    # pa decode axes
    p.add_argument("--pa-batch", type=int, nargs="*", default=[1, 3, 8, 64])
    p.add_argument(
        "--pa-kvh", type=int, nargs="*", default=[1, 8],
        help="PA kv_head_num (q_heads = kvh * 8). Default: 1 8",
    )
    p.add_argument("--pa-ctxlen", type=int, nargs="*", default=[7, 256, 1024, 4097, 16384])
    p.add_argument("--pa-mtp", type=int, nargs="*", default=[0])
    p.add_argument(
        "--pa-scales", type=float, nargs=3, default=None,
        metavar=("Q", "K", "V"),
        help="PA per-tensor q/k/v dequant scales (default: random)",
    )
    p.add_argument("--pa-sink", action="store_true", help="enable PA attention sink")
    # flydsl moe axes
    p.add_argument(
        "--moe-format", nargs="*", default=["a4w4", "a8w4"],
        choices=["a4w4", "a8w4"], help="MoE data formats (default: a4w4 a8w4)",
    )
    p.add_argument("--moe-experts", type=int, default=256)
    p.add_argument("--moe-tokens", type=int, default=64)
    p.add_argument("--moe-topk", type=int, default=6)
    p.add_argument("--moe-model-dim", type=int, default=4096)
    p.add_argument("--moe-inter-dim", type=int, default=2048)
    args = p.parse_args()

    for name in args.ops:
        OPS[name](args)


if __name__ == "__main__":
    main()
