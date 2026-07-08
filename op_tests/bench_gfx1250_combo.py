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
import re
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _silence():
    """Discard everything written to stdout/stderr — including native (C/C++)
    fd writes (ROCTracer, hipcc, aiter logger) — for the duration of the block.
    Redirects at the OS fd level so it catches more than sys.stdout swapping."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Flush any buffered Python-level output to the REAL fds BEFORE redirecting.
    # stdout is block-buffered when piped/redirected, so an earlier _print_table()
    # can still be sitting in the buffer; without this flush it would drain to
    # devnull once fd 1 is redirected here and the printed table would be lost.
    sys.stdout.flush()
    sys.stderr.flush()
    old1, old2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Flush again BEFORE restoring so anything printed inside the block goes
        # to devnull (not the real stdout after we restore it).
        sys.stdout.flush()
        sys.stderr.flush()
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


def _const_init(s):
    """Parse MoE const-init: 'random'/'rand'/'none' -> None (random init),
    else a float constant."""
    if isinstance(s, str) and s.lower() in ("random", "rand", "none"):
        return None
    return float(s)


def _int_pair(s):
    """Parse 'a,b' -> (int, int) — used for MLA (GQA,decode_qlen) pairs."""
    a, b = s.split(",")
    return int(a), int(b)


def _str2split(value):
    """Parse split_kv: 'auto' -> None (auto), else int."""
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return int(value)


def _tflops(flop, us):
    """TFLOPS from a FLOP count and microseconds (None-safe)."""
    return round(flop / us / 1e6, 2) if us else None


def _bw(nbytes, us):
    """Bandwidth (TB/s) from a byte count and microseconds (None-safe).
    bytes / (us*1e-6) / 1e12 == bytes / us / 1e6."""
    return round(nbytes / us / 1e6, 3) if us else None


# bytes-per-VALUE for the MoE quant formats (dims below are logical value counts,
# so fp4 must be 0.5 B/value, not the 1 B/element of the packed fp4x2 dtype).
#   a4w4 : fp4 act (0.5) x fp4 weight (0.5)
#   a8w4 : fp8 act (1.0) x fp4 weight (0.5)   (mxfp8 x mxfp4)
# The bf16 stage output is 2 B/value. (act_bpe, weight_bpe) per data_format.
_MOE_BPE = {"a4w4": (0.5, 0.5), "a8w4": (1.0, 0.5)}
_OUT_BPE = 2  # bf16 stage outputs


def _moe_stage_flops(token, topk, model_dim, inter_dim, use_g1u1=True):
    """Per-stage FLOP counts for the fused 2-stage MoE (matches gemm_moe_tune.py):
        stage1 GEMM: [token, model_dim] x [E, n, model_dim] -> token*n*model_dim*topk*2
                     n = inter_dim*2 (g1u1 gate+up) or inter_dim
        stage2 GEMM: [token, topk, inter_dim] x [E, model_dim, inter_dim]
                     -> topk*token*model_dim*inter_dim*2
    Returns (flop1, flop2)."""
    n = inter_dim * 2 if use_g1u1 else inter_dim
    flop1 = token * n * model_dim * topk * 2
    flop2 = topk * token * model_dim * inter_dim * 2
    return flop1, flop2


# per_1x32 microscale: every 32 quantized values share one e8m0 (1B) scale, so
# each quantized value carries an extra 1/32 B of scale traffic, on top of its
# own bpe. Applies to BOTH activations and weights (fp4 => bpe 0.5 => 17/16;
# fp8 => bpe 1.0 => 33/32). Output stays bf16 and is not microscaled.
# (gemm_moe_tune.py's stage1/stage2 omit scale entirely; we include it.)
_SCALE_PER_VALUE = 1 / 32


def _moe_stage_bytes(token, topk, model_dim, inter_dim, experts, aq_bpe, wq_bpe,
                     use_g1u1=True):
    """Per-stage MoE traffic (bytes), including per_1x32 e8m0 scale on every
    quantized operand (act + weight). The stage1 output / stage2 input is the
    expanded [token*topk, n] / [token*topk, inter] intermediate, so both carry
    topk; the stage1 input act is read once per token (reused across its topk
    experts):
        stage1: act[token,model_dim]@aq + out[token,topk,n]@bf16 + w1[E,n,model_dim]@wq
        stage2: act[token,topk,inter_dim]@aq + out[token,model_dim]@bf16
                + w2[E,model_dim,inter_dim]@wq
        n = inter_dim*2 (g1u1) or inter_dim.
    Returns (bytes1, bytes2)."""
    n = inter_dim * 2 if use_g1u1 else inter_dim
    bo = _OUT_BPE
    aq = aq_bpe + _SCALE_PER_VALUE  # quantized act: data + e8m0 scale per value
    wq = wq_bpe + _SCALE_PER_VALUE  # quantized weight: data + e8m0 scale per value
    bytes1 = (token * model_dim * aq
              + token * topk * n * bo
              + experts * n * model_dim * wq)
    bytes2 = (token * topk * inter_dim * aq
              + token * model_dim * bo
              + experts * model_dim * inter_dim * wq)
    return bytes1, bytes2


# Per-op column whitelists: keep shape identifiers + perf, drop the constant
# config/correctness columns @benchmark echoes (gfx/dtype/err/cos_diff/...).
_MXFP8_KEEP = [
    "batch", "nheads", "nheads_k", "seqlen", "d", "causal",
    "asm us", "asm TFLOPS", "asm TB/s",
]
_SINK_KEEP = [
    "head_dim", "hq", "hk", "sq", "sk", "batch", "is_causal", "init",
    "asm us", "asm TFLOPS", "asm TB/s",
]
_MLA_KEEP = [
    "batch", "ctx_len", "nhead", "decode_qlen", "split_kv", "mask", "init",
    "num_kv_splits", "mi400 us", "mi400 TFLOPS", "mi400 TB/s",
]
_MOE_KEEP = [
    "data_format", "token", "model_dim", "inter_dim", "E", "topk", "pass",
    "gemm1_us", "gemm1 TFLOPS", "gemm1 TB/s",
    "gemm2_us", "gemm2 TFLOPS", "gemm2 TB/s",
    "total us", "total TFLOPS", "total TB/s",
]
_GEMM_KEEP = [
    "intype", "M", "N", "K", "init",
    "gemm_a4w4 us", "gemm_a4w4 TFLOPS", "gemm_a4w4 TB/s",
    "asm us", "asm TFLOPS", "asm TB/s",
]
_GEMM84_KEEP = [
    "intype", "M", "N", "K", "init",
    "asm us", "asm TFLOPS", "asm TB/s",
]
_PA_KEEP = [
    "batch", "kv_head_num", "ctx_len", "mtp", "max_kv", "init",
    "us", "TFLOPS", "TB/s",
]
_MHA_BWD_KEEP = [
    "batch", "nheads", "nheads_k", "seqlen", "d", "mask",
    "us", "TFLOPS", "TB/s",
]


# Peak (max) TFLOPS / TB/s per op, accumulated across _print_table calls and
# written to CSV at the end of main() when --peak-csv is set.
_PEAK_ROWS = []


def _op_peaks(df):
    """Peak TFLOPS and peak TB/s over a table: max across every column whose
    name contains 'TFLOPS' / 'TB/s' (op tables name them differently, e.g.
    'asm TFLOPS', 'mi400 TB/s', 'total TFLOPS'). None if no such column/values."""
    def _mx(match):
        cols = [c for c in df.columns if match in c.lower()]
        if not cols:
            return None
        vals = pd.to_numeric(df[cols].stack(), errors="coerce")
        m = vals.max()
        return round(float(m), 2) if pd.notna(m) else None
    return _mx("tflops"), _mx("tb/s")


def _print_table(name, rows, keep=None):
    df = pd.DataFrame([r for r in rows if r is not None])
    if not df.empty:
        # Drop columns that are entirely empty, then whitelist/order via `keep`.
        # The @benchmark decorator dumps every call arg as a column, which makes
        # the tables wide; `keep` trims to shape ids + perf. ALWAYS surface any
        # err_msg / *err column so failures never get silently hidden.
        df = df.replace("", pd.NA).dropna(axis=1, how="all")
        if keep is not None:
            cols = [c for c in keep if c in df.columns]
            cols += [c for c in df.columns if "err_msg" in c and c not in cols]
            df = df[cols]
        peak_tflops, peak_bw = _op_peaks(df)
        _PEAK_ROWS.append(
            {"op": name, "peak_TFLOPS": peak_tflops, "peak_TB/s": peak_bw}
        )
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
    _print_table("fmha_fwd_mxfp8", rows, keep=_MXFP8_KEEP)


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
    _print_table("fmha_fwd_with_sink_asm", rows, keep=_SINK_KEEP)


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
    _print_table("mla_decode_pagesize64", rows, keep=_MLA_KEEP)


def run_pa(args):
    # test_pa_decode returns TB/s but no TFLOPS, so add it here. PA-decode FLOPs =
    # QK^T + P@V = 4 madd-flops per (q_token, q_head, kv_token, head_dim).
    #   q_tokens/seq = mtp+1 ; q_heads = kv_head_num * gqa(8) ; head_dim = 64.
    #   kv_tokens/seq = ctx_len padded up to page_size (matches the module's own
    #   attended-token count that drives its TB/s), non-varlen => ctx per seq.
    gqa = pa_mod.PA_GQA_RATIO
    head_dim = pa_mod.PA_HEAD_DIM
    page = pa_mod.PA_PAGE_SIZE
    rows = []
    scales = tuple(args.pa_scales) if args.pa_scales else None
    with _silence():
        for batch, kv_head_num, ctx_len, mtp, init in itertools.product(
            args.pa_batch, args.pa_kvh, args.pa_ctxlen, args.pa_mtp, args.pa_init
        ):
            # test_pa_decode reads QKV_CONST from the env (no function arg):
            # unset -> random (0.5*randn) Q/K/V; set -> fixed constant fill.
            if init == "random":
                os.environ.pop("QKV_CONST", None)
            else:
                os.environ["QKV_CONST"] = init.removeprefix("const")
            r = pa_mod.test_pa_decode(
                batch, kv_head_num, ctx_len, mtp, scales,
                varlen=False, use_sink=args.pa_sink,
            )
            if isinstance(r, dict):
                r["init"] = init
                qlen = mtp + 1
                q_head_num = kv_head_num * gqa
                kv_tokens = ((ctx_len + page - 1) // page) * page
                flops = batch * qlen * q_head_num * kv_tokens * head_dim * 4
                r["TFLOPS"] = _tflops(flops, r.get("us"))
            rows.append(r)
    os.environ.pop("QKV_CONST", None)
    _print_table("pa_decode_bf16_asm", rows, keep=_PA_KEEP)


def run_moe(args):
    rows = []
    for fmt, tokens in itertools.product(args.moe_format, args.moe_tokens):
        moe_mod.set_data_format(fmt)
        with _silence():
            metrics = moe_mod.run_moe(
                fmt,
                experts=args.moe_experts,
                tokens=tokens,
                topk=args.moe_topk,
                model_dim=args.moe_model_dim,
                inter_dim=args.moe_inter_dim,
                layout="gugu",
                activation=moe_mod.ActivationType.Swiglu,
                kernel_bench=True,
                warmup=5,
                iters=128,
                const_init=args.moe_const_init,
                check_aot_cache=False,
                raise_on_fail=False,
            )
        # kernel_bench returns per-GEMM us (gemm1/gemm2). Derive per-stage +
        # total TFLOPS / bw from the stage1/stage2 FLOP/byte split (Swiglu =>
        # g1u1 gate+up, so stage1 n = inter_dim*2). aq/wq bpe by data format.
        aq_bpe, wq_bpe = _MOE_BPE.get(fmt, (1, 1))
        flop1, flop2 = _moe_stage_flops(
            tokens, args.moe_topk, args.moe_model_dim, args.moe_inter_dim,
            use_g1u1=True,
        )
        bytes1, bytes2 = _moe_stage_bytes(
            tokens, args.moe_topk, args.moe_model_dim, args.moe_inter_dim,
            args.moe_experts, aq_bpe, wq_bpe, use_g1u1=True,
        )
        us1, us2 = metrics.get("gemm1_us"), metrics.get("gemm2_us")
        total_us = (us1 or 0) + (us2 or 0) if (us1 or us2) else None
        bw1, bw2, bwt = _bw(bytes1, us1), _bw(bytes2, us2), _bw(bytes1 + bytes2, total_us)
        rows.append({
            "data_format": fmt,
            "token": tokens,
            "model_dim": args.moe_model_dim,
            "inter_dim": args.moe_inter_dim,
            "E": args.moe_experts,
            "topk": args.moe_topk,
            "pass": metrics["passed"],
            "gemm1_us": us1,
            "gemm1 TFLOPS": _tflops(flop1, us1),
            "gemm1 TB/s": bw1,
            "gemm2_us": us2,
            "gemm2 TFLOPS": _tflops(flop2, us2),
            "gemm2 TB/s": bw2,
            "total us": round(total_us, 2) if total_us else None,
            "total TFLOPS": _tflops(flop1 + flop2, total_us),
            "total TB/s": bwt,
        })
    _print_table("flydsl_grouped_gemm (kernel)", rows, keep=_MOE_KEEP)


def run_gemm(args):
    rows = []
    with _silence():
        for intype, apre, init, (M, N, K) in itertools.product(
            ["mxfp4", "nvfp4"], [1], ["constant", "random"],
            [(16384, 16384, 16384)],
        ):
            rows.append(gemm_mod.test_gemm(intype, M, N, K, apre, init))
    _print_table("gemm_a4w4_mi400", rows, keep=_GEMM_KEEP)


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
    _print_table("mxfp8fp4gemm_mi400", rows, keep=_GEMM84_KEEP)


# mha_bwd_bf16: runs the compiled bwd.exe (asm v3 bf16 fmha backward) per shape
# and parses its result line, which the exe prints as:
#   ", <ms> ms, <tflops> TFlops, <bw> GB/s"
# The exe reports its own authoritative TFLOPS/BW, so we just parse them (ms->us,
# GB/s->TB/s). Fixed flags mirror the reference cmd: bwd_v3=1 kname=1
# v3_atomic_fp32=1 mode=0 v=0 init=3.
_MHA_BWD_RE = re.compile(
    r"([\d.]+)\s*ms,\s*([\d.]+)\s*TFlops,\s*([\d.]+)\s*GB/s"
)
_MHA_BWD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp", "mha")
_MHA_BWD_ASM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hsa"
)


def run_mha_bwd(args):
    rows = []
    exe = os.path.join(_MHA_BWD_DIR, "bwd.exe")
    env = {**os.environ, "AITER_ASM_DIR": _MHA_BWD_ASM}
    for b, h, hk, s, d, mask in itertools.product(
        args.mha_bwd_batch, args.mha_bwd_h, args.mha_bwd_hk,
        args.mha_bwd_seqlen, args.mha_bwd_d, args.mha_bwd_mask,
    ):
        row = {"batch": b, "nheads": h, "nheads_k": hk,
               "seqlen": s, "d": d, "mask": mask}
        cmd = [
            exe, "-prec=bf16", f"-b={b}", f"-h={h}", f"-h_k={hk}",
            f"-s={s}", f"-s_k={s}", f"-d={d}", "-bwd_v3=1", "-kname=1",
            "-v3_atomic_fp32=1", f"-mask={mask}", "-mode=0", "-v=0", "-init=3",
        ]
        try:
            out = subprocess.run(
                cmd, cwd=_MHA_BWD_DIR, env=env, capture_output=True,
                text=True, timeout=600,
            )
            m = _MHA_BWD_RE.search(out.stdout)
            if m:
                ms, tflops, gbps = (float(m.group(i)) for i in (1, 2, 3))
                row["us"] = round(ms * 1e3, 3)
                row["TFLOPS"] = round(tflops, 2)
                row["TB/s"] = round(gbps / 1e3, 3)
            else:
                row["err_msg"] = (out.stdout + out.stderr).strip()[-200:] or "no perf line"
        except Exception as e:  # noqa: BLE001 - keep the table alive
            row["err_msg"] = repr(e)
        rows.append(row)
    _print_table("mha_bwd_bf16 (asm v3)", rows, keep=_MHA_BWD_KEEP)


OPS = {
    "mxfp8": run_mxfp8,
    "sink": run_sink,
    "mla": run_mla,
    "pa": run_pa,
    "moe": run_moe,
    "gemm": run_gemm,
    "gemm84": run_gemm84,
    "mha_bwd": run_mha_bwd,
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
    # shared axis (mxfp8 fmha fwd). batch=2 lifts small-seqlen off the launch-bound
    # floor (seqlen=1024: ~418 TFLOPS @ batch1 -> ~814 @ batch2); seqlen=8192 is
    # already saturated (~4.2 PFLOPS) regardless of batch.
    p.add_argument("-b", "--batch", type=int, nargs="*", default=[2])
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
        default=[(8, 1), (16, 1)],
        help="MLA (GQA,decode_qlen) pairs (default: 8,1 16,1)",
    )
    # Defaults mirror test_mla_decode_pagesize64.py's own recommended sweep
    # (-n 8,1 8,2 16,1 32,1 -c 16384 --split_kv auto), batch extended to 256/512.
    p.add_argument("--mla-batch", type=int, nargs="*", default=[1024])
    p.add_argument("--mla-ctxlen", type=int, nargs="*", default=[16384])
    p.add_argument(
        "--mla-split-kv",
        type=_str2split,
        nargs="*",
        default=[None],
        help="MLA KV split count, or auto (default: auto)",
    )
    p.add_argument("--mla-mask", type=int, nargs="*", default=[1], choices=[0, 1])
    p.add_argument(
        "--mla-init",
        type=str,
        nargs="*",
        default=["randn", "const0.25"],
        choices=["randn", "const0.25"],
    )
    # pa decode axes
    p.add_argument("--pa-batch", type=int, nargs="*", default=[512])
    p.add_argument(
        "--pa-kvh", type=int, nargs="*", default=[8],
        help="PA kv_head_num (q_heads = kvh * 8). Default: 8",
    )
    p.add_argument("--pa-ctxlen", type=int, nargs="*", default=[65536])
    p.add_argument("--pa-mtp", type=int, nargs="*", default=[0])
    p.add_argument(
        "--pa-init", nargs="*", default=["random", "const0.25"],
        choices=["random", "const0.1", "const0.25", "const0.5"],
        help="PA Q/K/V init: 'random' (0.5*randn) or 'const<v>' fixed fill via "
        "QKV_CONST (default: random const0.25)",
    )
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
    p.add_argument("--moe-tokens", type=int, nargs="*", default=[64, 256, 1024, 8192])
    p.add_argument("--moe-topk", type=int, default=6)
    p.add_argument("--moe-model-dim", type=int, default=4096)
    p.add_argument("--moe-inter-dim", type=int, default=2048)
    p.add_argument(
        "--moe-const-init", type=_const_init, default=0.5,
        help="MoE init: a float constant for weights/act/bias, or "
        "'random'/'rand' for random init (default: 0.5)",
    )
    # mha_bwd_bf16 (asm v3 bwd.exe) axes
    p.add_argument("--mha-bwd-batch", type=int, nargs="*", default=[4])
    p.add_argument("--mha-bwd-h", type=int, nargs="*", default=[64],
                   help="mha_bwd nheads (default: 64)")
    p.add_argument("--mha-bwd-hk", type=int, nargs="*", default=[4],
                   help="mha_bwd nheads_k (default: 4)")
    p.add_argument("--mha-bwd-seqlen", type=int, nargs="*", default=[8192],
                   help="mha_bwd seqlen (used for both s and s_k; default: 8192)")
    p.add_argument("--mha-bwd-d", type=int, nargs="*", default=[128])
    p.add_argument("--mha-bwd-mask", type=int, nargs="*", default=[0], choices=[0, 1])
    p.add_argument(
        "--peak-csv", default="bench_gfx1250_peak.csv",
        help="CSV path to record each op's peak TFLOPS + peak TB/s "
        "(default: bench_gfx1250_peak.csv; pass '' to disable)",
    )
    args = p.parse_args()

    for name in args.ops:
        OPS[name](args)

    # Record per-op peaks (max TFLOPS / max TB/s) to CSV.
    if args.peak_csv and _PEAK_ROWS:
        pd.DataFrame(_PEAK_ROWS).to_csv(args.peak_csv, index=False)
        print(f"\n[peak] wrote {len(_PEAK_ROWS)} op(s) to {args.peak_csv}")


if __name__ == "__main__":
    main()
