# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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
    # Hardware-oriented single-op performance (no model-specific shape contract).
    python op_tests/bench_gfx1250_combo.py --perf                 # all perf ops
    python op_tests/bench_gfx1250_combo.py --perf --ops mha       # MHA
    python op_tests/bench_gfx1250_combo.py --perf --ops moe       # grouped MoE FC1/FC2 (a4w4 + a8w4)
    python op_tests/bench_gfx1250_combo.py --perf --ops gemm      # F4GEMM
    python op_tests/bench_gfx1250_combo.py --perf --ops f8gemm    # F8GEMM
    python op_tests/bench_gfx1250_combo.py --perf --ops mla_v4_decode  # MLA v4 decode

    # DeepSeek-V4 operators at the model shapes used by the DSv4 workload.
    python op_tests/bench_gfx1250_combo.py --dsv4                 # all DSv4 ops
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops moe       # grouped MoE FC1/FC2 (DSv4 a8w4)
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops a8w8_blockscale  # DSv4 FP8 linears
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops a16w16    # DSv4 BF16 linears
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mla_v4_decode   # sparse MLA v4 decode
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops inverse_rope  # inverse RoPE + group quant
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mla_v4_prefill  # MLA v4 prefill
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mla_v4_prefill_fp8  # FP8 MLA v4 prefill
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mhc       # mHC fused RMSNorm
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops qk_norm   # QK norm + RoPE
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops score_qk  # FP8 paged MQA logits
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mori_ep   # MORI EPv2 dispatch/combine
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mega_moe  # Mega on/off, 4 GPUs

The ``mori_ep`` op updates and installs ``${MORI:-/app/mori}``, then runs its
existing EPv2 benchmark. Environment variables select backend, token tiers,
eager/graph modes, EP size, dispatch dtype, and correctness checking:

    TOKENS=512 MODES=graph \
      python op_tests/bench_gfx1250_combo.py --dsv4 --ops mori_ep

The ``mhc`` op runs:

    python3 op_tests/test_mhc.py -n 7168 -m 512 --fuse_rmsnorm

The ``qk_norm`` op runs both DSv4 phases:

    python3 op_tests/test_flydsl_qk_norm_rope_quant.py \
      -T 16384 --H 128 --D 512 --RD 64 --no-quant --qweight

    python3 op_tests/test_flydsl_qk_norm_rope_quant.py \
      -T 1 2 4 8 16 32 64 --H 128 --D 512 --RD 64 --no-quant --qweight

The ``score_qk`` op runs two decode KV lengths at batch 512:

    python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py \
      --batch 512 --heads 64 --index_dim 128 -kv_length 384 -mtp 0 \
      --kv_preshuffle --blocksize 64

    python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py \
      --batch 512 --heads 64 --index_dim 128 -kv_length 10240 -mtp 0 \
      --kv_preshuffle --blocksize 64

For the 1K-input/1K-output workload, the average decode context is 1536 source
tokens. CSA compresses KV by 4, so score-QK scans an average KV length of 384.

The DSv4 ``mla_v4_decode`` op runs sparse decode with GQA/H=128, batch=512 and
q_seq=1 (M=512), sweeping KV lengths 256/512/1024 and split counts 1/2/4.

The DSv4 ``mla_v4_prefill`` op runs four FP8 performance cases at M=16384,
H=128 and D=512: compressed prefix-pool rows 4096/16384 crossed with
dense/sparse CSR modes. The current 16K-token chunk remains uncompressed:

    python3 op_tests/test_pa_sparse_prefill.py \
      -n 16384 --h_q 128 -d 512 \
      --total_pages 4096 16384 --total_tokens 16384 \
      --prec fp8 --mode dense sparse --no-verify

The separate ``mla_v4_prefill_fp8`` op runs:

    PYTHONPATH=. python3 op_tests/test_pa_sparse_prefill.py

The ``inverse_rope`` op runs the tp1 attention-output shape (-b is
(n_local_heads, n_local_groups); 128,16 is V4-Pro at dp/tp1):

    python3 op_tests/test_inverse_rope_group_quant.py \
      -b 128,16 -s <token sweep> -l n32k4 --group-size 32

The ``a8w8_blockscale`` op runs:

    python3 op_tests/test_gemm_a8w8_blockscale.py \
      -m 512 \
      -nk 2048,7168 7168,16384 6144,7168 \
          7168,3072 65536,1536 8192,1536 \
      --ck_preshuffle True --flydsl

The ``a16w16`` op uses ``test_opus_a16w16_gemm.py`` with batch=1, M=512,
K=7168 and N=64,384,1024,2048,32320,129280.

The ``mega_moe`` op runs both sides of the comparison:

    MORI_V2_KERNEL_BACKEND=hip MEGA_DISPATCH=mori \
    torchrun --standalone --nproc_per_node=4 \
      op_tests/multigpu_tests/test_mega_moe_gfx1250.py \
      -e 384 -k 6 -hd 7168 -id 3072 \
      --layers 61 -tpr 512 --combine scatter_fused \
      --acc_verify 0 --profile_table 1

    MORI_V2_KERNEL_BACKEND=hip MEGA_DISPATCH=mori \
    torchrun --standalone --nproc_per_node=4 \
      op_tests/multigpu_tests/test_mega_moe_gfx1250.py \
      -e 384 -k 6 -hd 7168 -id 3072 \
      --layers 61 -tpr 512 --combine gather \
      --acc_verify 0 --profile_table 1

Token sweeps are per-op and overridable from the environment, because the ops
do not share a supported range: score_qk asserts out past batch 1024, a16w16's
launcher refuses operands over 4 GiB, and inverse_rope faults the GPU at its
largest tokens. AITER_BENCH_TOKENS sets the default for every op and
AITER_BENCH_TOKENS_<OP> overrides one:

    AITER_BENCH_TOKENS=1,128,512 \
    AITER_BENCH_TOKENS_INVERSE_ROPE=1,8,128 \
      python op_tests/bench_gfx1250_combo.py --dsv4

gfx1250's bundled CK does not compile, so the asm JIT modules must be built with
ENABLE_CK=0. The script sets it (before importing aiter) so a plain run just
works; an explicit env override still wins.
"""

import os

# Must be set BEFORE `import aiter` so the JIT build picks it up. setdefault =>
# an explicitly-exported ENABLE_CK from the caller is respected.
os.environ.setdefault("ENABLE_CK", "0")

# FlyDSL MoE env vars — must be set before importing aiter / moe test module.
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_GROUPED_DEBUG", "0")
os.environ.setdefault("FLYDSL_DUMP_IR", "1")
os.environ.setdefault("AITER_LOG_MORE", "1")
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1")
os.environ.setdefault("AITER_FORCE_GFX1250", "1")

import argparse
import contextlib
import itertools
import subprocess
import sys
import tempfile
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
    import test_f4gemm as gemm_mod
    import test_flydsl_grouped_gemm_gfx1250 as moe_mod
    import test_fmha_fwd_with_sink_asm as mha_mod  # has __main__ guard
    import test_mla_v4_kargpreld as mla_v4_kargpreld_mod
    import test_mxfp8fp4gemm as f8gemm_mod
    import test_opus_a16w16_gemm as a16w16_mod
    import torch
    from triton_tests.attention import test_mla_v4_triton as mla_v4_triton_mod

    import aiter
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.test_common import run_perftest

SUPPORTED_GFX = ["gfx1250"]
# The gfx1250 launcher addresses gmem with 32-bit descriptors.
_GMEM_DESCRIPTOR_LIMIT = 4 << 30


def _tokens(name, default=None):
    """Token sweep for one op, from the environment.

    Ops do not share a supported range -- some shapes are refused by a kernel
    heuristic, some fault the GPU -- and the useful range also shifts while
    chasing a regression. AITER_BENCH_TOKENS sets the sweep for every op and
    AITER_BENCH_TOKENS_<OP> overrides one, e.g.

        AITER_BENCH_TOKENS=1,128,512 \\
        AITER_BENCH_TOKENS_INVERSE_ROPE=1,8,128 \\
          python op_tests/bench_gfx1250_combo.py --dsv4

    Returns None when nothing is set and the op has no default of its own: the
    op then passes no shape flag at all and the UT sweeps its own default, which
    is the range its owner keeps working.
    """
    raw = os.environ.get(f"AITER_BENCH_TOKENS_{name.upper()}") or os.environ.get(
        "AITER_BENCH_TOKENS"
    )
    if raw:
        return tuple(int(t) for t in raw.replace(",", " ").split())
    return tuple(default) if default is not None else None


# The in-process ops call their UT per shape, so they need a sweep to iterate;
# keep one here. The ops that shell out pass no shape flag unless asked, letting
# each UT sweep the range its owner maintains.
_TOKENS = _tokens("all", (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 65536))
_INVERSE_ROPE_TOKENS = _tokens("inverse_rope")
_SCORE_QK_TOKENS = _tokens("score_qk")
_A8W8_BLOCKSCALE_TOKENS = _tokens("a8w8_blockscale")
_MLA_DECODE_TOKENS = _tokens("mla_v4_decode", (t for t in _TOKENS if t <= 1024))
_MLA_PREFILL_TOKENS = _tokens("mla_v4_prefill", (t for t in _TOKENS if t >= 1024))


def _int_quad(s):
    """Parse 'a,b,c,d' -> (int, int, int, int) — MLA v4 kargpreld shape tuples."""
    a, b, c, d = s.split(",")
    return int(a), int(b), int(c), int(d)


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


def _moe_stage_bytes(
    token, topk, model_dim, inter_dim, experts, aq_bpe, wq_bpe, use_g1u1=True
):
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
    bytes1 = (
        token * model_dim * aq + token * topk * n * bo + experts * n * model_dim * wq
    )
    bytes2 = (
        token * topk * inter_dim * aq
        + token * model_dim * bo
        + experts * model_dim * inter_dim * wq
    )
    return bytes1, bytes2


# Per-op column whitelists: keep shape identifiers + perf, drop the constant
# config/correctness columns @benchmark echoes (gfx/dtype/err/cos_diff/...).
_MHA_KEEP = [
    "dtype",
    "head_dim",
    "hq",
    "hk",
    "sq",
    "sk",
    "batch",
    "is_causal",
    "init",
    "asm us",
    "asm TFLOPS",
    "asm TB/s",
]
# Curated (head_dim, seqlen, is_causal) grid — hq=64, hk=8(d64)/4(d128), batch=1.
_MHA_SHAPES = [
    (head_dim, tokens, causal)
    for head_dim in (64, 128)
    for tokens in _TOKENS
    for causal in (True, False)
]
_MOE_KEEP = [
    "data_format",
    "act",
    "token",
    "model_dim",
    "inter_dim",
    "E",
    "topk",
    "pass",
    "gemm1_us",
    "gemm1 TFLOPS",
    "gemm1 TB/s",
    "gemm2_us",
    "gemm2 TFLOPS",
    "gemm2 TB/s",
    "total us",
    "total TFLOPS",
    "total TB/s",
    "kernel",
]
# Fixed kernel-bench config (mirrors test_flydsl_grouped_gemm_gfx1250.py --scenario kernel).
_MOE_DATA_FORMATS = ["a4w4", "a8w4"]
_MOE_CONFIG = {
    "experts": 96,
    "tokens": _TOKENS,
    "topk": 6,
    "model_dim": 7168,
    "inter_dim": 3072,
    "activation": "silu",  # ActivationType.Silu
    "use_bias": False,
}
_GEMM_KEEP = [
    "workload",
    "intype",
    "M",
    "N",
    "K",
    "apre",
    "outtype",
    "data_init",
    "scale_init",
    "knl_name",
    "asm us",
    "asm TFLOPS",
    "asm TB/s",
    "asm err",
    "asm result",
]
# gemm_a4w4 throughput square.
_GEMM_A4W4_SHAPES = [(tokens, 16384, 16384) for tokens in _TOKENS]

_F8GEMM_PERF_SHAPES = {
    "a8w8": [(tokens, 16384, 8192) for tokens in _TOKENS]
    + [(tokens, 1048576, 16384) for tokens in _TOKENS],
    "a8w4": [(tokens, 16384, 16384) for tokens in _TOKENS]
    + [(tokens, 1048576, 16384) for tokens in _TOKENS],
}
# Curated (gqa_ratio, batch, kv_seq_lens, num_kv_splits) grid for MLA v4 nm
# kernarg-preload perf (mirrors op_tests/test_mla_v4_kargpreld.py sweep subset).
_MLA_V4_KARGPRELD_SHAPES = [
    (64, 64, 256, 1),
    (64, 64, 256, 2),
    (64, 64, 256, 4),
    (64, 64, 512, 1),
    (64, 64, 512, 2),
    (64, 64, 512, 4),
    (64, 64, 1024, 1),
    (64, 64, 1024, 2),
    (64, 64, 1024, 4),
    (128, 64, 256, 1),
    (128, 64, 256, 2),
    (128, 64, 256, 4),
    (128, 64, 512, 1),
    (128, 64, 512, 2),
    (128, 64, 512, 4),
    (128, 64, 1024, 1),
    (128, 64, 1024, 2),
    (128, 64, 1024, 4),
] + [
    (gqa, tokens, kv_seq_lens, num_kv_splits)
    for gqa in (64, 128)
    for tokens in _MLA_DECODE_TOKENS
    if tokens != 64
    for kv_seq_lens in (256, 512, 1024)
    for num_kv_splits in (1, 2, 4)
]
_MLA_V4_DSV4_SHAPES = [
    (128, 512, kv_seq_lens, num_kv_splits)
    for kv_seq_lens in (256, 512, 1024)
    for num_kv_splits in (1, 2, 4)
] + [
    (128, tokens, kv_seq_lens, num_kv_splits)
    for tokens in _MLA_DECODE_TOKENS
    if tokens != 512
    for kv_seq_lens in (256, 512, 1024)
    for num_kv_splits in (1, 2, 4)
]
_MLA_V4_COMPARE_KEEP = [
    "dtype",
    "gqa_ratio",
    "batch",
    "kv_seq_lens",
    "num_kv_splits",
    "asm_s1",
    "triton_s1",
    "s1 triton/asm",
    "asm_s2",
    "triton_s2",
    "s2 triton/asm",
    "asm_tot",
    "triton_tot",
    "tot triton/asm",
]


@contextlib.contextmanager
def _capture():
    """Like _silence, but hand the block's fd-level output back to the caller.

    Yields a one-element list that holds the captured text once the block ends.
    Backed by a temp file rather than a pipe: an op that emits more than the
    pipe buffer (64K) would otherwise deadlock with nobody draining it.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    old1, old2 = os.dup(1), os.dup(2)
    box = []
    with tempfile.TemporaryFile(mode="w+") as tmp:
        try:
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)
            yield box
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old1, 1)
            os.dup2(old2, 2)
            os.close(old1)
            os.close(old2)
            tmp.seek(0)
            box.append(tmp.read())


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
    print(f"\n===== {name} =====")
    print(df.to_markdown(index=False))


# Compiler / logger / IR-dump chatter the child UTs interleave with results.
_NOISE = (
    "[flydsl.compile]",
    "[aiter INFO]",
    "[aiter WARNING]",
    "import [module_",
    "In file included from",
    "torch/distributed/run.py",
    "Building extension",
    "Emitting ninja",
    "hipcc",
    "warning:",
    "UserWarning",
    "_warn_once",
)


def _md_row(line):
    """Markdown table row emitted by a child UT."""
    return line.startswith("|")


def _quiet(line):
    """Any non-empty line that is not compiler/logger noise."""
    return bool(line.strip()) and not any(n in line for n in _NOISE)


def _lines(pred):
    """Adapt a per-line predicate into a block extractor."""
    return lambda lines: [ln for ln in lines if pred(ln)]


def _md_tables(*labels):
    """Keep the markdown tables, labelling each by the columns it carries.

    A child UT often emits several tables in a row with different columns and
    nothing saying which is which. `labels` is ((column, ...), title) pairs; the
    first entry whose columns all appear in a header row names that table.
    """

    def extract(lines):
        md = [ln for ln in lines if _md_row(ln)]
        out = []
        for i, line in enumerate(md):
            is_header = i + 1 < len(md) and set(md[i + 1]) <= set("|-: ")
            if is_header:
                title = next(
                    (t for cols, t in labels if all(c in line for c in cols)),
                    None,
                )
                if title:
                    out.append(f"\n----- {title} -----")
                elif out:
                    out.append("")
            out.append(line)
        return out

    return extract


def _isnum(field):
    """Does this field parse as a number (thousands separators allowed)?"""
    try:
        float(field.replace(",", ""))
    except ValueError:
        return False
    return True


def _md_kernel_table(lines):
    """Render a rank-major kernel table as markdown, keeping the summary lines.

    mega_moe prints '[cfg] ...' / '# MEGA-MOE ...' lines around a space-aligned
    'Name rank0 rank1 rank2 rank3 avg calls' table. Kernel names contain spaces
    ("void at::native::reduce_kernel<512, 1, ...>"), so split from the right:
    the column count is fixed even when the name is not.
    """
    out, rows, cols = [], [], None

    def flush():
        if cols and rows:
            out.append(pd.DataFrame(rows, columns=cols).to_markdown(index=False))
            rows.clear()

    for line in lines:
        if not _quiet(line):
            continue
        if line.startswith("Name") and "rank0" in line:
            flush()
            cols = line.rsplit(maxsplit=6)
            continue
        fields = line.rsplit(maxsplit=6)
        if cols and len(fields) == 7 and all(_isnum(f) for f in fields[1:]):
            rows.append(fields)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out).splitlines()


def _md_from_pandas(marker, columns):
    """Re-emit a pandas-printed block as markdown.

    Some UTs print their result with DataFrame.__str__ (space aligned, leading
    index column) right after a marker line, which reads nothing like the
    markdown every other op produces.
    """

    def extract(lines):
        for i, line in enumerate(lines):
            if line.strip() != marker or i + 2 >= len(lines):
                continue
            values = lines[i + 2].split()[1 : len(columns) + 1]
            if len(values) != len(columns):
                continue
            df = pd.DataFrame([values], columns=list(columns))
            return df.to_markdown(index=False).splitlines()
        return []

    return extract


def _md_from_space_table(header_col):
    """Re-emit a UT's own DataFrame.to_string() summary as markdown.

    Anchors on the header row carrying `header_col` and takes the aligned rows
    that follow, so the table survives the aiter trace fragments and compiler
    warnings interleaved before it.
    """

    def extract(lines):
        for i, line in enumerate(lines):
            if header_col not in line.split():
                continue
            cols = line.split()
            rows = []
            for follower in lines[i + 1 :]:
                fields = follower.split()
                if len(fields) != len(cols):
                    break
                rows.append(fields)
            if rows:
                return (
                    pd.DataFrame(rows, columns=cols)
                    .to_markdown(index=False)
                    .splitlines()
                )
        return []

    return extract


def _table_row(*headers):
    """Whitespace-aligned table: the header line plus its numeric rows."""

    def keep(line):
        if not _quiet(line):
            return False
        if any(h in line for h in headers):
            return True
        # A data row starts with a bare number; "100% |####|" (pip) does not.
        head = line.split(maxsplit=1)[0]
        return head.strip("-").replace(",", "").replace(".", "").isdigit()

    return keep


def _gpu_trace_rows(lines):
    """(kernel, calls, device_us) for every GPU row in a profiler fragment.

    Rows look like '<idx> <kernel> <cnt> <host_us> <device_us> <avg_us> CUDA <id>';
    the kernel name carries spaces, the trailing column count does not. host_us
    is 0 on GPU rows, so the time to report is device_us.
    """
    for line in lines:
        fields = line.rsplit(maxsplit=6)
        if len(fields) != 7 or fields[-2] != "CUDA":
            continue
        _, cnt, _host, device_us, _avg, _, _ = fields
        if not (_isnum(cnt) and _isnum(device_us)):
            continue
        head = fields[0].split(None, 1)
        name = head[1].strip() if len(head) == 2 and head[0].isdigit() else fields[0]
        if name:
            yield name, float(cnt.replace(",", "")), float(device_us.replace(",", ""))


def _kernel_names(lines):
    """Distinct GPU kernel names in the order they first appear."""
    names = []
    for name, _, _ in _gpu_trace_rows(lines):
        if name not in names:
            names.append(name)
    return names


def _kernel_digest(lines):
    """Which GPU kernels actually ran, from the trace fragments in the output.

    A table of microseconds does not say which code path produced them, so a
    silent fallback (or a shape that quietly picked another kernel) reads as a
    normal result. The profiler fragments name every kernel that reached the
    GPU -- roll them up so each op states what it actually ran.
    """
    total, calls = {}, {}
    for name, n, us in _gpu_trace_rows(lines):
        total[name] = total.get(name, 0.0) + us
        calls[name] = calls.get(name, 0.0) + n
    if not total:
        return []
    ranked = sorted(total, key=total.get, reverse=True)
    table = pd.DataFrame(
        [
            {"kernel": k, "calls": round(calls[k]), "device us": round(total[k], 1)}
            for k in ranked
        ]
    )
    return ["", "----- kernels on GPU -----"] + table.to_markdown(
        index=False
    ).splitlines()


_DEFAULT_EXTRACT = _md_tables()


def _run_child(name, cmd, cwd, env=None, extract=None, timeout=None, tail=30,
               kernels=True):
    """Run a child UT with its output captured and surface only its results.

    Child UTs print their own progress, aiter INFO lines and (with FlyDSL) a
    couple of thousand IR-dump lines, which buries the numbers. Capture all of
    it, echo what `extract` pulls out, and fall back to the tail of the output
    when the child fails or emits nothing recognisable.
    """
    extract = extract or _DEFAULT_EXTRACT
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, env=env, text=True, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except subprocess.TimeoutExpired as exc:
        captured = exc.output or ""
        print(f"\n===== {name} =====", flush=True)
        print(f"--- timed out after {timeout}s, last {tail} lines ---")
        print("\n".join(captured.splitlines()[-tail:]), flush=True)
        raise
    lines = proc.stdout.splitlines()
    rows = extract(lines)
    if kernels:
        rows = list(rows) + _kernel_digest(lines)
    print(f"\n===== {name} =====", flush=True)
    print("\n".join(rows) if rows else "(no result rows recognised)", flush=True)
    if proc.returncode != 0 or not rows:
        print(f"--- {name}: exit={proc.returncode}, last {tail} lines ---")
        print("\n".join(lines[-tail:]), flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout)


# --- per-op runners: sweep axes silently, then print one table ---


def run_mha(args):
    # perf-only fn (no torch ref): sq==sk, hq=64, hk=8(d64)/4(d128), batch=1.
    rows = []
    with _silence():
        for init in args.mha_init:
            for head_dim, seqlen, causal in _MHA_SHAPES:
                hk = 8 if head_dim == 64 else 4
                rows.append(
                    mha_mod.test_fmha_fwd_with_sink_asm_perf(
                        head_dim, 64, hk, seqlen, seqlen, 1, causal, init
                    )
                )
    for row in rows:
        if row is not None:
            row["dtype"] = "bf16"
    _print_table("mha (bf16)", rows, keep=_MHA_KEEP)


def run_moe(args):
    cfg = _MOE_CONFIG
    activation = moe_mod.ActivationType.Silu
    rows = []
    data_formats = ["a8w4"] if args.suite == "dsv4" else _MOE_DATA_FORMATS
    for tokens, fmt in itertools.product(cfg["tokens"], data_formats):
        with _capture() as box:
            moe_mod.set_data_format(fmt)
            metrics = moe_mod.run_moe(
                fmt,
                experts=cfg["experts"],
                tokens=tokens,
                topk=cfg["topk"],
                model_dim=cfg["model_dim"],
                inter_dim=cfg["inter_dim"],
                activation=activation,
                use_bias=cfg["use_bias"],
                kernel_bench=True,
                check_aot_cache=False,
                raise_on_fail=False,
            )
        # stage1 n = inter_dim*2 (gate+up for silu/swiglu GUGU layout).
        aq_bpe, wq_bpe = _MOE_BPE.get(fmt, (1, 1))
        flop1, flop2 = _moe_stage_flops(
            tokens,
            cfg["topk"],
            cfg["model_dim"],
            cfg["inter_dim"],
            use_g1u1=True,
        )
        bytes1, bytes2 = _moe_stage_bytes(
            tokens,
            cfg["topk"],
            cfg["model_dim"],
            cfg["inter_dim"],
            cfg["experts"],
            aq_bpe,
            wq_bpe,
            use_g1u1=True,
        )
        us1, us2 = metrics.get("gemm1_us"), metrics.get("gemm2_us")
        total_us = (us1 or 0) + (us2 or 0) if (us1 or us2) else None
        bw1, bw2, bwt = (
            _bw(bytes1, us1),
            _bw(bytes2, us2),
            _bw(bytes1 + bytes2, total_us),
        )
        rows.append(
            {
                "data_format": fmt,
                "act": cfg["activation"],
                "token": tokens,
                "model_dim": cfg["model_dim"],
                "inter_dim": cfg["inter_dim"],
                "E": cfg["experts"],
                "topk": cfg["topk"],
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
                "kernel": " + ".join(_kernel_names(box[0].splitlines())) or None,
            }
        )
    _print_table("flydsl_grouped_gemm (kernel, silu)", rows, keep=_MOE_KEEP)


def run_gemm(args):
    # Hardware throughput sweep only. Functional/UT mode belongs in the source
    # op test and is intentionally not exposed by this performance driver.
    init_pairs = [("constant", "constant"), ("uniform", "auto")]
    rows = []
    with _silence():
        for (M, N, K), (di, si), intype, outtype in itertools.product(
            _GEMM_A4W4_SHAPES,
            init_pairs,
            ["mxfp4", "nvfp4"],
            ["bf16", "fp8"],
        ):
            rows.append(
                gemm_mod.test_gemm(
                    intype,
                    M,
                    N,
                    K,
                    1,
                    outtype,
                    di,
                    si,
                    mode="perf",
                )
            )
    _print_table("gemm_a4w4 (perf)", rows, keep=_GEMM_KEEP)


def run_f8gemm(args):
    # Generic MXFP8 hardware sweep. DSv4's projection path uses the separate
    # a8w8_blockscale runner below, not this F8GEMM kernel family.
    rows = []
    with _silence():
        cases = [
            ("hardware", intype, M, N, K, di, si)
            for (di, si), intype in itertools.product(
                [("constant", "constant"), ("uniform", "auto")],
                ["a8w8", "a8w4"],
            )
            for M, N, K in _F8GEMM_PERF_SHAPES[intype]
        ]
        for workload, intype, M, N, K, di, si in cases:
            row = f8gemm_mod.test_gemm(
                intype,
                M,
                N,
                K,
                1,
                data_init=di,
                scale_init=si,
                mode="perf",
            )
            if row is not None:
                row["workload"] = workload
            rows.append(row)
    _print_table(f"mxfp8fp4gemm ({args.suite})", rows, keep=_GEMM_KEEP)


def run_a8w8_blockscale(_args):
    """Run DSv4 FP8 blockscale linear projections at M=512."""
    _run_child(
        "gemm_a8w8_blockscale (DSv4)",
        [
            sys.executable,
            "op_tests/test_gemm_a8w8_blockscale.py",
            *(
                ["-m", *map(str, _A8W8_BLOCKSCALE_TOKENS)]
                if _A8W8_BLOCKSCALE_TOKENS
                else []
            ),
            "-nk",
            "2048,7168",
            "7168,16384",
            "6144,7168",
            "7168,3072",
            "65536,1536",
            "8192,1536",
            "--ck_preshuffle",
            "True",
            "--flydsl",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def run_a16w16(_args):
    """Run the DSv4 BF16 linear shapes through the Opus GEMM UT."""
    # test_a16w16 returns only the error; its timing is printed as
    #   [a16w16] batch=1 M=512 N=64 K=7168 dtype=... | 7.8us | 12.05 TFLOPs | err=0
    # so capture the block and parse that line back out.
    batch, K = 1, 7168
    rows = []
    for M, n in itertools.product(_TOKENS, (64, 384, 1024, 2048, 32320, 129280)):
        # The gfx1250 launcher addresses gmem with 32-bit descriptors, so the
        # heuristic refuses any operand past 4 GiB (opus_gemm_arch_gfx1250.cuh)
        # and raises instead of running. Skip those rather than fail the sweep.
        if max(M * K, n * K, M * n) * 2 > _GMEM_DESCRIPTOR_LIMIT:
            rows.append({"batch": batch, "M": M, "N": n, "K": K,
                         "err_msg": "skipped: >4GiB operand"})
            continue
        with _capture() as box:
            err = a16w16_mod.test_a16w16(batch=batch, M=M, N=n, K=K)
        captured = box[0].splitlines()
        row = {"batch": batch, "M": M, "N": n, "K": K, "err": err}
        for line in captured:
            if not line.startswith("[a16w16]"):
                continue
            fields = [f.strip() for f in line.split("|")]
            us = next((f for f in fields if f.endswith("us")), None)
            tflops = next((f for f in fields if f.endswith("TFLOPs")), None)
            row["us"] = float(us[:-2]) if us else None
            row["TFLOPS"] = float(tflops[:-7]) if tflops else None
            break
        # Which kernel served this shape: a16w16 switches between a splitk pair
        # and a 4wave_wl_co variant, and the timing alone does not say which.
        row["kernel"] = " + ".join(_kernel_names(captured)) or None
        rows.append(row)
    _print_table(
        "gemm_a16w16_opus (DSv4)",
        rows,
        keep=["batch", "M", "N", "K", "us", "TFLOPS", "kernel", "err"],
    )


def run_mega_moe(_args):
    """Run the four-rank DSv4 Mega MoE path vs its base combine, a4w4 and a8w4."""
    # The child ranks need GPU 0 as well. Release any cached allocations held by
    # this orchestration process before torchrun starts the four workers.
    torch.cuda.empty_cache()
    env = os.environ.copy()
    env.update({"MORI_V2_KERNEL_BACKEND": "hip", "MEGA_DISPATCH": "mori"})
    base_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        "op_tests/multigpu_tests/test_mega_moe_gfx1250.py",
        "-e",
        "384",
        "-k",
        "6",
        "-hd",
        "7168",
        "-id",
        "3072",
        "--layers",
        "61",
        "--acc_verify",
        "0",
        "--profile_table",
        "1",
    ]
    # AITER_FORCE_A8W4 selects the grouped kernel's ACTIVATION dtype (0 -> fp4,
    # 1 -> fp8); the weights are mxfp4 either way and -q only picks their layout,
    # so the env var and the quant key have to move together.
    for tokens, (quant, force_a8w4), (label, combine) in itertools.product(
        _TOKENS,
        (("a4w4_mxfp4", "0"), ("a8w4_mxfp4", "1")),
        (("non-Mega", "base"), ("Mega", "fused")),
    ):
        _run_child(
            f"mega_moe (tokens/rank={tokens}, {quant}, {label}, combine={combine})",
            [*base_cmd, "-tpr", str(tokens), "-q", quant, "--combine", combine],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env={**env, "AITER_FORCE_A8W4": force_a8w4},
            extract=_md_kernel_table,
                kernels=False,
        )


def run_mhc(_args):
    """Run the DSv4 mHC fused-RMSNorm benchmark at M=512, N=7168."""
    _run_child(
        "mhc (DSv4, fused RMSNorm)",
        [
            sys.executable,
            "op_tests/test_mhc.py",
            "-n",
            "7168",
            "-m",
            *map(str, _TOKENS),
            "--fuse_rmsnorm",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        extract=_md_tables(
            (("hip_nofuse_us",), "mhc: fused vs unfused RMSNorm"),
            (("unfused_us",), "mhc_post_pre"),
            (("hip_us",), "mhc_head"),
        ),
    )


def run_qk_norm(_args):
    """Run DSv4 QK norm + RoPE for prefill and decode token counts."""
    base_cmd = [
        sys.executable,
        "op_tests/test_flydsl_qk_norm_rope_quant.py",
        "--H",
        "128",
        "--D",
        "512",
        "--RD",
        "64",
        "--no-quant",
        "--qweight",
    ]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _run_child(
        "qk_norm",
        [*base_cmd, "-T", *map(str, _TOKENS)],
        cwd=repo_root,
        extract=_md_tables(
            (("quant_group_size",), "rope + quant"),
            (("rows_written",), "fused SWA write"),
        ),
    )


def run_score_qk(_args):
    """Run DSv4 decode score-QK at batch 512 for short and long CSA KV."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_cmd = [
        sys.executable,
        "op_tests/op_benchmarks/triton/bench_deepgemm_attention.py",
        "--heads",
        "64",
        "--index_dim",
        "128",
        "-mtp",
        "0",
        "--kv_preshuffle",
        "--blocksize",
        "64",
    ]
    # None => let the UT pick the batch, so run the KV lengths once each.
    for tokens, (label, kv_length) in itertools.product(
        _SCORE_QK_TOKENS or (None,), (("1K/1K average", "384"), ("long", "10240"))
    ):
        _run_child(
            f"score_qk (decode, B={tokens or 'UT default'}, {label} CSA KV={kv_length})",
            [
                *base_cmd,
                *(["--batch", str(tokens)] if tokens else []),
                "-kv_length",
                kv_length,
            ],
            cwd=repo_root,
            extract=_md_from_pandas(
                "paged_mqa_logits:",
                ("batch", "next_n", "heads", "index_dim", "avg_kv_len", "TFLOPS"),
            ),
        )


def run_mori_ep(_args):
    """Run MORI EPv2 dispatch/combine at the DSv4 MoE shape."""
    mori = os.environ.get("MORI", "/app/mori")
    # Bench whatever mori the image ships. Re-fetching main and reinstalling it
    # moved the measurement target between runs, and the rebuild needs a dev
    # ROCm toolchain that the pip-wheel images do not have.
    # for command in (
    #     ["git", "fetch", "origin", "main"],
    #     ["git", "switch", "main"],
    #     ["git", "pull", "--ff-only", "origin", "main"],
    #     [sys.executable, "-m", "pip", "install", "."],
    # ):
    #     subprocess.run(command, cwd=mori, check=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{mori}/python:{mori}"
    env["MORI_SOCKET_IFNAME"] = "lo"
    env["GLOO_SOCKET_IFNAME"] = "lo"
    env["PYTHONUNBUFFERED"] = "1"
    backend = env.get("BACKEND", "hip")
    env.update(
        {
            "BACKENDS": backend,
            "MORI_V2_KERNEL_BACKEND": backend,
            "HIDDEN": env.get("HIDDEN", "7168"),
            "TOPK": env.get("TOPK", "6"),
            "EPR": env.get("EPR", "96"),
            "SWEEP": env.get(
                "TOKENS", "64,128,256,512,1024,2048,4096,8192,16384"
            ),
            "ITERS": env.get("ITERS", "200"),
            "WARMUP": "10",
            "MODES": env.get("MODES", "eager,graph"),
            "COMBINE_IN": env.get("COMBINE_IN", "inplace"),
            "DISP": env.get("DISP", "bf16"),
            "CHECK": env.get("CHECK", "1"),
            "DBN": "",
            "DWPB": "",
            "CBN": "",
            "CWPB": "",
        }
    )
    _run_child(
        "mori_ep (DSv4 dispatch/combine)",
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={env.get('EP', '4')}",
            "tests/python/ops/dispatch_combine_v2/bench_ep.py",
        ],
        cwd=mori,
        env=env,
        extract=_lines(_quiet),
        timeout=3600,
    )


def _perf_ratio(num, den):
    """triton/asm speed ratio as '1.03x'; 'nanx' when undefined."""
    if num is None or den is None or den == 0:
        return "nanx"
    return f"{num / den:.2f}x"


def _bench_mla_v4_asm_staged(gqa, batch, ctx, split_kv, num_iters, num_warmup):
    """Asm kernel (s1) + merge (s2) + total; lives in combo bench only."""
    mod = mla_v4_kargpreld_mod
    q_seq = 1
    assert (gqa, q_seq) in mod._SHIPPED_TILE_VARIANTS
    if split_kv > 1:
        min_split = ctx // split_kv
        assert (
            min_split >= 16
        ), f"smallest KV split = floor({ctx}/{split_kv}) = {min_split} < 16"

    device = "cuda"
    inputs = mod._build_bf16_inputs(
        batch=batch,
        kv_seq_lens=ctx,
        q_seq_logical=q_seq,
        seed=mod._SEED,
        gqa_ratio=gqa,
        attn_sink=True,
    )
    sm_scale = 1.0 / (mod._QUANT_D**0.5)
    q_packed, q_rope = mod._native_to_2buff_for_asm(inputs["q_bf16"])
    kv_packed, kv_rope = mod._native_to_2buff_for_asm(inputs["kv_bf16"])

    total_q = inputs["q_bf16"].size(0)
    num_seqs = inputs["qo_indptr"].size(0) - 1
    num_heads = mod.NUM_KV_HEADS * gqa
    output_buf = torch.empty(
        (total_q, gqa, mod.V_HEAD_DIM), dtype=dtypes.bf16, device=device
    )
    split_indptr = torch.tensor(
        [i * split_kv for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device=device,
    )
    logits_buf = torch.empty(
        (total_q, split_kv, num_heads, mod.V_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    lse_buf = torch.empty(
        (total_q, split_kv, num_heads, 1), dtype=torch.float32, device=device
    )
    valid_split_count = torch.empty((num_seqs,), dtype=torch.int32, device=device)

    common_kwargs = {
        "q": q_packed,
        "qrope": q_rope.contiguous(),
        "kv_buffer": kv_packed,
        "kvrope": kv_rope.contiguous(),
        "output": output_buf,
        "qo_indptr": inputs["qo_indptr"],
        "kv_indptr": inputs["kv_indptr"],
        "kv_page_indices": inputs["kv_page_indices"],
        "kv_last_page_lens": inputs["kv_last_page_lens"],
        "split_indptr": split_indptr,
        "max_seqlen_q": inputs["max_seqlen_q"],
        "sink": inputs["sink"],
        "sm_scale": sm_scale,
        "num_kv_splits": split_kv,
        "logits": logits_buf,
        "attn_lse": lse_buf,
    }
    perf = {"num_iters": num_iters, "num_warmup": num_warmup, "num_rotate_args": 1}

    _, us_k = run_perftest(
        aiter.mla_decode_v4_asm,
        q_packed,
        q_rope.contiguous(),
        kv_packed,
        kv_rope.contiguous(),
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_page_indices"],
        split_indptr,
        inputs["sink"],
        inputs["max_seqlen_q"],
        sm_scale,
        0,
        split_kv,
        logits_buf,
        lse_buf,
        output_buf,
        valid_split_count,
        int(split_kv > 1),
        inputs["kv_last_page_lens"],
        **perf,
    )
    _, us_tot = run_perftest(
        aiter.mla.mla_decode_fwd_v4_nm,
        out_16_nosplit=0,
        **common_kwargs,
        **perf,
    )
    asm_s2 = max(0.0, us_tot - us_k) if split_kv > 1 else 0.0
    return {
        "asm_s1": round(us_k, 2),
        "asm_s2": round(asm_s2, 2),
        "asm_tot": round(us_tot, 2),
    }


def run_mla_v4_decode(args):
    # Side-by-side asm (kargpreld) vs Triton sparse decode on the same shape grid.
    iters = args.mla_v4_kargpreld_iters
    warmup = args.mla_v4_kargpreld_warmup
    mla_v4_triton_mod._PERF["num_iters"] = iters
    mla_v4_triton_mod._PERF["num_warmup"] = warmup
    default_shapes = (
        _MLA_V4_DSV4_SHAPES
        if args.suite == "dsv4"
        else _MLA_V4_KARGPRELD_SHAPES
    )
    shapes = args.mla_v4_kargpreld_shapes or default_shapes
    rows = []
    with _capture() as box:
        for gqa, batch, ctx, split_kv in shapes:
            row = {
                "gqa_ratio": gqa,
                "batch": batch,
                "kv_seq_lens": ctx,
                "num_kv_splits": split_kv,
            }
            try:
                asm = _bench_mla_v4_asm_staged(gqa, batch, ctx, split_kv, iters, warmup)
                tri = mla_v4_triton_mod.test_mla_v4_triton_staged(
                    gqa_ratio=gqa,
                    batch=batch,
                    kv_seq_lens=ctx,
                    num_kv_splits=split_kv,
                )
                row.update(asm)
                row.update(tri)
                row["s1 triton/asm"] = _perf_ratio(row["triton_s1"], row["asm_s1"])
                row["s2 triton/asm"] = _perf_ratio(row["triton_s2"], row["asm_s2"])
                row["tot triton/asm"] = _perf_ratio(row["triton_tot"], row["asm_tot"])
            except (RuntimeError, AssertionError, ValueError) as exc:
                msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                row["err_msg"] = msg
            rows.append(row)
    for row in rows:
        row["dtype"] = "bf16"
    _print_table(
        "mla_v4 decode (bf16, asm vs triton)",
        rows,
        keep=_MLA_V4_COMPARE_KEEP,
    )
    print("\n".join(_kernel_digest(box[0].splitlines())), flush=True)


def run_inverse_rope(_args):
    """Run DSv4 inverse RoPE + group quant at the tp1 attention-output shape."""
    # -b is (n_local_heads, n_local_groups); 128,16 is V4-Pro at dp/tp1. The UT
    # defaults to the two smallest configs instead, which never reach the shape
    # the model runs, so name it explicitly.
    _run_child(
        "inverse_rope_group_quant (DSv4, tp1)",
        [
            sys.executable,
            "op_tests/test_inverse_rope_group_quant.py",
            "-b",
            "128,16",
            *(["-s", *map(str, _INVERSE_ROPE_TOKENS)] if _INVERSE_ROPE_TOKENS else []),
            "-l",
            "n32k4",
            "--group-size",
            "32",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def run_mla_v4_prefill(_args):
    """Run DSv4 FP8 prefill at M=16K across two pools and CSR modes."""
    for tokens in (16384, *_MLA_PREFILL_TOKENS):
        _run_child(
            f"mla_v4 prefill (FP8, M={tokens}, prefix=4096/16384)",
            [
                sys.executable,
                "op_tests/test_pa_sparse_prefill.py",
                "-n",
                str(tokens),
                "--h_q",
                "128",
                "-d",
                "512",
                "--total_pages",
                "4096",
                "16384",
                "--total_tokens",
                str(tokens),
                "--prec",
                "fp8",
                "--mode",
                "dense",
                "sparse",
                "--no-verify",
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            extract=_lines(_table_row("latency_us")),
        )


def run_mla_v4_prefill_fp8(_args):
    """Run the default gfx1250 MLA v4 sparse-prefill FP8 sweep."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    # Keep the UT's own -n default ([512, 1024, 2048, 4096]). N=65536 faults the
    # GPU (HSA_STATUS_ERROR_MEMORY_FAULT) after its case verifies, and because
    # the UT prints its summary table only at the very end, that fault costs the
    # whole sweep: the smaller shapes run fine but never get reported.
    _run_child(
        "mla_v4 prefill FP8 (sparse-prefill default sweep)",
        [sys.executable, "op_tests/test_pa_sparse_prefill.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        extract=_md_from_space_table("n_tokens"),
    )


OPS = {
    "mha": run_mha,
    "moe": run_moe,
    "gemm": run_gemm,
    "f8gemm": run_f8gemm,
    "a8w8_blockscale": run_a8w8_blockscale,
    "a16w16": run_a16w16,
    "mla_v4_decode": run_mla_v4_decode,
    "inverse_rope": run_inverse_rope,
    "mla_v4_prefill": run_mla_v4_prefill,
    "mla_v4_prefill_fp8": run_mla_v4_prefill_fp8,
    "mhc": run_mhc,
    "qk_norm": run_qk_norm,
    "score_qk": run_score_qk,
    # "mori_ep" benchmarks mori, not aiter, and building it needs a dev ROCm
    # toolchain the pip-wheel images do not ship. run_mori_ep is kept below.
    # "mori_ep": run_mori_ep,
    "mega_moe": run_mega_moe,
}
# "gemm" (f4gemm a4w4) stays out: two back-to-back cases differing only in
# outtype (bf16 -> fp8) abort with HSA_STATUS_ERROR_MEMORY_FAULT, while each
# passes in a fresh process -- state leaking across cases, not a kernel bug.
#
# f8gemm has been seen reporting "0 us" / "inf TFLOPS" on part of its sweep.
# run_perftest times through the torch profiler, so a row like that means the
# profiler recorded no GPU work, not that the kernel was fast -- read it
# alongside the kernel digest, which says whether anything reached the GPU.
PERF_OPS = ["mha", "moe", "f8gemm", "mla_v4_decode"]
# "mori_ep" is out of the default sweep: it benchmarks mori rather than aiter,
# and it needs a dev ROCm toolchain to build. Run it explicitly when wanted.
DSV4_OPS = [
    "mega_moe",
    "moe",
    # "a8w8_blockscale" hits an intermittent HSA_STATUS_ERROR_MEMORY_FAULT on
    # gfx1250 at the (m=512, n=7168, k=16384) shape -- it passed one run and
    # aborted the next with the same binary.
    # "a8w8_blockscale",
    "a16w16",
    "mla_v4_decode",
    "inverse_rope",
    "mla_v4_prefill",
    "mla_v4_prefill_fp8",
    "mhc",
    "qk_norm",
    "score_qk",
]


def main():
    if get_gfx() not in SUPPORTED_GFX:
        print(
            f"combo bench targets {SUPPORTED_GFX} only; current {get_gfx()} — skipping"
        )
        return

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="combined gfx1250 asm-kernel perf bench (prints only summaries)",
    )
    suite = p.add_mutually_exclusive_group(required=True)
    suite.add_argument(
        "--perf",
        action="store_true",
        help=f"run hardware-oriented benchmarks (default ops: {', '.join(PERF_OPS)})",
    )
    suite.add_argument(
        "--dsv4",
        action="store_true",
        help=(
            "run the DeepSeek-V4 fixed-shape suite "
            f"(default ops: {', '.join(DSV4_OPS)})"
        ),
    )
    p.add_argument(
        "--ops",
        nargs="*",
        choices=list(OPS),
        default=None,
        help="select operations from the active suite (default: suite defaults)",
    )
    # mha (SWA fwd asm) — fixed 4-shape grid; init sweep only
    p.add_argument(
        "--mha-init",
        type=str,
        nargs="*",
        default=["randn", "const0.25"],
        choices=["randn", "const0.25"],
    )
    # flydsl moe — fixed kernel-bench config (see _MOE_CONFIG)
    # mla_v4 (v4 nm kernarg-preload decode) axes
    p.add_argument(
        "--mla-v4-kargpreld-shapes",
        type=_int_quad,
        nargs="*",
        default=None,
        metavar="GQA,BATCH,CTX,SPLIT",
        help="Override curated shape grid as gqa,batch,ctx,split tuples "
        "(default: suite-specific built-in grid)",
    )
    p.add_argument(
        "--mla-v4-kargpreld-iters",
        type=int,
        default=50,
        help="mla_v4_kargpreld timed iterations (default: 50)",
    )
    p.add_argument(
        "--mla-v4-kargpreld-warmup",
        type=int,
        default=2,
        help="mla_v4_kargpreld warmup iterations (default: 2)",
    )
    args = p.parse_args()

    args.suite = "dsv4" if args.dsv4 else "perf"
    default_ops = DSV4_OPS if args.dsv4 else PERF_OPS
    selected_ops = args.ops or default_ops
    invalid_ops = sorted(set(selected_ops) - set(default_ops))
    if invalid_ops:
        p.error(f"{args.suite} does not provide ops: {', '.join(invalid_ops)}")
    for name in selected_ops:
        OPS[name](args)


if __name__ == "__main__":
    main()
