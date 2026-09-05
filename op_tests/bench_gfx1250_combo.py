# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Combined gfx1250 asm-kernel perf bench.

Imports the top-level @benchmark sweep fns from the aiter op_tests (which the
aiter-op-test skill keeps importable for exactly this kind of combination
testing) and runs each over its own shape axes.

Output discipline: combo-owned pandas summaries are printed as record-oriented
JSON. Existing child-UT summaries are extracted without changing those UTs. All
the underlying noise (per-config "calling ..." logs, JIT build output, aiter
import banners, pandas/torch/ROCTracer warnings, including C-level fd writes) is
silenced via os-level fd redirection while the kernels run.

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
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mhc       # mHC fused RMSNorm
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops qk_norm   # QK norm + RoPE
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops score_qk  # FP8 paged MQA logits
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mori_ep   # MORI EPv2 dispatch/combine
    python op_tests/bench_gfx1250_combo.py --dsv4 --ops mega_moe  # Mega on/off, 4 GPUs

Environment
-----------

One variable, applied to every op that sweeps a token count:

    AITER_BENCH_TOKENS=1,128,512 python op_tests/bench_gfx1250_combo.py --dsv4

Leave it unset and every op runs the default in this file, which is the tested
configuration -- the shapes below are what the suites are expected to pass on.
Set it and it wins everywhere, with no second-guessing: an explicit request is
the caller's to make, including for shapes an op is known to fail.

The defaults are not one list, because a single token count does not mean the
same thing to every op:

    mla_v4_decode   1..1024. Decode carries one token per sequence, so the
                    axis is really the batch, and 65536 is not a shape the
                    model runs.
    inverse_rope    1..16384. The axis is -s at a fixed -b 128,16, and 65536
                    faults -- in the triton reference the UT compares against,
                    not in the kernel under test.
    mega_moe        1..2048. 65536 cannot allocate its symmetric arena; see
                    _MEGA_MOE_TOKENS.
    a8w8_blockscale 512..65536. M=512 covers a DSv4 decode batch of 512;
                    smaller M stays out because of a UT bug; see DSV4_OPS.
    mla_v4_prefill  1024..16384, the DSv4 prefill chunk. 65536 faults; see
                    _MLA_PREFILL_TOKENS.

With the variable unset, the child-UT ops (score_qk, a8w8_blockscale) pass no
shape flag at all, so each UT sweeps the range its owner maintains. The
in-process ops (moe, a16w16, mha, mla_v4_prefill) iterate shapes here and take
their default from the module.

Other variables:

    ENABLE_CK=0                        set before importing aiter; the module
                                       already setdefault()s it.
    GPU_ARCHS / CU_NUM                 detected once here and exported to every
                                       child, so no child runs rocminfo. Four
                                       ranks starting at once contend for
                                       rocminfo's rocm_smi mutex and a rank can
                                       lose it outright -- see _pin_arch. Set
                                       either yourself and yours wins.

Optional GPU telemetry replays each already-prepared benchmark case in its own
sampling window, after its normal latency measurement:

    # ROCm ships the binding here without a setup.py; make it importable first.
    export PYTHONPATH=/opt/rocm/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}
    python op_tests/bench_gfx1250_combo.py --dsv4 \
      --smi-monitor --smi-device 0 --smi-interval 0.05 --smi-duration 1.0

Input initialization, compilation, correctness and warmup are outside the SMI
window. The monitor uses the Python ``amdsmi`` package and prints a case-tagged
min/mean/median/max table for clocks, power, temperature, activity and VRAM.
``mega_moe`` has every rank monitor its local GPU around synchronized graph
replays, then gathers the four summaries to rank 0; ``mori_ep`` remains disabled
until its dispatch/combine loop exposes an aligned telemetry window.

Supported operator inputs can be overridden consistently with:

    --data-init zero|constant|uniform|norm [more ...]
    --scale-init zero|constant|uniform|norm|auto|pow2_binomial [more ...]
    --seed N

Operators whose underlying UT has no configurable initializer print an explicit
notice when these flags are supplied; the setting is never silently claimed.
The current passthrough matrix is:

    DATA + SCALE + seed   moe, gemm, f8gemm, a8w8_blockscale
    DATA + seed           a16w16, mega_moe, mhc, qk_norm, inverse_rope,
                          score_qk, mla_v4_decode, mla_v4_prefill
    DATA mapping only     mha (norm -> randn, constant -> const0.25)
    native init only      mori_ep

``--scale-init`` is reported as not applicable for operators without a scale
operand.

mega_moe at tokens/rank=65536 fails in setup(), asking 7.5 GB for cco's VMM
arena against a 4 GiB default. MORI_SHMEM_HEAP_SIZE does not reach that arena
(see run_mega_moe), so exporting it changes nothing -- and exporting it
sweep-wide takes the machine down, because that heap is preallocated per rank
for every case. The tier is out of the sweep; fixing it means passing
per_rank_vmm at Communicator.init().

Failures do not stop the sweep: a case that aborts is recorded and the run
moves to the next one, with a "N failed, M ops selected" list at the end and a
non-zero exit code. A GPU fault inside this process is the exception -- it
takes the interpreter down and no handler runs, which is why the child-UT ops
are the ones that survive their own crashes.

``--ops`` accepts any op, including one held out of a suite's defaults because
it is broken on the current arch, so it can be re-checked on a newer image.

The ``mori_ep`` op runs the EPv2 benchmark from ``${MORI:-/app/mori}`` as the
image provides it -- this script never updates or installs mori. Environment
variables select backend, token tiers, eager/graph modes, EP size, dispatch
dtype, and correctness checking:

    TOKENS=512 MODES=graph \
      python op_tests/bench_gfx1250_combo.py --dsv4 --ops mori_ep

It sweeps two dispatch wires by default, bf16 and fp4, one child process each
(bench_ep.py reads $DISP at import). fp4 is the wire DSv4 serves on; bf16 is
the reference. $DISP overrides, comma-separated -- DISP=fp8 or DISP=bf16,fp8,fp4.
Note that mori disables its own correctness check on fp4, so those rows are
unchecked rather than verified; the table label says so.

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

score-QK is a decode op, so its KV length is the average context one decode
step scans -- input + output/2 -- after CSA's 4x KV compression:

    1K in / 1K out   -> (1024  + 512)  / 4 =   384
    16K in / 4K out  -> (16384 + 2048) / 4 =  4608
    32K in / 16K out -> (32768 + 8192) / 4 = 10240

The DSv4 ``mla_v4_decode`` op runs sparse decode with GQA/H=128, batch=512 and
q_seq=1 (M=512), sweeping KV lengths 256/512/1024 and split counts 1/2/4.

The DSv4 ``mla_v4_prefill`` op runs eight performance cases at H=128 and
D=512: compressed prefix-pool rows 4096/16384, crossed with dense/sparse CSR
modes, crossed with fp8/bf16. The current chunk remains uncompressed:

    python3 op_tests/test_pa_sparse_prefill.py \
      -n <token sweep> --h_q 128 -d 512 \
      --total_pages 4096 16384 --total_tokens <token sweep> \
      --prec fp8 bf16 --mode dense sparse --no-verify

The UT compares the backends it has for each precision -- opus and asm on fp8,
opus and triton on bf16 -- so one run covers both precisions and all three
backends. There is no nnz axis to sweep: the CSR is generated from --mode
(sparse draws a random nnz per row, dense fills every row) under --seed, so
nnz is an outcome, not an input.

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

Token sweeps come from one variable, AITER_BENCH_TOKENS (see Environment
above). Unset, each op runs its own default -- the ops do not share a supported
range, so those defaults differ and each says why at its constant. Set, it
applies to every op that sweeps tokens, and the file does not argue with it.

``a16w16`` is not one of them: its range is a function of what opus has tuned,
not a fixed limit. Shapes with no tuned winner fall back to a split-K kid whose
launcher is 32-bit gmem-descriptor bound, which is both slow and, at M=65536,
wrong. Re-tuning through csrc/gemm_a16w16/gemm_a16w16_tune.py --libtype opus is
what widens the range, so the bench predicts nothing and reports what it gets.

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
import json
import subprocess
import sys
import tempfile
import warnings

from smi_monitor import SMI_RESULT_PREFIX

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
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.test_common import (
        DATA_DISTS,
        E8M0_SCALE_DISTS,
        make_generator,
        run_perftest,
    )

SUPPORTED_GFX = ["gfx1250"]
_SMI_ROWS = []


@contextlib.contextmanager
def _smi_case(label):
    """Set the case label consumed by the common perftest hook."""
    old = os.environ.get("AITER_SMI_LABEL")
    if os.environ.get("AITER_SMI_MONITOR") == "1":
        os.environ["AITER_SMI_LABEL"] = label
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("AITER_SMI_LABEL", None)
        else:
            os.environ["AITER_SMI_LABEL"] = old


def _without_smi(env):
    """Return a child environment with single-GPU telemetry disabled."""
    clean = env.copy()
    for key in tuple(clean):
        if key.startswith("AITER_SMI_"):
            clean.pop(key)
    return clean
# a16w16 N shapes at K=7168: attention/router projections, then lm_head twice
# (129280 is the DeepSeek vocab, 32320 is that sharded over TP4).
_A16W16_NS = (64, 384, 1024, 2048, 32320, 129280)
# lm_head cap. A shape judgement, not a kernel limit: lm_head runs one row per
# sequence, so M past this is not something the model produces, and M*N alone
# is 16 GB of bf16 output at (65536, 129280).
_A16W16_WIDE_N = 2048
_A16W16_WIDE_N_MAX_M = 2048
# a16w16 returns its own error ratio, and a wrong answer here is silent: the UT
# neither raises nor prints a warning. Measured on gfx1250 / 20260827, every
# shape that computed correctly came back 0 or ~1e-5, while M=65536 came back
# 0.96-0.99 on all four of its N -- an unrelated result, not a tolerance miss.
# Anything above this is reported as a failed op rather than printed as data.
_A16W16_MAX_ERR = 1e-2


def _tokens(default=None):
    """Token sweep from AITER_BENCH_TOKENS, else the op's own default.

    One variable for the whole bench. Ops whose axis is not a token count, or
    whose usable range is fixed, ignore it and pin their sweep in the source
    instead -- see _MLA_DECODE_TOKENS and _INVERSE_ROPE_TOKENS.

    Returns None when the variable is unset and the op has no default of its
    own: the op then passes no shape flag at all and the UT sweeps its own
    default, which is the range its owner keeps working.
    """
    raw = os.environ.get("AITER_BENCH_TOKENS")
    if raw:
        return tuple(int(t) for t in raw.replace(",", " ").split())
    return tuple(default) if default is not None else None


def _init_pairs(args, *, defaults):
    """Broadcast combo DATA/SCALE lists using the child-UT pairing contract."""

    data = (
        list(args.data_init)
        if args.data_init is not None
        else [pair[0] for pair in defaults]
    )
    scale = (
        list(args.scale_init)
        if args.scale_init is not None
        else [pair[1] for pair in defaults]
    )
    if len(data) == 1:
        data *= len(scale)
    if len(scale) == 1:
        scale *= len(data)
    if len(data) != len(scale):
        raise ValueError(
            "--data-init and --scale-init must have equal length "
            "(a length-1 side broadcasts)"
        )
    return tuple(zip(data, scale))


def _unsupported_init(args, op):
    """Make unsupported explicit init requests visible without blocking a sweep."""

    requested = []
    if args.data_init is not None:
        requested.append("--data-init")
    if args.scale_init is not None:
        requested.append("--scale-init")
    if requested:
        print(
            f"[data init] {op}: underlying UT does not support "
            f"{', '.join(requested)}; using its native initializer",
            flush=True,
        )


def _unused_scale_init(args, op):
    """Report a scale initializer passed to an op without a scale operand."""

    if args.scale_init is not None:
        print(
            f"[data init] {op}: --scale-init is not applicable; "
            "the operator has no scale input",
            flush=True,
        )


# The in-process ops call their UT per shape, so they need a sweep to iterate;
# keep one here. The ops that shell out pass no shape flag unless asked, letting
# each UT sweep the range its owner maintains.
_TOKENS = _tokens((1, 16, 32, 64, 128, 256, 512, 1024, 2048, 65536))
# a16w16's M is the token count, and the global sweep jumps 2048 -> 65536, so
# the prefill chunk sizes never got measured on the BF16 linears. Add them to
# this op's default; AITER_BENCH_TOKENS still overrides the whole thing.
_A16W16_MS = _tokens(tuple(sorted({*_TOKENS, 4096, 8192, 16384})))
# This axis is -s (sequence length) at a fixed -b 128,16, not the token count
# the other ops sweep, so the default is its own rather than _TOKENS. 65536 is
# left off it: that value faults, and in the triton reference the UT compares
# against rather than in the kernel under test. AITER_BENCH_TOKENS still wins if
# set -- what an explicit request sweeps is the caller's business.
_INVERSE_ROPE_TOKENS = _tokens((1, 16, 32, 64, 128, 256, 512, 1024, 2048, 16384))
_SCORE_QK_TOKENS = _tokens()
# score_qk is decode, so its KV length is the average context a decode step
# scans: input + output/2, then CSA's 4x compression.
#   1K in / 1K out  -> (1024  + 512)  / 4 =   384
#   16K in / 4K out -> (16384 + 2048) / 4 =  4608
#   32K in / 16K out-> (32768 + 8192) / 4 = 10240
_SCORE_QK_KV_LENGTHS = (
    ("1K/1K average", "384"),
    ("16K/4K average", "4608"),
    ("32K/16K average", "10240"),
)
# Was unset, which let the UT sweep its own 27-value default down to M=1. Two
# reasons to set it. First, M here is the token count of one step, so the small
# end of that default is decode batch and the large end is prefill chunk. This
# list retains the model-real decode point M=512, then covers the prefill side
# up to the 65536 the other DSv4 ops sweep and past the UT default's own ceiling
# of 10240. Second, the tiny M are what walk into
# the UT bug described at "a8w8_blockscale" below: get_CKGEMM_config retries the
# lookup as M -> get_padded_m(gl=0) -> nextPow2, so anything in [1, 16] or
# [33, 64] can land on one of #4773's M=16/M=64 gluon rows (gemm_common.cu:13).
# Starting at 512 clears both ranges by a wide margin.
#
# Two things remain outside coverage, both worth remembering: decode-side M
# below 512, and the 11 tuned rows that are the only shapes dispatching to
# gluon. This is a way around the UT bug, not a fix for it.
_A8W8_BLOCKSCALE_TOKENS = _tokens((512, 1024, 2048, 4096, 8192, 16384, 65536))
# Decode carries one token per sequence, so this axis is the batch, not a token
# count; past 1024 it stops being a shape the model runs, hence its own default
# rather than _TOKENS. AITER_BENCH_TOKENS overrides it like everywhere else.
_MLA_DECODE_TOKENS = _tokens((1, 16, 32, 64, 128, 256, 512, 1024))
# Up to 16384, the DSv4 prefill chunk. Re-measured on the #5084 UT (20260901,
# b45-2), one process per tier, all with the --no-verify below:
#   1024 .. 16384  clean, no coredump   (4096/8192/16384 faulted on the old UT)
#   65536          Memory access fault, and an 89 GB coredump with it
# 65536 stays out: it is past the chunk size the model prefills, and one fault
# costs a third of the host's free disk. Its fault also looks unrelated to the
# others -- address 0x7f2ddbec0000, a mapped high address, where the old UT's
# faults were low wild pointers like 0xc00000.
#
# "Clean" here means the kernel did not fault, NOT that it computed correctly.
# Correctness cannot be checked on gfx1250 at all right now: drop --no-verify
# and even n=1024 dies at the first case (fp8/dense, fault at 0x43000), so the
# reference or the comparison is what breaks, not the kernel under test. Until
# that is fixed these are timings from an unverified kernel -- the same footing
# as a16w16's M=65536 rows before _A16W16_MAX_ERR caught them.
_MLA_PREFILL_TOKENS = _tokens((1024, 2048, 4096, 8192, 16384))
# Default stops at 2048: tokens/rank=65536 dies in pipe.setup() building the
# symmetric arena -- cco sizes it from Communicator.DEFAULT_PER_RANK_VMM (4 GiB)
# and asks for 7.5 GB. That is a per_rank_vmm the UT never passes, not something
# MORI_SHMEM_HEAP_SIZE reaches, so the tier cannot run from here. Ask for it via
# AITER_BENCH_TOKENS anyway and you get it, along with that failure.
_MEGA_MOE_TOKENS = _tokens((1, 16, 32, 64, 128, 256, 512, 1024, 2048))
# What dispatch puts on the wire; combine is always bf16, so anything but bf16
# is an asymmetric pair. fp4 is the wire DSv4 actually serves on -- the receiver
# hands the payload straight to the expert GEMM as its A operand, and that GEMM
# is a4w4 (ATOM's serve script pins MEGA_WIRE=fp4, AITER_FORCE_A8W4=0) -- so
# measuring only bf16 measures a leg the model does not run. $DISP overrides,
# comma-separated, and is passed through unvalidated: mori owns the value set.
_MORI_EP_DISP = tuple(
    d.strip() for d in os.environ.get("DISP", "bf16,fp4").split(",") if d.strip()
)
# bench_ep.py forces its own correctness check off on fp4 ("fp4 combine is too
# lossy to compare"), so a passing fp4 row is unchecked, not verified. Labelled
# in the table rather than left for the reader to know that from mori's source.
_MORI_EP_UNCHECKED = ("fp4",)


def _int_quad(s):
    """Parse 'a,b,c,d' -> (int, int, int, int) — MLA v4 kargpreld shape tuples."""
    a, b, c, d = s.split(",")
    return int(a), int(b), int(c), int(d)


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
    "data_init",
    "scale_init",
    "seed",
    "act",
    "token",
    "model_dim",
    "inter_dim",
    "E",
    "topk",
    "pass",
    "gemm1_us",
    "gemm1 TFLOPS",
    "gemm1 GB/s",
    "gemm2_us",
    "gemm2 TFLOPS",
    "gemm2 GB/s",
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
    "seed",
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
    "data_init",
    "seed",
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
            _collect_smi_rows(box[0].splitlines())


def _collect_smi_rows(lines):
    """Collect structured per-case SMI records emitted by this or a child UT."""
    for line in lines:
        marker = line.find(SMI_RESULT_PREFIX)
        if marker < 0:
            continue
        try:
            record = json.loads(line[marker + len(SMI_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            continue
        base = {
            "case": record.get("label"),
            "rank": record.get("rank"),
            "device": record.get("device"),
            "duration_s": round(record.get("duration_s", 0.0), 3),
            "launches": record.get("launches"),
            "samples": record.get("samples"),
            "sample_status": record.get("sample_status"),
        }
        metrics = record.get("metrics", {})
        if not metrics:
            _SMI_ROWS.append({**base, "metric": "(no metrics)"})
        for metric, stats in metrics.items():
            _SMI_ROWS.append({**base, "metric": metric, **stats})


def _print_table(name, rows, keep=None):
    """Print one named DataFrame as a JSON object with record-oriented rows."""
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
    records = json.loads(df.to_json(orient="records"))
    print(json.dumps({"name": name, "rows": records}, indent=2), flush=True)


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
    seen = set()

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
        # Non-table lines are kept for the "[cfg] ..." summary, but the child
        # also emits "no grouped CSV config matched (...)" once per layer per
        # rank -- hundreds of byte-identical lines around one table. Keep the
        # first of each; a repeat carries nothing the first did not.
        if line in seen:
            continue
        seen.add(line)
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


def _space_table(header_col):
    """Keep a UT's own aligned summary table, verbatim.

    Anchors on the header row carrying `header_col` and takes the data rows that
    follow, so the table survives the trace fragments and compiler warnings
    interleaved before it.

    Emitted as the UT formatted it rather than rebuilt as markdown: pandas
    writes multi-word column names ("opus us", "asm TFLOPS"), so the header
    splits into 33 words against 21 data fields and cannot be mapped back to
    columns. A column name also appears in a UT's argument echo
    ("total_tokens = 1024,"), so require the next line to look like data.

    "Looks like data" counts numeric fields rather than testing the first one:
    the sparse-prefill table leads with prec/mode (bf16, dense), so a
    first-field test drops the whole table. An argument echo carries one
    number, a data row carries most of a row of them.
    """

    def is_data(fields):
        return sum(1 for f in fields if _isnum(f)) >= 4

    def extract(lines):
        for i, line in enumerate(lines):
            if header_col not in line.split() or i + 1 >= len(lines):
                continue
            first = lines[i + 1].split()
            if not first or not is_data(first):
                continue
            width = len(first)
            out = [line]
            for follower in lines[i + 1 :]:
                fields = follower.split()
                if len(fields) != width or not is_data(fields):
                    break
                out.append(follower)
            return out
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


_FAILURES = []


def _note_failure(label, why):
    """Record a dead case and let the sweep continue past it."""
    _FAILURES.append((label, why))
    print(f"--- {label}: FAILED ({why}), continuing ---", flush=True)


@contextlib.contextmanager
def _keep_going(label):
    """Op-level net for whatever _run_child cannot catch.

    A child UT that aborts is already handled inside _run_child; this catches
    the in-process ops raising Python exceptions. A GPU fault in this process
    is not recoverable -- it takes the interpreter with it, and no handler runs.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - a sweep must outlive one bad op
        _note_failure(label, f"{type(exc).__name__}: {exc}")
    finally:
        # A half-finished op can leave allocations behind; the next one should
        # not inherit them.
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not mask the failure
            pass


def _pin_arch(env):
    """Hand the child the arch we already know, so it never runs rocminfo.

    chip_info shells out to rocminfo twice -- once for the arch, once for the
    CU count -- and rocminfo takes a per-device rocm_smi mutex on the way in.
    One process is fine. A torchrun op starts four ranks at once, and they
    contend for that mutex: on 20260901/b45-1 a rank lost it and aborted
    ("init_mutex /rocm_smi_renderD128: unlock timed lock", surfacing as
    "Allgather operation failed" once the dead rank took the collective with
    it), and on b45-2 four rocminfo processes sat in it for minutes, one of
    them wedged in D state. Both cost a whole op; the nine single-GPU ops
    never noticed, because one process has nobody to contend with.

    GPU_ARCHS covers get_gfx_list, CU_NUM covers get_cu_num -- both are read
    from the environment before either shells out. Detected once here, in this
    process, where the call is serial. Not forced: an explicit setting from
    the caller wins.
    """
    env.setdefault("GPU_ARCHS", get_gfx())
    env.setdefault("CU_NUM", str(get_cu_num()))
    return env


def _run_child(name, cmd, cwd, env=None, extract=None, timeout=None, tail=30,
               kernels=True, smi=True):
    """Run a child UT with its output captured and surface only its results.

    Child UTs print their own progress, aiter INFO lines and (with FlyDSL) a
    couple of thousand IR-dump lines, which buries the numbers. Capture all of
    it, echo what `extract` pulls out, and fall back to the tail of the output
    when the child fails or emits nothing recognisable.
    """
    extract = extract or _DEFAULT_EXTRACT
    # Give every child invocation its combo-owned SMI case label. UTs remain
    # unaware of telemetry; the common perftest hook reads this environment.
    if smi and os.environ.get("AITER_SMI_MONITOR") == "1":
        env = os.environ.copy() if env is None else env.copy()
        env["AITER_SMI_LABEL"] = name
        old_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{cwd}{os.pathsep}{old_pythonpath}" if old_pythonpath else str(cwd)
        )
    # env=None means "inherit ours", which already carries these two.
    if env is not None:
        _pin_arch(env)
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
        _note_failure(name, f"timed out after {timeout}s")
        return
    lines = proc.stdout.splitlines()
    _collect_smi_rows(lines)
    # `results` decides whether the op reported anything; the kernel digest is
    # an annotation and must not stand in for a result table, or an extractor
    # that stops matching turns into a silent hole instead of a failure.
    results = extract(lines)
    rows = list(results) + (_kernel_digest(lines) if kernels else [])
    print(f"\n===== {name} =====", flush=True)
    print("\n".join(rows) if rows else "(no result rows recognised)", flush=True)
    if proc.returncode != 0 or not results:
        print(f"--- {name}: exit={proc.returncode}, last {tail} lines ---")
        print("\n".join(lines[-tail:]), flush=True)
    # Recorded, not raised: one dead shape used to take the rest of the sweep
    # with it -- a mega_moe case aborting at tokens/rank=65536 meant the seven
    # ops queued behind it never ran at all.
    if proc.returncode != 0:
        _note_failure(name, f"child exited {proc.returncode}")
    elif not results:
        _note_failure(name, "no result rows")


# --- per-op runners: sweep axes silently, then print one table ---


def run_mha(args):
    # perf-only fn (no torch ref): sq==sk, hq=64, hk=8(d64)/4(d128), batch=1.
    _unused_scale_init(args, "mha")
    if args.data_init is None:
        inits = args.mha_init
    else:
        init_map = {"norm": "randn", "constant": "const0.25"}
        unsupported = [dist for dist in args.data_init if dist not in init_map]
        if unsupported:
            raise ValueError(
                "mha only exposes randn and const0.25 initialization; "
                f"cannot map --data-init {' '.join(unsupported)}"
            )
        inits = [init_map[dist] for dist in args.data_init]
    rows = []
    with _silence():
        for init in inits:
            for head_dim, seqlen, causal in _MHA_SHAPES:
                with _smi_case(
                    f"mha/batch=1/hq=64/hk={8 if head_dim == 64 else 4}/"
                    f"sq={seqlen}/sk={seqlen}/d={head_dim}/causal={int(causal)}/"
                    f"data={init}"
                ):
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
    init_pairs = _init_pairs(args, defaults=(("uniform", "auto"),))
    for tokens, fmt, (data_init, scale_init) in itertools.product(
        cfg["tokens"], data_formats, init_pairs
    ):
        label = (
            f"moe/fmt={fmt}/tokens={tokens}/experts={cfg['experts']}/"
            f"topk={cfg['topk']}/hd={cfg['model_dim']}/id={cfg['inter_dim']}/"
            f"data={data_init}/scale={scale_init}/seed={args.seed}"
        )
        with _smi_case(label), _capture() as box:
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
                seed=args.seed,
                data_init=data_init,
                scale_init=scale_init,
                check_aot_cache=False,
                raise_on_fail=False,
            )
        us1, us2 = metrics.get("gemm1_us"), metrics.get("gemm2_us")
        rows.append(
            {
                "data_format": fmt,
                "act": cfg["activation"],
                "token": tokens,
                "model_dim": cfg["model_dim"],
                "inter_dim": cfg["inter_dim"],
                "E": cfg["experts"],
                "topk": cfg["topk"],
                "data_init": data_init,
                "scale_init": scale_init,
                "seed": args.seed,
                "pass": metrics["passed"],
                "gemm1_us": us1,
                "gemm1 TFLOPS": metrics.get("gemm1_tflops"),
                "gemm1 GB/s": metrics.get("gemm1_bandwidth_gbs"),
                "gemm2_us": us2,
                "gemm2 TFLOPS": metrics.get("gemm2_tflops"),
                "gemm2 GB/s": metrics.get("gemm2_bandwidth_gbs"),
                "kernel": " + ".join(_kernel_names(box[0].splitlines())) or None,
            }
        )
    _print_table("flydsl_grouped_gemm (kernel, silu)", rows, keep=_MOE_KEEP)


def run_gemm(args):
    # Hardware throughput sweep only. Functional/UT mode belongs in the source
    # op test and is intentionally not exposed by this performance driver.
    init_pairs = _init_pairs(
        args, defaults=(("constant", "constant"), ("uniform", "auto"))
    )
    rows = []
    with _silence():
        for (M, N, K), (di, si), intype, outtype in itertools.product(
            _GEMM_A4W4_SHAPES,
            init_pairs,
            ["mxfp4", "nvfp4"],
            ["bf16", "fp8"],
        ):
            with _smi_case(
                f"gemm_a4w4/intype={intype}/out={outtype}/M={M}/N={N}/K={K}/"
                f"data={di}/scale={si}/seed={args.seed}"
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
                        seed=args.seed,
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
                _init_pairs(
                    args,
                    defaults=(("constant", "constant"), ("uniform", "auto")),
                ),
                ["a8w8", "a8w4"],
            )
            for M, N, K in _F8GEMM_PERF_SHAPES[intype]
        ]
        for workload, intype, M, N, K, di, si in cases:
            with _smi_case(
                f"mxfp8fp4gemm/intype={intype}/M={M}/N={N}/K={K}/"
                f"data={di}/scale={si}/seed={args.seed}"
            ):
                row = f8gemm_mod.test_gemm(
                    intype,
                    M,
                    N,
                    K,
                    1,
                    data_init=di,
                    scale_init=si,
                    seed=args.seed,
                    mode="perf",
                )
            if row is not None:
                row["workload"] = workload
            rows.append(row)
    _print_table(f"mxfp8fp4gemm ({args.suite})", rows, keep=_GEMM_KEEP)


def run_a8w8_blockscale(args):
    """Run DSv4 FP8 blockscale linear projections at M=512."""
    # AITER_LOG_MORE=1 is set at module scope for the FlyDSL MoE ops, and a
    # child started with env=None inherits this process's whole environ. In this
    # UT that turned a clean sweep into an intermittent HSA memory fault, so
    # drop it for this child only -- every other op keeps it.
    env = os.environ.copy()
    env.pop("AITER_LOG_MORE", None)
    nk_shapes = (
        (2048, 7168),
        (7168, 16384),
        (6144, 7168),
        (7168, 3072),
        (65536, 1536),
        (8192, 1536),
    )

    def run_case(tokens, shapes, init_pairs, label):
        _run_child(
            label,
            [
                sys.executable,
                "op_tests/test_gemm_a8w8_blockscale.py",
                "-m",
                *map(str, tokens),
                "-nk",
                *(f"{n},{k}" for n, k in shapes),
                "--ck_preshuffle",
                "True",
                "--flydsl",
                "--data-init",
                *(data for data, _ in init_pairs),
                "--scale-init",
                *(scale for _, scale in init_pairs),
                "--seed",
                str(args.seed),
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
        )

    init_pairs = _init_pairs(
        args, defaults=(("constant", "constant"), ("uniform", "auto"))
    )
    if args.smi_monitor:
        for m, (n, k), pair in itertools.product(
            _A8W8_BLOCKSCALE_TOKENS, nk_shapes, init_pairs
        ):
            data_init, scale_init = pair
            run_case(
                (m,),
                ((n, k),),
                (pair,),
                f"a8w8_blockscale/M={m}/N={n}/K={k}/data={data_init}/"
                f"scale={scale_init}/seed={args.seed}",
            )
    else:
        run_case(
            _A8W8_BLOCKSCALE_TOKENS,
            nk_shapes,
            init_pairs,
            "gemm_a8w8_blockscale (DSv4)",
        )


def run_a16w16(args):
    """Run the DSv4 BF16 linear shapes through the Opus GEMM UT."""
    # test_a16w16 returns only the error; its timing is printed as
    #   [a16w16] batch=1 M=512 N=64 K=7168 dtype=... | 7.8us | 12.05 TFLOPs | err=0
    # so capture the block and parse that line back out.
    batch, K = 1, 7168
    rows = []
    _unused_scale_init(args, "a16w16")
    data_inits = args.data_init or ["norm"]
    generators = {dist: make_generator(args.seed) for dist in data_inits}
    for data_init, M, n in itertools.product(
        data_inits, _A16W16_MS, _A16W16_NS
    ):
        # N=32320/129280 is lm_head (the DeepSeek vocab, whole and TP4-sharded).
        # See _A16W16_WIDE_N: this is the one shape rule left here, and it is
        # about what DSv4 runs, not about what the kernel can do.
        if n > _A16W16_WIDE_N and M > _A16W16_WIDE_N_MAX_M:
            rows.append({"data_init": data_init, "seed": args.seed,
                         "batch": batch, "M": M, "N": n, "K": K,
                         "err_msg": f"skipped: N>{_A16W16_WIDE_N} is lm_head, "
                                    f"capped at M<={_A16W16_WIDE_N_MAX_M}"})
            continue
        # No >4 GiB pre-check. opus_dispatch_a16w16_gfx1250 tries the tuned
        # table FIRST and returns on a hit; check_shape_4g runs only after that
        # misses (opus_gemm_arch_gfx1250.cuh:161), on the way to the split-K
        # heuristic kid -- whose launcher is what builds the 32-bit gmem
        # descriptors. A tuned 4wave_wl_co winner never reaches it: that
        # pipeline addresses gmem through TDM descriptors, which clamp every
        # dimension and are not 32-bit bounded. So the limit belongs to one
        # fallback path, not to a16w16, and predicting it here would keep
        # skipping shapes that tuning has already made runnable. Let the kernel
        # raise and record that instead.
        try:
            with _smi_case(
                f"a16w16/batch={batch}/M={M}/N={n}/K={K}/"
                f"data={data_init}/seed={args.seed}"
            ), _capture() as box:
                err = a16w16_mod.test_a16w16(
                    batch=batch,
                    M=M,
                    N=n,
                    K=K,
                    dist=data_init,
                    gen=generators[data_init],
                )
        except Exception as exc:  # noqa: BLE001 - one shape must not end the sweep
            rows.append({"data_init": data_init, "seed": args.seed,
                         "batch": batch, "M": M, "N": n, "K": K,
                         "err_msg": f"{type(exc).__name__}: {exc}"})
            continue
        captured = box[0].splitlines()
        row = {"data_init": data_init, "seed": args.seed,
               "batch": batch, "M": M, "N": n, "K": K, "err": err}
        # float(): checkAllclose returns a bare 0 for a clean compare but a
        # numpy/torch scalar for a mismatch, and only one of those formats.
        if err is not None and float(err) > _A16W16_MAX_ERR:
            row["err_msg"] = (f"WRONG RESULT: err={float(err):g} "
                              f"> {_A16W16_MAX_ERR:g}")
            _note_failure(f"a16w16 M={M} N={n} K={K}", row["err_msg"])
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
        keep=["data_init", "seed", "batch", "M", "N", "K", "us", "TFLOPS", "kernel", "err"],
    )


def run_mega_moe(args):
    """Run the four-rank DSv4 Mega MoE path vs its base combine, a4w4 and a8w4."""
    # The child ranks need GPU 0 as well. Release any cached allocations held by
    # this orchestration process before torchrun starts the four workers.
    torch.cuda.empty_cache()
    env = os.environ.copy()
    # No MORI_SHMEM_HEAP_SIZE default here, for two independent reasons.
    #
    # Raising it sweep-wide took the machine down: the heap is preallocated per
    # rank for every case, not sized per case, so 16 GB became 64 GB reserved on
    # every one of them and b45-2 hard-rebooted at tokens/rank=512 -- long
    # before the case it was meant to help.
    #
    # And it would not have helped anyway. The 7.5 GB request at 65536 goes to
    # cco's VMM arena, not the shmem heap: "ccoMemAlloc: slot exhausted ... in
    # perRankSize=4294967296. Increase perRankVmmSize at ccoCommCreate". That
    # size is a ccoCommCreate argument with no environment variable behind it
    # (Communicator.DEFAULT_PER_RANK_VMM, 4 GiB), and
    # test_mega_moe_gfx1250.py:512 calls Communicator.init() without passing it.
    # MORI_SHMEM_HEAP_SIZE is read only in mori/src/shmem/init.cpp and feeds a
    # different allocator. The same error also prints "Hint: Increase via
    # MORI_SHMEM_HEAP_SIZE" -- that hint is what points the wrong way.
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
    data_inits = args.data_init or [None]
    _unused_scale_init(args, "mega_moe")
    for tokens, (quant, force_a8w4), (label, combine), data_init in itertools.product(
        _MEGA_MOE_TOKENS,
        (("a4w4_mxfp4", "0"), ("a8w4_mxfp4", "1")),
        (("non-Mega", "base"), ("Mega", "fused")),
        data_inits,
    ):
        init_label = data_init or "native-default"
        _run_child(
            f"mega_moe (tokens/rank={tokens}, {quant}, {label}, "
            f"combine={combine}, init={init_label}, seed={args.seed})",
            [
                *base_cmd,
                "-tpr",
                str(tokens),
                "-q",
                quant,
                "--combine",
                combine,
                *(["--data-init", data_init] if data_init else []),
                "--seed",
                str(args.seed),
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env={**env, "AITER_FORCE_A8W4": force_a8w4},
            extract=_md_kernel_table,
            kernels=False,
        )


def run_mhc(args):
    """Run the DSv4 mHC fused-RMSNorm benchmark at M=512, N=7168."""
    _unused_scale_init(args, "mhc")
    def run_case(tokens, data_inits, label):
        _run_child(
            label,
            [
                sys.executable,
                "op_tests/test_mhc.py",
                "-n",
                "7168",
                "-m",
                *map(str, tokens),
                "--fuse_rmsnorm",
                "--data-init",
                *data_inits,
                "--seed",
                str(args.seed),
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            extract=_md_tables(
                (("hip_nofuse_us",), "mhc: fused vs unfused RMSNorm"),
                (("unfused_us",), "mhc_post_pre"),
                (("hip_us",), "mhc_head"),
            ),
        )

    data_inits = args.data_init or ["norm"]
    if args.smi_monitor:
        for m, data_init in itertools.product(_TOKENS, data_inits):
            run_case(
                (m,),
                (data_init,),
                f"mhc/M={m}/N=7168/fuse_rmsnorm=1/data={data_init}/seed={args.seed}",
            )
    else:
        run_case(_TOKENS, data_inits, "mhc (DSv4, fused RMSNorm)")


def run_qk_norm(args):
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
    _unused_scale_init(args, "qk_norm")
    data_inits = args.data_init or [None]

    def run_case(tokens, data_init):
        qk_init = "normal" if data_init == "norm" else data_init
        _run_child(
            f"qk_norm/T={','.join(map(str, tokens))}/H=128/D=512/RD=64/"
            f"qweight=both/swa=direct,paged/init={qk_init or 'native-default'}/"
            f"seed={args.seed}",
            [
                *base_cmd,
                "-T",
                *map(str, tokens),
                *(["--init", qk_init] if qk_init else []),
                "--seed",
                str(args.seed),
            ],
            cwd=repo_root,
            extract=_md_tables(
                (("quant_group_size",), "rope + quant"),
                (("rows_written",), "fused SWA write"),
            ),
        )

    if args.smi_monitor:
        for token, data_init in itertools.product(_TOKENS, data_inits):
            run_case((token,), data_init)
    else:
        for data_init in data_inits:
            run_case(_TOKENS, data_init)


def run_score_qk(args):
    """Run DSv4 decode score-QK at batch 512 for short and long CSA KV."""
    _unused_scale_init(args, "score_qk")
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
    for tokens, (label, kv_length), data_init in itertools.product(
        _SCORE_QK_TOKENS or (None,),
        _SCORE_QK_KV_LENGTHS,
        args.data_init or ["norm"],
    ):
        _run_child(
            f"score_qk (decode, B={tokens or 'UT default'}, {label} "
            f"CSA KV={kv_length}, init={data_init}, seed={args.seed})",
            [
                *base_cmd,
                *(["--batch", str(tokens)] if tokens else []),
                "-kv_length",
                kv_length,
                "--data-init",
                data_init,
                "--seed",
                str(args.seed),
            ],
            cwd=repo_root,
            extract=_md_from_pandas(
                "paged_mqa_logits:",
                ("batch", "next_n", "heads", "index_dim", "avg_kv_len", "TFLOPS"),
            ),
        )


def run_mori_ep(args):
    """Run MORI EPv2 dispatch/combine at the DSv4 MoE shape."""
    _unsupported_init(args, "mori_ep")
    # Runs whatever mori the image provides; keeping it current is the image's
    # job. Updating it from here moved the measurement target between runs and
    # needed a dev ROCm toolchain the pip-wheel images do not ship.
    mori = os.environ.get("MORI", "/app/mori")
    env = _without_smi(os.environ)
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
            "CHECK": env.get("CHECK", "1"),
            "DBN": "",
            "DWPB": "",
            "CBN": "",
            "CWPB": "",
        }
    )
    # One child per wire: bench_ep.py reads $DISP once at import and builds the
    # transport for that dtype, so the tiers cannot share a process.
    for disp in _MORI_EP_DISP:
        env["DISP"] = disp
        note = " UNCHECKED" if disp in _MORI_EP_UNCHECKED else ""
        _run_child(
            f"mori_ep (DSv4 dispatch/combine, disp={disp}, combine=bf16{note})",
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
            smi=False,
        )


def _perf_ratio(num, den):
    """triton/asm speed ratio as '1.03x'; 'nanx' when undefined."""
    if num is None or den is None or den == 0:
        return "nanx"
    return f"{num / den:.2f}x"


def _bench_mla_v4_asm_staged(
    gqa, batch, ctx, split_kv, num_iters, num_warmup, data_init, seed
):
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
        seed=seed,
        data_init=data_init,
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
    _unused_scale_init(args, "mla_v4_decode")
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
    data_inits = args.data_init or ["norm"]
    rows = []
    with _capture() as box:
        for (gqa, batch, ctx, split_kv), data_init in itertools.product(
            shapes, data_inits
        ):
            row = {
                "data_init": data_init,
                "seed": args.seed,
                "gqa_ratio": gqa,
                "batch": batch,
                "kv_seq_lens": ctx,
                "num_kv_splits": split_kv,
            }
            try:
                with _smi_case(
                    f"mla_v4_decode/gqa={gqa}/batch={batch}/ctx={ctx}/"
                    f"split={split_kv}/data={data_init}/seed={args.seed}"
                ):
                    asm = _bench_mla_v4_asm_staged(
                        gqa,
                        batch,
                        ctx,
                        split_kv,
                        iters,
                        warmup,
                        data_init,
                        args.seed,
                    )
                    tri = mla_v4_triton_mod.test_mla_v4_triton_staged(
                        gqa_ratio=gqa,
                        batch=batch,
                        kv_seq_lens=ctx,
                        num_kv_splits=split_kv,
                        data_init=data_init,
                        seed=args.seed,
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


def run_inverse_rope(args):
    """Run DSv4 inverse RoPE + group quant at the tp1 attention-output shape."""
    _unused_scale_init(args, "inverse_rope")
    # -b is (n_local_heads, n_local_groups); 128,16 is V4-Pro at dp/tp1. The UT
    # defaults to the two smallest configs instead, which never reach the shape
    # the model runs, so name it explicitly.
    def run_case(tokens, data_inits, label):
        _run_child(
            label,
            [
                sys.executable,
                "op_tests/test_inverse_rope_group_quant.py",
                "-b",
                "128,16",
                "-s",
                *map(str, tokens),
                "-l",
                "n32k4",
                "--group-size",
                "32",
                "--data-init",
                *data_inits,
                "--seed",
                str(args.seed),
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    data_inits = args.data_init or ["norm"]
    if args.smi_monitor:
        for tokens, data_init in itertools.product(_INVERSE_ROPE_TOKENS, data_inits):
            run_case(
                (tokens,),
                (data_init,),
                f"inverse_rope/s={tokens}/heads=128/groups=16/layout=n32k4/"
                f"group_size=32/data={data_init}/seed={args.seed}",
            )
    else:
        run_case(
            _INVERSE_ROPE_TOKENS,
            data_inits,
            "inverse_rope_group_quant (DSv4, tp1)",
        )


def run_mla_v4_prefill(args):
    """Run DSv4 prefill across two precisions, pools and CSR modes."""
    _unused_scale_init(args, "mla_v4_prefill")
    data_inits = args.data_init or ["norm"]

    def run_case(tokens, pages, precs, modes, backends, init_values, label):
        _run_child(
            label,
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
                *map(str, pages),
                "--total_tokens",
                str(tokens),
                "--prec",
                *precs,
                # bf16 takes the single-tensor Q/K/V/O kernel; only fp8 has an
                # asm candidate, so the bf16 rows compare opus against triton
                # and leave the asm columns empty.
                "--mode",
                *modes,
                "--backend",
                *backends,
                "--no-verify",
                "--seed",
                str(args.seed),
                "--data-init",
                *init_values,
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            # Not _table_row: the UT has no "latency_us" column (it prints
            # "opus us"/"asm us"), so that predicate fell through to its
            # starts-with-a-number rule, swallowed the profiler's kernel lines
            # as data, and dropped the real header, which starts with "n".
            extract=_space_table("total_pages"),
        )

    if args.smi_monitor:
        backend_by_prec = {"fp8": ("opus", "asm"), "bf16": ("opus", "triton")}
        for tokens, pages, prec, mode, data_init in itertools.product(
            _MLA_PREFILL_TOKENS,
            (4096, 16384),
            ("fp8", "bf16"),
            ("dense", "sparse"),
            data_inits,
        ):
            for backend in backend_by_prec[prec]:
                run_case(
                    tokens,
                    (pages,),
                    (prec,),
                    (mode,),
                    (backend,),
                    (data_init,),
                    f"mla_v4_prefill/M={tokens}/H=128/D=512/pages={pages}/"
                    f"total_tokens={tokens}/prec={prec}/mode={mode}/backend={backend}/"
                    f"data={data_init}/seed={args.seed}",
                )
    else:
        for tokens in _MLA_PREFILL_TOKENS:
            run_case(
                tokens,
                (4096, 16384),
                ("fp8", "bf16"),
                ("dense", "sparse"),
                ("opus", "asm", "triton"),
                data_inits,
                f"mla_v4 prefill (M={tokens}, prec=fp8/bf16, pages=4096/16384)",
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
    "mhc": run_mhc,
    "qk_norm": run_qk_norm,
    "score_qk": run_score_qk,
    "mori_ep": run_mori_ep,
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
DSV4_OPS = [
    "mega_moe",
    # mori's own EPv2 bench, not an aiter kernel, but it is the dispatch and
    # combine either MoE path pays for -- the sweep is incomplete without the
    # two all2all legs beside the GEMMs. Reads the mori tree the image ships
    # (MORI=/app/mori); keeping that tree current is the image's job.
    "mori_ep",
    "moe",
    # "a8w8_blockscale" stays out of the default sweep, but NOT because the op is
    # dead on gfx1250 -- an earlier note here said that and it was wrong. The op
    # runs: 20260827, the six DSv4 (n,k) at -m 512 give err=0 at 1210-3564
    # TFLOPS, over ck / asm / flydsl.
    #
    # What kills it is one line of the UT. An earlier note here blamed #4773's
    # gluon tuning rows; that was wrong, and re-tuning would not have helped.
    # test_gemm_a8w8_blockscale.py:120 runs an extra "ck strided x_scale" check
    # on x_scale.transpose(0,1).contiguous().transpose(0,1) -- the same bytes
    # the measured call gets, but stride (1, M) instead of contiguous. #4406
    # added it to cover its own "honor strided x_scale" fix and gated it on
    # `if ck_preshuffle:` alone.
    #
    # Too wide a gate. The fp32 blockscale path does probe stride(0) != 1 to
    # learn the layout, so strided coverage means something there. The mxfp8_128
    # path this op runs (--flydsl --ck_preshuffle) does not: it declares the
    # layout with is_x_scale_transposed=True and never reads the stride
    # (gemm_op_a8w8.py:978-985 states the contract -- x_scale bytes are
    # column-major (K//128, M) inside a contiguous (M, K//128) tensor). So the
    # strided tensor exercises nothing real here; it only hands triton a
    # stride != 1 specialization that dies in make_llir.
    #
    # A/B 20260828, that line the only variable. The matrix is 162 cases: the
    # UT's 27-value default -m, times this op's six (n,k), M outer. As written
    # the sweep dies on case 2 (M=2, padded onto the M=16 row); with
    # .view(*x_scale.shape) it reaches case 160 -- so every M through 8192,
    # M=16 and M=64 among them. Those are exactly the M #4773's rows cover, so
    # the gluon kernel compiles and runs once the layout is right. (Case count
    # is derived from where the fault lands, not from a per-case log: _run_child
    # keeps only the last 30 lines, and a GPU fault is fatal, so reaching M's
    # 27th value is itself the proof the first 26 completed.)
    #
    # The patched sweep still ends in a GPU fault at its last M, but that is a
    # separate, older story: m=10240 n=7168 k=3072 run alone passes at err=0,
    # 1220 TFLOPS, split-K checks included. Same shape as a fresh process, so
    # state carried across cases -- see the f4gemm note above.
    #
    # The fix is upstream's call: that gate wants to be `ck_preshuffle and not
    # use_flydsl_fp8_scale`, matching how the asm/triton block below already
    # excludes this path. Narrowing the gate is the right shape of fix, not
    # rewriting the line as .view() -- the two calls differ only in stride, so
    # .view() would collapse them and drop coverage of the is_x_scale_tranposed
    # == False branch that #4406 added the line for. Evidence either way:
    # -m 16 -nk 2048,7168 --ck_preshuffle True passes the strided check with
    # the line untouched, and only adding --flydsl makes it crash.
    #
    # Back in the sweep because _A8W8_BLOCKSCALE_TOKENS now starts at 512,
    # which keeps every shape clear of the problematic tiny-M ranges while
    # retaining a real DSv4 decode batch. M=512 was verified above across all
    # six (n,k). The previous 1024..65536 sweep was verified on 20260828,
    # rocm/fw-bringup:gfx1250-atom--20260827-ubench: 36/36 cases, err=0 on all,
    # 2207-7003 TFLOPS. That run also clears M=10240, the shape the earlier
    # sweep faulted on -- more evidence that fault was cross-case state and not
    # the shape.
    "a8w8_blockscale",
    "a16w16",
    "mla_v4_decode",
    "inverse_rope",
    "mla_v4_prefill",
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
    # Before any child is spawned: children that inherit our environ (env=None)
    # get these too, not just the ones handed an explicit env. See _pin_arch.
    _pin_arch(os.environ)

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
        help=(
            "run these ops instead of the suite defaults. Any op is allowed, "
            "including ones held out of the defaults because they are broken "
            "on this arch (default: suite defaults)"
        ),
    )
    p.add_argument(
        "--data-init",
        nargs="+",
        choices=list(DATA_DISTS),
        default=None,
        help="override DATA initialization for supported ops",
    )
    p.add_argument(
        "--scale-init",
        nargs="+",
        choices=list(E8M0_SCALE_DISTS),
        default=None,
        help="override SCALE initialization for supported quantized ops",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed forwarded to supported ops (default: 0)",
    )
    p.add_argument(
        "--smi-monitor",
        action="store_true",
        help="replay and sample each timed benchmark case after latency measurement",
    )
    p.add_argument(
        "--smi-device",
        type=int,
        default=0,
        help="HIP device ordinal sampled by amdsmi (default: 0)",
    )
    p.add_argument(
        "--smi-interval",
        type=float,
        default=0.05,
        help="amdsmi sampling interval in seconds (default: 0.05)",
    )
    p.add_argument(
        "--smi-duration",
        type=float,
        default=1.0,
        help="minimum replay window in seconds for each benchmark case (default: 1.0)",
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
    if args.smi_device < 0:
        p.error("--smi-device must be non-negative")
    if args.smi_interval <= 0:
        p.error("--smi-interval must be positive")
    if args.smi_duration <= 0:
        p.error("--smi-duration must be positive")

    if args.smi_monitor:
        smi_output = tempfile.NamedTemporaryFile(
            prefix="aiter_smi_", suffix=".jsonl", delete=False
        )
        smi_output.close()
        os.environ.update(
            {
                "AITER_SMI_MONITOR": "1",
                "AITER_SMI_DEVICE": str(args.smi_device),
                "AITER_SMI_INTERVAL": str(args.smi_interval),
                "AITER_SMI_DURATION": str(args.smi_duration),
                "AITER_SMI_OUTPUT_PATH": smi_output.name,
                # Reference implementations timed by a few legacy UTs are not
                # hardware candidates and must not produce telemetry rows.
                "AITER_SMI_SKIP_FUNCTIONS": "run_torch,run_torch2",
            }
        )
    else:
        smi_output = None
        os.environ.pop("AITER_SMI_MONITOR", None)

    args.suite = "dsv4" if args.dsv4 else "perf"
    default_ops = DSV4_OPS if args.dsv4 else PERF_OPS
    # --ops selects from every op, not just the suite's defaults: an op pulled
    # out of the defaults because it is broken on this arch still has to be
    # runnable by name to check whether a newer image fixed it. argparse already
    # rejects names outside OPS.
    selected_ops = args.ops or default_ops
    for name in selected_ops:
        with _keep_going(name):
            OPS[name](args)

    if args.smi_monitor:
        try:
            with open(smi_output.name, encoding="utf-8") as output:
                _collect_smi_rows(output)
        finally:
            os.unlink(smi_output.name)
        _print_table(
            f"amdsmi per benchmark case (device={args.smi_device}, "
            f"interval={args.smi_interval}s, min_duration={args.smi_duration}s)",
            _SMI_ROWS,
            keep=[
                "case", "rank", "device", "duration_s", "launches", "samples",
                "sample_status",
                "metric", "min", "mean", "median", "max", "n",
            ],
        )

    if _FAILURES:
        print(f"\n===== {len(_FAILURES)} failed, "
              f"{len(selected_ops)} ops selected =====", flush=True)
        for label, why in _FAILURES:
            print(f"  {label}: {why}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
