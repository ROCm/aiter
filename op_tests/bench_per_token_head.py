"""Quick benchmark for FP8 PER_TOKEN_HEAD fmha_fwd on MI308X.

Calls fmha_fwd (run_fwd_per_token_head) and compares the measured numbers
against the H20 reference table baked into this script.

Sequence lengths to bench can be passed as positional args. Default is all
of {1024, 16384, 32768, 65536, 131072} (the H20 reference grid).

Examples:
    python op_tests/bench_per_token_head.py                  # all seqs
    python op_tests/bench_per_token_head.py 1024             # just one
    python op_tests/bench_per_token_head.py 1024 16384       # subset
    BENCH_VERIFY=1 python op_tests/bench_per_token_head.py 1024
"""

import argparse
import os
import sys

import torch

# Allow running as `python op_tests/bench_per_token_head.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import op_tests.test_batch_prefill as _tbp
from op_tests.test_batch_prefill import run_fwd_per_token_head

# The pytest helpers skip causal+soft_cap=0.0 on ROCm 7.2 + gfx950 out of an
# abundance of caution; the kernel actually runs fine in practice and we need
# this combo for the H20 reference grid. Allow bypassing for benchmarking.
if int(os.environ.get("BENCH_BYPASS_ROCM72_SKIP", "1")):
    _tbp.should_skip_rocm72_issue = lambda *a, **k: False


# (batch, qo_len, kv_len, nhq, nhk, head_dim, causal, soft_cap)
#
# H20 reference config (from ticket "H20 FP8+MTP Attention Kernel_perf.pdf"):
#   q_heads=8, kv_heads=1 (GQA ratio=8), head_dim=128, block_size=64,
#   quant_type=QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
#   q_len = kv_len, full causal prefill, batch (num_seqs) in {2, 4, 6, 8}.
#
# page_size is unused by fmha_fwd (non-paged) but kept here so the helper
# signature stays the same as the paged path.
PAGE_SIZE = int(os.environ.get("BENCH_PAGE_SIZE", "1024"))

# Enable correctness verification against the FP32 reference. Default off because
# the reference path materializes per-sequence SDPA in fp32 which is very slow
# for long sequences. Set BENCH_VERIFY=1 to turn it on.
VERIFY = bool(int(os.environ.get("BENCH_VERIFY", "0")))
#
# H20 TFLOPS reference table (rows = seq_len, cols = batch):
#                  batch=2   batch=4   batch=6   batch=8
#   seq_len=1024     146.2    169.4    186.7    188.7
#   seq_len=16384    266.5    269.8    270.1    268.0
#   seq_len=32768    269.6    272.3    272.5    270.3
#   seq_len=65536    270.8    273.6    273.6    271.5
#   seq_len=131072   272.8    274.2    272.1    272.1
H20_REF = {
    (2,   1024): 146.2, (4,   1024): 169.4, (6,   1024): 186.7, (8,   1024): 188.7,
    (2,  16384): 266.5, (4,  16384): 269.8, (6,  16384): 270.1, (8,  16384): 268.0,
    (2,  32768): 269.6, (4,  32768): 272.3, (6,  32768): 272.5, (8,  32768): 270.3,
    (2,  65536): 270.8, (4,  65536): 273.6, (6,  65536): 273.6, (8,  65536): 271.5,
    (2, 131072): 272.8, (4, 131072): 274.2, (6, 131072): 272.1, (8, 131072): 272.1,
}

# H20 latency reference (ms) from the same PDF "Prefill Latency (ms)" table.
H20_LAT_MS = {
    (2,   1024):   0.029, (4,   1024):   0.051, (6,   1024):   0.069, (8,   1024):   0.091,
    (2,  16384):   4.13,  (4,  16384):   8.15,  (6,  16384):  12.21,  (8,  16384):  16.41,
    (2,  32768):  16.31,  (4,  32768):  32.30,  (6,  32768):  48.43,  (8,  32768):  65.08,
    (2,  65536):  64.97,  (4,  65536): 128.61,  (6,  65536): 192.89,  (8,  65536): 259.22,
    (2, 131072): 257.92,  (4, 131072): 513.27,  (6, 131072): 775.93,  (8, 131072): 1034.50,
}

# Sequence lengths to bench. Default is the full H20 reference grid; the user
# can pass a subset (or single value) as positional args on the command line.
DEFAULT_SEQS = (1024, 16384, 32768, 65536, 131072)
_parser = argparse.ArgumentParser(
    description=(
        "Quick benchmark for FP8 PER_TOKEN_HEAD fmha_fwd on MI308X.\n"
        "Compares against the H20 reference table baked into this script\n"
        "(TFLOPS + latency from the H20 PDF)."
    ),
    epilog=(
        "examples:\n"
        "  # Smoke test: bench a single short seq, no verify, ~1s wallclock.\n"
        "  python op_tests/bench_per_token_head.py 1024\n"
        "\n"
        "  # Bench a subset of seq lengths with the H20 reference next to\n"
        "  # fmha_fwd for direct comparison.\n"
        "  python op_tests/bench_per_token_head.py 1024 16384\n"
        "\n"
        "  # Bench + correctness check (FP8 vs FP32 reference). Use this when\n"
        "  # you change the kernel and want both perf numbers AND a PASS/FAIL\n"
        "  # gate. Exits non-zero on failure, so it can be wired into CI.\n"
        "  BENCH_VERIFY=1 python op_tests/bench_per_token_head.py 1024 16384\n"
        "\n"
        "  # Full H20 grid run (5 seqs * 4 batches = 20 shapes). Slow on\n"
        "  # the long seqs; expect minutes.\n"
        "  python op_tests/bench_per_token_head.py\n"
        "\n"
        "sample output (one row, abbreviated):\n"
        "  Config: nhq=8, nhk=1, hd=128, causal=True, soft_cap=0.0\n"
        "  batch | PTH (fmha_fwd)             vrf | H20 (ref)              | H20/MI308\n"
        "  seq=1024\n"
        "      8 |    94 us   129.63 TFLOPS PASS |    91 us   188.70 TFLOPS |     0.97x\n"
        "  ...\n"
        "  Reading the row:\n"
        "    PTH (fmha_fwd) : measured on this MI308 (us + TFLOPS).\n"
        "    vrf            : PASS / FAIL only when BENCH_VERIFY=1.\n"
        "    H20 (ref)      : reference numbers from the H20 PDF.\n"
        "    H20/MI308      : H20 us / fmha_fwd us  (>1 = MI308 faster).\n"
        "\n"
        "environment variables:\n"
        "  BENCH_VERIFY=0|1            verify outputs vs FP32 reference\n"
        "                              (default 0; slow on long seqs)\n"
        "  BENCH_BYPASS_ROCM72_SKIP=1  bypass the causal+soft_cap=0 skip\n"
        "                              guard (default on; flip to 0 to\n"
        "                              honor the upstream guard)\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_parser.add_argument(
    "seqs",
    type=int,
    nargs="*",
    default=list(DEFAULT_SEQS),
    metavar="SEQ",
    help=(
        "sequence length(s) to bench (also used as qo_len = kv_len for full "
        "causal prefill). Pass one value for a smoke test or several to sweep. "
        "Defaults to the full H20 reference grid: "
        f"{' '.join(map(str, DEFAULT_SEQS))}."
    ),
)
_args = _parser.parse_args()
SEQS = tuple(_args.seqs)

SHAPES = [
    (b, s, s, 8, 1, 128, True, 0.0)
    for s in SEQS
    for b in (2, 4, 6, 8)
]


def fmt(r):
    if r.get("status") == "skipped":
        return "skip"
    if r.get("status") != "passed" or "tflops" not in r:
        return "n/a"
    return f"{r['time_us']:8.0f} us  {r['tflops']:7.2f} TFLOPS"


def fmt_h20(b, kv):
    """H20 reference rendered in the same '<us> us  <TFLOPS> TFLOPS' shape as
    the measured kernel columns."""
    h20_tf = H20_REF.get((b, kv))
    h20_ms = H20_LAT_MS.get((b, kv))
    if h20_tf is None or h20_ms is None:
        return "n/a"
    h20_us = h20_ms * 1000.0
    return f"{h20_us:8.0f} us  {h20_tf:7.2f} TFLOPS"


def verify_str(r):
    """Render verification status for a result dict."""
    v = r.get("verify")
    if v is None:
        return f"{'-':>6}"
    if v == "pass":
        return f"{'PASS':>6}"
    return f"{'FAIL':>6}"


def call_kernel(fn, **kwargs):
    """Run a kernel helper, capturing both bench numbers and verify status.

    Strategy when VERIFY=1: do two runs back-to-back so we always get clean
    timing even if verification fails.
      1) profile=True, skip_reference=True  -> bench numbers
      2) profile=False, skip_reference=False -> correctness check
    The reference build dominates runtime, so the extra kernel launch is noise.
    """
    bench_kwargs = dict(kwargs, profile=True, skip_reference=True)
    bench = fn(**bench_kwargs)
    if not VERIFY or bench.get("status") != "passed":
        return bench

    verify_kwargs = dict(kwargs, profile=False, skip_reference=False)
    try:
        fn(**verify_kwargs)
        bench["verify"] = "pass"
    except AssertionError as e:
        bench["verify"] = "fail"
        bench["error"] = str(e).splitlines()[0][:120]
    return bench


def _run_one(shape):
    b, qo, kv, nhq, nhk, hd, c, sc = shape
    common = dict(
        kvcache_layout="linear",
        table_layout="sglang",
        batch_size=b,
        qo_len=qo,
        kv_len=kv,
        page_size=PAGE_SIZE,
        num_qo_heads=nhq,
        num_kv_heads=nhk,
        head_dim=hd,
        causal=c,
        logits_soft_cap=sc,
        dtype=torch.bfloat16,
        contiguous_kv=True,
        seed=42,
    )
    return call_kernel(run_fwd_per_token_head, **common)


def _lat_ratio(r, h20_us):
    """H20 latency / MI308 latency = speedup factor (both in us).
    >1 means MI308 is faster than H20.
    """
    if r.get("status") != "passed" or h20_us is None or "time_us" not in r:
        return "-"
    return f"{h20_us / r['time_us']:.2f}x"


def _format_row(shape, fwd):
    b, qo, kv, nhq, nhk, hd, c, sc = shape
    h20_ms = H20_LAT_MS.get((b, kv))
    h20_us = h20_ms * 1000.0 if h20_ms is not None else None
    return (
        f"{b:>5} | "
        f"{fmt(fwd):>27} {verify_str(fwd)} | "
        f"{fmt_h20(b, kv):>27} | "
        f"{_lat_ratio(fwd, h20_us):>9}"
    )


def _run_silent(shape):
    """Run shape 0 with all stdout/stderr suppressed at the FD level.

    aiter prints '[aiter] import ...' / '[aiter] type hints mismatch ...' lines
    on first kernel call, and torch + ROCTracer emit one-time warnings via
    C-level logging. FD-level redirection captures both. Doing this for shape 0
    keeps the table header and data rows visually contiguous. On error we
    replay the captured output so the user sees what went wrong.
    """
    import tempfile

    capture = tempfile.TemporaryFile(mode="w+")
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)
    ok = True
    try:
        return _run_one(shape)
    except BaseException:
        ok = False
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        if not ok:
            capture.seek(0)
            sys.stderr.write(capture.read())
        capture.close()


# Trigger one-time JIT imports + first-call warnings BEFORE drawing the table,
# so the header and rows aren't split by aiter import lines / ROCTracer warnings.
print("Warming up kernels (one-time JIT setup, may take a moment)...", flush=True)
_fwd0 = _run_silent(SHAPES[0])

# Constants are pulled out of the table to keep it narrow. nhq/nhk/hd/causal/
# soft_cap are assumed constant across SHAPES; we sanity-check below.
_nhq, _nhk, _hd, _causal, _sc = SHAPES[0][3:8]
for _s in SHAPES:
    assert _s[3:8] == (_nhq, _nhk, _hd, _causal, _sc), (
        "Bench header assumes nhq/nhk/hd/causal/soft_cap are constant; "
        f"row {_s} differs from {SHAPES[0]}"
    )

if VERIFY:
    print("Verification: ON  (BENCH_VERIFY=1, FP8 outputs checked vs FP32 reference)")
else:
    print("Verification: OFF (set BENCH_VERIFY=1 to check outputs vs FP32 reference)")
print(
    f"Config: nhq={_nhq}, nhk={_nhk}, hd={_hd}, "
    f"causal={_causal}, soft_cap={_sc}"
)
_HEADER = (
    f"{'batch':>5} | "
    f"{'PTH (fmha_fwd)':>27} {'vrf':>6} | "
    f"{'H20 (ref)':>27} | "
    f"{'H20/MI308':>9}"
)
print(_HEADER)
print("-" * len(_HEADER))

failures = []


def _record_failures(shape, fwd):
    b, _qo, kv, *_ = shape
    if fwd.get("verify") == "fail":
        failures.append((b, kv, "PTH (fmha_fwd)", fwd.get("error", "")))


# Group shapes by seq so we print one "seq=N" sub-header per group instead of
# repeating seq on every row.
from collections import OrderedDict as _OrderedDict

_groups = _OrderedDict()
for _s in SHAPES:
    _groups.setdefault(_s[2], []).append(_s)

_warmup_done = False
for _seq, _shapes in _groups.items():
    print(f"seq={_seq}")
    for shape in _shapes:
        if shape == SHAPES[0] and not _warmup_done:
            fwd = _fwd0
            _warmup_done = True
        else:
            fwd = _run_one(shape)
        print(_format_row(shape, fwd), flush=True)
        _record_failures(shape, fwd)

if VERIFY:
    if failures:
        print("-" * len(_HEADER))
        print(f"VERIFY FAILED on {len(failures)} configuration(s):")
        for b, kv, label, err in failures:
            print(f"  batch={b} seq={kv} {label}: {err}")
        sys.exit(1)
    else:
        print("-" * len(_HEADER))
        print("VERIFY PASSED on all configurations.")
