"""Benchmark FP8 PER_TOKEN_HEAD batch_prefill kernel.

Sequence lengths can be passed as positional args.

Examples:
    python op_tests/bench_per_token_head.py 1024
    python op_tests/bench_per_token_head.py 1024 16384
    BENCH_VERIFY=1 python op_tests/bench_per_token_head.py 1024
    BENCH_KV_LAYOUT=vec_k_col_v python op_tests/bench_per_token_head.py 16384

Environment variables:
    BENCH_VERIFY=0|1            verify outputs vs FP32 reference (default 0)
    BENCH_PAGE_SIZE=N           paged-KV page size (default 1024)
    BENCH_BYPASS_ROCM72_SKIP=1  bypass causal+soft_cap=0 skip guard (default 1)
    BENCH_KV_LAYOUT=...         "linear" (default), "vectorized" (5D swizzled),
                                or "vec_k_col_v" (decode-aligned: 5D K + 4D ColumnMajor V)
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import op_tests.test_batch_prefill as _tbp
from op_tests.test_batch_prefill import run_batch_prefill_per_token_head

if int(os.environ.get("BENCH_BYPASS_ROCM72_SKIP", "1")):
    _tbp.should_skip_rocm72_issue = lambda *a, **k: False

PAGE_SIZE = int(os.environ.get("BENCH_PAGE_SIZE", "1024"))
VERIFY = bool(int(os.environ.get("BENCH_VERIFY", "0")))
# KV cache memory layout: "linear" (4D), "vectorized" (5D swizzled), or
# "vec_k_col_v" (decode-aligned: 5D vectorized K + 4D ColumnMajor V).
KV_LAYOUT = os.environ.get("BENCH_KV_LAYOUT", "linear")

DEFAULT_SEQS = (1024, 16384, 32768, 65536, 131072)
_parser = argparse.ArgumentParser(
    description="Benchmark FP8 PER_TOKEN_HEAD batch_prefill kernel.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_parser.add_argument(
    "seqs",
    type=int,
    nargs="*",
    default=list(DEFAULT_SEQS),
    metavar="SEQ",
    help="Sequence length(s) to bench (qo_len = kv_len). Default: all.",
)
_args = _parser.parse_args()
SEQS = tuple(_args.seqs)

SHAPES = [(b, s, s, 8, 1, 128, True, 0.0) for s in SEQS for b in (2, 4, 6, 8)]


def _fmt(r):
    if r.get("status") == "skipped":
        return "skip"
    if r.get("status") != "passed" or "tflops" not in r:
        return "n/a"
    return f"{r['time_us']:8.0f} us  {r['tflops']:7.2f} TFLOPS"


def _verify_str(r):
    v = r.get("verify")
    if v is None:
        return f"{'-':>4}"
    return f"{'PASS' if v == 'pass' else 'FAIL':>4}"


def _call_kernel(fn, **kwargs):
    """Run a benchmark pass and optional correctness pass."""
    bench = fn(**dict(kwargs, profile=True, skip_reference=True))
    if not VERIFY or bench.get("status") != "passed":
        return bench
    try:
        fn(**dict(kwargs, profile=False, skip_reference=False))
        bench["verify"] = "pass"
    except AssertionError as e:
        bench["verify"] = "fail"
        bench["error"] = str(e).splitlines()[0][:120]
    return bench


def _run_one(shape):
    b, qo, kv, nhq, nhk, hd, c, sc = shape
    common = dict(
        kvcache_layout=KV_LAYOUT,
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
    pth = _call_kernel(run_batch_prefill_per_token_head, **common)
    return pth


def _format_row(shape, pth):
    b = shape[0]
    return f"{b:>5} | {_fmt(pth):>27} {_verify_str(pth)}"


def _run_silent(shape):
    """Run one shape with FD-level stdout/stderr suppression."""
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


print("Warming up kernels (one-time JIT setup, may take a moment)...", flush=True)
_pth0 = _run_silent(SHAPES[0])

_nhq, _nhk, _hd, _causal, _sc = SHAPES[0][3:8]
for _s in SHAPES:
    assert _s[3:8] == (_nhq, _nhk, _hd, _causal, _sc)

if VERIFY:
    print("Verification: ON")
else:
    print("Verification: OFF (set BENCH_VERIFY=1 to enable)")
print(
    f"Config: nhq={_nhq}, nhk={_nhk}, hd={_hd}, "
    f"causal={_causal}, soft_cap={_sc}, page_size={PAGE_SIZE}, "
    f"kv_layout={KV_LAYOUT}"
)
_HEADER = f"{'batch':>5} | {'PER_TOKEN_HEAD':>27} {'vrf':>4}"
print(_HEADER)
print("-" * len(_HEADER))

failures = []


def _record_failures(shape, pth):
    b, _qo, kv, *_ = shape
    if pth.get("verify") == "fail":
        failures.append((b, kv, "PER_TOKEN_HEAD", pth.get("error", "")))


_groups = {}
for _s in SHAPES:
    _groups.setdefault(_s[2], []).append(_s)

_warmup_done = False
for _seq, _shapes in _groups.items():
    print(f"seq={_seq}")
    for shape in _shapes:
        if shape == SHAPES[0] and not _warmup_done:
            pth = _pth0
            _warmup_done = True
        else:
            pth = _run_one(shape)
        print(_format_row(shape, pth), flush=True)
        _record_failures(shape, pth)

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
