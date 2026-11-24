import subprocess
import sys
from pathlib import Path

import pytest

"""
Smoke test for every Triton kernel benchmark script in op_tests/op_benchmarks/triton.

This is intentionally shallow: it just runs each bench_*.py once with simple
CLI args and checks that the process exits successfully. It does not check
performance numbers or correctness - see test_kernel_benchmarks.py for that.
"""

TRITON_BENCH_DIR = Path(__file__).resolve().parent.parent

# Not a standalone benchmark: a shared helper module with no CLI entry point
# of its own (its argparse setup is exercised indirectly via the scripts that
# import it).
_EXCLUDE = {
    "bench_fused_gemm_commons.py",
}

# A handful of benchmarks have no default shape and require explicit args to
# run at all. Give them the smallest/fastest shape that still exercises the
# kernel. Every other script is run with no extra args and relies on its own
# built-in default shape.
_MOE_SHAPE_ARGS = ["--M", "1", "--shape", "4096", "14336", "--experts", "8", "2"]

_CUSTOM_ARGS = {
    "bench_fav3_sage.py": ["-b", "1", "-hq", "8", "-sq", "256", "-d", "128"],
    "bench_fav3_sage_mxfp4.py": ["-b", "1", "-hq", "8", "-sq", "256", "-d", "128"],
    "bench_mla_decode.py": ["--model", "deepseek-V3", "--seqlen", "1024"],
    "bench_moe_gemm_a16w4.py": _MOE_SHAPE_ARGS,
    "bench_moe_gemm_a4w4_cudagraph.py": _MOE_SHAPE_ARGS,
    "bench_moe_gemm_a4w4_proton.py": _MOE_SHAPE_ARGS,
    "bench_moe_gemm_a8w4_cudagraph.py": _MOE_SHAPE_ARGS,
    "bench_moe_gemm_a8w4_proton.py": _MOE_SHAPE_ARGS,
    "bench_moe_gemm_a8w8.py": _MOE_SHAPE_ARGS,
    "bench_moe_gemm_a8w8_blockscale.py": _MOE_SHAPE_ARGS,
    "bench_moe_gemm_int8_smoothquant.py": [
        "--shape",
        "4096",
        "14336",
        "--experts",
        "8",
        "2",
    ],
    "bench_na3d_flash.py": ["-s", "(1,18,32,48,32,64,3,7,7)"],
    "bench_pa_mqa_logits_mxfp4.py": ["--batch", "1", "--q-len", "1", "--kv-len", "128"],
    "bench_sage.py": ["--b", "1", "--hq", "8", "--sq", "256", "--d", "128"],
}

BENCH_SCRIPTS = sorted(
    p.name for p in TRITON_BENCH_DIR.glob("bench_*.py") if p.name not in _EXCLUDE
)


@pytest.mark.parametrize("script", BENCH_SCRIPTS)
def test_benchmark_smoke(script, tmp_path):
    args = _CUSTOM_ARGS.get(script, [])
    result = subprocess.run(
        [sys.executable, str(TRITON_BENCH_DIR / script), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} exited with code {result.returncode}\n"
        f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
    )
