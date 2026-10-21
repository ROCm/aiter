# ruff: noqa: S102
# Execute only the documented pinned local benchmark adapter.
from pathlib import Path

b = Path("/home/jun_chen2_qle/yzhou_aiter")
w = b / "fp8-qk-h128-20261021"
src = (b / "fp8-qk-20261021/shared_bench.py").read_text()
src = src.replace(
    "work=Path('/home/jun_chen2_qle/yzhou_aiter/fp8-qk-20261021')",
    f"work=Path({str(w)!r})",
)
src = src.replace("(work/'benchmark.py')", "(work/'full_benchmark_generated.py')")
exec(compile(src, "shared_compare_generated", "exec"))
