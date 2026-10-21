# Reuse pinned local fixture/adapter source to preserve the comparison contract.
# ruff: noqa: S102
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
work = base / "fp8-qk-pv-opt-20261021/release"
src = (base / "fp8-qk-20261021/shared_bench.py").read_text()
src = src.replace(
    "work=Path('/home/jun_chen2_qle/yzhou_aiter/fp8-qk-20261021')",
    f"work=Path({str(work)!r})",
)
src = src.replace("(work/'benchmark.py')", "(work/'full_benchmark_generated.py')")
src = src.replace(
    "    q8=torch.empty_like(q,dtype=torch.float8_e4m3fn);",
    """    for x,y in zip(outputs['fp8'],outputs['fp8old']):
        torch.testing.assert_close(x.view(torch.uint8),y.view(torch.uint8),rtol=0,atol=0)
    q8=torch.empty_like(q,dtype=torch.float8_e4m3fn);""",
)
exec(compile(src, "shared_final", "exec"))
