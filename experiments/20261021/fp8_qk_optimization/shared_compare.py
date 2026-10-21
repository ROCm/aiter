# ruff: noqa: S102
# Execute only the pinned local fixture helper after the documented AST adaptation.
from pathlib import Path

b = Path("/home/jun_chen2_qle/yzhou_aiter")
w = b / "fp8-qk-opt-20261021"
src = (b / "fp8-qk-20261021/shared_bench.py").read_text()
src = src.replace(
    "work=Path('/home/jun_chen2_qle/yzhou_aiter/fp8-qk-20261021')",
    f"work=Path({str(w)!r})",
)
src = src.replace("(work/'benchmark.py')", "(work/'full_benchmark_generated.py')")
src = src.replace(
    "torch.manual_seed(20261021)",
    "modules['vasm2']=b.load_module('shared_vasm2',work/'jit/vasm2/module_fused_dsv41_mxfp8.so')\ntorch.manual_seed(20261021)",
)
anchor = "    q8=torch.empty_like(q,dtype=torch.float8_e4m3fn);"
src = src.replace(
    anchor,
    "    for x,y in zip(outputs['fp8'][:2],outputs['fp8old'][:2]):\n        torch.testing.assert_close(x.view(torch.uint8),y.view(torch.uint8),rtol=0,atol=0)\n"
    + anchor,
)
exec(compile(src, "shared_compare_generated", "exec"))
