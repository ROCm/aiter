# ruff: noqa: S102
# Execute only the pinned local fixture helper after the documented AST adaptation.
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
tag = sys.argv[1]
src = (base / "fp8-qk-20261021/benchmark.py").read_text()
src = src.replace("WORK=BASE/'fp8-qk-20261021'", "WORK=BASE/'fp8-qk-opt-20261021'")
old = "fp8=WORK/'jit/v4/module_fused_dsv41_mxfp8.so')"
new = f"fp8old=BASE/'fp8-qk-20261021/jit/v4/module_fused_dsv41_mxfp8.so', fp8=WORK/'jit/{tag}/module_fused_dsv41_mxfp8.so')"
assert old in src
src = src.replace(old, new).replace(
    "if module is modules['fp8']:vals.extend",
    "if module is not modules['bf16']:vals.extend",
)
src = src.replace(
    "for precision in ['bf16','fp8']:", "for precision in ['bf16','fp8old','fp8']:"
)
anchor = "    precision_delta = pr.relative"
src = src.replace(
    anchor,
    "    for layout in ['paged','row']:\n        for x,y in zip(outputs['fp8_'+layout][:2],outputs['fp8old_'+layout][:2]):\n            torch.testing.assert_close(x.view(torch.uint8),y.view(torch.uint8),rtol=0,atol=0)\n        for x,y in zip(outputs['fp8_'+layout][2:],outputs['fp8old_'+layout][2:]):\n            torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-6)\n"
    + anchor,
)
(base / "fp8-qk-opt-20261021/full_benchmark_generated.py").write_text(src)
exec(compile(src, "full_benchmark_generated", "exec"))
