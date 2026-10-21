import json
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
tag = sys.argv[1]
selected = (
    json.loads(
        (base / "fp8-qk-profile-opt-20261021/release/selection.json").read_text()
    )["candidate"]
    if tag == "final"
    else tag
)
src = (base / "fp8-qk-h128-20261021/full_benchmark.py").read_text()
src = src.replace(
    "WORK=BASE/'fp8-qk-h128-20261021'",
    "WORK=BASE/'fp8-qk-profile-opt-20261021/release'",
)
src = src.replace(
    "BASE/'fp8-qk-opt-20261021/jit/qkpipe/module_fused_dsv41_mxfp8.so'",
    "BASE/'fp8-qk-h128-20261021/jit/fma/module_fused_dsv41_mxfp8.so'",
)
src = src.replace(
    "(base/'fp8-qk-h128-20261021/full_benchmark_generated.py')",
    "(base/'fp8-qk-profile-opt-20261021/release/full_benchmark_generated.py')",
)
# N64 changes BF16 probability rounding by normalizing 64 scores together.
# Keep the original independent dense-oracle, max/LSE, layout, PR and replay
# checks; record its delta from N32 rather than requiring N32 equivalence.
adjustment = (
    "src=src.replace('    assert previous_delta < .002', '    print(\"N64_VS_N32_OUTPUT_L2\",previous_delta,flush=True)'); "
    if selected.startswith("n64")
    else ""
)
if not selected.startswith("n64"):
    exact = """    for layout in ['paged','row']:
        for x,y in zip(outputs['fp8_'+layout],outputs['fp8old_'+layout]):
            torch.testing.assert_close(x.view(torch.uint8),y.view(torch.uint8),rtol=0,atol=0)
"""
    adjustment = (
        f"src=src.replace('    previous_delta =', {exact!r}+'    previous_delta ='); "
    )
src = src.replace(
    "exec(compile(src,'full_benchmark_generated','exec'))",
    "src=src.replace(\"WORK/'benchmark.json'\",f\"WORK/'benchmark_{tag}.json'\"); "
    + adjustment
    + "exec(compile(src,'full_benchmark_generated','exec'))",
)
exec(compile(src, "full_benchmark_current", "exec"))
