# Reuse pinned local fixture/adapter source to preserve the comparison contract.
# ruff: noqa: S102
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
src = (base / "fp8-qk-h128-20261021/compare.py").read_text()
src = src.replace(
    "work=base/'fp8-qk-h128-20261021'", "work=base/'fp8-qk-profile-opt-20261021'"
)
src = src.replace(
    "base/'fp8-qk-opt-20261021/jit/qkpipe/module_fused_dsv41_mxfp8.so'",
    "base/'fp8-qk-h128-20261021/jit/fma/module_fused_dsv41_mxfp8.so'",
)
src = src.replace("for i in [4,5,2]:", "for i in [4,5,0,2]:")
exec(compile(src, "compare_current_fp8", "exec"))
