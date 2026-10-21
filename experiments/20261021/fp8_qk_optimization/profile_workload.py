# ruff: noqa: S102
# Execute only the pinned local fixture helper after the documented AST adaptation.
import ast
import importlib.util
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
work = base / "fp8-qk-opt-20261021"
tag = sys.argv[1]
spec = importlib.util.spec_from_file_location(
    "profile_bench", base / "decode-pipeline-opt-20261021/bench.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
torch = b.torch
binary = (
    base / "fp8-qk-20261021/jit/v4/module_fused_dsv41_mxfp8.so"
    if tag == "baseline"
    else work / "jit" / tag / "module_fused_dsv41_mxfp8.so"
)
module = b.load_module("profile_" + tag, binary)
src = ast.unparse(
    next(
        n
        for n in ast.parse((base / "fp8-qk-20261021/benchmark.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "make_call"
    )
)
src = src.replace("if module is modules['fp8']:", "if True:")
exec(compile(src, "profile_make_call", "exec"), {"b": b, "torch": torch}, (ns := {}))
b.make_call = ns["make_call"]
src = ast.unparse(
    next(
        n
        for n in ast.parse(
            (base / "decode-pipeline-opt-20261021/bench.py").read_text()
        ).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_case"
    )
)
src = (
    src[: src.index("    calls, outputs =")]
    + "    call,out=make_call(module,q,paged.flatten(),ids,ptr,cos,sin,pos,sink,fixture['sm_scale'],kw)\n    for _ in range(5):\n        call()\n        torch.cuda.synchronize()\n    print('PROFILE_DONE',name)\n"
)
b.module = module
exec(compile(src, "profile_case", "exec"), b.__dict__)
b.run_case(b.matched.CASES[2], {})
