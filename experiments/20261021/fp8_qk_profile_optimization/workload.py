# Reuse pinned local fixture/adapter source to preserve the comparison contract.
# ruff: noqa: S102
"""Pinned decode fixture; profile one selected full path with ordinary launches."""

import ast
import importlib.util
import os
import sys
from pathlib import Path

BASE = Path("/home/jun_chen2_qle/yzhou_aiter")
WORK = BASE / "fp8-qk-profile-opt-20261021/release"
mode, case_index = sys.argv[1], int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else "baseline"
os.environ["AITER_JIT_DIR"] = str(BASE / "pr6675-results-20261021/jit")
sys.path.insert(0, str(BASE / "aiter-pr6675-20261021"))
sys.path.insert(0, str(BASE / "pr6675-results-20261021"))
import bench_compare as pr

spec = importlib.util.spec_from_file_location(
    "profile_bench", BASE / "decode-pipeline-opt-20261021/bench.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
torch = b.torch
binary = (
    BASE / "fp8-qk-h128-20261021/jit/fma/module_fused_dsv41_mxfp8.so"
    if tag == "baseline"
    else WORK / "jit" / tag / "module_fused_dsv41_mxfp8.so"
)
module = b.load_module("profile_" + tag, binary)
src = ast.unparse(
    next(
        n
        for n in ast.parse((BASE / "fp8-qk-20261021/benchmark.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "make_call"
    )
)
src = src.replace("if module is modules['fp8']:", "if True:")
exec(compile(src, "profile_make_call", "exec"), {"b": b, "torch": torch}, ns := {})
b.make_call = ns["make_call"]
src = ast.unparse(
    next(
        n
        for n in ast.parse(
            (BASE / "decode-pipeline-opt-20261021/bench.py").read_text()
        ).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_case"
    )
)
src = src[: src.index("    calls, outputs =")] + """
    if mode == 'paged':
        call, out = make_call(module, q, paged.flatten(), ids, ptr, cos, sin, pos, sink, fixture['sm_scale'], kw)
    elif mode == 'row':
        call, out = make_call(module, q, packed.flatten(), ids.long()*1056+1, ptr, cos, sin, pos, sink, fixture['sm_scale'], {})
    else:
        q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        qs = torch.empty((t,h,16), device='cuda', dtype=torch.uint8)
        att = torch.empty_like(q)
        cast = [torch.empty((t,h//8,4096),device='cuda',dtype=torch.float8_e4m3fn), torch.empty((t,h//8,128),device='cuda',dtype=torch.float8_e8m0fnu)]
        pos64 = pos.long()
        data, codes = packed[:,:512].view(torch.float8_e4m3fn), packed[:,512:]
        pids, eids = [s.flatten() for s in scopes]
        pp = torch.arange(t+1,device='cuda',dtype=torch.int32)*scopes[0].shape[1]
        ep = torch.arange(t+1,device='cuda',dtype=torch.int32)*scopes[1].shape[1]
        def call():
            pr.q_prepare[(t*h//8,)](q,q8,qs,pos,cos,sin,h,True,ROWS=8,num_warps=4,enable_fp_fusion=False)
            pr.mla_sparse_prefill_v41(q8,qs,data,codes,pids,pp,data,codes,eids,ep,sink,fixture['sm_scale'],out=att,backend='opus')
            pr.inverse_rope_group_quant(att,pos64,cos,sin,num_groups=h//8,quant_group_size=32,scale_layout='row',x_fp8=cast[0],x_scale=cast[1])
            return cast
    for _ in range(5):
        call()
        torch.cuda.synchronize()
    print('PROFILE_DONE',mode,name,flush=True)
"""
b.module, b.mode, b.pr = module, mode, pr
exec(compile(src, "profile_case", "exec"), b.__dict__)
b.run_case(b.matched.CASES[case_index], {}, pr=pr)
