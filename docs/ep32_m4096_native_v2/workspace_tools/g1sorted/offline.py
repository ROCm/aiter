"""Compile only the one-shape sorted-A prototype and its pack kernel."""
import os
import sys
import importlib.util
from pathlib import Path


os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["FLYDSL_GPU_ARCH"] = "gfx950"
os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["FLYDSL_DUMP_IR"] = "1"
ROOT = Path(__file__).resolve().parent
case = sys.argv[1]
assert case in ("baseline", "sorted", "pack")
os.environ["FLYDSL_DUMP_DIR"] = str(ROOT / f"build_{case}")
import flydsl.compiler as flyc
import flydsl.expr as fx

file = "ep32_sorted_a_kernel.py" if case == "pack" else f"{case}_kernel.py"
name = "aiter.ops.flydsl.kernels._cx_g1_sorted_" + case
spec = importlib.util.spec_from_file_location(name, ROOT / "source" / file)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
if case == "pack":
    fn = module.compile_sorted_a(None)
    flyc.compile(fn, *([flyc.from_c_void_p(fx.Uint8, 0)] * 4), 4096, fx.Stream(None))
else:
    fn = module.compile_gemm1_ep32(BM=80, D_HIDDEN=3584, D_INTER=3072, NE=28,
        situ_beta=4.0, situ_linear_beta=25.0)
    flyc.compile(fn, *([0] * 13), fx.Stream(None))
print(f"offline {case} passed", flush=True)
