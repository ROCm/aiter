"""Compile the original and NT-output kernels without using a GPU."""
import os
import sys
import importlib.util
from pathlib import Path


os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["FLYDSL_GPU_ARCH"] = "gfx950"
os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["FLYDSL_DUMP_IR"] = "1"
root = Path(__file__).resolve().parent
case = sys.argv[1]
assert case in ("baseline", "nt")
os.environ["FLYDSL_DUMP_DIR"] = str(root / f"build_{case}")
import flydsl.compiler as flyc
import flydsl.expr as fx

path = root / "source" / f"{case}_kernel.py"
name = "aiter.ops.flydsl.kernels.ep32_continuity_" + case
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
fn = module.compile_gemm2_ep32(HIDDEN_MAX=3584, INTER_MAX=3072, BM=80,
    GRID_WGS=512, B_NT=True, REDUCE=True, NATIVE_C=True)
flyc.compile(fn, *([0] * 16), fx.Stream(None))
print(f"offline {case} passed", flush=True)
