"""Profile the pre-change kernel with the unchanged benchmark driver."""
import importlib.util
from pathlib import Path
import runpy
import sys

import aiter.ops.flydsl.kernels

root = Path(__file__).resolve().parents[2]
name = "aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942"
spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("baseline_kernel.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
runpy.run_path(str(root / "scripts/bench_unified_attention_gemma4.py"), run_name="__main__")
