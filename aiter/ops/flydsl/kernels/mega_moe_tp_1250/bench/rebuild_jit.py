# CPU-only rebuild of the stale aiter JIT modules after an upstream merge (AITER_REBUILD=1 in the env)
import sys, time
from aiter.jit import core
for md in sys.argv[1:]:
    t = time.time()
    d = core.get_args_of_build(md)
    core.build_module(md, d["srcs"], d["flags_extra_cc"], d["flags_extra_hip"], d["blob_gen_cmd"], d["extra_include"],
                      d["extra_ldflags"], d["verbose"], d["is_python_module"], d["is_standalone"], d["torch_exclude"],
                      d.get("third_party", []), d.get("hipify", False),
                      flags_extra_hip_per_source=d.get("flags_extra_hip_per_source", {}))
    print(f"built {md} in {time.time()-t:.0f}s", flush=True)
