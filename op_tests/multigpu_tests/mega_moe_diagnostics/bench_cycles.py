"""Compare native E2E with/without probes and save final-replay cycle records."""

import json
import os
from pathlib import Path
import sys

import install_overlay

TRACE = os.environ.get("CYCLE_TRACE", "1") == "1"
if TRACE:
    install_overlay.install()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_mega_moe as benchmark

original_bench = benchmark.DeviceMoEPipeline.bench


def bench(self, *args, **kwargs):
    result = original_bench(self, *args, **kwargs)
    output = Path(os.environ["PROFILE_OUT"])
    (output / f"native_rank{self.dist_ctx.rank}.json").write_text(
        json.dumps(dict(native_e2e_us=result, cycle_trace=TRACE), indent=2)
    )
    if TRACE:
        install_overlay.save(output, self.dist_ctx.rank)
    self.comm.barrier()
    return result


benchmark.DeviceMoEPipeline.bench = bench
if __name__ == "__main__":
    benchmark.main()
