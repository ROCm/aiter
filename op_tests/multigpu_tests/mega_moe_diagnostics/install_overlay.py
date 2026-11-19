"""Install fine S1F probes in this process; leave repository modules untouched."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREFIX = "aiter.ops.flydsl.kernels.mega_moe_gfx1250"
REGISTRY = {}


def install():
    package = importlib.import_module(PREFIX)
    for leaf in ("cycle_trace", "dispatch_pull", "s1f_stage1", "s1f_pull"):
        name = PREFIX + "." + leaf
        spec = importlib.util.spec_from_file_location(name, ROOT / (leaf + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(package, leaf, module)
        spec.loader.exec_module(module)
    import torch

    module = importlib.import_module(PREFIX + ".stages.pull")
    cls = module.PullDispatch
    original_init, original_run, original_gemm = (
        cls.__init__,
        module._run_compiled,
        cls.gemm1,
    )

    def initialize(self, dispatch):
        original_init(self, dispatch)
        if not self.fused:
            return
        assert not self.ready_first and not self.packed and self.producer_group == 1
        cap = dispatch.compact.cap
        flags, m_tiles = (cap + 63) // 64, (cap + 255) // 256
        words = flags * 16 + m_tiles * 18 * 4 * 20
        trace = torch.zeros(words, dtype=torch.int64, device=self.ctl.device)
        REGISTRY[self.ctl.data_ptr()] = (self, trace, flags, m_tiles)

    def gemm(self, *args, **kwargs):
        assert kwargs["tile_m"] == 256 and kwargs["tile_n"] == 256
        assert self.config.hidden_dim == 5120 and self.config.geometry.inter_dim == 2304
        return original_gemm(self, *args, **kwargs)

    def run(function, *args):
        if len(args) == 19 and args[14] in REGISTRY:
            assert args[9] == 0
            args = list(args)
            args[9] = REGISTRY[args[14]][1].data_ptr()
        return original_run(function, *args)

    cls.__init__, cls.gemm1, module._run_compiled = initialize, gemm, run


def save(output, rank):
    import numpy as np
    import torch

    torch.cuda.synchronize()
    # Inspect one final layer/replay, not a mixture of stale generations.
    helper, trace, flags, m_tiles = list(REGISTRY.values())[-1]
    np.savez_compressed(
        output / f"cycles_rank{rank}.npz",
        trace=trace.cpu().numpy(),
        info=helper.tile_info.cpu().numpy(),
        rowmap=helper.dispatch._ep_rowmap.cpu().numpy(),
    )
    metadata = dict(
        rank=rank,
        flags=flags,
        m_tiles=m_tiles,
        max_routes=helper.dispatch.compact.max_routes,
        traced_layers=len(REGISTRY),
        consumer_words=20,
        producer_words=16,
    )
    (output / f"cycles_rank{rank}.json").write_text(json.dumps(metadata, indent=2))
