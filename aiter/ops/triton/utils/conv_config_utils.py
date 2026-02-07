from typing import Optional
import os
import triton

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER: AiterTritonLogger = AiterTritonLogger()

_CONV3D_DEFAULT_KEYS = [
        "N",
        "D",
        "H",
        "W",
        "OC",
        "OD",
        "OH",
        "OW",
        "K_C",
        "K_D",
        "K_H",
        "K_W",
        "STRIDE_D",
        "STRIDE_H",
        "STRIDE_W",
        "PAD_D",
        "PAD_H",
        "PAD_W",
        "DIL_D",
        "DIL_H",
        "DIL_W",
        "GROUPS",
    ]

_CONV3D_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32, "BLOCK_CI": 16, "BLOCK_CO": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64, "BLOCK_CI": 16, "BLOCK_CO": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_CI": 16, "BLOCK_CO": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 64, "BLOCK_CI": 32, "BLOCK_CO": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_CI": 32, "BLOCK_CO": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_N": 64, "BLOCK_CI": 16, "BLOCK_CO": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_CI": 16, "BLOCK_CO": 128}, num_warps=8, num_stages=4),
]

def get_default_conv3d_config(Config: Optional[dict] = None) -> dict:
    return {
        "BLOCK_N": 128,
        "BLOCK_CI": 16,
        "BLOCK_CO": 128,
        "num_warps": 8,
        "num_ctas": 1,
        "num_stages": 4,
    }

def _print_conv3d_autotune_timings(
    kernel: triton.runtime.jit.JITFunction, config: Optional[dict]
) -> None:
    if config:
        _LOGGER.info(f"CONV3D_STD autotune best config: {config}")
    else:
        timings = getattr(kernel, "configs_timings", None)
        if isinstance(timings, dict) and timings:
            # timings are in milliseconds (from triton.testing.do_bench)
            sorted_items = sorted(timings.items(), key=lambda kv: kv[1])
            _LOGGER.info("CONV3D_STD autotune per-config timings (ms):")
            for cfg, t_ms in sorted_items:
                if isinstance(t_ms, (list, tuple)):
                    # triton.testing.do_bench returns [median, p20, p80] by default
                    if len(t_ms) == 3:
                        _LOGGER.info(
                            "  %s -> median: %.4f ms, p20: %.4f ms, p80: %.4f ms"
                            % (cfg, t_ms[0], t_ms[1], t_ms[2])
                        )
                    else:
                        _LOGGER.info(f"  {cfg} -> {t_ms}")
                else:
                    _LOGGER.info(f"  {cfg} -> {t_ms:.4f} ms")
        else:
            _LOGGER.info(
                "CONV3D_STD autotune timings not available (cached result or autotune not run yet)."
            )
