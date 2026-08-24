# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-arch tuned tiles for kernels that carry a Python autotune search space.

The GEMM and MOE families resolve a tuned entry per shape at launch time. These
kernels need less: their search is opt-in, so all that has to be decided is the
one config registered when it is off. Keeping that in a config file rather than
in Python means pinning a tile for a new device is a file and not a branch.
"""

import functools

import triton

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import (
    AITER_TRITON_CONFIGS_PATH,
    USE_LRU_CACHE,
    load_config_json,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

logger = AiterTritonLogger()


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def _get_tuned_kernel_entry(
    op: str, config_name: str, kernel_name: str, backend: str
) -> tuple[str, dict | None]:
    """Internal cached lookup returning ``(config path, entry or None)``.

    Do NOT use this directly — the entry is the shared cached object, so
    ``get_tuned_kernel_config()`` copies it before handing it out.
    """
    arch = arch_info.get_arch()
    # Nested layout of configs/CLAUDE.md: <arch>/<backend>/<op>/<d_type>/DEFAULT.json,
    # <d_type> being the config name lowercased with dashes folded to underscores.
    dtype_dir = config_name.lower().replace("-", "_")
    config_path = (
        f"{AITER_TRITON_CONFIGS_PATH}/{arch}/{backend}/{op}/{dtype_dir}/DEFAULT.json"
    )
    published = load_config_json(config_path, required=False) or {}
    return config_path, published.get(kernel_name)


def get_tuned_kernel_config(
    op: str,
    config_name: str,
    kernel_name: str,
    fallback: triton.Config,
    backend: str = "triton",
) -> triton.Config:
    """The tile pinned for this device, or ``fallback`` where none is published.

    What fits is not portable: the same tile can compile to 16KB of LDS on one
    arch and to more than the 64KB another one has. A device nobody has measured
    therefore gets the fallback, which has to be launchable anywhere rather than
    fastest somewhere, and stays on it until a measured entry is published.

    Args:
        op: Op family directory, e.g. ``"attention"``.
        config_name: Config family, e.g. ``"CHUNK_DELTA_ATTN"``.
        kernel_name: Key of the kernel's entry within the config file.
        fallback: Config to register when this device has no published entry.
        backend: ``"triton"`` or ``"gluon"``.
    """
    try:
        config_path, entry = _get_tuned_kernel_entry(
            op, config_name, kernel_name, backend
        )
    except BaseException as error:  # noqa: BLE001 -- no accelerator/unreadable file
        logger.warning(
            f"Unable to load tuned Triton config '{config_name}' for "
            f"kernel '{kernel_name}'; using fallback {fallback}: {error}"
        )
        return fallback
    if not entry:
        logger.warning(
            f"No tuned Triton config for kernel '{kernel_name}' in "
            f"'{config_path}'; using fallback {fallback}"
        )
        return fallback
    entry = dict(entry)
    num_warps = entry.pop("num_warps", fallback.num_warps)
    num_stages = entry.pop("num_stages", fallback.num_stages)
    return triton.Config(entry, num_warps=num_warps, num_stages=num_stages)
