# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared core of the config machinery: the config-tree paths, the cached
JSON loading, and the validated, deterministic nested-layout directory
builder every loader goes through.

The per-family loaders build on this module and keep their own files:
``gemm_config_utils``, ``conv_config_utils``, ``mhc_config_utils``,
``moe_config_utils`` and ``tuned_config_utils``.
"""

import functools
import json
import os
import re

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

logger = AiterTritonLogger()


this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_TRITON_OPS_PATH = os.path.abspath(f"{this_dir}/../")
AITER_TRITON_CONFIGS_PATH = os.path.abspath(f"{this_dir}/../configs")

# This flag should be set to True, unless it is being used for debugging.
# When False, config JSON files are re-read on every call, so live edits to
# the JSON are picked up.
USE_LRU_CACHE = True


@functools.lru_cache(maxsize=None if USE_LRU_CACHE else 0)
def load_config_json(fpath: str, required: bool = True) -> dict | None:
    """Load a config JSON file, cached per path (including negative results —
    add config files before process start, or call
    ``load_config_json.cache_clear()``). Raises FileNotFoundError if the file
    doesn't exist, consistently on every call (exceptions are never cached);
    pass required=False for probe/fallback lookups to get None instead.

    The returned dict is the shared cached object — copy before mutating:
    a shallow ``.copy()`` suffices for flat bucket dicts (scalar values),
    ``copy.deepcopy`` when nested sub-dicts will be mutated."""
    try:
        with open(fpath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(
                f"Required config file doesn't exist: {fpath}"
            ) from None
        return None


def _dtype_dir(config_name: str) -> str:
    """Nested-layout directory for a config family:
    ``GEMM-AFP4WFP4`` -> ``gemm_afp4wfp4``."""
    return config_name.lower().replace("-", "_")


_VALID_BACKENDS = ("triton", "gluon")

# Every argument below becomes a filesystem path component, so each one is
# validated against a whitelist and the function fails closed: a bad value
# can never traverse outside the config tree or silently resolve to a wrong
# directory. The override arch is programmer-written (a literal like
# "gfx942"), so it gets the strict identifier form; the running arch comes
# from the driver, so it tolerates vendor formats (e.g. feature-suffixed
# targets) while still rejecting anything path-unsafe.
_OP_RE = re.compile(r"[a-z][a-z0-9_]*")
_CONFIG_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_ARCH_OVERRIDE_RE = re.compile(r"[a-z][a-z0-9_]*")
_ARCH_SAFE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+:-]*")


def resolve_config_dir(
    op: str,
    config_name: str,
    backend: str = "triton",
    arch: str | None = None,
) -> str:
    """Build the directory that holds one config family's JSON files.

    The nested layout (see configs/CLAUDE.md) keys every family by
    architecture, backend, op and family:
    ``<configs>/<arch>/<backend>/<op>/<d_type>``, with the ``<d_type>`` leaf
    derived by ``_dtype_dir()`` (``GEMM-AFP4WFP4`` -> ``gemm_afp4wfp4``).
    The path is deterministic: it is built, never probed for, and there is
    no search across backends, so a family tuned only for the other backend
    on this arch is never silently borrowed.

    Args:
        op: op directory in the layout -- ``"gemm"``, ``"conv"``, ``"mhc"``,
            ``"moe"``, ``"attention"``, ``"gmm"``, ... Must match
            ``[a-z][a-z0-9_]*``.
        config_name: config family name exactly as spelled in the JSON file
            stems, e.g. ``"GEMM-A8W8_BLOCKSCALE"`` or ``"CONV-PREPACK"``.
            Must match ``[A-Za-z0-9][A-Za-z0-9_-]*``. Specialized files in
            the directory keep this stem (``<config_name>-N=...-K=....json``).
        backend: which backend's tuning to load -- the caller declares it.
            Gluon kernels and gluon dispatch paths pass ``"gluon"``;
            everything else takes the ``"triton"`` default. The two backends
            take disjoint config params, so a config from the wrong backend
            is not usable. Must be one of ``("triton", "gluon")``.
        arch: overrides the running architecture (``arch_info.get_arch()``)
            -- for loaders that retry under another arch when the running one
            ships no tuned configs (e.g. MHC's documented gfx942 fallback).
            Must match ``[a-z][a-z0-9_]*`` when given.

    Returns:
        The directory path (a plain ``str``). Existence is deliberately not
        checked here; whether a given file inside it exists is the loader's
        decision. The family default is ``<dir>/DEFAULT.json``: loading it
        with ``load_config_json(path)`` raises ``FileNotFoundError`` naming
        this exact path when a required table is missing, while optional
        tables pass ``required=False`` and handle ``None``.

    Raises:
        AssertionError: if any argument falls outside its whitelist above,
            or the running architecture resolves to a path-unsafe string --
            the arguments become path components, so resolution fails closed
            instead of building an escaped or wrong directory.

    Example -- specialized file first, family default as fallback::

        cfg_dir = resolve_config_dir("gemm", "GEMM-A8W8_BLOCKSCALE",
                                     backend="gluon")
        config = load_config_json(
            f"{cfg_dir}/GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json", required=False
        )
        if config is None:
            config = load_config_json(f"{cfg_dir}/DEFAULT.json")
    """
    assert isinstance(op, str) and _OP_RE.fullmatch(
        op
    ), f"op must match [a-z][a-z0-9_]* (e.g. 'gemm', 'conv'), got {op!r}"
    assert isinstance(config_name, str) and _CONFIG_NAME_RE.fullmatch(config_name), (
        "config_name must match [A-Za-z0-9][A-Za-z0-9_-]* "
        f"(e.g. 'GEMM-A8W8_BLOCKSCALE'), got {config_name!r}"
    )
    assert (
        backend in _VALID_BACKENDS
    ), f"unknown backend {backend!r}; expected one of {_VALID_BACKENDS}"
    assert arch is None or (
        isinstance(arch, str) and _ARCH_OVERRIDE_RE.fullmatch(arch)
    ), f"arch override must match [a-z][a-z0-9_]* (e.g. 'gfx942'), got {arch!r}"
    dev = arch if arch is not None else arch_info.get_arch()
    assert isinstance(dev, str) and _ARCH_SAFE_RE.fullmatch(
        dev
    ), f"arch_info.get_arch() returned a path-unsafe architecture: {dev!r}"
    return f"{AITER_TRITON_CONFIGS_PATH}/{dev}/{backend}/{op}/{_dtype_dir(config_name)}"
