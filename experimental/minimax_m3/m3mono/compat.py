# SPDX-License-Identifier: MIT
"""vLLM stand-ins for the ATOM engine hooks the vendored mono code imports.

The mono kernels and framework come from ROCm/ATOM (``atom/mono``,
``atom/models/minimax_m3/mono``, HEAD 67c54da); in ATOM they read ATOM's envs,
forward context, top-k metadata and linear helpers. Here those resolve to vLLM's
objects (or to the values vLLM's MiniMax-M3 model implies), and two helpers newer
than this image's aiter are vendored.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import flydsl.expr as fx


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False")


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ATOM_MONO_* -> VLLM_M3_MONO_*
envs = SimpleNamespace(
    ATOM_MONO_ENABLE=_flag("VLLM_M3_MONO", "0"),
    ATOM_MONO_CHECK=_flag("VLLM_M3_MONO_CHECK"),
    ATOM_MONO_TRACE=_str("VLLM_M3_MONO_TRACE"),
    ATOM_MONO_TIMELINE=_flag("VLLM_M3_MONO_TIMELINE"),
    ATOM_MONO_DEBUG=_flag("VLLM_M3_MONO_DEBUG"),
    ATOM_FORCE_ATTN_TRITON=False,
)


def is_vllm() -> bool:
    # ATOM refuses mono "in plugin mode" (its model under vLLM's engine). Here the
    # kernels serve vLLM's own model through this package's runner, not ATOM's.
    return False


def is_sglang() -> bool:
    return False


def indexer_cp_enabled() -> bool:
    return False


def weight_is_stored_preshuffled(*_a, **_k) -> bool:
    # the runner preshuffles its FP8 copies itself (weights_vllm.py)
    return True


def get_forward_context():
    from vllm.forward_context import get_forward_context as _gfc

    return _gfc()


def tensor_model_parallel_all_reduce(x):
    from vllm.distributed import tensor_model_parallel_all_reduce as _ar

    return _ar(x)


class _TopkMeta:
    """vLLM keeps the fused-shared-expert routing table in
    ``rocm_aiter_moe.aiter_topK_meta_data`` as ATOM does in ``atom.model_ops.topK``."""

    @property
    def aiter_topK_meta_data(self):
        from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

        return rocm_aiter_moe.aiter_topK_meta_data


topk_meta = _TopkMeta()

# aiter.ops.flydsl.kernels.kernels_common.LOG2E (newer aiter)
LOG2E = 1.4426950408889634


def update_dpp_i32(old, src, dpp_ctrl: int, row_mask: int = 0xF, bank_mask: int = 0xF,
                   bound_ctrl: bool = False, **kw):
    """``llvm.amdgcn.update.dpp.i32`` (aiter.ops.flydsl.kernels.dpp_utils, newer aiter)."""
    from flydsl._mlir.dialects import llvm as _llvm

    return _llvm.call_intrinsic(
        fx.Int32.ir_type,
        "llvm.amdgcn.update.dpp.i32",
        [
            fx.Int32(old).ir_value(),
            fx.Int32(src).ir_value(),
            fx.Int32(dpp_ctrl).ir_value(),
            fx.Int32(row_mask).ir_value(),
            fx.Int32(bank_mask).ir_value(),
            fx.Boolean(bound_ctrl).ir_value(),
        ],
        [],
        [],
        **kw,
    )
