# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# --- Triton < 3.6 import-time backward-compatibility shim ---
# Some gluon kernels under aiter/ops/triton/{gluon,_gluon_kernels}/ decorate
# helper functions with `@gluon.constexpr_function`, a symbol that only exists
# in `triton.experimental.gluon` on Triton >= 3.6. Those kernel modules are
# imported eagerly at module load (and across GPU archs), so on Triton < 3.6
# (e.g. the ROCm 7.0 image ships Triton 3.4) merely importing them raises:
#     AttributeError: module 'triton.experimental.gluon'
#                     has no attribute 'constexpr_function'
# This breaks downstream importers (e.g. SGLang's model registry, which sweeps
# and imports every model module) for any model whose import chain reaches a
# gluon kernel -- even though the gluon kernels are never *executed* on
# Triton < 3.6 (aiter selects non-gluon code paths there). Installing a no-op
# `constexpr_function` keeps `import` working on old Triton; it is a no-op on
# Triton >= 3.6 where the real symbol is present.
try:
    import triton as _triton
    from packaging.version import Version as _Version

    if _Version(_triton.__version__.split("+")[0]) < _Version("3.6.0"):
        from triton.experimental import gluon as _gluon

        if not hasattr(_gluon, "constexpr_function"):

            def _aiter_compat_constexpr_function(fn=None, *args, **kwargs):
                # Supports both `@gluon.constexpr_function` and
                # `@gluon.constexpr_function(...)`. Returns the decorated
                # function unchanged so the module imports; it is never called
                # on Triton < 3.6 because the gluon kernels are not dispatched.
                if callable(fn):
                    return fn
                return lambda _fn: _fn

            _gluon.constexpr_function = _aiter_compat_constexpr_function
except Exception:
    pass

import importlib.util
import sys
from types import SimpleNamespace

# Try to import quant module
try:
    from . import quant
except (ImportError, AttributeError):
    quant = None

# Try to import comms module (requires iris)
try:
    from . import comms

    # Re-export communication primitives at this level for convenience
    from .comms import (
        IrisCommContext,
        reduce_scatter,
        all_gather,
        reduce_scatter_rmsnorm_quant_all_gather,
        IRIS_COMM_AVAILABLE,
    )

    _COMMS_AVAILABLE = True
except ImportError:
    # Iris not available - comms module won't be available
    _COMMS_AVAILABLE = False
    IRIS_COMM_AVAILABLE = False
    comms = None

__all__ = []
if quant is not None:
    __all__.append("quant")

if _COMMS_AVAILABLE:
    __all__.extend(
        [
            "comms",
            "IrisCommContext",
            "reduce_scatter",
            "all_gather",
            "reduce_scatter_rmsnorm_quant_all_gather",
            "IRIS_COMM_AVAILABLE",
        ]
    )

"""
These following help implement backward-compatibility
for modules that were reorganized so that external repos (like sglang for example),
which depend on the old module names, can still import it the old "way" of importing.
"""
# This is a mapping of the old module names to the new module names
_BACKWARD_COMPAT_MAP = {
    # Batched GEMM modules (gemm/batched/)
    "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant": "gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
    "batched_gemm_a16wfp4": "gemm.batched.batched_gemm_a16wfp4",
    "batched_gemm_a8w8": "gemm.batched.batched_gemm_a8w8",
    "batched_gemm_afp4wfp4_pre_quant": "gemm.batched.batched_gemm_afp4wfp4_pre_quant",
    "batched_gemm_afp4wfp4": "gemm.batched.batched_gemm_afp4wfp4",
    "batched_gemm_bf16": "gemm.batched.batched_gemm_bf16",
    # Basic GEMM modules (gemm/basic/)
    "gemm_a16w16_agnostic": "gemm.basic.gemm_a16w16_agnostic",
    "gemm_a16w16_atomic": "gemm.basic.gemm_a16w16_atomic",
    "gemm_a16w16_gated": "gemm.basic.gemm_a16w16_gated",
    "gemm_a16w16": "gemm.basic.gemm_a16w16",
    "gemm_a16w8_blockscale": "gemm.basic.gemm_a16w8_blockscale",
    "gemm_a16wfp4": "gemm.basic.gemm_a16wfp4",
    "gemm_a8w8_blockscale": "gemm.basic.gemm_a8w8_blockscale",
    "gemm_a8w8_per_token_scale": "gemm.basic.gemm_a8w8_per_token_scale",
    "gemm_a8w8": "gemm.basic.gemm_a8w8",
    "gemm_a8wfp4": "gemm.basic.gemm_a8wfp4",
    "gemm_afp4wfp4_pre_quant_atomic": "gemm.basic.gemm_afp4wfp4_pre_quant_atomic",
    "gemm_afp4wfp4": "gemm.basic.gemm_afp4wfp4",
    # Feed-forward modules (gemm/feed_forward/)
    "ff_a16w16_fused_gated": "gemm.feed_forward.ff_a16w16_fused_gated",
    "ff_a16w16_fused_ungated": "gemm.feed_forward.ff_a16w16_fused_ungated",
    "ff_a16w16": "gemm.feed_forward.ff_a16w16",
    # Fused GEMM modules (gemm/fused/)
    "fused_gemm_a8w8_blockscale_a16w16": "gemm.fused.fused_gemm_a8w8_blockscale_a16w16",
    "fused_gemm_a8w8_blockscale_mul_add": "gemm.fused.fused_gemm_a8w8_blockscale_mul_add",
    "fused_gemm_afp4wfp4_a16w16": "gemm.fused.fused_gemm_afp4wfp4_a16w16",
    "fused_gemm_afp4wfp4_mul_add": "gemm.fused.fused_gemm_afp4wfp4_mul_add",
    "fused_gemm_afp4wfp4_split_cat": "gemm.fused.fused_gemm_afp4wfp4_split_cat",
    "fused_gemm_a8w8_blockscale_split_cat": "gemm.fused.fused_gemm_a8w8_blockscale_split_cat",
    # Attention modules (attention/)
    "chunked_pa_prefill": "attention.chunked_pa_prefill",
    "extend_attention": "attention.extend_attention",
    "fp8_mqa_logits": "attention.fp8_mqa_logits",
    "hstu_attention": "attention.hstu_attention",
    "lean_atten_paged": "attention.lean_atten_paged",
    "lean_atten": "attention.lean_atten",
    "mha_fused_bwd": "attention.mha_fused_bwd",
    "mha_onekernel_bwd": "attention.mha_onekernel_bwd",
    "mha_v3": "attention.mha_v3",
    "mha": "attention.mha",
    "mla_decode": "attention.mla_decode",
    "mla_decode_rope": "attention.mla_decode_rope",
    "pa_decode": "attention.pa_decode",
    "pa_mqa_logits": "attention.pa_mqa_logits",
    "pa_prefill": "attention.pa_prefill",
    "pod_attention": "attention.pod_attention",
    "prefill_attention": "attention.prefill_attention",
    "unified_attention_sparse_mla": "attention.unified_attention_sparse_mla",
    "unified_attention": "attention.unified_attention",
    # Fusions modules (fusions/)
    "fused_kv_cache": "fusions.fused_kv_cache",
    "fused_mul_add": "fusions.fused_mul_add",
    "fused_qk_concat": "fusions.fused_qk_concat",
    # MOE modules (moe/)
    "moe_align_block_size": "moe.moe_align_block_size",
    "moe_op_e2e": "moe.moe_op_e2e",
    "moe_op_gelu": "moe.moe_op_gelu",
    "moe_op_gemm_a8w4": "moe.moe_op_gemm_a8w4",
    "moe_op_gemm_a8w8": "moe.moe_op_gemm_a8w8",
    "moe_op_mxfp4_silu_fused": "moe.moe_op_mxfp4_silu_fused",
    "moe_op_mxfp4": "moe.moe_op_mxfp4",
    "moe_op_silu_fused": "moe.moe_op_silu_fused",
    "moe_op": "moe.moe_op",
    "moe_routing_sigmoid_top1_fused": "moe.moe_routing_sigmoid_top1_fused",
    "moe_routing": "moe.moe_routing",
    "quant_moe": "moe.quant_moe",
    # Normalization modules (normalization/)
    "fused_add_rmsnorm_pad": "normalization.fused_add_rmsnorm_pad",
    "norm": "normalization.norm",
    "rmsnorm": "normalization.rmsnorm",
    "fused_qkv_split_qk_rope": "rope.fused_qkv_split_qk_rope",
    # Utils modules (utils/)
    "common_utils": "utils.common_utils",
    "core": "utils.core",
    "device_info": "utils.device_info",
    "gmm_common": "utils.gmm_common",
    "la_kernel_utils": "utils.la_kernel_utils",
    "logger": "utils.logger",
    "mha_kernel_utils": "utils.mha_kernel_utils",
    "moe_common": "utils.moe_common",
    "moe_config_utils": "utils.moe_config_utils",
    "types": "utils.types",
    # Quant modules (quant/)
    "fused_fp8_quant": "quant.fused_fp8_quant",
    "fused_mxfp4_quant": "quant.fused_mxfp4_quant",
}


def __getattr__(name):
    """
    Handles attribute access to the triton module
    example -
    import aiter.ops.triton
    x = aiter.ops.triton.gemm_afp4wfp4
    """
    if name in _BACKWARD_COMPAT_MAP:
        new_path = f"aiter.ops.triton.{_BACKWARD_COMPAT_MAP[name]}"
        module = importlib.import_module(new_path)
        sys.modules[f"aiter.ops.triton.{name}"] = module
        return module
    raise AttributeError(f"module 'aiter.ops.triton' has no attribute '{name}'")


def _backward_compat_find_spec(fullname, path, target=None):
    """
    Handles the import of the triton module
    examples -
     from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
     import aiter.ops.triton.gemm_afp4wfp4
    """
    if fullname.startswith("aiter.ops.triton.") and fullname.count(".") == 3:
        name = fullname.split(".")[-1]
        if name in _BACKWARD_COMPAT_MAP:
            new_path = f"aiter.ops.triton.{_BACKWARD_COMPAT_MAP[name]}"
            try:
                sys.modules[fullname] = importlib.import_module(new_path)
                return importlib.util.find_spec(new_path)
            except ImportError:
                pass
    return None


# The SimpleNamespace is just to avoid creating a class and then creating an object
# This keeps it simple by letting us treat it like an object and makes it much more readable
sys.meta_path.insert(0, SimpleNamespace(find_spec=_backward_compat_find_spec))
