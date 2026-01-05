# This file implements USP with torch version >= '2.5.0'
import torch
from torch.nn import functional as F

import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention

if torch.cuda.is_available():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_rank,
)

from packaging.version import parse
from xfuser.envs import PACKAGES_CHECKER
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed import get_runtime_state



env_info = PACKAGES_CHECKER.get_packages_info()
HAS_FLASH_ATTN = env_info["has_flash_attn"]
if HAS_FLASH_ATTN:
    import flash_attn

HAS_AITER = env_info["has_aiter"]
if HAS_AITER:
    import aiter
    import inspect
    try:
        HAS_ROUND_MODE = inspect.signature(aiter.flash_attn_func).parameters.get("how_v3_bf16_cvt") is not None
    except (AttributeError, TypeError):
        HAS_ROUND_MODE = False
    if HAS_ROUND_MODE:
        import os
        HOW_V3_BF16_CVT = int(os.environ.get("HOW_V3_BF16_CVT", "2"))


_fav3_fp8_func = None
def _get_fav3_fp8_func():
    global _fav3_fp8_func
    if _fav3_fp8_func is None:
        import importlib.util
        import sys
        import types
        
        # Create minimal mock modules for the aiter package hierarchy
        if 'aiter.ops.triton._triton_kernels.flash_attn_triton_amd' not in sys.modules:
            # Load the flash_attn_triton_amd package __init__.py
            fa3_init_path = "/workspace/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/__init__.py"
            fa3_spec = importlib.util.spec_from_file_location(
                "aiter.ops.triton._triton_kernels.flash_attn_triton_amd", 
                fa3_init_path
            )
            fa3_module = importlib.util.module_from_spec(fa3_spec)
            sys.modules[fa3_spec.name] = fa3_module
            
            # Create mock parent modules
            sys.modules['aiter.ops.triton._triton_kernels'] = types.ModuleType('aiter.ops.triton._triton_kernels')
            sys.modules['aiter.ops.triton'] = types.ModuleType('aiter.ops.triton')
            sys.modules['aiter.ops'] = types.ModuleType('aiter.ops')
            sys.modules['aiter.ops.triton._triton_kernels.flash_attn_triton_amd'] = fa3_module
            
            fa3_spec.loader.exec_module(fa3_module)
        
        # Load mha_v3 which will use the already-loaded flash_attn_triton_amd
        mha_v3_path = "/workspace/aiter/aiter/ops/triton/mha_v3.py"
        spec = importlib.util.spec_from_file_location("mha_v3_workspace", mha_v3_path)
        mha_v3_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mha_v3_module
        spec.loader.exec_module(mha_v3_module)
        
        _fav3_fp8_func = mha_v3_module.flash_attn_fp8_func
    
    return _fav3_fp8_func


def _load_workspace_module(module_name: str, module_path: str, raise_on_error: bool = False):
    """Load a module directly from a workspace path and return it, or None on failure."""
    import importlib.util
    import sys

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
        if raise_on_error:
            raise ImportError(f"spec/loader missing for {module_path}")
    except Exception as exc:
        if raise_on_error:
            raise
        return None
    return None


def _ensure_attn_qk_module():
    """Ensure aiter.ops.triton.attn_qk_int8_per_block is importable from workspace."""
    import importlib
    import importlib.util
    import sys
    import types

    module_name = "aiter.ops.triton.attn_qk_int8_per_block"
    if module_name in sys.modules:
        return

    # Ensure workspace path is on sys.path so package imports resolve
    workspace_root = "/workspace/aiter"
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    # Create parent packages with stubs like fav3
    if "aiter" not in sys.modules:
        sys.modules["aiter"] = types.ModuleType("aiter")
    if "aiter.ops" not in sys.modules:
        sys.modules["aiter.ops"] = types.ModuleType("aiter.ops")
    if "aiter.ops.triton" not in sys.modules:
        sys.modules["aiter.ops.triton"] = types.ModuleType("aiter.ops.triton")
    if "aiter.ops.triton._triton_kernels" not in sys.modules:
        sys.modules["aiter.ops.triton._triton_kernels"] = types.ModuleType("aiter.ops.triton._triton_kernels")

    # Load the _triton_kernels submodule that attn_qk_int8_per_block depends on
    kernel_module_name = "aiter.ops.triton._triton_kernels.attn_qk_int8_per_block"
    if kernel_module_name not in sys.modules:
        kernel_path = "/workspace/aiter/aiter/ops/triton/_triton_kernels/attn_qk_int8_per_block.py"
        kernel_spec = importlib.util.spec_from_file_location(kernel_module_name, kernel_path)
        if kernel_spec and kernel_spec.loader:
            kernel_mod = importlib.util.module_from_spec(kernel_spec)
            sys.modules[kernel_module_name] = kernel_mod
            kernel_spec.loader.exec_module(kernel_mod)

    # Now load the main attn_qk_int8_per_block module
    attn_path = "/workspace/aiter/aiter/ops/triton/attn_qk_int8_per_block.py"
    spec = importlib.util.spec_from_file_location(module_name, attn_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)


_sagev1_func = None
def _get_sagev1_func():
    """Lazy-load sagev1 attention from the canonical workspace path."""

    global _sagev1_func
    if _sagev1_func is not None:
        return _sagev1_func

    module_path = "/workspace/aiter/op_tests/sagev1_tests/core.py"
    _ensure_attn_qk_module()
    try:
        mod = _load_workspace_module("sagev1_tests.core_workspace", module_path, raise_on_error=True)
    except Exception as exc:
        raise RuntimeError(f"kernel 'sagev1' load failed from {module_path}: {exc}") from exc

    fn = getattr(mod, "sageattn", None)
    if fn is None:
        raise RuntimeError(f"kernel 'sagev1' missing sageattn in {module_path}")

    _sagev1_func = fn
    return _sagev1_func


_fav3_sage_fn = None
def _ensure_sage_attn_module():
    """Preload sage_attn_triton_amd dependencies for fav3_sage."""
    import importlib.util
    import sys
    import types

    # Create parent module chain
    if "aiter" not in sys.modules:
        sys.modules["aiter"] = types.ModuleType("aiter")
    if "aiter.ops" not in sys.modules:
        sys.modules["aiter.ops"] = types.ModuleType("aiter.ops")
    if "aiter.ops.triton" not in sys.modules:
        sys.modules["aiter.ops.triton"] = types.ModuleType("aiter.ops.triton")
    if "aiter.ops.triton._triton_kernels" not in sys.modules:
        sys.modules["aiter.ops.triton._triton_kernels"] = types.ModuleType("aiter.ops.triton._triton_kernels")

    # Load sage_attn_triton_amd package
    sage_init_path = "/workspace/aiter/aiter/ops/triton/_triton_kernels/sage_attn_triton_amd/__init__.py"
    spec = importlib.util.spec_from_file_location("aiter.ops.triton._triton_kernels.sage_attn_triton_amd", sage_init_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

    # Load sage_attn_triton_amd.utils
    utils_path = "/workspace/aiter/aiter/ops/triton/_triton_kernels/sage_attn_triton_amd/utils.py"
    spec = importlib.util.spec_from_file_location("aiter.ops.triton._triton_kernels.sage_attn_triton_amd.utils", utils_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)


def _get_fav3_sage_attn():
    """Lazy-load fav3_sage attention from /workspace/aiter/aiter/ with dependency preloading."""
    global _fav3_sage_fn
    if _fav3_sage_fn is not None:
        return _fav3_sage_fn

    import importlib.util
    import sys

    # Preload dependencies that fav3_sage.py imports
    _ensure_attn_qk_module()  # Loads attn_qk_int8_per_block
    _ensure_sage_attn_module()  # Loads sage_attn_triton_amd and utils

    module_path = "/workspace/aiter/aiter/ops/triton/fav3_sage.py"
    spec = importlib.util.spec_from_file_location("aiter.ops.triton.fav3_sage_workspace", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to create spec for {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise RuntimeError(f"Failed to execute module {module_path}: {e}")

    _fav3_sage_fn = getattr(mod, "fav3_sage_wrapper_func", None)
    if _fav3_sage_fn is None:
        raise RuntimeError(f"Module {module_path} loaded but 'fav3_sage_wrapper_func' not found")

    return _fav3_sage_fn


aten = torch.ops.aten


def ring_attn(query, key, value, dropout_p=0.0, is_causal=False):
    if parse(torch.__version__).release >= parse("2.6.0").release:
        from torch.distributed.tensor.experimental._attention import _cp_options
        _cp_options.enable_load_balance = False
        kwargs = {
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if HAS_AITER or HAS_FLASH_ATTN:
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                1,
                _aiter_bf16_attn_call if HAS_AITER else _flash_attn_call,
                query,
                key,
                value,
                **kwargs,
            )
        else:
            kwargs = {
                **kwargs,
                "attn_bias": None,
                "compute_log_sumexp": True,
            }
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                1,
                aten._scaled_dot_product_efficient_attention,
                query,
                key,
                value,
                **kwargs,
            )
    else:
        kwargs = {
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if HAS_AITER:
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                1,
                _aiter_bf16_attn_call,
                query,
                key,
                value,
                **kwargs,
            )
        elif HAS_FLASH_ATTN:
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                _flash_attn_call,
                query,
                key,
                value,
                **kwargs
            )
        else:
            kwargs = {
                **kwargs,
                "attn_bias": None,
                "compute_log_sumexp": True,
            }
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                aten._scaled_dot_product_efficient_attention,
                query,
                key,
                value,
                **kwargs,
            )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor

def _check_if_use_fp8_attn():
    use_fp8_attn = False
    kernel = "default_fp8"
    try:
        use_fp8_attn = get_runtime_state().use_fp8_attn
        kernel = get_runtime_state().kernel
    except:
        pass
    if not HAS_AITER and use_fp8_attn:
        raise RuntimeError("FP8 attention requested but AITER is not available.")
    return use_fp8_attn, kernel


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x

def _aiter_fp8_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    softmax_lse = None
    quant_dtype = aiter.dtypes.fp8
    # Descale not yet supported in AITER.
    quant_q, _ = aiter.per_tensor_quant(query, scale=torch.tensor(1, device=query.device), quant_dtype=quant_dtype)
    quant_k, _ = aiter.per_tensor_quant(key, scale=torch.tensor(1, device=key.device), quant_dtype=quant_dtype)
    quant_v, _ = aiter.per_tensor_quant(value, scale=torch.tensor(1, device=value.device), quant_dtype=quant_dtype)
    torch._dynamo.graph_break() # Without this line, fp8 call will crash due to q and k dont have the same dtype, which is not true.
    output = aiter.flash_attn_fp8_pertensor_func(
        quant_q, quant_k, quant_v,
        causal=is_causal,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

def _fav3_fp8_call(query, key, value, dropout_p, is_causal):
    fn = _get_fav3_fp8_func()
    if fn is None:
        raise RuntimeError("kernel 'fav3_fp8' requested but flash_attn_fp8_func not found")
    
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    output = fn(
        query, key, value,
        causal=is_causal,
    )
    
    output = torch.permute(output, [0, 2, 1, 3])
    return output


def _sagev1_call(query, key, value, dropout_p, is_causal):
    """Call local sagev1 (expects HND) without caching the fn."""
    fn = _get_sagev1_func()
    if fn is None:
        raise RuntimeError("kernel 'sagev1' requested but sageattn is not available")
    output = fn(query, key, value, tensor_layout="HND", is_causal=is_causal)
    return output


def _fav3_sage_call(query, key, value, dropout_p, is_causal):
    """Call fav3_sage wrapper (expects BSHD), converting from BHSD and back."""
    fav3_sage_fn = _get_fav3_sage_attn()
    if fav3_sage_fn is None:
        raise RuntimeError("kernel 'fav3_sage' requested but fav3_sage_wrapper_func not found")

    # Convert BHSD -> BSHD
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    output = fav3_sage_fn(
        query,
        key,
        value,
        causal=is_causal,
        layout="bshd",
    )

    # Convert back BSHD -> BHSD
    output = torch.permute(output, [0, 2, 1, 3])
    return output, None

def _aiter_bf16_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    attn_kwargs = {
        "dropout_p": dropout_p,
        "causal": is_causal,
        "return_attn_probs": False,
        "return_lse": True,
    }
    if HAS_ROUND_MODE:
        attn_kwargs["how_v3_bf16_cvt"] = HOW_V3_BF16_CVT
    output, softmax_lse = aiter.flash_attn_func(
        query,
        key,
        value,
        **attn_kwargs
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

def _flash_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn
    """

    query = torch.permute(query, [0, 2, 1, 3])
    key = torch.permute(key, [0, 2, 1, 3])
    value = torch.permute(value, [0, 2, 1, 3])
    output, softmax_lse, S_mask = flash_attn.flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=is_causal,
        return_attn_probs=True,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

def _attention(query, key, value, dropout_p, is_causal):
    """
    Calls the correct attention mechanism based on the available libraries
    """
    use_fp8_attn, kernel = _check_if_use_fp8_attn()
    if HAS_AITER:
        if use_fp8_attn:
            if kernel == "fav3_fp8":
                output = _fav3_fp8_call(query, key, value, dropout_p, is_causal)
            elif kernel == "sagev1":
                output = _sagev1_call(query, key, value, dropout_p, is_causal)
            elif kernel == "fav3_sage":
                output, _ = _fav3_sage_call(query, key, value, dropout_p, is_causal)
            else:
                output, _ = _aiter_fp8_attn_call(query, key, value, dropout_p, is_causal)
        else:
            output, _ = _aiter_bf16_attn_call(query, key, value, dropout_p, is_causal)
        return output
    elif HAS_FLASH_ATTN:
        output, _ = _flash_attn_call(query, key, value, dropout_p, is_causal)
        return output
    else:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal
        )


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = (
        joint_key.shape[1] // ulysses_world_size
    )
    joint_key = joint_key.transpose(1,2)
    joint_value = joint_value.transpose(1,2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :, ].transpose(1,2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :,
    ].transpose(1,2)
    return joint_key, joint_value

def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor

def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value

def USP(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        joint_query: torch.Tensor | None = None,
        joint_key: torch.Tensor | None = None,
        joint_value: torch.Tensor | None = None,
        joint_strategy: str | None = None,
        attn_layer=None,
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    """

    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)

    if get_ulysses_parallel_world_size() > 1:
        query = _ft_c_input_all_to_all(query)
        key = _ft_c_input_all_to_all(key)
        value = _ft_c_input_all_to_all(value)

    if attn_layer:
        key, value = _update_and_get_kv_cache(key, value, attn_layer)
    if joint_strategy:
        key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
        value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)

    if get_sequence_parallel_world_size() == 1: # No SP
        out = _attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out = _attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
        else: # USP
            out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
        out = _ft_c_output_all_to_all(out)

    return out


