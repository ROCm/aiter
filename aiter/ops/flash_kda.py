# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The native kernel implementation wrapped by this module is adapted from
# MoonshotAI/FlashKDA (MIT, Copyright (c) 2026 MoonshotAI).

"""Native HIP/MFMA FlashKDA forward for gfx942 and gfx950.

This module owns the small aiter adapter around the architecture-specialized
FlashKDA kernels.  It deliberately exposes an allocation-owning Python API:
the C++ entry point only launches kernels into buffers allocated here, which
keeps it compatible with aiter's lightweight ``aiter_tensor_t`` JIT ABI.

The native implementation serves the Kimi-K3 prefill contract and its grouped
value-attention (GVA) extension: BF16 Q/K/V/raw-gate, FP32 beta logits and gate
parameters, K=V=128, and V-first recurrent state.  GVA maps each value head to
``hv // (HV // H)`` in the smaller Q/K head set.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from ..jit import core as _jit_core
from ..jit.core import compile_ops, get_module

MD_NAME = "module_flash_kda_hip"

FLASH_KDA_NATIVE_ARCHS = frozenset({"gfx942", "gfx950"})
FLASH_KDA_NATIVE_CHUNK = 16
FLASH_KDA_NATIVE_DIM = 128

# Workspace ABI shared with the imported FlashKDA K1/K2 launchers.  Keep these
# names next to the sizing function so changes to the native hand-off layout
# cannot be mistaken for ordinary tuning constants.
_K_DECAYED = FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_DIM * 2
_Q_DECAYED = FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_DIM * 2
_K_RESTORED = FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_DIM * 2
_G_TOTAL = FLASH_KDA_NATIVE_DIM * 4
_INV = FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_CHUNK * 2
_MQK = FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_CHUNK * 2
_PER_TILE = _K_DECAYED + _Q_DECAYED + _K_RESTORED + _G_TOTAL + _INV + _MQK
_CSPLIT_U = FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_DIM * 2
_CSPLIT_SIN = FLASH_KDA_NATIVE_DIM * FLASH_KDA_NATIVE_DIM * 2
_CSPLIT_CROSS = FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_CHUNK * 2
_CSPLIT_CROSS64 = 4 * FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_CHUNK * 2
_CSPLIT_BETA = 64 * 4
_CSPLIT_SEGMENT_A = (
    10 * FLASH_KDA_NATIVE_CHUNK * FLASH_KDA_NATIVE_CHUNK * 2
)


@compile_ops(MD_NAME, develop=True)
def flash_kda_fwd_hip(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    out: Tensor,
    workspace: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    initial_state: Tensor,
    final_state: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    lower_bound: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
) -> None: ...


def flash_kda_workspace_size(total_tokens: int, num_heads: int, num_seqs: int) -> int:
    """Return the native workspace size in bytes.

    The conservative ``+ num_seqs`` terms are part of FlashKDA's dense/varlen
    allocation contract and intentionally over-allocate partial tiles.
    """

    if total_tokens < 0 or num_heads <= 0 or num_seqs <= 0:
        raise ValueError(
            "Expected total_tokens >= 0, num_heads > 0 and num_seqs > 0, got "
            f"{total_tokens}, {num_heads}, {num_seqs}."
        )
    total_tiles = (total_tokens + 15) // 16 + num_seqs
    total_pairs = (total_tokens + 31) // 32 + num_seqs
    total_segments = (total_tokens + 63) // 64 + num_seqs
    # Three (N + 1) prefix arrays, an N-entry persistent worklist, a device
    # sequence count, and an atomic task counter. Keep this byte-for-byte in
    # sync with WorkspaceSizes::prefix_bytes in csrc/include/flash_kda.h.
    prefix_bytes = ((4 * num_seqs + 5) * 4 + 127) // 128 * 128
    return (
        num_heads * total_tiles * _PER_TILE
        + prefix_bytes
        + num_heads * total_tiles * _CSPLIT_U
        + num_heads * total_segments * _CSPLIT_SIN
        + num_heads * total_pairs * _CSPLIT_CROSS
        + num_heads * total_segments * _CSPLIT_CROSS64
        + num_heads * total_segments * _CSPLIT_BETA
        + num_heads * total_segments * _CSPLIT_SEGMENT_A
    )


def _normalize_max_seqlen_upper_bound(
    value: int | None,
    *,
    total_tokens: int,
    num_seqs: int,
    dense_seqlen: int,
    is_varlen: bool,
) -> int | None:
    """Validate a host-only packed routing hint without reading device data."""

    if value is not None and type(value) is not int:
        raise TypeError(
            "max_seqlen_upper_bound must be a Python int or None, "
            f"got {value!r}"
        )
    if not is_varlen:
        # Dense geometry already carries the exact per-sequence length.  Do
        # not let a caller-provided estimate alter its route.
        return dense_seqlen
    if value is None:
        return None
    minimum = (total_tokens + num_seqs - 1) // num_seqs
    if not minimum <= value <= total_tokens:
        raise ValueError(
            "packed max_seqlen_upper_bound must satisfy "
            f"ceil(total_tokens / num_seqs) <= bound <= total_tokens; got "
            f"bound={value}, total_tokens={total_tokens}, num_seqs={num_seqs}"
        )
    return value


@lru_cache(maxsize=None)
def _device_arch(device: torch.device) -> str | None:
    """Return the immutable GCN architecture without a per-call HIP query."""

    if (
        torch.device(device).type != "cuda"
        or torch.version.hip is None
        or not torch.cuda.is_available()
    ):
        return None
    try:
        props = torch.cuda.get_device_properties(device)
    except (AssertionError, RuntimeError, ValueError):
        return None
    arch = getattr(props, "gcnArchName", None)
    return arch.split(":", 1)[0] if arch else None


def _native_rejection_reason(
    *,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor | None,
    dt_bias: Tensor | None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: float | None = -5.0,
    state_v_first: bool = True,
    chunk_size: int | None = None,
    cu_seqlens: Tensor | None = None,
    max_seqlen_upper_bound: int | None = None,
    **_: Any,
) -> str | None:
    """Return why a call cannot use native FlashKDA, or ``None`` if it can."""

    if os.getenv("AITER_TRITON_ONLY", "0") == "1":
        return "AITER_TRITON_ONLY=1 disables native HIP operators"
    arch = _device_arch(q.device)
    if arch not in FLASH_KDA_NATIVE_ARCHS:
        supported = sorted(FLASH_KDA_NATIVE_ARCHS)
        return f"device architecture {arch!r} is not one of {supported}"
    if chunk_size is not None:
        return "an explicit chunk_size belongs to the Triton implementation"
    if q.ndim != 4 or k.shape != q.shape:
        return "q and k must have matching [B,T,H,K] shapes"
    if v.ndim != 4 or g.ndim != 4 or beta.ndim != 3:
        return "v/g/beta ranks must be 4/4/3"
    B, T, H, K = q.shape
    if B <= 0 or T <= 0 or H <= 0:
        return "B, T and H must be positive"
    if K != FLASH_KDA_NATIVE_DIM or v.shape[-1] != FLASH_KDA_NATIVE_DIM:
        return "native FlashKDA requires K=V=128"
    if tuple(v.shape[:2]) != (B, T):
        return "v must match q's batch/token dimensions"
    HV = int(v.shape[2])
    if HV < H or HV % H != 0:
        return "native FlashKDA requires HV >= H and HV divisible by H"
    if HV != H:
        loaded_binding = _get_raw_pointer_binding()
        if loaded_binding is not None and loaded_binding[1] < 3:
            return (
                "the loaded native FlashKDA extension predates the raw-v3 "
                "GVA ABI; rebuild module_flash_kda_hip"
            )
    if tuple(g.shape) != (B, T, HV, K) or tuple(beta.shape) != (B, T, HV):
        return "g and beta must match v's batch/token/value-head dimensions"
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, g)):
        return "q/k/v/g must be bfloat16"
    if beta.dtype not in (torch.float32, torch.bfloat16):
        return "beta must be float32 or bfloat16"
    if (
        A_log is None
        or A_log.dtype != torch.float32
        or A_log.ndim != 1
        or A_log.numel() != HV
    ):
        return "A_log must be float32 [HV]"
    if (
        dt_bias is None
        or dt_bias.dtype != torch.float32
        or dt_bias.ndim not in (1, 2)
        or dt_bias.numel() != HV * K
        or (dt_bias.ndim == 2 and tuple(dt_bias.shape) != (HV, K))
    ):
        return "dt_bias must be contiguous-compatible float32 [HV*K] or [HV,K]"
    if not (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and use_beta_sigmoid_in_kernel
        and safe_gate
    ):
        return (
            "native FlashKDA requires fused l2norm, raw gate, beta sigmoid "
            "and safe_gate"
        )
    if lower_bound is None or not -5.0 <= float(lower_bound) < 0.0:
        return "lower_bound must be in [-5, 0)"
    if (initial_state is not None or output_final_state) and not state_v_first:
        return "native FlashKDA recurrent state is V-first"
    if initial_state is not None:
        if initial_state.dtype not in (torch.float32, torch.bfloat16):
            return "initial_state must be float32 or bfloat16"
        N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
        if tuple(initial_state.shape) != (N, HV, 128, 128):
            return "initial_state must be [N,HV,V,K] with V=K=128"
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            return "packed varlen mode requires B=1 and 1D cu_seqlens [N+1]"
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            return "cu_seqlens must be int32 or int64"
    N = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else B
    _normalize_max_seqlen_upper_bound(
        max_seqlen_upper_bound,
        total_tokens=B * T,
        num_seqs=N,
        dense_seqlen=T,
        is_varlen=cu_seqlens is not None,
    )
    tensors = [q, k, v, g, beta, A_log, dt_bias]
    if initial_state is not None:
        tensors.append(initial_state)
    if cu_seqlens is not None:
        tensors.append(cu_seqlens)
    if any(t.device != q.device for t in tensors):
        return "all tensors must be on q.device"
    return None


def flash_kda_native_supported(**kwargs: Any) -> bool:
    """Whether the native gfx942/gfx950 implementation can serve ``kwargs``."""

    try:
        return _native_rejection_reason(**kwargs) is None
    except (AttributeError, TypeError, ValueError):
        return False


_RAW_POINTER_BINDING: tuple[Any, int] | None = None


def _get_raw_pointer_binding() -> tuple[Any, int] | None:
    """Return ``(callable, ABI version)`` once the JIT module is available.

    The first source-only invocation may still need ``compile_ops`` to build
    the extension.  In that case the caller uses the descriptor ABI once and
    retries this lookup on its next invocation.  Respect ``AITER_REBUILD`` as
    well, otherwise a developer rebuild could be skipped merely because an
    older module was importable.
    """

    global _RAW_POINTER_BINDING
    if (
        _jit_core.AITER_REBUILD
        and MD_NAME not in _jit_core.rebuilded_list
    ):
        # A cached callable belongs to the pre-rebuild extension.  Drop it so
        # the first post-build lookup can discover the newest additive symbol.
        _RAW_POINTER_BINDING = None
        return None
    if _RAW_POINTER_BINDING is not None:
        return _RAW_POINTER_BINDING
    try:
        module = get_module(MD_NAME)
    except ModuleNotFoundError:
        return None
    for version, symbol in (
        (3, "flash_kda_fwd_hip_raw_v3"),
        (2, "flash_kda_fwd_hip_raw_v2"),
        (1, "flash_kda_fwd_hip_raw"),
    ):
        candidate = getattr(module, symbol, None)
        if callable(candidate):
            _RAW_POINTER_BINDING = (candidate, version)
            return _RAW_POINTER_BINDING
    return None


def _get_raw_pointer_op() -> Any | None:
    """Compatibility accessor returning the best direct pybind callable."""

    binding = _get_raw_pointer_binding()
    return None if binding is None else binding[0]


def _call_raw_pointer_binding(
    binding: tuple[Any, int],
    raw_v1_args: tuple[Any, ...],
    max_seqlen_upper_bound: int | None,
    num_qk_heads: int,
) -> None:
    """Invoke a versioned raw binding while preserving the 25-argument v1 ABI."""

    if len(raw_v1_args) != 25:
        raise RuntimeError(
            "internal FlashKDA raw-v1 argument tuple must contain 25 values, "
            f"got {len(raw_v1_args)}"
        )
    raw_op, raw_version = binding
    if raw_version == 3:
        raw_op(
            *raw_v1_args,
            0 if max_seqlen_upper_bound is None else max_seqlen_upper_bound,
            num_qk_heads,
        )
    elif raw_version == 2:
        raw_op(
            *raw_v1_args,
            0 if max_seqlen_upper_bound is None else max_seqlen_upper_bound,
        )
    elif raw_version == 1:
        raw_op(*raw_v1_args)
    else:
        raise RuntimeError(f"unsupported FlashKDA raw ABI version {raw_version}")


def _torch_is_compiling() -> bool:
    """Whether pointer extraction must be avoided for a torch.compile trace."""

    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return True
    dynamo = getattr(torch, "_dynamo", None)
    return bool(dynamo is not None and dynamo.is_compiling())


def _flash_kda_fwd_prevalidated(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    lower_bound: float = -5.0,
    cu_seqlens: Tensor | None = None,
    max_seqlen_upper_bound: int | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Allocation-owning native path after ``_native_rejection_reason``.

    K3's auto/native router has already performed the complete metadata
    admission check before entering here.  Keeping this private contract lets
    that hot path use the raw-pointer ABI without repeating the Python tensor
    walk, while the public ``flash_kda_fwd`` entry below remains defensive.
    """

    B, T, H, K = q.shape
    HV = int(v.shape[2])
    N = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else B
    normalized_max_seqlen = _normalize_max_seqlen_upper_bound(
        max_seqlen_upper_bound,
        total_tokens=B * T,
        num_seqs=N,
        dense_seqlen=T,
        is_varlen=cu_seqlens is not None,
    )
    scale = K**-0.5 if scale is None else float(scale)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")

    # The imported kernels use contiguous pointer arithmetic.  K3 serving
    # tensors already satisfy this contract, so these branches are normally
    # no-ops; standalone callers retain view and dtype normalization.
    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and g.is_contiguous()
    ):
        q, k, v, g = (x.contiguous() for x in (q, k, v, g))
    if beta.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("beta must be float32 or bfloat16")
    if beta.dtype != torch.float32:
        beta = beta.float()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    if not A_log.is_contiguous():
        A_log = A_log.contiguous()
    if dt_bias.ndim != 1:
        dt_bias = dt_bias.reshape(-1)
    if not dt_bias.is_contiguous():
        dt_bias = dt_bias.contiguous()
    if initial_state is not None and not initial_state.is_contiguous():
        initial_state = initial_state.contiguous()

    is_varlen = cu_seqlens is not None
    if is_varlen:
        # ATOM's native metadata contract is int32.  Accepting int64 here keeps
        # the standalone API friendly while avoiding any conversion in K3.
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be int32 or int64")
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(dtype=torch.int32)
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()

    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    final_state = (
        torch.empty((N, HV, 128, 128), device=q.device, dtype=state_dtype)
        if output_final_state
        else None
    )
    out = torch.empty_like(v)
    workspace = torch.empty(
        flash_kda_workspace_size(B * T, HV, N),
        device=q.device,
        dtype=torch.uint8,
    )

    raw_binding = None if _torch_is_compiling() else _get_raw_pointer_binding()
    # Raw ABI v1/v2 carry a single head count and therefore cannot describe
    # GVA.  A source build reaches the descriptor entry once, then discovers
    # the additive v3 symbol on subsequent calls.
    if raw_binding is not None and raw_binding[1] < 3 and H != HV:
        raise RuntimeError(
            "the loaded native FlashKDA extension predates the raw-v3 GVA "
            "ABI; rebuild module_flash_kda_hip"
        )

    def launch() -> None:
        if raw_binding is not None:
            device_index = q.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            raw_args = (
                q.data_ptr(),
                k.data_ptr(),
                v.data_ptr(),
                g.data_ptr(),
                beta.data_ptr(),
                out.data_ptr(),
                workspace.data_ptr(),
                A_log.data_ptr(),
                dt_bias.data_ptr(),
                initial_state.data_ptr() if initial_state is not None else 0,
                final_state.data_ptr() if final_state is not None else 0,
                cu_seqlens.data_ptr() if cu_seqlens is not None else 0,
                B,
                T,
                HV,
                N,
                workspace.nbytes,
                scale,
                float(lower_bound),
                initial_state is not None,
                output_final_state,
                is_varlen,
                state_dtype == torch.float32,
                device_index,
                torch.cuda.current_stream(q.device).cuda_stream,
            )
            _call_raw_pointer_binding(
                raw_binding,
                raw_args,
                normalized_max_seqlen,
                H,
            )
            return

        # JIT cold start and torch.compile use the tensor-descriptor custom op.
        # Its boolean guards make serving tensors safe zero-cost placeholders
        # for absent optional arguments.
        initial_arg = initial_state if initial_state is not None else q
        final_arg = final_state if final_state is not None else initial_arg
        cu_arg = cu_seqlens if cu_seqlens is not None else q
        flash_kda_fwd_hip(
            q,
            k,
            v,
            g,
            beta,
            out,
            workspace,
            A_log,
            dt_bias,
            initial_arg,
            final_arg,
            cu_arg,
            scale,
            float(lower_bound),
            initial_state is not None,
            output_final_state,
            is_varlen,
        )

    # The raw C++ ABI checks both the active device and stream ownership.  The
    # common path avoids entering a redundant device context; mixed-device
    # processes get the same restore-on-exit semantics as the descriptor ABI.
    if q.device.index == torch.cuda.current_device():
        launch()
    else:
        with torch.cuda.device(q.device):
            launch()
    return out, final_state


def flash_kda_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    lower_bound: float = -5.0,
    cu_seqlens: Tensor | None = None,
    max_seqlen_upper_bound: int | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Run native FlashKDA and return ``(output, final_state)``.

    ``initial_state`` and the returned state use the Kimi-K3 V-first layout
    ``[N,HV,V,K]``.  Both dense inputs and B=1 packed-varlen inputs are
    supported, including GVA when ``HV`` is an integer multiple of the Q/K
    head count ``H``. Unsupported devices or shapes raise rather than silently
    selecting another implementation; use ``chunk_kimi_delta_attn`` with
    ``backend="auto"`` when a fallback is desired.

    ``max_seqlen_upper_bound`` is a route-only host hint for packed inputs.
    It must be a built-in Python ``int`` (not ``bool`` or an ``int`` subclass),
    lie in ``[ceil(total_tokens / N), total_tokens]``, and should be a static
    bucket bound when this call is captured in a CUDA/HIP graph.  Dense calls
    always use the exact shape-derived sequence length.  Older native modules
    without the additive raw-v3 symbol retain their legacy equal-head route.
    """

    # Validate the public scalar before architecture admission so malformed
    # API input has stable TypeError/ValueError behavior on every device.
    if (
        max_seqlen_upper_bound is not None
        and type(max_seqlen_upper_bound) is not int
    ):
        raise TypeError(
            "max_seqlen_upper_bound must be a Python int or None, "
            f"got {max_seqlen_upper_bound!r}"
        )
    if q.ndim == 4:
        B, T = int(q.shape[0]), int(q.shape[1])
        N = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else B
        if B > 0 and T > 0 and N > 0:
            _normalize_max_seqlen_upper_bound(
                max_seqlen_upper_bound,
                total_tokens=B * T,
                num_seqs=N,
                dense_seqlen=T,
                is_varlen=cu_seqlens is not None,
            )

    reason = _native_rejection_reason(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        max_seqlen_upper_bound=max_seqlen_upper_bound,
    )
    if reason is not None:
        raise ValueError(f"Native FlashKDA cannot serve this call: {reason}.")
    return _flash_kda_fwd_prevalidated(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        max_seqlen_upper_bound=max_seqlen_upper_bound,
    )


__all__ = [
    "FLASH_KDA_NATIVE_ARCHS",
    "FLASH_KDA_NATIVE_CHUNK",
    "flash_kda_fwd",
    "flash_kda_native_supported",
    "flash_kda_workspace_size",
]
