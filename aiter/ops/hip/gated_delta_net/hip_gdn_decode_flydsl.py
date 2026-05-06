from typing import Optional
import math

import torch
from aiter.ops.flydsl.kernels.gdr_decode import create_shuffle_gdr_decode_kernel
from aiter.ops.flydsl.kernels.tensor_shim import get_dtype_str

_flydsl_kernels = {}
_flydsl_compiled_launchers = {}


def _prepare_flydsl_tensor(
    tensor: torch.Tensor, *, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    out = tensor.detach() if tensor.requires_grad else tensor
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    if not out.is_contiguous():
        out = out.contiguous()
    return out


def prepare_flydsl_gdn_decode_static_inputs(
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
):
    dt_bias_bf16 = _prepare_flydsl_tensor(dt_bias, dtype=torch.bfloat16)
    return (
        dt_bias_bf16.contiguous(),
        _prepare_flydsl_tensor(A_log),
    )


def prepare_flydsl_gdn_decode_dynamic_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_indices: torch.Tensor,
    initial_state_source: torch.Tensor,
):
    indices_int32 = _prepare_flydsl_tensor(initial_state_indices, dtype=torch.int32)
    return (
        _prepare_flydsl_tensor(q),
        _prepare_flydsl_tensor(k),
        _prepare_flydsl_tensor(v),
        _prepare_flydsl_tensor(a),
        _prepare_flydsl_tensor(b),
        _prepare_flydsl_tensor(initial_state_source),
        indices_int32,
    )


def prepare_flydsl_gdn_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    initial_state_indices: torch.Tensor,
    initial_state_source: torch.Tensor,
):
    (
        q_launch,
        k_launch,
        v_launch,
        a_launch,
        b_launch,
        state_launch,
        indices_int32,
    ) = prepare_flydsl_gdn_decode_dynamic_inputs(
        q,
        k,
        v,
        a,
        b,
        initial_state_indices,
        initial_state_source,
    )
    dt_bias_launch, A_log_launch = prepare_flydsl_gdn_decode_static_inputs(
        dt_bias,
        A_log,
    )

    return (
        q_launch,
        k_launch,
        v_launch,
        a_launch,
        b_launch,
        dt_bias_launch,
        A_log_launch,
        state_launch,
        indices_int32,
    )


def _device_cache_key(device: torch.device):
    return (device.type, device.index)


def _require_same_device(*tensors: torch.Tensor) -> torch.device:
    device = tensors[0].device
    for tensor in tensors[1:]:
        if tensor.device != device:
            raise ValueError(
                "FlyDSL GDN decode expects all tensors on one device, "
                f"but got {device} and {tensor.device}."
            )
    return device


def get_flydsl_gdn_kernel_cache_key(
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_blocks_per_v_dim: int,
    seq_length: int,
    dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
    device: torch.device,
):
    return (
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        num_blocks_per_v_dim,
        seq_length,
        str(dtype),
        use_qk_l2norm_in_kernel,
        _device_cache_key(device),
    )


def _validate_flydsl_gdn_decode_args(
    *, head_k_dim: int, scale: Optional[float], softplus_beta: float, softplus_threshold: float
):
    expected_scale = head_k_dim**-0.5
    if scale is not None and not math.isclose(scale, expected_scale, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "FlyDSL GDN decode only supports the default scale "
            f"({expected_scale}), but got {scale}."
        )
    if not math.isclose(softplus_beta, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "FlyDSL GDN decode only supports softplus_beta=1.0, "
            f"but got {softplus_beta}."
        )
    if not math.isclose(softplus_threshold, 20.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "FlyDSL GDN decode only supports softplus_threshold=20.0, "
            f"but got {softplus_threshold}."
        )


def get_flydsl_num_blocks_per_v_dim(batch_size: int) -> int:
    if batch_size <= 3:
        return 4
    if batch_size <= 10:
        return 2
    return 1


def get_flydsl_gdn_launch_callable(
    cache_key,
    kernel,
    q,
    k,
    v,
    a,
    b,
    dt_bias,
    A_log,
    indices,
    state,
    out,
    batch_size,
    stream,
):
    launch_key = (cache_key, id(kernel), batch_size)
    if launch_key not in _flydsl_compiled_launchers:
        _flydsl_compiled_launchers[launch_key] = kernel.compile(
            q,
            k,
            v,
            a,
            b,
            dt_bias,
            A_log,
            indices,
            state,
            out,
            batch_size,
            stream,
        )
    return _flydsl_compiled_launchers[launch_key]


def launch_flydsl_gdn_decode(
    cache_key,
    kernel,
    q,
    k,
    v,
    a,
    b,
    dt_bias,
    A_log,
    indices,
    state,
    out,
    batch_size,
    stream,
):
    launch = get_flydsl_gdn_launch_callable(
        cache_key,
        kernel,
        q,
        k,
        v,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        state,
        out,
        batch_size,
        stream,
    )
    launch(
        q,
        k,
        v,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        state,
        out,
        batch_size,
        stream,
    )


class FlydslGdnDecodeLayerRunner:
    def __init__(
        self,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        device: torch.device,
        use_qk_l2norm_in_kernel: bool,
    ):
        self._device = device
        self._num_k_heads = num_k_heads
        self._num_v_heads = num_v_heads
        self._head_k_dim = head_k_dim
        self._head_v_dim = head_v_dim
        self._use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self._dt_bias_launch, self._A_log_launch = (
            prepare_flydsl_gdn_decode_static_inputs(dt_bias, A_log)
        )
        self._kernels = {}

    def _get_kernel(self, dtype: torch.dtype, num_blocks_per_v_dim: int):
        kernel_key = (str(dtype), num_blocks_per_v_dim)
        if kernel_key not in self._kernels:
            self._kernels[kernel_key] = create_shuffle_gdr_decode_kernel(
                dtype=get_dtype_str(dtype),
                seq_length=1,
                num_k_heads=self._num_k_heads,
                num_v_heads=self._num_v_heads,
                head_k_dim=self._head_k_dim,
                head_v_dim=self._head_v_dim,
                use_qk_l2norm=self._use_qk_l2norm_in_kernel,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                NUM_BLOCKS_PER_V_DIM=num_blocks_per_v_dim,
                device=str(self._device),
            )
        return self._kernels[kernel_key]

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        initial_state_source: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        active_batch_size: Optional[int] = None,
    ):
        if cu_seqlens is None:
            raise ValueError("FlyDSL layer runner is only supported for decode mode with cu_seqlens.")

        device = _require_same_device(
            q,
            k,
            v,
            a,
            b,
            initial_state_indices,
            initial_state_source,
        )
        if device != self._device:
            raise ValueError(
                "FlyDSL layer runner was created for "
                f"{self._device}, but got runtime tensors on {device}."
            )

        batch_size = len(cu_seqlens) - 1
        if active_batch_size is None:
            active_batch_size = batch_size
        active_batch_size = int(active_batch_size)
        if active_batch_size < 0 or active_batch_size > batch_size:
            raise ValueError(
                "FlyDSL layer runner got an invalid active batch size: "
                f"{active_batch_size} (batch size {batch_size})."
            )
        full_out = None
        if active_batch_size < batch_size:
            if active_batch_size == 0:
                return torch.zeros_like(v, memory_format=torch.contiguous_format)
            # The backend precomputes active_batch_size to avoid tensor-to-host
            # syncs during CUDA graph capture.
            full_out = torch.zeros_like(v, memory_format=torch.contiguous_format)
            q = q[:, :active_batch_size]
            k = k[:, :active_batch_size]
            v = v[:, :active_batch_size]
            a = a[:, :active_batch_size]
            b = b[:, :active_batch_size]
            initial_state_indices = initial_state_indices[:active_batch_size]
            cu_seqlens = cu_seqlens[: active_batch_size + 1]
            batch_size = active_batch_size

        (
            q_launch,
            k_launch,
            v_launch,
            a_launch,
            b_launch,
            state_launch,
            indices_int32,
        ) = prepare_flydsl_gdn_decode_dynamic_inputs(
            q,
            k,
            v,
            a,
            b,
            initial_state_indices,
            initial_state_source,
        )
        out = torch.empty_like(v_launch, memory_format=torch.contiguous_format)

        num_blocks_per_v_dim = get_flydsl_num_blocks_per_v_dim(batch_size)
        kernel = self._get_kernel(q_launch.dtype, num_blocks_per_v_dim)
        kernel_key = (str(q_launch.dtype), num_blocks_per_v_dim)
        stream = torch.cuda.current_stream(device=device).cuda_stream
        launch_flydsl_gdn_decode(
            kernel_key,
            kernel,
            q_launch,
            k_launch,
            v_launch,
            a_launch,
            b_launch,
            self._dt_bias_launch,
            self._A_log_launch,
            indices_int32,
            state_launch,
            out,
            batch_size,
            stream,
        )
        if full_out is not None:
            full_out[:, :batch_size].copy_(out)
            return full_out
        return out


def create_flydsl_gdn_decode_layer_runner(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    device: torch.device,
    use_qk_l2norm_in_kernel: bool,
):
    return FlydslGdnDecodeLayerRunner(
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        device=device,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def flydsl_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
):
    del is_kda

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    _validate_flydsl_gdn_decode_args(
        head_k_dim=K,
        scale=scale,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    device = _require_same_device(
        q, k, v, a, b, dt_bias, A_log, initial_state_indices, initial_state_source
    )

    if scale is None:
        scale = K**-0.5

    N = B * T if cu_seqlens is None else len(cu_seqlens) - 1

    (
        q_launch,
        k_launch,
        v_launch,
        a_launch,
        b_launch,
        dt_bias_launch,
        A_log_launch,
        state_launch,
        indices_int32,
    ) = prepare_flydsl_gdn_decode_inputs(
        q,
        k,
        v,
        a,
        b,
        dt_bias,
        A_log,
        initial_state_indices,
        initial_state_source,
    )
    o = torch.empty_like(v_launch, memory_format=torch.contiguous_format)

    batch_size = N
    seq_length = 1 if cu_seqlens is not None else T
    num_k_heads = H
    num_v_heads = HV

    nvb = get_flydsl_num_blocks_per_v_dim(batch_size)

    cache_key = get_flydsl_gdn_kernel_cache_key(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=K,
        head_v_dim=V,
        num_blocks_per_v_dim=nvb,
        seq_length=seq_length,
        dtype=q_launch.dtype,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        device=device,
    )
    if cache_key not in _flydsl_kernels:
        _flydsl_kernels[cache_key] = create_shuffle_gdr_decode_kernel(
            dtype=get_dtype_str(q_launch.dtype),
            seq_length=seq_length,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=K,
            head_v_dim=V,
            use_qk_l2norm=use_qk_l2norm_in_kernel,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            NUM_BLOCKS_PER_V_DIM=nvb,
            device=str(device),
        )

    fly_kernel = _flydsl_kernels[cache_key]
    stream = torch.cuda.current_stream(device=device).cuda_stream
    launch_flydsl_gdn_decode(
        cache_key,
        fly_kernel,
        q_launch,
        k_launch,
        v_launch,
        a_launch,
        b_launch,
        dt_bias_launch,
        A_log_launch,
        indices_int32,
        state_launch,
        o,
        batch_size,
        stream,
    )

    return o
