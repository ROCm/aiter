import torch
import functools


@functools.cache
def _get_compiled_gdr_decode(
    dtype: torch.dtype,
    b: int,
    sq: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_qk_l2norm: bool,
):
    from .kernels.gdr_decode import Args, get_func

    args_ = Args(
        dtype=dtype,
        b=b,
        sq=sq,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        use_qk_l2norm=use_qk_l2norm,
    )
    exe = get_func(args_)

    def tensor_api(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        A_log: torch.Tensor,
        indices: torch.Tensor,
        initial_state: torch.Tensor,
        out: torch.Tensor,
        batch_size: int,
        scale: float,
    ) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        exe(
            q,
            k,
            v,
            a,
            b,
            dt_bias,
            A_log,
            indices,
            initial_state,
            out,
            batch_size,
            scale,
            stream,
        )

    return tensor_api


def flydsl_gdr_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    bc, sq, num_k_heads, head_k_dim = q.shape
    num_v_heads = v.shape[-2]
    head_v_dim = v.shape[-1]
    out = torch.empty((bc, sq, num_v_heads, head_v_dim), dtype=q.dtype, device=q.device)
    tensor_api = _get_compiled_gdr_decode(
        q.dtype, bc, sq, num_k_heads, num_v_heads, head_k_dim, head_v_dim, use_qk_l2norm
    )
    tensor_api(
        q,
        k,
        v,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        initial_state,
        out,
        bc,
        float(1.0 / (head_k_dim**0.5)),
    )
    return out
