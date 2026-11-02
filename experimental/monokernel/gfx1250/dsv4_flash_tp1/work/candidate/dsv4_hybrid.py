"""Opt-in gfx1250 mHC epilogue quantization for DSV4 decode."""

import os
import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

_CHECK = os.environ.get("ATOM_DSV4_HYBRID_CHECK", "0") == "1"
_checked = 0
_MAX_TOKENS = min(128, int(os.environ.get("ATOM_DSV4_HYBRID_MAX_TOKENS", "128")))


def _hybrid_fake(
    x,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    norm_weight,
    rms_eps=1e-6,
    hc_eps=1e-6,
    sinkhorn_iters=20,
    res_preshuffle=True,
):
    m, c = x.shape
    return (
        torch.empty((m, 4, 1), device=x.device, dtype=torch.float32),
        torch.empty((m, 4, 4), device=x.device, dtype=torch.float32),
        torch.empty_like(x),
        torch.empty_like(residual),
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((m, c // 128), device=x.device, dtype=torch.float8_e8m0fnu),
    )


@torch_compile_guard(gen_fake=_hybrid_fake, mutates_args=[])
def dsv4_hybrid_hc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    res_preshuffle: bool = True,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    from aiter.ops.triton.fusions.mhc import mhc_post_pre_quant_gfx1250

    # Concrete shape dispatch belongs inside the opaque op: Dynamo compiles
    # the model at a large warmup shape and reuses it for small graph captures.
    if not 0 < x.shape[0] <= _MAX_TOKENS:
        import aiter
        from aiter.ops.quant import per_group_quant_hip

        ref = aiter.mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_iters,
            norm_weight,
            rms_eps,
            w_preshuffle_bf16=fn.dtype == torch.int32,
            res_preshuffle=res_preshuffle,
            backend="gluon",
        )
        q, qs = per_group_quant_hip(
            ref[2],
            quant_dtype=torch.float8_e4m3fn,
            group_size=128,
            transpose_scale=True,
            scale_type=torch.float8_e8m0fnu,
        )
        return (*ref, q, qs)

    result = mhc_post_pre_quant_gfx1250(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        norm_weight,
        eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_post_mult_value=2.0,
        sinkhorn_iters=sinkhorn_iters,
        hc_sinkhorn_eps=hc_eps,
        norm_eps=rms_eps,
        res_preshuffle=res_preshuffle,
    )
    global _checked
    if _CHECK and _checked < 16 and not torch.cuda.is_current_stream_capturing():
        import aiter
        from aiter.ops.quant import per_group_quant_hip

        ref = aiter.mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_iters,
            norm_weight,
            rms_eps,
            w_preshuffle_bf16=fn.dtype == torch.int32,
            res_preshuffle=res_preshuffle,
            backend="gluon",
        )
        for actual, expected in zip(result[:4], ref):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        q, qs = per_group_quant_hip(
            ref[2],
            quant_dtype=torch.float8_e4m3fn,
            group_size=128,
            transpose_scale=True,
            scale_type=torch.float8_e8m0fnu,
        )
        torch.testing.assert_close(
            result[4].view(torch.uint8), q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result[5].view(torch.uint8), qs.view(torch.uint8), rtol=0, atol=0
        )
        _checked += 1
        if _checked == 16:
            print("DSV4 hybrid mHC: 16 exact live checks passed", flush=True)
    return result


_projection_checked = 0


def _projection_fake(x, weight, x_scale, weight_scale):
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )


@torch_compile_guard(gen_fake=_projection_fake, mutates_args=[])
def dsv4_hybrid_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor | None,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Use AITER's tuned gfx1250 split-K projection for small decode batches."""
    from aiter.ops.quant import per_group_quant_hip
    from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle

    if x_scale is None:
        x, x_scale = per_group_quant_hip(
            x,
            quant_dtype=torch.float8_e4m3fn,
            group_size=128,
            transpose_scale=True,
            scale_type=torch.float8_e8m0fnu,
        )
    if not 0 < x.shape[0] <= 64:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle

        return gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, weight_scale)
    result = gemm_afp8wfp8_preshuffle(
        x,
        weight,
        x_scale.view(torch.uint8),
        weight_scale.view(torch.uint8),
        backend="gluon",
        is_x_scale_transposed=True,
    )
    global _projection_checked
    if (
        _CHECK
        and _projection_checked < 16
        and not torch.cuda.is_current_stream_capturing()
    ):
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle

        reference = gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, weight_scale)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        _projection_checked += 1
        if _projection_checked == 16:
            print("DSV4 hybrid projection: 16 exact live checks passed", flush=True)
    return result
