# SPDX-License-Identifier: MIT
"""Graph-safe regional operations; public AITER math and measured configurations."""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.utils import envs

logger = logging.getLogger("atom")
_checks = {}


def _check(key, actual, reference):
    if envs.ATOM_MONO_CHECK and not torch.cuda.is_current_stream_capturing():
        count = _checks.get(key, 0)
        if count < 16:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            _checks[key] = count + 1
            if count + 1 in (2, 16):
                logger.info(
                    "DSV4 gfx1250 hybrid exact checks=%d key=%s", count + 1, key
                )


@lru_cache(None)
def configs(profile):
    return json.loads(
        (Path(__file__).parent.parent / "configs" / f"{profile}.json").read_text()
    )


def _projection_fake(x, weight, x_scale, weight_scale, profile, role):
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )


@torch_compile_guard(gen_fake=_projection_fake, mutates_args=[])
def projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    profile: str,
    role: str,
) -> torch.Tensor:
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle

    key = f"{x.shape[0]},{weight.shape[0]},{weight.shape[1]}"
    config = configs(profile).get(key)
    allowed = (
        profile == "flash_tp1"
        or (role == "wqkv_a" and weight.shape == (2048, 7168))
        or (role in ("wq_b", "indexer.wq_b") and weight.shape[1] == 1536)
    )
    if config is None or not allowed:
        return gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, weight_scale)
    from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle

    result = gemm_afp8wfp8_preshuffle(
        x,
        weight,
        x_scale.view(torch.uint8),
        weight_scale.view(torch.uint8),
        backend="gluon",
        is_x_scale_transposed=True,
        config=dict(config),
    )
    check_key = (profile, role, weight.data_ptr(), key)
    if (
        envs.ATOM_MONO_CHECK
        and _checks.get(check_key, 0) < 16
        and not torch.cuda.is_current_stream_capturing()
    ):
        reference = gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, weight_scale)
        try:
            _check(check_key, result, reference)
        except AssertionError:
            logger.exception("DSV4 gfx1250 exact projection failure key=%s", check_key)
            directory = os.environ.get("ATOM_MONO_FAILURE_DIR")
            if directory:
                target = Path(directory)
                target.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "x": x.cpu(),
                        "weight": weight.cpu(),
                        "x_scale": x_scale.cpu(),
                        "weight_scale": weight_scale.cpu(),
                        "actual": result.cpu(),
                        "reference": reference.cpu(),
                        "profile": profile,
                        "role": role,
                        "config": config,
                    },
                    target
                    / f"projection-{weight.data_ptr()}-{key.replace(',', '-')}.pt",
                )
            raise
    return result


def _mhc_fake(
    x,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    norm_weight,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    res_preshuffle: bool,
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


@torch_compile_guard(gen_fake=_mhc_fake, mutates_args=[])
def mhc_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    res_preshuffle: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    from aiter.ops.triton.fusions.mhc import mhc_post_pre_quant_gfx1250

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
    key = ("mhc", fn.data_ptr(), x.shape[0])
    if (
        envs.ATOM_MONO_CHECK
        and _checks.get(key, 0) < 16
        and not torch.cuda.is_current_stream_capturing()
    ):
        import aiter
        from aiter.ops.quant import per_group_quant_hip

        reference = aiter.mhc_fused_post_pre(
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
            reference[2],
            quant_dtype=torch.float8_e4m3fn,
            group_size=128,
            transpose_scale=True,
            scale_type=torch.float8_e8m0fnu,
        )
        _check(key, result, (*reference, q, qs))
    return result
