# SPDX-License-Identifier: MIT
"""Task 5: the fused dispatch+GEMM1 stage-1 kernel must build.

Tracing is where every FlyDSL-level mistake in the overlap helpers shows up --
a dynamic ``if`` in a helper the AST rewriter never saw, a role branch whose two
sides disagree on types, an scf loop nested wrong. Codegen is cheap compared to
standing up four ranks, so it runs first and on a single GPU.

The kernel is built but never launched: with no peers to answer the count
exchange, its phase-2 spin would never retire.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")


def _is_gfx1250():
    try:
        from flydsl.runtime.device import get_rocm_arch

        return "gfx1250" in get_rocm_arch().lower()
    except Exception:
        return False


def _overlap_consts(**over):
    from aiter.ops.flydsl.kernels.push_group_overlap_stage1_gfx1250 import (
        OverlapConsts,
    )

    kw = dict(
        rank=0,
        npes=4,
        experts_per_rank=8,
        experts_per_token=2,
        token_nbytes=512,
        max_tok_per_rank=64,
        cap=256,
        tile_m=64,
        tiles_per_expert=4,
        scale_num_i32=4,
        scale_wmma_rep=4,
        dispatch_ctas=8,
        grid_ctas=64,
        arena_handle=1 << 40,
    )
    for i, n in enumerate(
        ("count", "count_done", "tile_arrived", "disp_out", "out_scales",
         "pg_rowmap", "tis")
    ):
        kw[f"off_{n}"] = 1 << (16 + i)
    for i, n in enumerate(
            ("entry", "bar", "plan_ready", "my_base", "hist", "route_slot",
         "route_order", "count", "count_done", "tile_arrived", "tile_expected")
    ):
        kw[f"ptr_{n}"] = (1 << 41) + (i << 16)
    kw.update(over)
    return OverlapConsts(**kw)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250 hardware")
def test_fused_stage1_kernel_builds(monkeypatch):
    from aiter.ops.flydsl.kernels import mxfp4_preshuffle_gfx1250_tdm as tdm
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    launched = {}
    dev = torch.device("cuda")
    E, cap, K, two_inter = 8, 256, 512, 256
    tile_m, tile_n, tile_k = 64, 128, 256
    cm = E * cap
    C = _overlap_consts(cap=cap, tile_m=tile_m, tiles_per_expert=cap // tile_m,
                        scale_wmma_rep=tile_m // 16, token_nbytes=K,
                        scale_num_i32=K // 32)

    a = torch.zeros(cm * K, dtype=torch.uint8, device=dev)
    b = torch.zeros(E * two_inter * K // 2, dtype=torch.uint8, device=dev)
    sa = torch.zeros(cm // C.wmma_rep, (K // 32) * C.wmma_rep, dtype=torch.int32,
                     device=dev)
    sb = torch.zeros(E, two_inter * K // 32 // 4, dtype=torch.int32, device=dev)
    c = torch.zeros(cm, two_inter, dtype=torch.bfloat16, device=dev)
    i32 = lambda n: torch.zeros(n, dtype=torch.int32, device=dev)

    # Patch out the launch: tracing + codegen is the whole point, and the phase-2
    # spin has no peers to satisfy it here.
    import flydsl.compiler as flyc

    orig = flyc.kernel

    def _kernel(*a_, **k_):
        deco = orig(*a_, **k_)

        def wrap(fn):
            built = deco(fn)

            def call(*args, **kwargs):
                obj = built(*args, **kwargs)
                obj.launch = lambda **kw: launched.update(kw)
                return obj

            return call

        return wrap

    monkeypatch.setattr(flyc, "kernel", _kernel)

    tdm.launch_gemm_a8w4_tdm(
        arg_c=c, arg_a=ptr_arg(a), arg_b=ptr_arg(b), arg_scale_a=sa, arg_scale_b=sb,
        i32_m=cm, N=two_inter, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, m_warp=2, n_warp=2,
        out_is_f16=0, num_buffers=2, a_is_fp4=0, arg_m_tile_map=ptr_arg(i32(cm // tile_m)),
        n_experts=E, stage1_act=1, has_bias=0, arg_bias=ptr_arg(a),
        arg_quant_scale=c, arg_ep_rowmap=c,
        f32_swiglu_limit=float("inf"),
        push_group=1, ep_persistent_gemm1=1, persistent_workers=64,
        arg_tile_row_base=ptr_arg(i32(cm // tile_m)),
        arg_expert_ids=ptr_arg(i32(cm // tile_m)),
        arg_tile_valid=ptr_arg(i32(cm // tile_m)),
        arg_num_valid_rows=ptr_arg(i32(1)),
        ep_overlap=C.as_tuple(),
        i64_ov_inp_tok=a.data_ptr(), i64_ov_inp_idx=i32(64 * 2).data_ptr(),
        i64_ov_inp_wts=i32(64 * 2).data_ptr(),
        i64_ov_inp_scales=i32(64 * (K // 32)).data_ptr(),
        i32_ov_cur_tok=64,
        stream=torch.cuda.current_stream().cuda_stream,
    )

    assert launched["grid"] == (C.grid_ctas, 1, 1), (
        "the fused grid is the role space, not the GEMM tile count"
    )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250 hardware")
def test_overlap_requires_persistent_schedule():
    from aiter.ops.flydsl.kernels import mxfp4_preshuffle_gfx1250_tdm as tdm
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    dev = torch.device("cuda")
    E, cap, K, two_inter = 8, 256, 512, 256
    cm = E * cap
    u8 = torch.zeros(cm * K, dtype=torch.uint8, device=dev)
    c = torch.zeros(cm, two_inter, dtype=torch.bfloat16, device=dev)
    i32 = torch.zeros(cm, dtype=torch.int32, device=dev)

    with pytest.raises(ValueError, match="ep_persistent_gemm1"):
        tdm.launch_gemm_a8w4_tdm(
            arg_c=c, arg_a=ptr_arg(u8), arg_b=ptr_arg(u8),
            arg_scale_a=i32, arg_scale_b=i32,
            i32_m=cm, N=two_inter, K=K, tile_m=64, tile_n=128, tile_k=256,
            m_warp=2, n_warp=2, out_is_f16=0, num_buffers=2, a_is_fp4=0,
            arg_m_tile_map=ptr_arg(i32), n_experts=E, stage1_act=1, has_bias=0,
            arg_bias=ptr_arg(u8), arg_quant_scale=c, arg_ep_rowmap=c,
            f32_swiglu_limit=float("inf"),
            arg_tile_row_base=ptr_arg(i32), arg_expert_ids=ptr_arg(i32),
            arg_tile_valid=ptr_arg(i32), arg_num_valid_rows=ptr_arg(i32),
            push_group=1, ep_persistent_gemm1=0,
            stream=torch.cuda.current_stream().cuda_stream,
            ep_overlap=_overlap_consts(token_nbytes=512, scale_num_i32=16,
                                       cap=256, tile_m=64, tiles_per_expert=4,
                                       scale_wmma_rep=4).as_tuple(),
        )
