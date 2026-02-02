# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import torch
import aiter
from aiter.test_common import checkAllclose


def sinkhorn_ref(M, iters):
    for _ in range(iters):
        row_sum = M.sum(dim=-1, keepdim=True)
        M = torch.where(row_sum > 1e-8, M / row_sum, M)
        col_sum = M.sum(dim=-2, keepdim=True)
        M = torch.where(col_sum > 1e-8, M / col_sum, M)
    return M


def mhc_layer_ref(
    x_expanded,
    rmsnorm_weight,
    phi_pre,
    phi_post,
    phi_res,
    b_pre,
    b_post,
    b_res,
    alpha_pre,
    alpha_post,
    alpha_res,
    sinkhorn_iters,
    eps,
):
    B, n, C = x_expanded.shape
    nC = n * C
    n2 = n * n

    x_flat = x_expanded.reshape(B, nC)
    x_flat_bf16 = x_flat.to(torch.bfloat16)
    phi_pre_bf16 = phi_pre.to(torch.bfloat16)
    phi_post_bf16 = phi_post.to(torch.bfloat16)
    phi_res_bf16 = phi_res.to(torch.bfloat16)

    rms = torch.sqrt(x_flat_bf16.float().pow(2).mean(dim=1, keepdim=True) + eps)
    rms_inv = 1.0 / rms

    proj_pre = torch.matmul(x_flat_bf16.float(), phi_pre_bf16.float().t())
    proj_post = torch.matmul(x_flat_bf16.float(), phi_post_bf16.float().t())
    proj_res = torch.matmul(x_flat_bf16.float(), phi_res_bf16.float().t())

    H_pre = torch.sigmoid(alpha_pre * proj_pre * rms_inv + b_pre.view(1, n))
    H_post = 2.0 * torch.sigmoid(alpha_post * proj_post * rms_inv + b_post.view(1, n))
    H_res = torch.exp(alpha_res * proj_res * rms_inv + b_res.view(1, n2))

    M = sinkhorn_ref(H_res.view(B, n, n), sinkhorn_iters)

    x_agg = torch.sum(H_pre.unsqueeze(-1) * x_expanded, dim=1)
    x_agg_bf16 = x_agg.to(torch.bfloat16)
    rms_agg = torch.sqrt(x_agg_bf16.float().pow(2).mean(dim=1, keepdim=True) + eps)
    x_normed = x_agg_bf16.float() / rms_agg * rmsnorm_weight.to(torch.bfloat16).float().view(1, C)

    mix = torch.matmul(M, x_expanded)
    dist = H_post.unsqueeze(-1) * x_normed.unsqueeze(1)
    out = mix + dist
    return out


def mhc_layer_ref_fp32(
    x_expanded,
    rmsnorm_weight,
    phi_pre,
    phi_post,
    phi_res,
    b_pre,
    b_post,
    b_res,
    alpha_pre,
    alpha_post,
    alpha_res,
    sinkhorn_iters,
    eps,
):
    B, n, C = x_expanded.shape
    nC = n * C
    n2 = n * n

    x_flat = x_expanded.reshape(B, nC)
    rms = torch.sqrt(x_flat.pow(2).mean(dim=1, keepdim=True) + eps)
    rms_inv = 1.0 / rms

    proj_pre = torch.matmul(x_flat, phi_pre.t())
    proj_post = torch.matmul(x_flat, phi_post.t())
    proj_res = torch.matmul(x_flat, phi_res.t())

    H_pre = torch.sigmoid(alpha_pre * proj_pre * rms_inv + b_pre.view(1, n))
    H_post = 2.0 * torch.sigmoid(alpha_post * proj_post * rms_inv + b_post.view(1, n))
    H_res = torch.exp(alpha_res * proj_res * rms_inv + b_res.view(1, n2))

    M = sinkhorn_ref(H_res.view(B, n, n), sinkhorn_iters)

    x_agg = torch.sum(H_pre.unsqueeze(-1) * x_expanded, dim=1)
    rms_agg = torch.sqrt(x_agg.pow(2).mean(dim=1, keepdim=True) + eps)
    x_normed = x_agg / rms_agg * rmsnorm_weight.view(1, C)

    mix = torch.matmul(M, x_expanded)
    dist = H_post.unsqueeze(-1) * x_normed.unsqueeze(1)
    out = mix + dist
    return out


def mhc_layer_ref_intermediates(
    x_expanded,
    rmsnorm_weight,
    phi_pre,
    phi_post,
    phi_res,
    b_pre,
    b_post,
    b_res,
    alpha_pre,
    alpha_post,
    alpha_res,
    sinkhorn_iters,
    eps,
):
    B, n, C = x_expanded.shape
    nC = n * C
    n2 = n * n

    x_flat = x_expanded.reshape(B, nC)
    x_flat_bf16 = x_flat.to(torch.bfloat16)
    phi_pre_bf16 = phi_pre.to(torch.bfloat16)
    phi_post_bf16 = phi_post.to(torch.bfloat16)
    phi_res_bf16 = phi_res.to(torch.bfloat16)

    rms = torch.sqrt(x_flat_bf16.float().pow(2).mean(dim=1, keepdim=True) + eps)
    H_proj_pre = torch.matmul(x_flat_bf16.float(), phi_pre_bf16.float().t())
    H_proj_post = torch.matmul(x_flat_bf16.float(), phi_post_bf16.float().t())
    H_proj_res = torch.matmul(x_flat_bf16.float(), phi_res_bf16.float().t())
    H_proj_raw = torch.cat([H_proj_pre, H_proj_post, H_proj_res], dim=1) / rms

    phi_combined = torch.cat([phi_pre_bf16, phi_post_bf16, phi_res_bf16], dim=0)
    H_proj_raw_alt = torch.matmul(phi_combined.float(), x_flat_bf16.float().t()).t() / rms
    H_proj_raw_alt_nodiv = torch.matmul(phi_combined.float(), x_flat_bf16.float().t()).t()

    proj_pre = H_proj_raw[:, :n]
    proj_post = H_proj_raw[:, n : 2 * n]
    proj_res = H_proj_raw[:, 2 * n :]

    H_pre = torch.sigmoid(alpha_pre * proj_pre + b_pre.view(1, n))
    H_post = 2.0 * torch.sigmoid(alpha_post * proj_post + b_post.view(1, n))
    H_res_tilde = alpha_res * proj_res + b_res.view(1, n2)
    M = sinkhorn_ref(torch.exp(H_res_tilde).view(B, n, n), sinkhorn_iters)

    x_agg = torch.sum(H_pre.unsqueeze(-1) * x_expanded, dim=1)
    x_agg_bf16 = x_agg.to(torch.bfloat16)
    rms_values = torch.sqrt(x_agg_bf16.float().pow(2).mean(dim=1, keepdim=True) + eps).squeeze(1)
    layer_out = x_agg_bf16.float() / rms_values.view(B, 1) * rmsnorm_weight.to(
        torch.bfloat16
    ).float().view(1, C)
    layer_out_bf16 = layer_out.to(torch.bfloat16)

    return {
        "H_proj_raw": H_proj_raw,
        "H_proj_raw_alt": H_proj_raw_alt,
        "H_proj_raw_alt_nodiv": H_proj_raw_alt_nodiv,
        "H_pre": H_pre,
        "H_post": H_post,
        "M": M,
        "x_agg_bf16": x_agg_bf16,
        "layer_out_bf16": layer_out_bf16,
        "rms_values": rms_values,
    }


def run_case(
    B,
    C,
    n,
    sinkhorn_iters=20,
    eps=1e-5,
    ref_device="cuda",
    debug_intermediates=False,
):
    device = "cuda"
    torch.manual_seed(42)

    x_expanded = torch.rand(B, n, C, device=device, dtype=torch.float32) * 2.0 - 1.0
    rmsnorm_weight = (
        torch.rand(C, device=device, dtype=torch.float32) * 0.5 + 0.75
    ).to(torch.bfloat16)

    nC = n * C
    total_H_dim = n + n + n * n
    phi = (torch.rand(total_H_dim, nC, device=device, dtype=torch.float32) * 0.1 - 0.05).to(
        torch.bfloat16
    )
    phi_pre = phi[0:n]
    phi_post = phi[n : 2 * n]
    phi_res = phi[2 * n :]

    b_pre = torch.zeros(n, device=device, dtype=torch.float32)
    b_post = torch.zeros(n, device=device, dtype=torch.float32)
    b_res = (torch.rand(n * n, device=device, dtype=torch.float32) * 2.0 - 1.0) * 0.01

    out = aiter.mhc_layer(
        x_expanded,
        rmsnorm_weight,
        phi_pre,
        phi_post,
        phi_res,
        b_pre,
        b_post,
        b_res,
        alpha_pre=0.01,
        alpha_post=0.01,
        alpha_res=0.01,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )

    ref_bf16_like = mhc_layer_ref(
        x_expanded.to(ref_device),
        rmsnorm_weight.to(ref_device),
        phi_pre.to(ref_device),
        phi_post.to(ref_device),
        phi_res.to(ref_device),
        b_pre.to(ref_device),
        b_post.to(ref_device),
        b_res.to(ref_device),
        0.01,
        0.01,
        0.01,
        sinkhorn_iters,
        eps,
    ).to(device)

    ref_fp32_cuda = mhc_layer_ref_fp32(
        x_expanded,
        rmsnorm_weight.float(),
        phi_pre.float(),
        phi_post.float(),
        phi_res.float(),
        b_pre,
        b_post,
        b_res,
        0.01,
        0.01,
        0.01,
        sinkhorn_iters,
        eps,
    )

    ref_fp32_cpu = mhc_layer_ref_fp32(
        x_expanded.cpu(),
        rmsnorm_weight.cpu().float(),
        phi_pre.cpu().float(),
        phi_post.cpu().float(),
        phi_res.cpu().float(),
        b_pre.cpu(),
        b_post.cpu(),
        b_res.cpu(),
        0.01,
        0.01,
        0.01,
        sinkhorn_iters,
        eps,
    ).to(device)

    torch.cuda.synchronize()
    msg = f"[mhc_layer bf16_like] B={B}, C={C}, n={n}"
    checkAllclose(ref_bf16_like, out, atol=0.1, rtol=1e-2, msg=msg)
    msg = f"[mhc_layer fp32_cuda] B={B}, C={C}, n={n}"
    checkAllclose(ref_fp32_cuda, out, atol=0.1, rtol=1e-2, msg=msg)
    msg = f"[mhc_layer fp32_cpu] B={B}, C={C}, n={n}"
    checkAllclose(ref_fp32_cpu, out, atol=0.1, rtol=1e-2, msg=msg)
    msg = f"[mhc_layer fp32 cpu vs cuda] B={B}, C={C}, n={n}"
    checkAllclose(ref_fp32_cpu, ref_fp32_cuda, atol=1e-5, rtol=1e-5, msg=msg)

    if debug_intermediates:
        debug = aiter.mhc_layer_debug(
            x_expanded,
            rmsnorm_weight,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            alpha_pre=0.01,
            alpha_post=0.01,
            alpha_res=0.01,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )
        ref_dbg = mhc_layer_ref_intermediates(
            x_expanded,
            rmsnorm_weight,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            0.01,
            0.01,
            0.01,
            sinkhorn_iters,
            eps,
        )

        checkAllclose(
            ref_dbg["H_proj_raw"],
            debug["H_proj_raw"],
            atol=1e-2,
            rtol=1e-2,
            msg=f"[dbg H_proj_raw] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["H_proj_raw_alt"],
            debug["H_proj_raw"],
            atol=1e-2,
            rtol=1e-2,
            msg=f"[dbg H_proj_raw_alt] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["H_proj_raw_alt_nodiv"],
            debug["H_proj_raw"],
            atol=1e-2,
            rtol=1e-2,
            msg=f"[dbg H_proj_raw_alt_nodiv] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["H_pre"],
            debug["H_pre"],
            atol=1e-3,
            rtol=1e-3,
            msg=f"[dbg H_pre] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["H_post"],
            debug["H_post"],
            atol=1e-3,
            rtol=1e-3,
            msg=f"[dbg H_post] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["M"],
            debug["M"],
            atol=1e-2,
            rtol=1e-2,
            msg=f"[dbg M] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["x_agg_bf16"].float(),
            debug["x_agg_bf16"].float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"[dbg x_agg_bf16] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["layer_out_bf16"].float(),
            debug["layer_out_bf16"].float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"[dbg layer_out_bf16] B={B}, C={C}, n={n}",
        )
        checkAllclose(
            ref_dbg["rms_values"],
            debug["rms_values"],
            atol=1e-3,
            rtol=1e-3,
            msg=f"[dbg rms_values] B={B}, C={C}, n={n}",
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MHC layer tests aligned with bench_mhc_layer configs",
    )
    parser.add_argument("--B", type=int, default=None, help="batch size")
    parser.add_argument("--C", type=int, default=None, help="hidden dim")
    parser.add_argument("--n", type=int, default=None, help="expansion rate")
    parser.add_argument(
        "--ref-device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="reference device (default: cuda)",
    )
    parser.add_argument(
        "--debug-intermediates",
        action="store_true",
        help="compare debug intermediates from kernel vs reference",
    )
    args = parser.parse_args()

    configs = [
        (64, 1280, 4),
        (128, 1280, 4),
        (256, 1280, 4),
        (320, 1280, 4),
        (64, 1920, 4),
        (128, 1920, 4),
        (64, 2560, 4),
        (128, 2560, 4),
        (32, 1280, 32),
        (64, 1280, 32),
        (128, 1280, 32),
    ]

    if args.B is not None or args.C is not None or args.n is not None:
        configs = [
            (args.B or B, args.C or C, args.n or n) for (B, C, n) in configs
        ]
        configs = list(dict.fromkeys(configs))

    for B, C, n in configs:
        run_case(
            B,
            C,
            n,
            ref_device=args.ref_device,
            debug_intermediates=args.debug_intermediates,
        )


if __name__ == "__main__":
    main()
