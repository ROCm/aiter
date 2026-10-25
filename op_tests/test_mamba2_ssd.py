# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import math

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.mamba2_ssd import SUPPORTED_GFX, mamba2_chunk_forward_flydsl
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

try:
    from fla.ops.simple_gla import chunk_simple_gla
except ImportError:
    chunk_simple_gla = None

torch.set_default_device("cuda")


def _ref_decay_cumsum(a, dt, chunk_size):
    """Per-chunk cumsum of a * dt."""
    B, T, H = dt.shape
    decay = (a * dt.float()).view(B, T // chunk_size, chunk_size, H)
    return decay.cumsum(dim=2).view(B, T, H)


def _ref_pass_state(x, b, dt, decay_cumsum, init_state, chunk_size):
    """Chunk-entry states and final state."""
    H, G = x.shape[2], b.shape[2]
    x, b, dt = x.float(), b.float().repeat_interleave(H // G, dim=2), dt.float()
    hid, hiddens = init_state, []
    for ci in range(x.shape[1] // chunk_size):
        hiddens.append(hid.clone())
        sl = slice(ci * chunk_size, (ci + 1) * chunk_size)
        xc, bc = x[:, sl].transpose(1, 2), b[:, sl].transpose(1, 2)
        dtc, dc = dt[:, sl].transpose(1, 2), decay_cumsum[:, sl].transpose(1, 2)
        g_last = dc[:, :, -1]
        bw = bc * (dtc * torch.exp(g_last[..., None] - dc))[..., None]
        hid = hid * torch.exp(g_last[..., None, None]) + xc.transpose(-1, -2) @ bw
    return torch.stack(hiddens, dim=2), hid


def _ref_out(x, b, c, d, dt, decay_cumsum, hiddens, chunk_size):
    """Fused chunk output."""
    H, G = x.shape[2], b.shape[2]
    x, dt = x.float(), dt.float()
    b = b.float().repeat_interleave(H // G, dim=2)
    c = c.float().repeat_interleave(H // G, dim=2)
    out = torch.zeros_like(x)
    for ci in range(hiddens.shape[2]):
        sl = slice(ci * chunk_size, (ci + 1) * chunk_size)
        xc, bc, cc = (t[:, sl].transpose(1, 2) for t in (x, b, c))
        dtc, dc = dt[:, sl].transpose(1, 2), decay_cumsum[:, sl].transpose(1, 2)
        cb = cc @ bc.transpose(-1, -2)
        cb = torch.tril(cb * torch.tril(torch.exp(dc[..., None] - dc[..., None, :])))
        y = (cc * torch.exp(dc[..., None])) @ hiddens[:, :, ci].float().transpose(
            -1, -2
        )
        y = y + cb @ (dtc[..., None] * xc) + d[None, :, None, None] * xc
        out[:, sl] = y.transpose(1, 2)
    return out


def run_torch(x, a, b, c, d, dt, init_state, chunk_size):
    """Chunked Mamba-2 forward reference, in fp32."""
    decay_cumsum = _ref_decay_cumsum(a, dt, chunk_size)
    hiddens, final_state = _ref_pass_state(
        x, b, dt, decay_cumsum, init_state, chunk_size
    )
    out = _ref_out(x, b, c, d, dt, decay_cumsum, hiddens, chunk_size)
    return out.to(x.dtype), final_state


def _bf16_ulp(x):
    """Spacing of bf16 values at magnitude x (8 significant bits)."""
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 0.0


def make_inputs(B, T, G, H, N, P, L, dtype):
    """Random inputs; the initial state comes from one chunk of the recurrence
    so it is in distribution."""
    x = F.silu(torch.randn(B, T, H, P, dtype=dtype))
    a = -torch.empty(H, dtype=dtypes.fp32).uniform_(1, 16)
    b = F.silu(torch.randn(B, T, G, N, dtype=dtype))
    c = F.silu(torch.randn(B, T, G, N, dtype=dtype))
    d = torch.randn(H, dtype=dtypes.fp32)
    dt = F.softplus(torch.randn(B, T, H, dtype=dtype))
    x0 = F.silu(torch.randn(B, L, H, P, dtype=dtype))
    b0 = F.silu(torch.randn(B, L, G, N, dtype=dtype))
    dt0 = F.softplus(torch.randn(B, L, H, dtype=dtype))
    a0 = -torch.empty(H, dtype=dtypes.fp32).uniform_(1, 16)
    zeros = torch.zeros(B, H, P, N, dtype=dtypes.fp32)
    _, init_state = _ref_pass_state(x0, b0, dt0, (a0 * dt0).cumsum(dim=1), zeros, L)
    return x, a, b, c, d, dt, init_state


@benchmark()
def test_mamba2_chunk_forward(
    batch, num_tokens, num_groups, num_heads, state_size, head_size, chunk_size, dtype
):
    B, T, G, H, N, P, L = (
        batch,
        num_tokens,
        num_groups,
        num_heads,
        state_size,
        head_size,
        chunk_size,
    )
    x, a, b, c, d, dt, init_state = make_inputs(B, T, G, H, N, P, L, dtype)
    ref_out, ref_state = run_torch(x, a, b, c, d, dt, init_state, L)

    candidates = {
        "flydsl": lambda: mamba2_chunk_forward_flydsl(
            x, a, b, c, d, dt, init_state, chunk_size=L
        ),
    }
    if chunk_simple_gla is not None:
        # Mamba-2 SSD == simple GLA with q=C, k=B, v=dt*x, g=a*dt, scale=1, state [N, P].
        # Head expansion and v/g prep are hoisted out of the timed call, so only
        # fla's kernels (plus the D*x skip) are measured.
        q_f = c.repeat_interleave(H // G, dim=2).contiguous()
        k_f = b.repeat_interleave(H // G, dim=2).contiguous()
        v_f = (x.float() * dt.float()[..., None]).to(dtype)
        g_f = a * dt.float()
        h0_f = init_state.transpose(-1, -2).contiguous()

        def fla_fn():
            o, ht = chunk_simple_gla(
                q_f,
                k_f,
                v_f,
                g=g_f,
                scale=1.0,
                initial_state=h0_f,
                output_final_state=True,
                chunk_size=L,
            )
            return (o + d[:, None] * x).to(dtype), ht.transpose(-1, -2)

        candidates["fla_triton"] = fla_fn

    nc = T // L
    # Per (chunk, head): state update 2LPN, carried-state term 2LPN, causal C.B^T and P@x ~ L^2 (N + P).
    flops = B * H * nc * (4 * L * P * N + L * L * (N + P))
    nbytes = (
        2 * B * T * H * P * x.element_size()  # x in, out
        + 2 * B * T * G * N * b.element_size()  # b, c
        + 2 * B * H * nc * P * N * 2  # bf16 chunk-entry states written then read
        + 2 * B * H * P * N * 4  # init / final state
    )

    # bf16 operands round at the scale of the largest output terms, and those
    # errors survive cancellation into small outputs, so allow one bf16 ulp of
    # the output's magnitude on top of the base atol.
    out_atol = 2e-2 + _bf16_ulp(ref_out.abs().max().item())

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (out, state), us = run_perftest(fn)
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=out_atol,
            msg=f"{name}: out",
        )
        checkAllclose(
            ref_state,
            state.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: final_state",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mamba2_chunk_forward (FlyDSL) unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default="bf16,",
        help="input dtype",
    )
    # Default sweep: B=4, H=8, P=64, chunk 128, state 64-512.
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[4], help="batch size"
    )
    parser.add_argument(
        "-t",
        "--num_tokens",
        type=int,
        nargs="*",
        default=[1024, 4096, 16384],
        help="sequence length",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(1, 8, s, 64) for s in (64, 128, 256, 512)],
        help="(num_groups, num_heads, state_size, head_size), e.g. -s 1,8,128,64",
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, nargs="*", default=[128], help="chunk size"
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for b, t, (g, h, n, p), cs in itertools.product(
            args.batch, args.num_tokens, args.shape, args.chunk_size
        ):
            df.append(test_mamba2_chunk_forward(b, t, g, h, n, p, cs, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mamba2_chunk_forward summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
