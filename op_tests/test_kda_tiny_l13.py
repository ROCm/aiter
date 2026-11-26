# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tiny-L13 (ZhangDanyang-AMD/Lumen#35) KDA layers: BF16 forward + backward of
the fused backend against the torch reference, one layer at a time.

Checkpoint: --ckpt or AITER_KIMI_TINY_L13_PATH (local snapshot of
Qinghao/Kimi-K3-Tiny-L13); random weights at the real shapes otherwise.
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import kda_fused_layer, kda_tiny_l13
from aiter.ops.kda_fused_layer import KDALayerWeights, kda_layer_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]

# loaded one layer at a time so only one layer's weights (~1.6 GB bf16) are resident
_CUR = {}


def run_layer(x, dy, w, num_heads, chunk, eps, backend):
    xs = x.detach().requires_grad_(True)
    ws = [p.detach().requires_grad_(True) for p in w.tensors()]
    y = kda_layer_fwd(
        xs, KDALayerWeights.from_tensors(ws), num_heads, chunk, eps, backend=backend
    )
    return y.detach(), torch.autograd.grad(y, [xs, *ws], dy)


@benchmark()
def test_tiny_l13_kda_layer(layer, b, t, chunk):
    w, cfg, source = _CUR["w"], _CUR["cfg"], _CUR["source"]
    H, D, eps = cfg["heads"], cfg["hidden"], cfg["eps"]
    g = torch.Generator(device="cuda").manual_seed(layer)
    x = torch.randn(b, t, D, generator=g).to(dtypes.bf16)
    dy = torch.randn(b, t, D, generator=g).to(dtypes.bf16)

    (ref_y, ref_g), ref_us = run_perftest(
        run_layer, x, dy, w, H, chunk, eps, "torch", num_iters=1, num_warmup=0
    )
    (y, grads), us = run_perftest(
        run_layer, x, dy, w, H, chunk, eps, "fused", num_iters=3, num_warmup=1
    )

    names = ["y", "dx"] + [f for f in KDALayerWeights.__dataclass_fields__]
    tol = 5e-2
    worst = 0.0
    for name, ref, out in zip(names, [ref_y, *ref_g], [y, *grads]):
        # grade each output against its own scale so a small gradient is not hidden
        s = ref.float().abs().max().clamp_min(1e-6)
        err = checkAllclose(
            ref.float() / s, out.float() / s, rtol=tol, atol=tol,
            msg=f"layer {layer}: {name}",
        )  # fmt: skip
        assert err == 0, f"layer {layer}: {name} {err:.2%} outside tolerance"
        worst = max(worst, err)
    return {
        "weights": source,
        "torch us": ref_us,
        "fused us": us,
        "err": worst,
        "fused pieces": " ".join(
            f"{k}={v}" for k, v in sorted(kda_fused_layer.LAST_PIECES.items())
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Tiny-L13 KDA layer fwd+bwd parity (bf16, fused vs torch)",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help=f"Tiny-L13 snapshot dir (default: ${kda_tiny_l13.TINY_L13_PATH_ENV}; "
        "random weights if unset)",
    )
    parser.add_argument(
        "--a_log",
        choices=["error", "random"],
        default="random",
        help="checkpoint A_log is [head_dim], not [heads]: 'random' substitutes it",
    )
    parser.add_argument("-l", "--layers", type=int, nargs="*", default=None)
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument(
        "-t", "--seqlen", type=int, nargs="*", default=[100, 256],
        help="tokens per sequence; 100 exercises a partial last chunk",
    )  # fmt: skip
    parser.add_argument("-c", "--chunk", type=int, default=64)
    args = parser.parse_args()

    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("Tiny-L13 KDA parity targets gfx950; skipping")
        return

    path = kda_tiny_l13.checkpoint_path(args.ckpt)
    source = "checkpoint" if path else "random"
    cfg = kda_tiny_l13.load_config(path)
    layers = args.layers if args.layers is not None else list(cfg["kda_layers"])
    df = []
    for layer in layers:
        _CUR.update(
            cfg=cfg,
            source=source,
            w=kda_tiny_l13.load_kda_layer(
                layer, path, dtypes.bf16, "cuda", cfg, a_log=args.a_log
            ),
        )
        for t in args.seqlen:
            df.append(test_tiny_l13_kda_layer(layer, args.batch, t, args.chunk))
        _CUR.clear()
        torch.cuda.empty_cache()
    df = pd.DataFrame(df)
    aiter.logger.info(
        "Tiny-L13 KDA fwd+bwd summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
