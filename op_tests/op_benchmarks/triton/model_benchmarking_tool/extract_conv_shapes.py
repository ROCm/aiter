# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Offline extraction of Conv2d layer shapes from real model checkpoints.

Run once per model. Walks the model with forward hooks, captures the
per-layer (N, C, H, W, K, R, S, stride, pad, dilation) tuples, dedupes,
and prints JSON ready to paste under ``"<ModelName>": {"conv2d": [...]}``
in ``model_shapes.json``.

Consumed by ``bench_conv2d.py --model NAME`` (sweep mode).

Usage::

    python -m op_tests.op_benchmarks.triton.model_benchmarking_tool.extract_conv_shapes \\
        --model resnet50

    python -m op_tests.op_benchmarks.triton.model_benchmarking_tool.extract_conv_shapes \\
        --model sd35_vae --model-path /app/models/stable-diffusion-3.5-medium

    python -m op_tests.op_benchmarks.triton.model_benchmarking_tool.extract_conv_shapes \\
        --model flux2_vae --model-path /app/models/FLUX.2-klein-9B

Weight values do not affect the captured shapes (Conv2d dimensions are
fixed by the layer definition, not the weight tensor contents). Concretely:
- resnet50 uses ``weights=None`` (random init, no checkpoint download).
- sd35_vae / flux2_vae use ``AutoencoderKL[Flux2].from_pretrained(...)``,
  which loads the local checkpoint at ``--model-path`` (architecture +
  weights). The weights are unused for shape extraction; they come along
  because diffusers' easiest API loads both.

Skips ``groups != 1`` layers (kernel doesn't support grouped/depthwise yet)
and ``padding_mode != 'zeros'`` layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import torch
import torch.nn as nn


def _conv_shape_dict(mod: nn.Conv2d, x_shape: torch.Size) -> dict:
    _, _, H, W = x_shape
    R, S = (
        mod.kernel_size
        if isinstance(mod.kernel_size, tuple)
        else (mod.kernel_size, mod.kernel_size)
    )
    sh, sw = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
    ph, pw = (
        mod.padding if isinstance(mod.padding, tuple) else (mod.padding, mod.padding)
    )
    dh, dw = (
        mod.dilation
        if isinstance(mod.dilation, tuple)
        else (mod.dilation, mod.dilation)
    )
    return {
        "N": int(x_shape[0]),
        "C": int(mod.in_channels),
        "H": int(H),
        "W": int(W),
        "K": int(mod.out_channels),
        "R": int(R),
        "S": int(S),
        "stride_h": int(sh),
        "stride_w": int(sw),
        "pad_h": int(ph),
        "pad_w": int(pw),
        "dilation_h": int(dh),
        "dilation_w": int(dw),
    }


def _walk(module: nn.Module, run_forward, batch_size: int) -> list[dict]:
    """Hook every Conv2d, run forward once, return deduped shape dicts."""
    captured: dict[str, torch.Size] = {}

    def make_hook(key):
        def hook(m, inp, out):
            if key not in captured and inp and isinstance(inp[0], torch.Tensor):
                captured[key] = inp[0].shape

        return hook

    handles = []
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            run_forward()
    finally:
        for h in handles:
            h.remove()

    shapes = []
    skipped_groups = 0
    skipped_padmode = 0
    skipped_no_shape = 0
    for name, mod in module.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        if mod.groups != 1:
            skipped_groups += 1
            continue
        if getattr(mod, "padding_mode", "zeros") != "zeros":
            skipped_padmode += 1
            continue
        if name not in captured:
            skipped_no_shape += 1
            continue
        shapes.append(_conv_shape_dict(mod, captured[name]))

    print(
        f"# captured {len(shapes)} conv layers "
        f"(skipped: {skipped_groups} grouped, {skipped_padmode} non-zero pad-mode, "
        f"{skipped_no_shape} unreached)",
        file=sys.stderr,
    )

    # Dedupe — many models have the same shape repeated across blocks.
    seen = set()
    deduped = []
    for s in shapes:
        key = tuple(s.items())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    print(f"# {len(deduped)} unique shapes after dedupe", file=sys.stderr)
    return deduped


# Hardcoded internally — these don't affect captured shape values, only the
# forward-pass runtime. fp16 + cuda is fine for every model we extract from.
_DTYPE = torch.float16
_DEVICE = "cuda"


def extract_resnet50(N: int, H: int, W: int) -> list[dict]:
    from torchvision.models import resnet50

    model = resnet50(weights=None).to(device=_DEVICE, dtype=_DTYPE).eval()
    x = torch.randn(N, 3, H, W, device=_DEVICE, dtype=_DTYPE)

    def fwd():
        model(x)

    return _walk(model, fwd, N)


def extract_sd35_vae(model_path: str, N: int, H: int, W: int) -> list[dict]:
    from diffusers import AutoencoderKL

    sub = model_path if model_path.rstrip("/").endswith("/vae") else model_path + "/vae"
    vae = AutoencoderKL.from_pretrained(sub, torch_dtype=_DTYPE).to(_DEVICE).eval()
    img = torch.randn(N, 3, H, W, device=_DEVICE, dtype=_DTYPE)

    def fwd():
        latent = vae.encode(img).latent_dist.sample()
        vae.decode(latent)

    return _walk(vae, fwd, N)


def extract_flux2_vae(model_path: str, N: int, H: int, W: int) -> list[dict]:
    from diffusers import AutoencoderKLFlux2

    sub = model_path if model_path.rstrip("/").endswith("/vae") else model_path + "/vae"
    vae = AutoencoderKLFlux2.from_pretrained(sub, torch_dtype=_DTYPE).to(_DEVICE).eval()
    img = torch.randn(N, 3, H, W, device=_DEVICE, dtype=_DTYPE)

    def fwd():
        latent = vae.encode(img).latent_dist.sample()
        vae.decode(latent)

    return _walk(vae, fwd, N)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="extract_conv_shapes", description=__doc__)
    p.add_argument(
        "--model", required=True, choices=["resnet50", "sd35_vae", "flux2_vae"]
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="local checkpoint dir (required for sd35_vae / flux2_vae)",
    )
    p.add_argument("--N", type=int, default=1, help="batch size")
    p.add_argument(
        "--H", type=int, default=None, help="input H (default: model-typical)"
    )
    p.add_argument(
        "--W", type=int, default=None, help="input W (default: model-typical)"
    )
    args = p.parse_args(argv)

    if args.model == "resnet50":
        H = args.H if args.H is not None else 224
        W = args.W if args.W is not None else 224
        shapes = extract_resnet50(args.N, H, W)
        model_key = "resnet50"
    elif args.model == "sd35_vae":
        if not args.model_path:
            p.error("--model-path required for sd35_vae")
        H = args.H if args.H is not None else 512
        W = args.W if args.W is not None else 512
        shapes = extract_sd35_vae(args.model_path, args.N, H, W)
        model_key = "stable-diffusion-3.5-medium"
    elif args.model == "flux2_vae":
        if not args.model_path:
            p.error("--model-path required for flux2_vae")
        H = args.H if args.H is not None else 512
        W = args.W if args.W is not None else 512
        shapes = extract_flux2_vae(args.model_path, args.N, H, W)
        model_key = "FLUX.2-klein-9B"
    else:
        raise AssertionError("unreachable")

    print(json.dumps({model_key: {"conv2d": shapes}}, indent=4))


if __name__ == "__main__":
    main()
