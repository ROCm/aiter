# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .suite import TestSuite, run_all_methods


def _normalize_conv_attrs(conv: nn.Conv2d):
    """Extract and normalize conv attributes to 2-tuples."""
    stride = (
        tuple(conv.stride)
        if isinstance(conv.stride, tuple)
        else (conv.stride, conv.stride)
    )
    padding = (
        tuple(conv.padding)
        if isinstance(conv.padding, tuple)
        else (conv.padding, conv.padding)
    )
    dilation = (
        tuple(conv.dilation)
        if isinstance(conv.dilation, tuple)
        else (conv.dilation, conv.dilation)
    )
    R, S = (
        conv.kernel_size
        if isinstance(conv.kernel_size, tuple)
        else (conv.kernel_size, conv.kernel_size)
    )
    return R, S, stride, padding, dilation


def _extract_conv_layers(components, device, dtype, pretrained: bool = False):
    """Extract Conv2d layers (groups=1) with real input shapes.

    Args:
        components: list of (name, module, forward_fn) tuples.
                    forward_fn(module) runs a dummy forward pass to capture shapes.
        device: target device
        dtype: target dtype
        pretrained: if True, use the model's real (loaded) weights/biases.
                    If False, replace each with `torch.randn_like(...)` (default).
                    Bench TFLOPS doesn't depend on weight values; correctness
                    still compares Triton vs PyTorch with the same tensors.

    Returns:
        list of (C, K_out, R, S, stride, padding, dilation, H, W, desc, w_t, b_t)
    """
    conv_input_shapes: Dict[str, tuple] = {}
    hooks = []

    for comp_name, comp, _ in components:
        for mod_name, mod in comp.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.groups == 1:
                key = f"{comp_name}.{mod_name}" if comp_name else mod_name

                def _make_hook(k):
                    def _hook(m, inp, out):
                        if k not in conv_input_shapes:
                            conv_input_shapes[k] = inp[0].shape

                    return _hook

                hooks.append(mod.register_forward_hook(_make_hook(key)))

    for comp_name, comp, forward_fn in components:
        comp.to(device=device, dtype=dtype)
        with torch.no_grad():
            forward_fn(comp)
        comp.cpu()

    for h in hooks:
        h.remove()

    bench_layers = []
    n_total_conv = 0
    n_skipped_grouped = 0
    n_skipped_padding = 0
    n_skipped_no_shape = 0
    no_shape_keys = []
    for comp_name, comp, _ in components:
        for mod_name, mod in comp.named_modules():
            if not isinstance(mod, nn.Conv2d):
                continue
            n_total_conv += 1
            if mod.groups != 1:
                n_skipped_grouped += 1
                continue
            if getattr(mod, "padding_mode", "zeros") != "zeros":
                n_skipped_padding += 1
                continue
            key = f"{comp_name}.{mod_name}" if comp_name else mod_name
            if key not in conv_input_shapes:
                n_skipped_no_shape += 1
                no_shape_keys.append(key)
                continue
            _, _, H, W = conv_input_shapes[key]
            R, S, stride, padding, dilation = _normalize_conv_attrs(mod)
            C = mod.in_channels
            K_out = mod.out_channels
            w_real = mod.weight.to(dtype=dtype, device=device)
            w_t = w_real if pretrained else torch.randn_like(w_real)
            if mod.bias is not None:
                b_real = mod.bias.to(dtype=dtype, device=device)
                b_t = b_real if pretrained else torch.randn_like(b_real)
            else:
                b_t = None
            bench_layers.append(
                (C, K_out, R, S, stride, padding, dilation, H, W, key, w_t, b_t)
            )

    if n_skipped_grouped or n_skipped_padding or n_skipped_no_shape:
        bar = "=" * 72
        covered = (
            n_total_conv - n_skipped_grouped - n_skipped_padding - n_skipped_no_shape
        )
        pct = 100.0 * covered / n_total_conv if n_total_conv else 0.0
        print(bar)
        print(
            f"[test_models] Benchmarking {covered}/{n_total_conv} Conv2d layers "
            f"({pct:.1f}% coverage)."
        )
        if n_skipped_grouped:
            print(
                f"[test_models] !! {n_skipped_grouped} layer(s) SKIPPED — "
                f"groups>1 (grouped / depthwise) NOT YET SUPPORTED in aiter conv2d."
            )
            print(
                "[test_models]    These layers are excluded from both correctness "
                "and benchmark measurements."
            )
        if n_skipped_padding:
            print(
                f"[test_models] !! {n_skipped_padding} layer(s) SKIPPED — "
                f"padding_mode != 'zeros' (reflect / replicate / circular)."
            )
        if n_skipped_no_shape:
            print(
                f"[test_models] !! {n_skipped_no_shape} layer(s) SKIPPED — "
                f"input shape not captured (dummy forward did not reach them)."
            )
            print(
                "[test_models]    Fix the forward_fn to exercise these layers, "
                "or accept reduced coverage."
            )
            preview = ", ".join(no_shape_keys[:5])
            if len(no_shape_keys) > 5:
                preview += f", ... (+{len(no_shape_keys)-5} more)"
            print(f"[test_models]    Layers: {preview}")
        print(bar)

    torch.cuda.empty_cache()
    return bench_layers


def _bench_conv_layers(
    suite, bench_layers, N, model_name, method, limit, spatial_h=None, spatial_w=None
):
    """Pre-compute MIOpen solvers, pre-warm, and run benchmarks on extracted conv layers."""
    layers = bench_layers[:limit]

    # Override spatial dimensions if requested
    if spatial_h is not None or spatial_w is not None:
        layers = [
            (
                C,
                K_out,
                R,
                S,
                stride,
                padding,
                dilation,
                spatial_h if spatial_h is not None else H,
                spatial_w if spatial_w is not None else W,
                desc,
                w_t,
                b_t,
            )
            for C, K_out, R, S, stride, padding, dilation, H, W, desc, w_t, b_t in layers
        ]

    if suite.bench_enabled and layers:
        from .bench import precompute_miopen_solvers

        miopen_shapes = [
            (N, C, H, W, K_out, R, S, stride, padding, dilation)
            for C, K_out, R, S, stride, padding, dilation, H, W, desc, w_t, b_t in layers
        ]
        print("  (pre-computing MIOpen solver names for all shapes...)")
        precompute_miopen_solvers(miopen_shapes, dtype=suite.dtype)
        print("  (MIOpen solver detection complete)")

    if suite.bench_enabled and layers:
        print(f"  (pre-warming MIOpen for {model_name} layer shapes...)")
        for C, K_out, R, S, stride, padding, dilation, H, W, desc, w_t, b_t in layers:
            x_w = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
            for _ in range(3):
                _ = F.conv2d(
                    x_w, w_t, None, stride=stride, padding=padding, dilation=dilation
                )
            torch.cuda.synchronize()
        del x_w
        torch.cuda.synchronize()
        print("  (MIOpen pre-warm complete)")

    for li, (
        C,
        K_out,
        R,
        S,
        stride,
        padding,
        dilation,
        H,
        W,
        desc,
        w_t,
        b_t,
    ) in enumerate(layers):
        x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite,
            x,
            w_t,
            b_t,
            stride,
            padding,
            dilation,
            name=f"{model_name} L{li} ({desc})",
            method=method,
            activation="none",
        )


def _resolve_torchvision_weights(model_name: str):
    """Return torchvision DEFAULT weights enum for ``model_name`` (or None if unavailable)."""
    from torchvision.models import get_model_weights

    ws = get_model_weights(model_name)
    if ws is not None and len(ws):
        return getattr(ws, "DEFAULT", next(iter(ws)))
    return None


def test_models(
    suite: TestSuite,
    activation: str = "none",
    models: Optional[str] = None,
    num_layers: int = 5,
    method: str = "default",
    model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    spatial_h: Optional[int] = None,
    spatial_w: Optional[int] = None,
    pretrained: bool = False,
):
    tvm = None
    try:
        from torchvision import models as tvm
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("======================== MODEL TESTING =========================")
    print("=" * 80)
    print("Using HYBRID kernel: 1x1->Specialized, 3x3->Specialized, other->General")

    model_names = [
        m.strip() for m in (models.split(",") if models else ["resnet18", "resnet50"])
    ]

    for name in model_names:
        if name == "sd_unet":
            if model_path is None:
                print("  (skip sd_unet: --model-path required)")
                continue

            print(f"  Loading SD U-Net from {model_path} ...")
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(
                model_path, torch_dtype=suite.dtype
            )

            N = batch_size if batch_size is not None else 1
            img_h = spatial_h if spatial_h is not None else 512
            img_w = spatial_w if spatial_w is not None else 512
            lat_h, lat_w = img_h // 8, img_w // 8
            in_ch = pipe.unet.config.in_channels
            cross_dim = pipe.unet.config.cross_attention_dim

            def _unet_forward(unet):
                dummy = torch.randn(
                    N, in_ch, lat_h, lat_w, device=suite.device, dtype=suite.dtype
                )
                dummy_t = torch.tensor([1.0], device=suite.device)
                dummy_enc = torch.randn(
                    N, 77, cross_dim, device=suite.device, dtype=suite.dtype
                )
                unet(dummy, dummy_t, encoder_hidden_states=dummy_enc)

            bench_layers = _extract_conv_layers(
                [("unet", pipe.unet, _unet_forward)],
                suite.device,
                suite.dtype,
                pretrained=pretrained,
            )
            del pipe

            limit = min(num_layers, len(bench_layers))
            print(f"  Found {len(bench_layers)} Conv2d layers, benchmarking {limit}")
            _bench_conv_layers(
                suite,
                bench_layers,
                N,
                "sd_unet",
                method,
                limit,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            )
            continue

        if name == "sd35_vae":
            if model_path is None:
                print("  (skip sd35_vae: --model-path required)")
                continue

            print(f"  Loading SD 3.5 VAE from {model_path} ...")
            from diffusers import AutoencoderKL

            # Load only the VAE subfolder. Avoids pulling the (large)
            # transformer + text encoders just to time conv layers, and
            # lets us bench from a partial checkpoint.
            vae = AutoencoderKL.from_pretrained(
                (
                    model_path + "/vae"
                    if not model_path.rstrip("/").endswith("/vae")
                    else model_path
                ),
                torch_dtype=suite.dtype,
            )

            N = batch_size if batch_size is not None else 1
            img_h = spatial_h if spatial_h is not None else 512
            img_w = spatial_w if spatial_w is not None else 512
            lat_h, lat_w = img_h // 8, img_w // 8

            def _vae_forward(vae_mod):
                dummy_img = torch.randn(
                    1, 3, img_h, img_w, device=suite.device, dtype=suite.dtype
                )
                latent = vae_mod.encode(dummy_img).latent_dist.sample()
                _ = vae_mod.decode(latent)

            bench_layers = _extract_conv_layers(
                [("vae", vae, _vae_forward)],
                suite.device,
                suite.dtype,
                pretrained=pretrained,
            )
            del vae

            limit = min(num_layers, len(bench_layers))
            print(f"  Found {len(bench_layers)} Conv2d layers, benchmarking {limit}")
            _bench_conv_layers(
                suite,
                bench_layers,
                N,
                "sd35",
                method,
                limit,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            )
            continue

        if name == "flux2_vae":
            if model_path is None:
                print("  (skip flux2_vae: --model-path required)")
                continue

            print(f"  Loading FLUX.2-klein-9B model from {model_path} ...")
            from diffusers import AutoencoderKLFlux2

            vae = AutoencoderKLFlux2.from_pretrained(
                model_path + "/vae" if not model_path.endswith("/vae") else model_path,
                torch_dtype=suite.dtype,
            )

            N = batch_size if batch_size is not None else 1
            img_h = spatial_h if spatial_h is not None else 512
            img_w = spatial_w if spatial_w is not None else 512
            # FLUX.2 VAE has 4 down blocks (scale_factor=16) and patch_size=[2,2]
            lat_h, lat_w = img_h // 16, img_w // 16

            def _vae_forward(vae_mod):
                dummy_img = torch.randn(
                    1, 3, img_h, img_w, device=suite.device, dtype=suite.dtype
                )
                latent = vae_mod.encode(dummy_img).latent_dist.sample()
                _ = vae_mod.decode(latent)

            bench_layers = _extract_conv_layers(
                [("vae", vae, _vae_forward)],
                suite.device,
                suite.dtype,
                pretrained=pretrained,
            )
            del vae

            limit = min(num_layers, len(bench_layers))
            print(f"  Found {len(bench_layers)} Conv2d layers, benchmarking {limit}")
            _bench_conv_layers(
                suite,
                bench_layers,
                N,
                "flux2_vae",
                method,
                limit,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            )
            continue

        # Torchvision models
        if tvm is None:
            print(f"  (skip {name}: torchvision not available)")
            continue

        weights = _resolve_torchvision_weights(name) if pretrained else None
        if weights is not None:
            print(f"  (using pretrained weights for {name})")
        try:
            net = getattr(tvm, name)(weights=weights).to(device=suite.device).eval()
        except Exception as e:
            print(f"  (skip {name}: could not construct model: {e})")
            continue

        N = batch_size if batch_size is not None else 1
        H_in = spatial_h if spatial_h is not None else 224
        W_in = spatial_w if spatial_w is not None else H_in

        def _tv_forward(model):
            dummy = torch.randn(
                N, 3, H_in, W_in, device=suite.device, dtype=suite.dtype
            )
            model(dummy)

        bench_layers = _extract_conv_layers(
            [("", net, _tv_forward)],
            suite.device,
            suite.dtype,
            pretrained=pretrained,
        )
        del net

        if not bench_layers:
            print(f"  (skip {name}: no Conv2d layers)")
            continue

        limit = min(num_layers, len(bench_layers))
        _bench_conv_layers(
            suite,
            bench_layers,
            N,
            name,
            method,
            limit,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )
