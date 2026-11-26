# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Kimi-K3 Tiny-L13 (``Qinghao/Kimi-K3-Tiny-L13``, ZhangDanyang-AMD/Lumen#35) KDA
layer weights as ``KDALayerWeights``.

Tiny-L13 keeps the real K3 widths (hidden 7168, 96 KDA heads x 128, conv width
4, F-decay rank 128) and has 13 decoder layers. ``linear_attn_config`` in its
config.json is 1-indexed: KDA layers [1,2,3,5,6,7,9,10,11] and full-attention
(gated MLA) layers [4,8,12,13], i.e. 0-indexed KDA 0-2, 4-6, 8-10 and MLA 3, 7,
11, 12.

The checkpoint directory comes from the ``path`` argument or the
``AITER_KIMI_TINY_L13_PATH`` env var (a local snapshot, e.g. from
``huggingface-cli download Qinghao/Kimi-K3-Tiny-L13``). Without either, weights
are random-initialized at the same shapes.
"""

import json
import os

import torch

from aiter.ops.kda_fused_layer import KDALayerWeights

TINY_L13_REPO = "Qinghao/Kimi-K3-Tiny-L13"
TINY_L13_PATH_ENV = "AITER_KIMI_TINY_L13_PATH"

HIDDEN, HEADS, HEAD_DIM, CONV, GATE_RANK = 7168, 96, 128, 4, 128
NUM_LAYERS = 13
KDA_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10)
MLA_LAYERS = (3, 7, 11, 12)
RMS_NORM_EPS = 1e-6

_PREFIX = "language_model.model.layers.{}.self_attn."
_KEYS = {
    "w_q": "q_proj.weight",
    "w_k": "k_proj.weight",
    "w_v": "v_proj.weight",
    "conv_q": "q_conv1d.weight",
    "conv_k": "k_conv1d.weight",
    "conv_v": "v_conv1d.weight",
    "f_a": "f_a_proj.weight",
    "f_b": "f_b_proj.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "w_beta": "b_proj.weight",
    "g_proj": "g_proj.weight",
    "norm_w": "o_norm.weight",
    "w_o": "o_proj.weight",
}


def checkpoint_path(path=None):
    return path or os.environ.get(TINY_L13_PATH_ENV) or None


def load_config(path=None):
    """KDA geometry and layer split. Falls back to the published values."""
    cfg = {
        "hidden": HIDDEN,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "conv": CONV,
        "gate_rank": GATE_RANK,
        "num_layers": NUM_LAYERS,
        "kda_layers": KDA_LAYERS,
        "mla_layers": MLA_LAYERS,
        "eps": RMS_NORM_EPS,
        "gate_lower_bound": -5.0,
    }
    path = checkpoint_path(path)
    if path is None:
        return cfg
    with open(os.path.join(path, "config.json")) as f:
        tc = json.load(f)
    tc = tc.get("text_config", tc)
    la = tc["linear_attn_config"]
    cfg.update(
        hidden=tc["hidden_size"],
        heads=la["num_heads"],
        head_dim=la["head_dim"],
        conv=la["short_conv_kernel_size"],
        gate_rank=la["head_dim"],
        num_layers=tc["num_hidden_layers"],
        kda_layers=tuple(i - 1 for i in la["kda_layers"]),
        mla_layers=tuple(i - 1 for i in la["full_attn_layers"]),
        eps=tc.get("rms_norm_eps", RMS_NORM_EPS),
        gate_lower_bound=la.get("gate_lower_bound"),
    )
    return cfg


def random_weights(cfg, dtype, seed=0):
    D, H, K, W, R = (cfg[n] for n in ("hidden", "heads", "head_dim", "conv", "gate_rank"))
    g = torch.Generator(device="cpu").manual_seed(seed)

    def p(*shape, std):
        return (torch.randn(*shape, generator=g) * std).to(dtype)

    return KDALayerWeights(
        w_q=p(H * K, D, std=D**-0.5),
        w_k=p(H * K, D, std=D**-0.5),
        w_v=p(H * K, D, std=D**-0.5),
        conv_q=p(H * K, W, std=0.5),
        conv_k=p(H * K, W, std=0.5),
        conv_v=p(H * K, W, std=0.5),
        f_a=p(R, D, std=D**-0.5),
        f_b=p(H * K, R, std=R**-0.5),
        A_log=torch.log(torch.empty(H).uniform_(1, 16, generator=g)).to(dtype),
        dt_bias=p(H * K, std=1.0),
        w_beta=p(H, D, std=D**-0.5),
        g_proj=p(H * K, D, std=D**-0.5),
        norm_w=(1.0 + torch.randn(K, generator=g) * 0.1).to(dtype),
        w_o=p(D, H * K, std=(H * K) ** -0.5),
    )


def _weight_map(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"]
    return None


def _read(path, wmap, names):
    from safetensors import safe_open

    by_file = {}
    for n in names:
        fname = wmap[n] if wmap is not None else "model.safetensors"
        by_file.setdefault(fname, []).append(n)
    out = {}
    for fname, ns in by_file.items():
        with safe_open(os.path.join(path, fname), framework="pt", device="cpu") as f:
            for n in ns:
                out[n] = f.get_tensor(n)
    return out


def load_kda_layer(layer, path=None, dtype=torch.bfloat16, device="cuda", cfg=None, a_log="error"):
    """KDALayerWeights for 0-indexed decoder ``layer`` of Tiny-L13.

    The published K3 checkpoints store ``A_log`` as [head_dim] (128), while
    ``KDALayerWeights`` (and HF ``KimiDeltaAttention.__init__``) use one decay
    per head [num_heads] (96). ``a_log="error"`` raises on that mismatch;
    ``a_log="random"`` keeps every other real tensor and draws A_log the way the
    HF init does.
    """
    cfg = cfg or load_config(path)
    if layer not in cfg["kda_layers"]:
        raise ValueError(f"layer {layer} is not a KDA layer; KDA layers are {cfg['kda_layers']}")
    path = checkpoint_path(path)
    if path is None:
        w = random_weights(cfg, dtype, seed=layer)
        return KDALayerWeights.from_tensors([t.to(device) for t in w.tensors()])

    H, K = cfg["heads"], cfg["head_dim"]
    prefix = _PREFIX.format(layer)
    names = {field: prefix + key for field, key in _KEYS.items()}
    raw = _read(path, _weight_map(path), names.values())
    ts = {field: raw[name] for field, name in names.items()}
    for c in ("conv_q", "conv_k", "conv_v"):
        ts[c] = ts[c].reshape(ts[c].shape[0], -1)  # [C, 1, W] -> [C, W]
    if ts["A_log"].numel() != H:
        if a_log != "random":
            raise ValueError(
                f"{names['A_log']} has shape {tuple(ts['A_log'].shape)}, expected [{H}] "
                "(one decay per head); pass a_log='random' to substitute it"
            )
        g = torch.Generator(device="cpu").manual_seed(layer)
        ts["A_log"] = torch.log(torch.empty(H).uniform_(1, 16, generator=g))
    D, W, R, HK = cfg["hidden"], cfg["conv"], cfg["gate_rank"], H * K
    expect = {
        "w_q": (HK, D), "w_k": (HK, D), "w_v": (HK, D),
        "conv_q": (HK, W), "conv_k": (HK, W), "conv_v": (HK, W),
        "f_a": (R, D), "f_b": (HK, R), "A_log": (H,), "dt_bias": (HK,),
        "w_beta": (H, D), "g_proj": (HK, D), "norm_w": (K,), "w_o": (D, HK),
    }  # fmt: skip
    for n, t in ts.items():
        if tuple(t.shape) != (want := expect[n]):
            raise ValueError(f"{names[n]}: shape {tuple(t.shape)}, expected {want}")
    return KDALayerWeights(
        **{n: t.to(device=device, dtype=dtype).contiguous() for n, t in ts.items()}
    )


def load_kda_layers(path=None, dtype=torch.bfloat16, device="cuda", a_log="error"):
    """{layer: KDALayerWeights} for every KDA layer of Tiny-L13."""
    cfg = load_config(path)
    return cfg, {
        i: load_kda_layer(i, path, dtype, device, cfg, a_log) for i in cfg["kda_layers"]
    }
