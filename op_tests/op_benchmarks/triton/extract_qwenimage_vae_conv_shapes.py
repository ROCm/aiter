#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extract Qwen-Image VAE Conv2d/Conv3d shapes via forward hooks.

Downloads only the vae/ subfolder (~254 MB), runs encode+decode at the
official 7 T2I resolutions, writes a full CSV + markdown report, and
optionally merges deduplicated conv2d shapes into conv_shapes.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "qwenimage_vae_conv_shapes.csv"
DEFAULT_REPORT = SCRIPT_DIR / "qwenimage_vae_conv_shapes_report.md"
CONV_SHAPES_JSON = SCRIPT_DIR / "conv_shapes.json"

HF_REPO = "Qwen/Qwen-Image"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"

OFFICIAL_RESOLUTIONS = [
    (1328, 1328),
    (1664, 928),
    (928, 1664),
    (1472, 1140),
    (1140, 1472),
    (1584, 1056),
    (1056, 1584),
]

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

EXPECTED_CAUSAL_CONV3D = {
    "encoder": 26,
    "decoder": 33,
    "quant_conv": 1,
    "post_quant_conv": 1,
}
EXPECTED_TOTAL_CAUSAL = sum(EXPECTED_CAUSAL_CONV3D.values())  # 61
EXPECTED_CONV2D = 10  # 5 encoder + 5 decoder (attention + resample)
EXPECTED_TIME_CONV = 4


@dataclass
class ConvRecord:
    resolution: str
    direction: str  # encode | decode
    module_path: str
    cls: str
    in_ch: int
    out_ch: int
    kernel_t: int
    kernel_h: int
    kernel_w: int
    stride_t: int
    stride_h: int
    stride_w: int
    declared_padding: str
    causal_padding: str
    logical_in_shape: str
    padded_in_shape: str
    out_shape: str
    macs: int
    in_dtype: str
    weight_dtype: str
    bias_dtype: str
    out_dtype: str
    has_bias: bool
    in_bytes: int
    out_bytes: int
    weight_bytes: int
    is_dead_path: bool = False
    section: str = ""  # encoder | decoder | top


@dataclass
class HookState:
    records: list[ConvRecord] = field(default_factory=list)
    call_counts: Counter = field(default_factory=Counter)
    handles: list = field(default_factory=list)


def latent_hw(h: int, w: int) -> tuple[int, int]:
    return 2 * (h // 16), 2 * (w // 16)


def _dtype_str(t: torch.dtype) -> str:
    return str(t).removeprefix("torch.")


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _section_from_path(path: str) -> str:
    if path.startswith("encoder"):
        return "encoder"
    if path.startswith("decoder"):
        return "decoder"
    return "top"


def _is_time_conv(path: str) -> bool:
    return path.endswith("time_conv")


def _compute_padded_shape(
    mod: nn.Module, x: torch.Tensor, cache_x: torch.Tensor | None
) -> tuple[int, ...]:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageCausalConv3d,
    )

    if not isinstance(mod, QwenImageCausalConv3d):
        return tuple(x.shape)

    padding = list(mod._padding)
    t_in = x.shape[2]
    if cache_x is not None and padding[4] > 0:
        t_in += cache_x.shape[2]
        padding[4] -= cache_x.shape[2]

    n, c, _, h, w = x.shape
    padded = (
        n,
        c,
        t_in + padding[4] + padding[5],
        h + padding[2] + padding[3],
        w + padding[0] + padding[1],
    )
    return padded


def _conv_macs(
    mod: nn.Module,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> int:
    if isinstance(mod, nn.Conv2d):
        n, _, h_out, w_out = out_shape
        c_in = mod.in_channels
        k_out = mod.out_channels
        r, s = (
            mod.kernel_size
            if isinstance(mod.kernel_size, tuple)
            else (mod.kernel_size, mod.kernel_size)
        )
        return 2 * n * h_out * w_out * k_out * c_in * r * s

    if isinstance(mod, nn.Conv3d):
        n, _, t_out, h_out, w_out = out_shape
        c_in = mod.in_channels
        k_out = mod.out_channels
        kt, kh, kw = mod.kernel_size
        return 2 * n * t_out * h_out * w_out * k_out * c_in * kt * kh * kw

    return 0


def _declared_padding(mod: nn.Module) -> str:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageCausalConv3d,
    )

    if isinstance(mod, QwenImageCausalConv3d):
        p = mod._padding
        t_pad = p[4] // 2 if p[4] else 0
        s_pad = p[2]
        return str((t_pad, s_pad, s_pad))
    p = mod.padding
    if isinstance(p, tuple):
        return str(p)
    return str((p, p, p) if isinstance(mod, nn.Conv3d) else (p, p))


def _causal_padding(mod: nn.Module) -> str:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageCausalConv3d,
    )

    if isinstance(mod, QwenImageCausalConv3d):
        return str(tuple(mod._padding))
    return ""


def _register_hooks(vae: nn.Module, state: HookState) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageCausalConv3d,
    )

    ctx: dict[str, Any] = {"resolution": "", "direction": ""}

    def make_hook(path: str, mod: nn.Module):
        def hook(_mod, inp, out):
            x = inp[0]
            cache_x = inp[1] if len(inp) > 1 else None
            if not torch.is_tensor(x):
                return

            logical = tuple(x.shape)
            if isinstance(mod, QwenImageCausalConv3d):
                padded = _compute_padded_shape(
                    mod, x, cache_x if torch.is_tensor(cache_x) else None
                )
            else:
                padded = logical

            out_shape = tuple(out.shape)
            kt, kh, kw = (
                mod.kernel_size
                if isinstance(mod, nn.Conv3d)
                else (
                    (1, mod.kernel_size[0], mod.kernel_size[1])
                    if isinstance(mod.kernel_size, tuple)
                    else (1, mod.kernel_size, mod.kernel_size)
                )
            )
            st = mod.stride
            if isinstance(mod, nn.Conv3d):
                stride_t, stride_h, stride_w = st
            else:
                stride_t = 1
                stride_h, stride_w = st if isinstance(st, tuple) else (st, st)

            bias = mod.bias
            rec = ConvRecord(
                resolution=ctx["resolution"],
                direction=ctx["direction"],
                module_path=path,
                cls=type(mod).__name__,
                in_ch=mod.in_channels,
                out_ch=mod.out_channels,
                kernel_t=kt,
                kernel_h=kh,
                kernel_w=kw,
                stride_t=stride_t,
                stride_h=stride_h,
                stride_w=stride_w,
                declared_padding=_declared_padding(mod),
                causal_padding=_causal_padding(mod),
                logical_in_shape=str(logical),
                padded_in_shape=str(padded),
                out_shape=str(out_shape),
                macs=_conv_macs(
                    mod, padded if isinstance(mod, nn.Conv3d) else logical, out_shape
                ),
                in_dtype=_dtype_str(x.dtype),
                weight_dtype=_dtype_str(mod.weight.dtype),
                bias_dtype=_dtype_str(bias.dtype) if bias is not None else "none",
                out_dtype=_dtype_str(out.dtype),
                has_bias=bias is not None,
                in_bytes=_tensor_bytes(x),
                out_bytes=_tensor_bytes(out),
                weight_bytes=_tensor_bytes(mod.weight),
                is_dead_path=_is_time_conv(path),
                section=_section_from_path(path),
            )
            state.records.append(rec)
            state.call_counts[path] += 1

        return hook

    for path, mod in vae.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv3d)):
            h = mod.register_forward_hook(make_hook(path, mod))
            state.handles.append(h)

    state.ctx = ctx  # type: ignore[attr-defined]


def _run_pass(
    vae,
    state: HookState,
    resolution: tuple[int, int],
    direction: str,
    dtype: torch.dtype,
    device: str,
):
    h, w = resolution
    state.ctx["resolution"] = f"{h}x{w}"  # type: ignore[attr-defined]
    state.ctx["direction"] = direction  # type: ignore[attr-defined]

    if direction == "encode":
        x = torch.randn(1, 3, 1, h, w, dtype=dtype, device=device)
        with torch.no_grad():
            vae.encode(x)
    else:
        lh, lw = latent_hw(h, w)
        z = torch.randn(1, 16, 1, lh, lw, dtype=dtype, device=device)
        with torch.no_grad():
            vae.decode(z)


def verify_causal_conv3d_equivalence(vae, device: str, dtype: torch.dtype) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageCausalConv3d,
    )

    conv3d = vae.encoder.conv_in
    assert isinstance(conv3d, QwenImageCausalConv3d) and conv3d.kernel_size == (3, 3, 3)
    h, w = 64, 64
    x = torch.randn(1, conv3d.in_channels, 1, h, w, dtype=dtype, device=device)
    with torch.no_grad():
        y3 = conv3d(x)
        y2 = F.conv2d(
            x.squeeze(2),
            conv3d.weight[:, :, -1, :, :],
            conv3d.bias,
            stride=(conv3d.stride[1], conv3d.stride[2]),
            padding=(1, 1),
        ).unsqueeze(2)
    max_err = (y3 - y2).abs().max().item()
    assert max_err < 1e-2, f"conv3d!=conv2d equivalence failed: max_err={max_err}"


def verify_counts(vae, state: HookState) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageCausalConv3d,
    )

    causal_modules = [
        p for p, m in vae.named_modules() if isinstance(m, QwenImageCausalConv3d)
    ]
    conv2d_modules = [p for p, m in vae.named_modules() if isinstance(m, nn.Conv2d)]

    assert (
        len(causal_modules) == EXPECTED_TOTAL_CAUSAL
    ), f"Expected {EXPECTED_TOTAL_CAUSAL} CausalConv3d, got {len(causal_modules)}"
    assert (
        len(conv2d_modules) == EXPECTED_CONV2D
    ), f"Expected {EXPECTED_CONV2D} Conv2d, got {len(conv2d_modules)}"

    cached = vae._cached_conv_counts
    assert cached["encoder"] == EXPECTED_CAUSAL_CONV3D["encoder"]
    assert cached["decoder"] == EXPECTED_CAUSAL_CONV3D["decoder"]

    time_conv_paths = [p for p in causal_modules if _is_time_conv(p)]
    assert len(time_conv_paths) == EXPECTED_TIME_CONV
    for p in time_conv_paths:
        assert (
            state.call_counts[p] == 0
        ), f"time_conv {p} called {state.call_counts[p]} times, expected 0"


def verify_dtypes(records: list[ConvRecord], expected: str) -> list[ConvRecord]:
    mismatches = []
    for r in records:
        assert r.has_bias, f"{r.module_path} has no bias"
        if not (
            r.in_dtype == r.weight_dtype == r.bias_dtype == r.out_dtype == expected
        ):
            mismatches.append(r)
    assert not mismatches, "dtype mismatch layers: " + ", ".join(
        f"{r.module_path}({r.in_dtype},{r.weight_dtype},{r.bias_dtype},{r.out_dtype})"
        for r in mismatches[:5]
    )
    return mismatches


def conv3d_to_conv2d_entry(rec: ConvRecord) -> dict | None:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageCausalConv3d,
    )

    if rec.cls != "QwenImageCausalConv3d":
        return None
    if (rec.kernel_t, rec.kernel_h, rec.kernel_w) != (3, 3, 3):
        return None

    logical = eval(rec.logical_in_shape)  # noqa: S307 — trusted hook output
    n, c, t, h, w = logical
    if t != 1:
        return None

    out = eval(rec.out_shape)
    _, k, _, h_out, w_out = out

    return {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "K": k,
        "R": 3,
        "S": 3,
        "stride_h": rec.stride_h,
        "stride_w": rec.stride_w,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "_source": rec.module_path,
    }


def conv2d_to_entry(rec: ConvRecord) -> dict:
    logical = eval(rec.logical_in_shape)  # noqa: S307
    n, c, h, w = logical
    out = eval(rec.out_shape)
    k = out[1]

    pad_h = pad_w = 0
    if rec.kernel_h == 3 and rec.stride_h == 1:
        pad_h = pad_w = 1
    elif rec.kernel_h == 3 and rec.stride_h == 2:
        # ZeroPad2d((0,1,0,1)) folded into spatial dims
        h += 1
        w += 1
        pad_h = pad_w = 0

    return {
        "N": n,
        "C": c,
        "H": h,
        "W": w,
        "K": k,
        "R": rec.kernel_h,
        "S": rec.kernel_w,
        "stride_h": rec.stride_h,
        "stride_w": rec.stride_w,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "dilation_h": 1,
        "dilation_w": 1,
        "_source": rec.module_path,
    }


def dedupe_conv2d_shapes(records: list[ConvRecord], call_counts: Counter) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []

    ref_res = "1328x1328"
    ref_recs = [
        r for r in records if r.resolution == ref_res and r.direction == "decode"
    ]

    for rec in ref_recs:
        if rec.is_dead_path and call_counts[rec.module_path] == 0:
            continue
        if rec.cls == "Conv2d":
            entry = conv2d_to_entry(rec)
        else:
            entry = conv3d_to_conv2d_entry(rec)
        if entry is None:
            continue
        key = tuple(
            entry[k]
            for k in (
                "N",
                "C",
                "H",
                "W",
                "K",
                "R",
                "S",
                "stride_h",
                "stride_w",
                "pad_h",
                "pad_w",
            )
        )
        if key in seen:
            continue
        seen.add(key)
        entry_clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        result.append(entry_clean)
    return result


def dedupe_conv3d_shapes(records: list[ConvRecord], call_counts: Counter) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    ref_res = "1328x1328"
    for rec in records:
        if rec.resolution != ref_res or rec.direction != "decode":
            continue
        if rec.cls != "QwenImageCausalConv3d":
            continue
        if rec.is_dead_path and call_counts[rec.module_path] == 0:
            continue
        logical = eval(rec.logical_in_shape)
        padded = eval(rec.padded_in_shape)
        out = eval(rec.out_shape)
        key = (rec.module_path, logical, padded, out)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "module_path": rec.module_path,
                "N": logical[0],
                "C": logical[1],
                "T": logical[2],
                "H": logical[3],
                "W": logical[4],
                "padded_T": padded[2],
                "padded_H": padded[3],
                "padded_W": padded[4],
                "K": out[1],
                "kernel_t": rec.kernel_t,
                "kernel_h": rec.kernel_h,
                "kernel_w": rec.kernel_w,
                "stride_t": rec.stride_t,
                "stride_h": rec.stride_h,
                "stride_w": rec.stride_w,
            }
        )
    return result


def load_csv(path: Path) -> list[ConvRecord]:
    with path.open() as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            row["has_bias"] = row["has_bias"] in ("True", "true", "1")
            row["is_dead_path"] = row["is_dead_path"] in ("True", "true", "1")
            row["macs"] = int(row["macs"])
            row["in_bytes"] = int(row["in_bytes"])
            row["out_bytes"] = int(row["out_bytes"])
            row["weight_bytes"] = int(row["weight_bytes"])
            for k in (
                "in_ch",
                "out_ch",
                "kernel_t",
                "kernel_h",
                "kernel_w",
                "stride_t",
                "stride_h",
                "stride_w",
            ):
                row[k] = int(row[k])
            records.append(ConvRecord(**row))
        return records


def rebuild_call_counts(records: list[ConvRecord]) -> Counter:
    counts: Counter = Counter()
    for r in records:
        counts[r.module_path] += 1
    return counts


def write_csv(records: list[ConvRecord], path: Path) -> None:
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(asdict(records[0]).keys()), lineterminator="\n"
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def write_report(
    records: list[ConvRecord], path: Path, dtype_label: str, param_info: dict
) -> None:
    ref = [
        r for r in records if r.resolution == "1328x1328" and r.direction == "decode"
    ]
    ref_alive = [r for r in ref if not (r.is_dead_path and r.macs == 0)]

    by_macs = sorted(ref_alive, key=lambda r: r.macs, reverse=True)

    dtype_combos = Counter(
        (r.in_dtype, r.weight_dtype, r.bias_dtype, r.out_dtype) for r in records
    )

    lines = [
        "# Qwen-Image VAE Conv Shape Report",
        "",
        f"- Observed dtype: **{dtype_label}**",
        f"- Resolutions: {len(OFFICIAL_RESOLUTIONS)} official T2I sizes × encode/decode",
        f"- Total hook records: {len(records)}",
        "",
        "## Parameter counts (safetensors header, no full download)",
        "",
        "| Component | Params | Disk dtype (sample) |",
        "|-----------|--------|---------------------|",
    ]
    for comp, info in param_info.items():
        lines.append(
            f"| {comp} | {info['params']:,} | {info.get('sample_dtype', 'n/a')} |"
        )

    lines.extend(["", "## Conv layer counts", ""])
    lines.append(f"- CausalConv3d (static): {EXPECTED_TOTAL_CAUSAL}")
    lines.append(f"- Conv2d (static): {EXPECTED_CONV2D}")
    lines.append(f"- time_conv dead paths: {EXPECTED_TIME_CONV} (0 calls each)")

    lines.extend(["", "## Top layers by MACs @ 1328×1328 decode", ""])
    lines.append("| section | path | cls | logical_in | out | MACs | dead |")
    lines.append("|---------|------|-----|------------|-----|------|------|")
    for r in by_macs[:20]:
        lines.append(
            f"| {r.section} | `{r.module_path}` | {r.cls} | {r.logical_in_shape} | {r.out_shape} | {r.macs:,} | {r.is_dead_path} |"
        )

    conv3d_recs = [
        r for r in ref if r.cls == "QwenImageCausalConv3d" and not r.is_dead_path
    ]
    conv2d_recs = [r for r in ref if r.cls == "Conv2d"]

    lines.extend(["", "## Conv3d layers @ 1328 decode", ""])
    lines.append("| path | logical_in | padded_in | out | MACs |")
    lines.append("|------|------------|-----------|-----|------|")
    for r in sorted(conv3d_recs, key=lambda x: x.module_path):
        lines.append(
            f"| `{r.module_path}` | {r.logical_in_shape} | {r.padded_in_shape} | {r.out_shape} | {r.macs:,} |"
        )

    lines.extend(["", "## Conv2d layers @ 1328 decode", ""])
    lines.append("| path | logical_in | out | MACs |")
    lines.append("|------|------------|-----|------|")
    for r in sorted(conv2d_recs, key=lambda x: x.module_path):
        lines.append(
            f"| `{r.module_path}` | {r.logical_in_shape} | {r.out_shape} | {r.macs:,} |"
        )

    lines.extend(["", "## Dtype combinations (all records)", ""])
    lines.append("| in | weight | bias | out | count |")
    lines.append("|----|--------|------|-----|-------|")
    for combo, cnt in dtype_combos.most_common():
        lines.append(f"| {combo[0]} | {combo[1]} | {combo[2]} | {combo[3]} | {cnt} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Single-frame T=1: all 3×3×3 CausalConv3d equivalent to Conv2d with `W[:,:,-1]`.",
            "- MIOpen/cuDNN accumulate bf16 conv in fp32 internally (not visible in module dtypes).",
            "- RMS_norm and Upsample run normalize/interp in fp32 but cast back before conv input.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def fetch_safetensors_header_url(url: str) -> dict:
    req = Request(url, headers={"Range": "bytes=0-7"})
    with urlopen(req, timeout=60) as resp:
        header_len = struct.unpack("<Q", resp.read(8))[0]

    req2 = Request(url, headers={"Range": f"bytes=8-{8 + header_len - 1}"})
    with urlopen(req2, timeout=60) as resp:
        header_json = resp.read().decode("utf-8")

    return json.loads(header_json)


def fetch_safetensors_header(
    subfolder: str, filename: str = "diffusion_pytorch_model.safetensors"
) -> dict:
    url = f"{HF_BASE}/{subfolder}/{filename}"
    return fetch_safetensors_header_url(url)


def _list_subfolder_files(subfolder: str) -> list[str]:
    api_url = f"https://huggingface.co/api/models/{HF_REPO}/tree/main/{subfolder}"
    with urlopen(api_url, timeout=60) as resp:
        entries = json.loads(resp.read())
    return [e["path"].split("/")[-1] for e in entries]


def _shard_files_from_index(subfolder: str) -> list[str]:
    index_name = None
    for name in _list_subfolder_files(subfolder):
        if name.endswith(".safetensors.index.json"):
            index_name = name
            break
    if index_name is None:
        return []

    index_url = f"{HF_BASE}/{subfolder}/{index_name}"
    with urlopen(index_url, timeout=60) as resp:
        index = json.loads(resp.read())
    return sorted(set(index["weight_map"].values()))


def count_params_for_subfolder(subfolder: str) -> tuple[int, Counter]:
    files = _list_subfolder_files(subfolder)
    shard_names = [f for f in files if f.endswith(".safetensors")]
    if not shard_names:
        shard_names = _shard_files_from_index(subfolder)
    if not shard_names:
        raise FileNotFoundError(f"No safetensors in {subfolder}")

    total = 0
    dtypes: Counter = Counter()
    seen_tensors: set[str] = set()
    for fname in shard_names:
        header = fetch_safetensors_header(subfolder, fname)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in seen_tensors:
                continue
            seen_tensors.add(name)
            shape = meta["shape"]
            n = 1
            for d in shape:
                n *= d
            total += n
            dtypes[meta.get("dtype", "unknown")] += n
    return total, dtypes


def fetch_param_counts() -> dict:
    components = {
        "vae": "vae",
        "transformer": "transformer",
        "text_encoder": "text_encoder",
    }
    result = {}
    for label, sub in components.items():
        try:
            params, dtypes = count_params_for_subfolder(sub)
            result[label] = {
                "params": params,
                "dtypes": dict(dtypes),
                "sample_dtype": dtypes.most_common(1)[0][0] if dtypes else "unknown",
            }
        except Exception as e:
            result[label] = {"params": -1, "error": str(e)}
    return result


def merge_conv_shapes(
    conv2d: list[dict], conv3d: list[dict], observed_dtype: str
) -> None:
    with CONV_SHAPES_JSON.open() as f:
        data = json.load(f)

    data["Qwen-Image-VAE"] = {
        "conv2d": conv2d,
        "conv3d": conv3d,
        "observed_dtype": observed_dtype,
    }

    with CONV_SHAPES_JSON.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--merge-json", action="store_true", help="Update conv_shapes.json"
    )
    parser.add_argument(
        "--from-csv", action="store_true", help="Skip extraction; load existing CSV"
    )
    parser.add_argument("--skip-download-test", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    dtype = DTYPE_MAP[args.dtype]
    dtype_label = args.dtype

    from diffusers import AutoencoderKLQwenImage

    print(f"Loading VAE ({HF_REPO}/vae) dtype={dtype_label} device={args.device} ...")
    vae = (
        AutoencoderKLQwenImage.from_pretrained(
            HF_REPO,
            subfolder="vae",
            torch_dtype=dtype,
        )
        .to(args.device)
        .eval()
    )

    state = HookState()

    if args.from_csv:
        if not args.csv.is_file():
            print(f"ERROR: --from-csv but {args.csv} not found", file=sys.stderr)
            sys.exit(1)
        records = load_csv(args.csv)
        state.call_counts = rebuild_call_counts(records)
        print(f"Loaded CSV: {args.csv} ({len(records)} rows)")
    else:
        _register_hooks(vae, state)
        print("Running shape extraction ...")
        for res in OFFICIAL_RESOLUTIONS:
            print(f"  encode {res[0]}x{res[1]}")
            _run_pass(vae, state, res, "encode", dtype, args.device)
            print(f"  decode {res[0]}x{res[1]} (latent {latent_hw(*res)})")
            _run_pass(vae, state, res, "decode", dtype, args.device)
        records = state.records
        write_csv(records, args.csv)
        print(f"Wrote CSV: {args.csv} ({len(records)} rows)")

    print("Verifying ...")
    verify_causal_conv3d_equivalence(vae, args.device, dtype)
    verify_counts(vae, state)
    verify_dtypes(records, dtype_label if dtype_label != "bf16" else "bfloat16")

    print("Fetching safetensors param counts ...")
    param_info = fetch_param_counts()

    write_report(records, args.report, dtype_label, param_info)
    print(f"Wrote report: {args.report}")

    if args.merge_json:
        conv2d_shapes = dedupe_conv2d_shapes(records, state.call_counts)
        conv3d_shapes = dedupe_conv3d_shapes(records, state.call_counts)
        merge_conv_shapes(conv2d_shapes, conv3d_shapes, dtype_label)
        print(
            f"Merged {len(conv2d_shapes)} conv2d + {len(conv3d_shapes)} conv3d shapes into conv_shapes.json"
        )

    if not args.from_csv:
        for h in state.handles:
            h.remove()

    print("Done.")


if __name__ == "__main__":
    main()
