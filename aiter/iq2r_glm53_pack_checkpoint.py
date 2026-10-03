# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Write a self-contained GLM-5.3 IQ2R checkpoint in the packed layout.

usage: python -m aiter.iq2r_glm53_pack_checkpoint SRC_MODEL OUT_DIR {iq2r|base|meta} [--part P --parts N]

SRC_MODEL is a GLM-5.3 IQ2R model in the generic per-expert IQ2R layout, as
written by ``aiter.iq2r_overlay`` (config.json, model.safetensors.index.json,
the base FP8 shards and iq2r-layer-NNNN-{gate-up,down}.safetensors). Run the
three stages in order:

- ``iq2r`` (GPU): relayout every IQ2R layer with
  ``aiter.iq2r_glm53.iq2r_glm53_pack``. ``--part P --parts N`` splits the
  layers across N processes.
- ``base`` (CPU): copy the non-expert tensors into fresh ~5 GiB shards.
- ``meta``: write the index and config.json (``iq2r_layout`` =
  ``glm53-packed-v1``) and copy the tokenizer, chat template and license.

ATOM loads the result as is and only slices the packed experts per TP rank.
Every stage skips outputs that already exist, so an interrupted run can be
restarted.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYOUT = "glm53-packed-v1"
SHARD_BYTES = 5 << 30
DTYPE_BYTES = {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}


def _read_index(src):
    index = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    iq2r_files = sorted({f for f in index.values() if f.startswith("iq2r-layer-")})
    base_keys = sorted(k for k, f in index.items() if not f.startswith("iq2r-layer-"))
    return index, iq2r_files, base_keys


def pack_iq2r_layers(src, out, part=0, parts=1):
    from aiter.iq2r_glm53 import iq2r_glm53_pack

    _, iq2r_files, _ = _read_index(src)
    config = json.load(open(f"{src}/config.json"))
    dev = torch.device("cuda", torch.cuda.current_device())
    layers = sorted({int(f.split("-")[2]) for f in iq2r_files})[part::parts]
    for layer in layers:
        names = {
            p: f"iq2r-layer-{layer:04d}-{p}.safetensors" for p in ("gate-up", "down")
        }
        if all(os.path.exists(f"{out}/{n}") for n in names.values()):
            continue
        tensors, meta = {}, {}
        for proj, name in names.items():
            with safe_open(f"{src}/{name}", "pt", device="cpu") as f:
                meta[proj] = dict(f.metadata())
                if meta[proj].get("iq2r_layout") == LAYOUT:
                    raise SystemExit(
                        f"{name} is already packed; SRC_MODEL must be unpacked"
                    )
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        key = {
            s: next(k for k in tensors if k.endswith(s))
            for s in (
                "up_gate_proj.0.iq2r_data",
                "up_gate_proj.0.iq2r_auxiliary",
                "down_proj.0.iq2r_data",
                "down_proj.0.iq2r_auxiliary",
            )
        }
        packed = iq2r_glm53_pack(
            *(tensors[key[s]].to(dev) for s in key),
            intermediate_size=config["moe_intermediate_size"],
        )
        # (gate_quad, gate_auxiliary, down_data, down_auxiliary) keep the source names.
        for s, value in zip(key, packed):
            tensors[key[s]] = value.cpu().contiguous()
        for proj, name in names.items():
            subset = {
                k: v
                for k, v in tensors.items()
                if (".down_proj." in k) == (proj == "down")
            }
            tmp = f"{out}/.{name}.tmp"
            save_file(subset, tmp, metadata={**meta[proj], "iq2r_layout": LAYOUT})
            os.replace(tmp, f"{out}/{name}")
        print("packed layer", layer, flush=True)


def write_base_shards(src, out, part=0, parts=1):
    index, _, base_keys = _read_index(src)
    # Group keys by source shard so each source file is opened once.
    by_file = {}
    for key in base_keys:
        by_file.setdefault(index[key], []).append(key)
    groups, current, size = [], [], 0
    for src_file in sorted(by_file):
        with safe_open(f"{src}/{src_file}", "pt") as f:
            for key in by_file[src_file]:
                sl = f.get_slice(key)
                nbytes = DTYPE_BYTES[sl.get_dtype()]
                for d in sl.get_shape():
                    nbytes *= d
                if current and size + nbytes > SHARD_BYTES:
                    groups.append(current)
                    current, size = [], 0
                current.append((src_file, key))
                size += nbytes
    if current:
        groups.append(current)
    total = len(groups)
    for i, group in enumerate(groups):
        if i % parts != part:
            continue
        name = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        if os.path.exists(f"{out}/{name}"):
            continue
        tensors = {}
        for src_file, key in group:
            with safe_open(f"{src}/{src_file}", "pt") as f:
                tensors[key] = f.get_tensor(key)
        tmp = f"{out}/.{name}.tmp"
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, f"{out}/{name}")
        print("wrote", name, len(tensors), flush=True)
    json.dump(
        [[k for _, k in g] for g in groups], open(f"{out}/.base-groups.json", "w")
    )


def write_metadata(src, out):
    index, _, _ = _read_index(src)
    groups = json.load(open(f"{out}/.base-groups.json"))
    weight_map = {}
    for i, keys in enumerate(groups):
        for key in keys:
            weight_map[key] = f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors"
    for key, f in index.items():
        if f.startswith("iq2r-layer-"):
            weight_map[key] = f
    total_size = sum(os.path.getsize(f"{out}/{f}") for f in set(weight_map.values()))
    json.dump(
        {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        },
        open(f"{out}/model.safetensors.index.json", "w"),
        indent=2,
    )
    config = json.load(open(f"{src}/config.json"))
    config["quantization_config"]["iq2r_layout"] = LAYOUT
    json.dump(config, open(f"{out}/config.json", "w"), indent=2)
    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
    ):
        if os.path.exists(f"{src}/{name}"):
            shutil.copy(os.path.realpath(f"{src}/{name}"), f"{out}/{name}")
    missing = [f for f in set(weight_map.values()) if not os.path.exists(f"{out}/{f}")]
    print(
        "index tensors",
        len(weight_map),
        "files",
        len(set(weight_map.values())),
        "missing",
        missing,
        "total GB",
        total_size / 1e9,
    )
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("stage", choices=("iq2r", "base", "meta"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.stage == "iq2r":
        pack_iq2r_layers(args.src, args.out, args.part, args.parts)
    elif args.stage == "base":
        write_base_shards(args.src, args.out, args.part, args.parts)
    else:
        write_metadata(args.src, args.out)


if __name__ == "__main__":
    main()
