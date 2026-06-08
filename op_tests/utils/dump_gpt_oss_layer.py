"""Extract a single MoE layer's expert weights from gpt-oss safetensors.

Downloads only the safetensors shard that contains the requested layer
(typically ~4-5 GB for gpt-oss-120b) plus the small index/config files,
then writes each expert tensor into <output-dir>/ as a torch .pt:

  model.layers.{L}.mlp.experts.gate_up_proj_blocks.pt   # mxfp4 packed
  model.layers.{L}.mlp.experts.gate_up_proj_scales.pt   # e8m0 (uint8)
  model.layers.{L}.mlp.experts.gate_up_proj_bias.pt     # fp32
  model.layers.{L}.mlp.experts.down_proj_blocks.pt
  model.layers.{L}.mlp.experts.down_proj_scales.pt
  model.layers.{L}.mlp.experts.down_proj_bias.pt
  model.layers.{L}.mlp.router.weight.pt
  model.layers.{L}.mlp.router.bias.pt

It also dumps a small ``_meta.json`` describing the layout assumption.
gpt-oss expert MLPs use the GUGU interleaved layout: gate = [..., 0::2],
up = [..., 1::2] along the 2*intermediate axis.

Usage:
  python op_tests/utils/dump_gpt_oss_layer.py \
      --repo openai/gpt-oss-120b \
      --layer 0 \
      --output-dir /home/carhuang/wjx/atom_moe_all_dump/rank0/weights/model.layers.0.mlp.experts \
      [--cache-dir /tmp/hf-cache] [--token $HF_TOKEN]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _hf_download(repo: str, filename: str, cache_dir: str | None, token: str | None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
        resume_download=True,
    )


def _find_layer_shards(index_path: str, layer: int) -> tuple[dict[str, str], set[str]]:
    """Return (key->shard) for matching keys and the set of shards needed."""
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]
    prefixes = (
        f"model.layers.{layer}.mlp.experts.",
        f"model.layers.{layer}.mlp.router.",
    )
    selected = {k: v for k, v in weight_map.items() if k.startswith(prefixes)}
    shards = set(selected.values())
    return selected, shards


def _save_tensor(t, path: str) -> None:
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    t = t.contiguous().cpu()
    torch.save(t, str(p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="openai/gpt-oss-120b",
                        help="HuggingFace repo id (default: %(default)s).")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index to extract (default: %(default)s).")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the per-tensor .pt files.")
    parser.add_argument("--cache-dir", default=None,
                        help="HF cache dir (default: ~/.cache/huggingface).")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HF token (default: $HF_TOKEN).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print which shards would be downloaded and exit.")
    parser.add_argument("--print-shapes", action="store_true",
                        help="After download, print every tensor's shape/dtype.")
    args = parser.parse_args()

    print(f"[dump_gpt_oss_layer] repo={args.repo} layer={args.layer} -> {args.output_dir}", flush=True)

    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        sys.exit(f"missing dep: {e}. Run: pip install huggingface_hub safetensors torch")

    print("[dump_gpt_oss_layer] fetching index...", flush=True)
    idx_path = _hf_download(args.repo, "model.safetensors.index.json", args.cache_dir, args.token)
    cfg_path = _hf_download(args.repo, "config.json", args.cache_dir, args.token)
    selected, shards = _find_layer_shards(idx_path, args.layer)
    if not selected:
        sys.exit(f"no keys matched layer {args.layer} in {idx_path}")

    print(f"[dump_gpt_oss_layer] {len(selected)} tensors across {len(shards)} shard(s):", flush=True)
    for k in sorted(selected):
        print(f"   {selected[k]}  {k}", flush=True)
    if args.dry_run:
        return

    shard_paths: dict[str, str] = {}
    for shard in sorted(shards):
        print(f"[dump_gpt_oss_layer] downloading {shard} ...", flush=True)
        shard_paths[shard] = _hf_download(args.repo, shard, args.cache_dir, args.token)

    from safetensors import safe_open
    import torch

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written = []
    shapes = {}
    for shard, path in shard_paths.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            keys_here = [k for k, v in selected.items() if v == shard]
            for k in keys_here:
                t = f.get_tensor(k)
                shapes[k] = (tuple(t.shape), str(t.dtype))
                fname = k + ".pt"
                fpath = out_root / fname
                _save_tensor(t, str(fpath))
                written.append((k, fpath))
                print(f"[dump_gpt_oss_layer] wrote {fpath} shape={tuple(t.shape)} dtype={t.dtype}", flush=True)

    cfg = {}
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        pass

    meta = {
        "layer_name": f"model.layers.{args.layer}.mlp.experts",
        "repo": args.repo,
        "layer": args.layer,
        "layout": "gugu",
        "notes": (
            "gpt-oss expert MLP uses GUGU interleaving along the 2*intermediate axis: "
            "gate = (..., 0::2), up = (..., 1::2). Set "
            "AITER_GROUPED_STAGE1_WEIGHT_LAYOUT=gugu when running grouped GEMM."
        ),
        "global_num_experts": cfg.get("num_experts") or cfg.get("num_local_experts"),
        "top_k": cfg.get("num_experts_per_tok") or cfg.get("num_active_experts"),
        "hidden_size": cfg.get("hidden_size"),
        "intermediate_size": cfg.get("intermediate_size"),
        "tensors": {k: {"shape": list(s), "dtype": d} for k, (s, d) in shapes.items()},
    }
    with open(out_root / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dump_gpt_oss_layer] wrote {out_root / '_meta.json'}", flush=True)

    if args.print_shapes:
        print("\n[dump_gpt_oss_layer] summary:")
        for k in sorted(shapes):
            s, d = shapes[k]
            print(f"   {k:60s}  {d:20s}  {s}")


if __name__ == "__main__":
    main()
