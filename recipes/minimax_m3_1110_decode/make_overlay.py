# SPDX-License-Identifier: MIT
"""Generate a guarded TP-only M16 overlay for the MiniMax-M3 2026-11-10 image.

Pure CPU; does not import aiter or touch GPUs. Never edits the supplied baseline.
"""

import argparse
import csv
import hashlib
import io
import pathlib
import subprocess

PINNED_HEAD = "55cde6a0c49cc04f39f6141e312938a8c48e61fb"
PINNED_FILES = {
    "aiter/ops/flydsl/grouped_moe_gfx1250.py": "1efc90f3d35cc5ec337cfaab08d9116f45977c89cc4acfde1f1e79986f8f4089",
    "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py": "76e9c5662536b35167bacab5d748d2c031b416033c69b0238cadc48bd5019f01",
}
KEY = {
    "gfx": "gfx1250",
    "cu_num": "256",
    "token": "16",
    "model_dim": "6144",
    "inter_dim": "768",
    "expert": "129",
    "topk": "5",
    "act_type": "ActivationType.Swiglu",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
    "use_g1u1": "1",
    "doweight_stage1": "0",
}
EXPECTED = {
    "tile_m": "16",
    "tile_n": "384",
    "tile_k": "128",
    "m_warp": "1",
    "n_warp": "2",
    "num_buffers": "3",
    "tile_m2": "16",
    "tile_n2": "128",
    "tile_k2": "128",
    "m_warp2": "1",
    "n_warp2": "2",
    "num_buffer_stage2": "2",
    "split_k1": "",
    "split_k2": "",
    "waves_per_tensor_tdm": "-1",
    "waves_per_tensor_tdm2": "",
    "tdm_b_th": "0",
    "a_preshuffle": "0",
    "persistent_workers": "0",
    "persistent_workers2": "0",
    "cluster_m": "-1",
    "cluster_n": "-1",
    "cluster_m2": "",
    "cluster_n2": "",
    "next_stage_prefetch": "0",
    "tdm_as_in_prologue": "0",
    "lds_soa_load_interleave": "0",
    "rolled_drain": "0",
    "ep_fused": "",
}
UPDATE = {
    "tile_n": "128",
    "n_warp": "1",
    "split_k1": "2",
    "waves_per_tensor_tdm": "1",
    "tdm_b_th": "1",
    "ep_fused": "0",
    "us": "32.60",
}


def check_source(root):
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PINNED_HEAD:
        raise ValueError(f"requires recipe AITER {PINNED_HEAD}, got {head}")
    for name, expected in PINNED_FILES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source mismatch: {name}; refusing unmeasured kernel")


def make_overlay(text):
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames
    rows = list(reader)
    if fields is None or not set(KEY | EXPECTED | UPDATE) <= set(fields):
        raise ValueError("baseline CSV lacks required fields")
    matched = [r for r in rows if all(r[k] == v for k, v in KEY.items())]
    if len(matched) != 1:
        raise ValueError(f"expected one M16 FP4 row, found {len(matched)}")
    row = matched[0]
    for k, v in EXPECTED.items():
        if row[k] != v:
            raise ValueError(f"unmeasured baseline field {k}: {row[k]!r} != {v!r}")
    # ep-specific TP row wins; wildcard row remains an unchanged EP fallback.
    new = dict(row, **UPDATE)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows + [new])
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-root", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    check_source(args.aiter_root)
    result = make_overlay(args.baseline.read_text())
    with args.output.open("x") as f:
        f.write(result)
    print(
        f"Created {args.output}; baseline unchanged. Pass "
        f"AITER_CONFIG_GROUPED_FMOE={args.output.resolve()} to the server."
    )


if __name__ == "__main__":
    main()
