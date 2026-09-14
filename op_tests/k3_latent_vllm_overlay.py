# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Install the decode-only K3 latent FHMoE call into a vLLM source tree.

This is an intentionally small, idempotent prototype overlay.  It patches the
runner after the IX recipe has installed its own K3 performance patches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "maybe_run_vllm_k3_latent_fhmoe"
OLD = """        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )
"""
NEW = """        from aiter.latent_fhmoe_vllm import maybe_run_vllm_k3_latent_fhmoe

        result = maybe_run_vllm_k3_latent_fhmoe(
            self,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )
        if result is None:
            result = self._forward_entry(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                self._encode_layer_name(),
                self.moe_config.hidden_dim_unpadded
                if self._quant_method.has_unpadded_output
                else 0,
            )
"""


def _replace_once(target: Path, old: str, new: str, marker: str) -> bool:
    text = target.read_text()
    if marker in text:
        return False
    matches = text.count(old)
    if matches != 1:
        raise RuntimeError(f"Expected one patch site in {target}, found {matches}")
    target.write_text(text.replace(old, new, 1))
    return True


def install(vllm_root: Path) -> bool:
    target = vllm_root / "models/kimi_k3/amd/latent_moe_runner.py"
    return _replace_once(target, OLD, NEW, MARKER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    args = parser.parse_args()
    changed = install(args.vllm_root)
    print("installed K3 latent FHMoE overlay" if changed else "overlay already installed")


if __name__ == "__main__":
    main()
