# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kid table of the gfx950 FP8 blockscale grouped GEMMs (gen_co/co_kernels_gfx950_a8w8_blockscale_grouped.json).

Read by gen_co/build_co.py --family a8w8_blockscale_grouped (offline .co images), gen_instances.py (the host
kid table header) and aiter/ops/opus/gemm_op_a8w8_blockscale_grouped.py (kid choice).
"""

import json
import os
from typing import NamedTuple

A8W8_BLOCKSCALE_GROUPED_CO_JSON = os.path.join(
    os.path.dirname(__file__),
    "gen_co",
    "co_kernels_gfx950_a8w8_blockscale_grouped.json",
)
A8W8_BLOCKSCALE_GROUPED_CO_DIR = "gfx950/a8w8_blockscale_grouped"
# Images of a kid: M-grouped NT / NN; K-grouped TN with bf16 / fp32 C.
A8W8_BLOCKSCALE_GROUPED_LAYOUTS = {
    "m_grouped": ("nt", "nn"),
    "k_grouped": ("tn", "tn_f32"),
}
# Scales E8M0: M-grouped A 1x128, B 128x128; K-grouped A and B 128x1 (along the rows).
_A8W8_BS_NAME_TAGS = {
    "m_grouped": ("m_grouped", "1x128x128"),
    "k_grouped": ("k_grouped", "128x1x128x1"),
}


class A8W8BlockscaleGroupedKid(NamedTuple):
    kid: int
    kind: str  # m_grouped / k_grouped
    B_M: int
    B_N: int

    @property
    def name(self) -> str:
        """Image name stem (= ELF entry symbol without the layout suffix)."""
        tag, scales = _A8W8_BS_NAME_TAGS[self.kind]
        return (
            f"opus_gemm_gfx950_mxfp8_{tag}_256x{self.B_M}x{self.B_N}"
            f"x128_2x2_16x16x128_{scales}"
        )

    def images(self) -> list[tuple[str, str]]:
        """[(layout, symbol)] of the kid's images <name>_<layout>.co."""
        return [
            (lay, f"{self.name}_{lay}")
            for lay in A8W8_BLOCKSCALE_GROUPED_LAYOUTS[self.kind]
        ]


def _load_a8w8_blockscale(path):
    if not os.path.exists(path):
        return {}
    with open(path) as stream:
        entries = json.load(stream)["gfx950"]
    kids = {}
    for e in entries:
        assert e["kind"] in _A8W8_BS_NAME_TAGS and e["kid"] not in kids, e
        kids[e["kid"]] = A8W8BlockscaleGroupedKid(e["kid"], e["kind"], *e["tile"])
    return kids


def _a8w8_blockscale_images_present(k: A8W8BlockscaleGroupedKid) -> bool:
    root = os.path.join(
        os.path.dirname(A8W8_BLOCKSCALE_GROUPED_CO_JSON), A8W8_BLOCKSCALE_GROUPED_CO_DIR
    )
    return all(os.path.isfile(os.path.join(root, f"{s}.co")) for _, s in k.images())


a8w8_blockscale_grouped_kids_declared = _load_a8w8_blockscale(
    A8W8_BLOCKSCALE_GROUPED_CO_JSON
)
a8w8_blockscale_m_grouped_kids = {
    kid: k
    for kid, k in a8w8_blockscale_grouped_kids_declared.items()
    if k.kind == "m_grouped" and _a8w8_blockscale_images_present(k)
}
a8w8_blockscale_k_grouped_kids = {
    kid: k
    for kid, k in a8w8_blockscale_grouped_kids_declared.items()
    if k.kind == "k_grouped" and _a8w8_blockscale_images_present(k)
}
