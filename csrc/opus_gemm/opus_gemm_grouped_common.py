# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kid table of the gfx950 bf16 grouped GEMMs (gen_co/co_kernels_gfx950_a16w16_grouped.json).

Read by gen_co/build_co.py --family a16w16_grouped (offline .co images), gen_instances.py (the host kid
table header) and aiter/ops/opus/gemm_op_a16w16_grouped.py (tile choice). The grouped kids are
launched by their own entries (opus_gemm_grouped.cu), outside KERNEL_REGISTRY.
"""

import json
import os
from typing import NamedTuple

A16W16_GROUPED_CO_JSON = os.path.join(
    os.path.dirname(__file__), "gen_co", "co_kernels_gfx950_a16w16_grouped.json"
)
A16W16_GROUPED_CO_DIR = "gfx950/a16w16_grouped"
# Images of a kid: M-grouped NT / NN (B [G, N, K] / [G, K, N]), one kernel with tiles 256 / 192 / 128
# rows high picked per group; K-grouped TN with bf16 / fp32 C.
A16W16_GROUPED_LAYOUTS = {
    "m_grouped": ("nt", "nn"),
    "k_grouped": ("tn", "tn_f32"),
}
_NAME_TAGS = {
    "m_grouped": "m_grouped",
    "k_grouped": "k_grouped",
}


class GroupedKid(NamedTuple):
    kid: int
    kind: str  # m_grouped / k_grouped
    B_M: int  # m_grouped: the largest tile height (256)
    B_N: int

    @property
    def name(self) -> str:
        """Image name stem (= ELF entry symbol without the layout suffix)."""
        return (
            f"opus_gemm_gfx950_bf16_{_NAME_TAGS[self.kind]}_256x{self.B_M}x{self.B_N}"
            "x64_2x2_16x16x32_0x0x0"
        )

    def images(self) -> list[tuple[str, str]]:
        """[(layout, symbol)] of the kid's images <name>_<layout>.co."""
        return [
            (lay, f"{self.name}_{lay}") for lay in A16W16_GROUPED_LAYOUTS[self.kind]
        ]


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as stream:
        entries = json.load(stream)["gfx950"]
    kids = {}
    for e in entries:
        assert e["kind"] in A16W16_GROUPED_LAYOUTS and e["kid"] not in kids, e
        kids[e["kid"]] = GroupedKid(e["kid"], e["kind"], *e["tile"])
    return kids


def _images_present(k: GroupedKid) -> bool:
    root = os.path.join(os.path.dirname(A16W16_GROUPED_CO_JSON), A16W16_GROUPED_CO_DIR)
    return all(os.path.isfile(os.path.join(root, f"{s}.co")) for _, s in k.images())


# Declared kids, and those whose images are on disk (the launchable ones).
a16w16_grouped_kids_declared = _load(A16W16_GROUPED_CO_JSON)
a16w16_m_grouped_kids = {
    kid: k
    for kid, k in a16w16_grouped_kids_declared.items()
    if k.kind == "m_grouped" and _images_present(k)
}
a16w16_k_grouped_kids = {
    kid: k
    for kid, k in a16w16_grouped_kids_declared.items()
    if k.kind == "k_grouped" and _images_present(k)
}
