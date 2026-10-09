# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Iterable

from torch import Tensor

from ..jit.core import compile_ops

# Keep synchronized with moe_aux/codegen/gen_instances.py::SHAPES.
MXFP4_MOE_SUPPORTED_SHAPES = frozenset(
    {
        (385, 7168, 512, 9),
        (385, 7168, 1024, 9),
        (257, 7168, 512, 9),
        (257, 7168, 256, 9),
        (384, 7168, 512, 8),
        (385, 7168, 256, 9),
        (32, 7168, 2048, 8),
        (33, 7168, 2048, 8),
        (256, 3072, 1536, 8),
        (256, 3072, 768, 8),
        (512, 4096, 256, 10),
        (48, 7168, 3072, 6),
        (24, 7168, 3072, 6),
        (384, 7168, 1536, 6),
        (384, 7168, 768, 6),
        (384, 7168, 512, 6),
        (384, 5120, 768, 6),
        (128, 5120, 768, 3),
        (128, 5120, 1280, 3),
        (384, 5120, 1280, 6),
        (96, 5120, 2304, 6),
        (32, 5120, 2304, 2),
        (32, 5120, 2304, 3),
        (256, 4096, 256, 6),
        (256, 4096, 2048, 6),
        (385, 7168, 1536, 7),
        (385, 7168, 768, 7),
        (385, 7168, 512, 7),
        (257, 6144, 2048, 9),
        (257, 6144, 1024, 9),
        (257, 6144, 512, 9),
        (257, 6144, 256, 9),
        (896, 3584, 512, 16),
        (896, 3584, 3072, 16),
        (56, 3584, 3072, 16),
        (64, 7168, 2048, 8),
        (128, 3072, 512, 4),
        (128, 3072, 1536, 4),
        (128, 3072, 3072, 4),
        (129, 6144, 512, 5),
        (129, 6144, 768, 5),
        (256, 3072, 256, 8),
        (256, 3072, 512, 8),
        (256, 7168, 256, 8),
        (256, 7168, 512, 8),
        (384, 7168, 256, 8),
        (512, 4096, 512, 10),
        (513, 4096, 512, 11),
    }
)
_MXFP4_MOE_SCATTER_KEYS = frozenset(
    (hidden, topk) for _expert, hidden, _inter, topk in MXFP4_MOE_SUPPORTED_SHAPES
)


def is_mxfp4_moe_scatter_supported(model_dim: int, topk: int) -> bool:
    """Static generated scatter/scatterq support, independent of expert/inter_dim.

    The generator emits both NT variants for each catalog (model_dim, topk)
    key. This does not probe or build the extension during candidate enumeration.
    """
    return (int(model_dim), int(topk)) in _MXFP4_MOE_SCATTER_KEYS


def is_mxfp4_moe_shape_supported(
    expert: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> bool:
    """Return whether the generated MXFP4 auxiliary kernels cover this shape."""
    padded_inter = ((int(inter_dim) + 255) // 256) * 256
    return (int(expert), int(model_dim), padded_inter, int(topk)) in (
        MXFP4_MOE_SUPPORTED_SHAPES
    )


@compile_ops("module_moe_mxfp4_aux", develop=True)
def _mxfp4_moe_sort_internal_is_supported(
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
    zero_init: bool,
) -> bool:
    """Private dispatch probe; not exported through ``aiter.ops`` or ``aiter``."""


def prepare_mxfp4_moe_aux(shapes: "Iterable[tuple[int, int, int, int]]") -> None:
    """Load/build and verify the BM16 sort instances before tuning workers start.

    ``shapes`` contains (expert, model_dim, inter_dim, topk) tuples. Instances
    depend on the auxiliary key, so a new inter_dim can reuse compiled support
    without adding a model whitelist to the tuner. An existing stale module
    must be regenerated and rebuilt in a fresh process before workers launch.
    """
    keys = {(int(ne), int(h), int(topk)) for ne, h, _inter, topk in shapes}
    missing = []
    for expert, hidden, topk in sorted(keys):
        for zero_init in (False, True):
            if not _mxfp4_moe_sort_internal_is_supported(
                expert, topk, hidden, 16, zero_init
            ):
                operation = "sortzi" if zero_init else "sortonly"
                missing.append(f"aux_{operation}_NE{expert}_TOPK{topk}_MB16_H{hidden}")
    if missing:
        raise RuntimeError(
            "module_moe_mxfp4_aux is missing generated instances: "
            + ", ".join(missing)
            + ". Add support in moe_aux/codegen/gen_instances.py if needed, "
            "then rerun this preparation in a fresh process with AITER_REBUILD=1 "
            "before launching tuning workers."
        )


@compile_ops("module_moe_mxfp4_aux", develop=True)
def mxfp4_moe_sort_quant(
    a_input: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    cumsum_tensor: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    a_quant: Tensor,
    a_scale: Tensor,
    m_indices: Tensor,
    bf16_zero_out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux", develop=True)
def mxfp4_moe_sort(
    topk_ids: Tensor,
    topk_weight: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    cumsum_tensor: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    m_indices: Tensor,
    bf16_zero_out: Tensor,
    bf16_zero_workspace: Tensor,
    sort3stage_ws: Tensor,
    M_logical: int,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    D_INTER: int,
    MB: int,
    prologue: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux", develop=True)
def mxfp4_moe_quant(
    a_input: Tensor,
    a_quant: Tensor,
    a_scale: Tensor,
    bf16_zero_out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux", develop=True)
def mxfp4_moe_sort_scales(
    a_scale: Tensor,
    sorted_token_ids: Tensor,
    cumsum_tensor: Tensor,
    a_scale_sorted_shuffled: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
    max_sorted: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux", develop=True)
def mxfp4_moe_scatter_reduce(
    flat_out: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux", develop=True)
def mxfp4_moe_scatter_reduce_q(
    flat_out_q: Tensor,
    flat_out_scale: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...
