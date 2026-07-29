# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE v2 Stage1 autotune candidates and pruning."""

import os

from flydsl.autotune import Config

_SHAPES = (
    (32, 128, 2),
    (32, 128, 4),
    (32, 256, 2),
    (32, 256, 4),
    (32, 256, 8),
    (32, 512, 4),
    (32, 512, 8),
    (64, 128, 2),
    (64, 128, 4),
    (64, 256, 4),
    (64, 256, 8),
    (64, 512, 8),
    (128, 256, 4),
    (128, 256, 8),
    (128, 512, 8),
)
_ANCHORS = {
    (32, 128, 4),
    (32, 256, 4),
    (64, 256, 4),
    (128, 256, 4),
    (128, 512, 8),
}
_GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16)
_DISPATCH_CU_VALUES = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224)
_GEOMETRY_SHAPES = {(32, 128, 4), (32, 256, 4), (32, 512, 8), (64, 256, 4), (64, 512, 8), (128, 512, 8)}
_GEOMETRY_GRIDS = (1, 2, 3, 4)
_GEOMETRY_DISPATCH = (64, 96, 128, 160, 192, 224)
_INTERACTION_SHAPES = {(32, 256, 4), (32, 512, 8), (64, 512, 8), (128, 512, 8)}
_INTERACTION_GRIDS = (1, 2, 3)
_INTERACTION_DISPATCH = (160, 192, 224)
_B_INTERACTIONS = tuple((True, False, b_nt) for b_nt in (0, 3))
_ASYNC_INTERACTIONS = tuple((resource, True, b_nt) for resource in (True, False) for b_nt in (-1, 0, 3))
_INTERACTIONS = _B_INTERACTIONS + _ASYNC_INTERACTIONS
_CALIBRATED_VARIANTS = {
    (32, 256, 4): (
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": False,
            "mfma_amajor": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 16,
            "pipe_weights": False,
            "mfma_amajor": False,
            "active_expert_producer": True,
            "use_tile_resource": False,
        },
    ),
    (32, 512, 8): (
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": False,
            "mfma_amajor": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": False,
            "mfma_amajor": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": False,
            "mfma_amajor": False,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": True,
            "mfma_amajor": True,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": True,
            "mfma_amajor": True,
            "async_a_copy": True,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": True,
            "mfma_amajor": True,
            "cooperative_payload_copy": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": True,
            "mfma_amajor": True,
            "async_a_copy": True,
            "cooperative_payload_copy": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": False,
            "mfma_amajor": False,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 16,
            "pipe_weights": False,
            "mfma_amajor": False,
            "active_expert_producer": True,
            "use_tile_resource": False,
        },
    ),
    (64, 512, 8): (
        {
            "grid_mult": 1,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 192,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 128,
            "use_tile_resource": False,
        },
    ),
    (128, 512, 8): (
        {
            "grid_mult": 3,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 3,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
    ),
}


def _candidate_variants(shape):
    variants = [{}, *_CALIBRATED_VARIANTS.get(shape, ())]
    if shape[0] == 32:
        variants += [
            {"mfma_amajor": True},
            {
                "grid_mult": 1,
                "num_dispatch_cu": 64,
                "mfma_amajor": True,
                "async_a_copy": True,
                "use_tile_resource": False,
            },
        ]
    if shape == (128, 512, 8):
        variants.append({"async_a_copy": True})
    if shape in _GEOMETRY_SHAPES:
        for grid_mult in _GEOMETRY_GRIDS:
            for num_dispatch_cu in _GEOMETRY_DISPATCH:
                geometry = {"grid_mult": grid_mult, "num_dispatch_cu": num_dispatch_cu}
                variants += [geometry, dict(geometry, use_tile_resource=False)]
                if num_dispatch_cu >= 128:
                    variants += [
                        dict(geometry, use_tile_resource=False, b_nt=0),
                        dict(geometry, use_tile_resource=False, b_nt=3),
                    ]
                    if shape[0] >= 64 and shape[1] == 512:
                        variants.append(dict(geometry, use_tile_resource=False, async_a_copy=True))
                if (
                    shape in _INTERACTION_SHAPES
                    and grid_mult in _INTERACTION_GRIDS
                    and num_dispatch_cu in _INTERACTION_DISPATCH
                ):
                    variants += [
                        dict(geometry, use_tile_resource=resource, async_a_copy=async_copy, b_nt=b_nt)
                        for resource, async_copy, b_nt in _INTERACTIONS
                    ]
    if shape not in _ANCHORS:
        return variants
    variants += [{"grid_mult": value} for value in _GRID_MULT_VALUES if value != 4]
    variants += [{"num_dispatch_cu": value} for value in _DISPATCH_CU_VALUES if value != 64]
    variants += [
        {"pipe_weights": False, "mfma_amajor": False},
        {"mfma_amajor": False},
        {"swizzle_a": False},
        {"use_tile_resource": False},
        {"waves_per_eu_hint": 1},
        {"b_nt": 0},  # cached B-load (L2 reuse); best at large bs
        {"b_nt": 3},  # streamed B-load (bypass); best at small/decode bs
    ]
    return variants


def get_stage1_autotune_configs(dispatch_cu=None, grid_mult=None, tile_m_values=(32,)):
    tile_m_values = {int(value) for value in tile_m_values}
    if os.environ.get("MEGA_S1_NOTUNE") == "1":
        # Pin one valid Stage1 config when isolating Stage2 performance.
        return [
            Config(
                sort_block_m=max(tile_m_values),
                tile_n=256,
                tile_k=256,
                num_waves=4,
                grid_mult=4 if grid_mult is None else int(grid_mult),
                pipe_weights=True,
                mfma_amajor=True,
                swizzle_a=True,
                async_a_copy=False,
                active_expert_producer=False,
                cooperative_payload_copy=False,
                num_dispatch_cu=64 if dispatch_cu is None else int(dispatch_cu),
                use_tile_resource=True,
                waves_per_eu_hint=2,
                b_nt=-1,
            )
        ]
    configs = []
    seen = set()
    for sort_block_m, tile_n, num_waves in _SHAPES:
        if sort_block_m not in tile_m_values:
            continue
        base = dict(
            sort_block_m=sort_block_m,
            tile_n=tile_n,
            tile_k=256,
            num_waves=num_waves,
            grid_mult=4,
            pipe_weights=True,
            mfma_amajor=sort_block_m >= 64,
            swizzle_a=True,
            async_a_copy=False,
            active_expert_producer=False,
            cooperative_payload_copy=False,
            num_dispatch_cu=64,
            use_tile_resource=True,
            waves_per_eu_hint=2,
            b_nt=-1,  # -1 = per-bucket default policy (stream<=512, cached>=1024)
        )
        for update in _candidate_variants((sort_block_m, tile_n, num_waves)):
            values = dict(base, **update)
            if dispatch_cu is not None:
                values["num_dispatch_cu"] = int(dispatch_cu)
            if grid_mult is not None:
                values["grid_mult"] = int(grid_mult)
            signature = tuple(sorted(values.items()))
            if signature not in seen:
                configs.append(Config(**values))
                seen.add(signature)
    return configs


def prune_stage1_autotune_configs(configs, sig_args):
    """Prune invalid and batch-irrelevant configs before collective compilation."""
    tokens = int(sig_args["tune_tokens"])
    model_dim = int(sig_args["model_dim"])
    inter_dim = int(sig_args["inter_dim"])
    num_cu = int(sig_args["num_cu"])
    fuse_npes = int(sig_args["fuse_npes"])
    fuse_topk = int(sig_args["fuse_topk"])
    fuse_cap = int(sig_args["fuse_cap"])
    fuse_mtpr = int(sig_args["fuse_mtpr"])
    experts_per_rank = int(sig_args["experts_per_rank"])
    fixed_slot_dispatch = bool(sig_args["fixed_slot_dispatch"])
    out = []
    for config in configs:
        values = config.kwargs
        block_m = int(values["sort_block_m"])
        tile_n = int(values["tile_n"])
        tile_k = int(values["tile_k"])
        num_waves = int(values["num_waves"])
        grid_mult = int(values["grid_mult"])
        dispatch_cu = int(values["num_dispatch_cu"])
        b_nt = int(values["b_nt"])
        use_tile_resource = bool(values["use_tile_resource"])
        direct_fixed_slot = (
            fixed_slot_dispatch
            and fuse_npes == 8
            and experts_per_rank == 48
            and fuse_cap == ((fuse_npes * fuse_mtpr + block_m - 1) // block_m) * block_m
        )
        if b_nt == (3 if fuse_mtpr <= 512 else 0):
            continue
        if direct_fixed_slot and (values["active_expert_producer"] or values["cooperative_payload_copy"]):
            continue
        num_acc_n = tile_n // num_waves // 16
        m_repeat = block_m // 16
        lds_pool = max(2 * block_m * tile_k, 2 * block_m * tile_n)
        lds_scale = block_m * (model_dim // 32)
        max_rows = fuse_npes * fuse_mtpr * fuse_topk + experts_per_rank * block_m
        payload_bytes = max_rows * model_dim
        output_bytes = max_rows * inter_dim
        if (
            model_dim % tile_k
            or (2 * inter_dim) % tile_n
            or tile_n % (num_waves * 32)
            or block_m % 32
            or m_repeat * num_acc_n * 4 > 256
            or lds_pool + lds_scale > 160 * 1024
            or not 0 < dispatch_cu < num_cu
            or num_cu * grid_mult - 1 - dispatch_cu <= 0
            or (payload_bytes >= 1 << 32 and not use_tile_resource)
            or (output_bytes >= 1 << 32 and not use_tile_resource)
        ):
            continue
        if os.environ.get("MEGA_S1_NOTUNE") == "1":
            keep = True
        elif tokens <= 64:
            # Keep explicit M64/M128 joint-SBM sweeps valid for small batches.
            keep = tile_n <= 512 and grid_mult <= 4 and dispatch_cu >= 32 and not (block_m > 32 and b_nt == 0)
        elif tokens <= 1024:
            keep = tile_n <= 512 and grid_mult <= 8 and dispatch_cu >= 24
        else:
            # Limit large-batch M32 to the gfx950-safe N512/w8 family.
            keep = (
                tile_n == 512
                and num_waves == 8
                and ((block_m == 32 and grid_mult == 1 and dispatch_cu >= 32) or (block_m >= 64 and grid_mult <= 3))
            )
        if keep:
            out.append(config)
    if not out:
        raise ValueError(f"no valid stage1 configs for tokens={tokens}")
    return out
