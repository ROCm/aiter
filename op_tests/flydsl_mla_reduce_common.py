"""Shared helpers for FlyDSL MLA reduce test + benchmark harnesses."""

import torch
import aiter

from aiter.ops.flydsl.kernels.mla_reduce import (
    LDS_MAX_SPLITS,
    Tier,
    compile_mla_reduce,
    compile_mla_reduce_splitk,
    plan_splitk,
    should_use_persistent_launch,
    splitk_enabled,
    waves_per_eu_from_env,
    _get_splitk_scratch,
)

SERVING_NUM_REDUCE_TILE = 16384
SERVING_PARTIAL_POOL = 606
# gfx950 not yet a target for this kernel; keep tests scoped to gfx942 until it is.
MLA_REDUCE_SUPPORTED_GFX = ["gfx942"]


def mla_reduce_out_dtype(dt: str) -> torch.dtype:
    return torch.bfloat16 if dt == "bf16" else torch.float16


def mla_reduce_out_atol(dt: str | torch.dtype) -> float:
    return 6.3e-2 if dt in ("bf16", torch.bfloat16) else 2e-3


def build_irregular_inputs(
    splits_per_tile,
    H,
    Dv,
    out_dtype,
    M=1,
    gap_stride=1,
    pool_slack=0,
    device="cuda",
    seed=0,
):
    """Build reduce inputs mirroring real decode metadata (variable ``n_splits``
    per tile, non-dense ``reduce_partial_map``, over-sized pool). ``build_inputs``
    is the dense special case (``[S]*num_tiles``, ``gap_stride=1``, ``pool_slack=0``).

    Args:
        splits_per_tile: per-tile ``n_splits``; ``0`` (or ``1``) marks an empty tile
            whose ``reduce_final_map`` q-range is garbage (exercises the empty-tile
            guard, which must never deref it).
        gap_stride: spacing between partial-pool base rows (``1`` dense, ``>1`` holes).
        pool_slack: extra unused rows appended to the pool (over-allocated buffer).

    For ``M > 1`` each split owns ``M`` contiguous partial rows and each tile's final
    q-range spans ``[tile*M, tile*M + M)`` (``get_mla_metadata_v1`` layout).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_tiles = len(splits_per_tile)
    total_splits = int(sum(int(s) for s in splits_per_tile))

    indptr_host = [0]
    for s in splits_per_tile:
        indptr_host.append(indptr_host[-1] + int(s))
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    if total_splits > 0:
        slot = torch.arange(total_splits, dtype=torch.int32, device=device)
        reduce_partial_map = slot * (gap_stride * M)
        max_base = int(reduce_partial_map.max().item())
    else:
        reduce_partial_map = torch.zeros(1, dtype=torch.int32, device=device)
        max_base = 0
    num_partial_rows = max_base + M + pool_slack * M

    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    q_start = torch.arange(num_tiles, dtype=torch.int32, device=device) * M
    reduce_final_map = torch.stack([q_start, q_start + M], dim=1).contiguous()
    for t, s in enumerate(splits_per_tile):
        if int(s) <= 1:
            reduce_final_map[t, 0] = 1 << 24
            reduce_final_map[t, 1] = (1 << 24) + M

    final_output = torch.empty(num_tiles * M, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.empty(num_tiles * M, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, M=1, device="cuda", seed=0):
    """Dense/uniform reduce inputs: every tile has ``num_splits`` splits and the
    gather map is contiguous."""
    return build_irregular_inputs(
        [num_splits] * num_tiles,
        H,
        Dv,
        out_dtype,
        M=M,
        gap_stride=1,
        device=device,
        seed=seed,
    )


def build_degenerate_inputs(num_tiles, H, Dv, out_dtype, device="cuda", seed=0):
    """All-empty (``n_splits=0``) metadata for the empty-tile guard regression."""
    return build_irregular_inputs(
        [0] * num_tiles, H, Dv, out_dtype, gap_stride=1, device=device, seed=seed
    )


def build_serving_decode_inputs(
    active_tiles,
    splits,
    out_dtype,
    H=16,
    Dv=512,
    num_reduce_tile=SERVING_NUM_REDUCE_TILE,
    partial_pool=SERVING_PARTIAL_POOL,
    device="cuda",
    seed=0,
):
    """Sparse serving decode grid with active-sized outputs."""
    active_splits = active_tiles * splits
    pool_slack = max(0, partial_pool - active_splits)
    splits_per_tile = [splits] * active_tiles + [0] * (num_reduce_tile - active_tiles)
    po, pl, indptr, fmap, pmap, _, _ = build_irregular_inputs(
        splits_per_tile,
        H,
        Dv,
        out_dtype,
        M=1,
        gap_stride=1,
        pool_slack=pool_slack,
        device=device,
        seed=seed,
    )
    return (
        po,
        pl,
        indptr,
        fmap,
        pmap,
        torch.empty(active_tiles, H, Dv, dtype=out_dtype, device=device),
        torch.empty(active_tiles, H, dtype=torch.float32, device=device),
    )


def torch_ref(
    partial_output, partial_lse, num_tiles, num_splits, H, Dv, out_dtype, M=1
):
    """Vectorized online-softmax reduce reference (any max_seqlen_q M)."""
    po = partial_output.view(num_tiles, num_splits, M, H, Dv).double()
    pl = partial_lse.view(num_tiles, num_splits, M, H).double()
    max_lse = pl.max(dim=1, keepdim=True).values
    w = torch.exp(pl - max_lse)
    denom = w.sum(dim=1)
    num = (w.unsqueeze(-1) * po).sum(dim=1)
    out = (num / denom.unsqueeze(-1)).to(out_dtype)
    lse = (max_lse.squeeze(1) + torch.log(denom)).float()
    return out.reshape(num_tiles * M, H, Dv), lse.reshape(num_tiles * M, H)


def torch_ref_gather(
    po,
    pl,
    indptr,
    fmap,
    pmap,
    H,
    Dv,
    out_dtype,
    M=1,
    num_partial_rows=None,
    num_final_rows=None,
):
    """Gather-based online-softmax reference for irregular metadata.

    Follows the kernel's CSR + gather contract: per tile it gathers partial rows via
    ``pmap[indptr[t]:indptr[t+1]]`` and merges them; tiles with ``n_splits <= 1`` are
    skipped (output rows stay zero). When ``num_partial_rows`` / ``num_final_rows``
    are set, out-of-range pmap rows / q_start are skipped (mirrors guards-ON).
    """
    num_tiles = fmap.shape[0]
    ref_out = torch.zeros(num_tiles * M, H, Dv, dtype=out_dtype, device=po.device)
    ref_lse = torch.zeros(num_tiles * M, H, dtype=torch.float32, device=po.device)
    indptr_h = indptr.tolist()
    pmap_h = pmap.tolist()
    fmap_h = fmap.tolist()
    pod = po.double()
    pld = pl.double()
    for t in range(num_tiles):
        s0, s1 = indptr_h[t], indptr_h[t + 1]
        if s1 - s0 <= 1:
            continue
        q_start = fmap_h[t][0]
        if num_final_rows is not None and (q_start < 0 or q_start >= num_final_rows):
            continue
        bases = pmap_h[s0:s1]
        for local in range(M):
            rows = []
            for b in bases:
                row = b + local
                if num_partial_rows is not None and (
                    row < 0 or row >= num_partial_rows
                ):
                    continue
                rows.append(row)
            if not rows:
                continue
            o = pod[rows]
            lg = pld[rows]
            max_lse = lg.max(dim=0, keepdim=True).values
            w = torch.exp(lg - max_lse)
            denom = w.sum(dim=0)
            num = (w.unsqueeze(-1) * o).sum(dim=0)
            ref_out[q_start + local] = (num / denom.unsqueeze(-1)).to(out_dtype)
            ref_lse[q_start + local] = (max_lse.squeeze(0) + torch.log(denom)).float()
    return ref_out, ref_lse


def hip_ref(po, pl, indptr, fmap, pmap, num_tiles, H, Dv, out_dtype, M=1):
    """Reference output from HIP kn_mla_reduce_v1. Outputs are zero-initialized so
    skipped (empty) tiles match a zero-initialized final buffer under test.
    """
    ref_out = torch.zeros(num_tiles * M, H, Dv, dtype=out_dtype, device=po.device)
    ref_lse = torch.zeros(num_tiles * M, H, dtype=torch.float32, device=po.device)
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, LDS_MAX_SPLITS, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse, M=1):
    """HIP reference sized to ``fout`` / ``flse`` (serving grids with sparse tiles)."""
    ref_out = torch.zeros_like(fout)
    ref_lse = torch.zeros_like(flse)
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, LDS_MAX_SPLITS, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def build_serving_sparse_grid_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """batch=8 steady-state: 16384-tile grid, 8 active tiles × 32 splits.

    Mirrors 131K-context serving metadata: partial pool 606 rows, CSR sentinel
    at 256, garbage ``reduce_final_map`` / ``reduce_partial_map`` tail slots.

    Tail tiles are flat at sentinel (``n_splits == 0``), already skipped by the
    ``n_splits > 1`` clamp, so this fixture does **not** discriminate the
    gather/store guards.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_reduce_tile = SERVING_NUM_REDUCE_TILE
    active_tiles = 8
    splits_per_active = 32
    total_splits = active_tiles * splits_per_active
    sentinel = total_splits
    num_partial_rows = SERVING_PARTIAL_POOL

    indptr_host = list(range(0, sentinel + 1, splits_per_active))
    while len(indptr_host) <= num_reduce_tile:
        indptr_host.append(sentinel)
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    reduce_partial_map = torch.empty(num_partial_rows, dtype=torch.int32, device=device)
    reduce_partial_map[:total_splits] = torch.arange(
        total_splits, dtype=torch.int32, device=device
    )
    garbage_pmap = torch.randint(
        -(1 << 30),
        1 << 30,
        (num_partial_rows - total_splits,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    reduce_partial_map[total_splits:] = garbage_pmap

    reduce_final_map = torch.empty(num_reduce_tile, 2, dtype=torch.int32, device=device)
    q = torch.arange(active_tiles, dtype=torch.int32, device=device)
    reduce_final_map[:active_tiles, 0] = q
    reduce_final_map[:active_tiles, 1] = q + 1
    reduce_final_map[active_tiles:, 0] = 1 << 24
    reduce_final_map[active_tiles:, 1] = (1 << 24) + 1

    final_output = torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(active_tiles, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_serving_mapped_slack_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
    num_tiles=64,
    splits_per_active=32,
    slack_p=4,
    slack_f=2,
):
    """Small serving grid with allocation slack for in-process guard differentials.

    Allocates ``partial_output`` / ``final_output`` with extra rows but returns
    smaller *logical* ``num_partial_rows`` / ``num_final_rows`` for the kernel.
    The fake-active tail (non-sentinel ``n_splits``) exercises the gather/store
    guards adversarially.

    Returns tensors plus a ``meta`` dict with logical bounds and discriminator
    tile indices for tests.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    active_tiles = 3
    gather_tile = 0
    store_tile = 10
    store_splits = 4
    total_active_splits = active_tiles * splits_per_active
    total_splits = total_active_splits + store_splits

    indptr_host = [0]
    for _ in range(active_tiles):
        indptr_host.append(indptr_host[-1] + splits_per_active)
    base_flat = indptr_host[-1]
    while len(indptr_host) <= store_tile:
        indptr_host.append(base_flat)
    indptr_host.append(base_flat + store_splits)
    while len(indptr_host) <= num_tiles:
        indptr_host.append(indptr_host[-1])
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    logical_partial_rows = 256
    alloc_partial_rows = logical_partial_rows + slack_p
    logical_final_rows = active_tiles
    alloc_final_rows = logical_final_rows + slack_f

    partial_output = torch.randn(
        alloc_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            alloc_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    reduce_partial_map = torch.arange(total_splits, dtype=torch.int32, device=device)
    # Gather discriminator: one split on tile 0 points into mapped slack.
    slack_p_row = logical_partial_rows
    bad_split = splits_per_active // 2
    reduce_partial_map[bad_split] = slack_p_row
    partial_output[slack_p_row].fill_(1000.0)
    partial_lse[slack_p_row].fill_(50.0)
    tile0_lse_max = partial_lse[:splits_per_active].max().item()
    partial_lse[slack_p_row].fill_(tile0_lse_max + 5.0)

    reduce_final_map = torch.empty(num_tiles, 2, dtype=torch.int32, device=device)
    q = torch.arange(active_tiles, dtype=torch.int32, device=device)
    reduce_final_map[:active_tiles, 0] = q
    reduce_final_map[:active_tiles, 1] = q + 1
    reduce_final_map[active_tiles:store_tile, 0] = 1 << 24
    reduce_final_map[active_tiles:store_tile, 1] = (1 << 24) + 1

    store_slack_q = logical_final_rows
    reduce_final_map[store_tile, 0] = store_slack_q
    reduce_final_map[store_tile, 1] = store_slack_q + 1
    reduce_final_map[store_tile + 1 :, 0] = 1 << 24
    reduce_final_map[store_tile + 1 :, 1] = (1 << 24) + 1

    last = int(reduce_indptr[num_tiles].item())
    t0_store = int(reduce_indptr[store_tile].item())
    n_splits_store = int(
        reduce_indptr[store_tile + 1].item() - reduce_indptr[store_tile].item()
    )
    assert n_splits_store >= 2, "store discriminator must be fake-active"
    assert t0_store != last, "store discriminator must not hit sentinel skip"

    final_output = torch.zeros(alloc_final_rows, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(alloc_final_rows, H, dtype=torch.float32, device=device)
    fout_slack_seed = 42.0
    final_output[store_slack_q:].fill_(fout_slack_seed)

    meta = {
        "logical_partial_rows": logical_partial_rows,
        "logical_final_rows": logical_final_rows,
        "gather_tile": gather_tile,
        "gather_q_row": gather_tile,
        "store_tile": store_tile,
        "store_slack_q": store_slack_q,
        "fout_slack_seed": fout_slack_seed,
    }
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
        meta,
    )


def build_serving_true_oob_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
    num_tiles=64,
):
    """Fake-active tail with genuine OOB indices (guards-ON no-fault regression).

    Cannot be run with ``disable_guards=True`` in-process (would GPU-fault).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    active_tiles = 2
    splits_per_active = 32
    tail_splits = 4
    total_active = active_tiles * splits_per_active
    sentinel = total_active + tail_splits

    indptr_host = list(range(0, total_active + 1, splits_per_active))
    while len(indptr_host) < num_tiles:
        indptr_host.append(total_active)
    tail_tile = num_tiles - 2
    indptr_host[tail_tile + 1] = sentinel
    while len(indptr_host) <= num_tiles:
        indptr_host.append(sentinel)
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    num_partial_rows = 128
    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    reduce_partial_map = torch.arange(sentinel, dtype=torch.int32, device=device)
    tail_t0 = int(reduce_indptr[tail_tile].item())
    for i in range(tail_splits):
        reduce_partial_map[tail_t0 + i] = num_partial_rows + i

    reduce_final_map = torch.empty(num_tiles, 2, dtype=torch.int32, device=device)
    reduce_final_map[:active_tiles, 0] = torch.arange(
        active_tiles, dtype=torch.int32, device=device
    )
    reduce_final_map[:active_tiles, 1] = torch.arange(
        1, active_tiles + 1, dtype=torch.int32, device=device
    )
    reduce_final_map[active_tiles:tail_tile, 0] = 1 << 24
    reduce_final_map[active_tiles:tail_tile, 1] = (1 << 24) + 1
    reduce_final_map[tail_tile, 0] = 1 << 24
    reduce_final_map[tail_tile, 1] = (1 << 24) + 1
    reduce_final_map[tail_tile + 1 :, 0] = 1 << 24
    reduce_final_map[tail_tile + 1 :, 1] = (1 << 24) + 1

    final_output = torch.zeros(active_tiles, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(active_tiles, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_serving_stale_indptr_inputs(
    H=16,
    Dv=512,
    out_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """Batch-transition metadata: batch=8 sparse grid patched to batch=1 layout.

    Tile 0 becomes ``n_splits=128`` for batch=1; tiles 1..7 keep stale batch=8 CSR
    (``n_splits=32``, stale pmap rows beyond the logical bound, stale fmap q 1..7).
    Uses the allocation-slack trick so logical bounds are smaller than the buffers.

    Returns tensors plus ``meta`` with logical bounds for guard differentials.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_reduce_tile = SERVING_NUM_REDUCE_TILE
    active_tiles_batch8 = 8
    splits_per_active = 32
    batch1_splits = 128
    sentinel = batch1_splits + (active_tiles_batch8 - 1) * splits_per_active

    indptr_host = [0, batch1_splits]
    for t in range(1, active_tiles_batch8):
        indptr_host.append(indptr_host[-1] + splits_per_active)
    while len(indptr_host) <= num_reduce_tile:
        indptr_host.append(sentinel)
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    logical_partial_rows = batch1_splits
    slack_p = 256
    alloc_partial_rows = logical_partial_rows + slack_p
    logical_final_rows = 1
    slack_f = 8
    alloc_final_rows = logical_final_rows + slack_f

    partial_output = torch.randn(
        alloc_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            alloc_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    total_pmap = sentinel
    reduce_partial_map = torch.arange(total_pmap, dtype=torch.int32, device=device)
    # Stale tiles 1..7: pmap rows from batch=8 now >= logical_partial_rows.
    stale_base = batch1_splits
    for t in range(1, active_tiles_batch8):
        t0 = indptr_host[t]
        for s in range(splits_per_active):
            reduce_partial_map[t0 + s] = stale_base + s
    partial_output[stale_base : stale_base + splits_per_active].fill_(500.0)
    tile1_lse_max = partial_lse[stale_base : stale_base + splits_per_active].max()
    partial_lse[stale_base].fill_(tile1_lse_max + 5.0)

    reduce_final_map = torch.empty(num_reduce_tile, 2, dtype=torch.int32, device=device)
    reduce_final_map[0, 0] = 0
    reduce_final_map[0, 1] = 1
    for t in range(1, active_tiles_batch8):
        reduce_final_map[t, 0] = t
        reduce_final_map[t, 1] = t + 1
    reduce_final_map[active_tiles_batch8:, 0] = 1 << 24
    reduce_final_map[active_tiles_batch8:, 1] = (1 << 24) + 1

    final_output = torch.zeros(alloc_final_rows, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(alloc_final_rows, H, dtype=torch.float32, device=device)
    fout_slack_seed = 42.0
    final_output[logical_final_rows:].fill_(fout_slack_seed)

    meta = {
        "logical_partial_rows": logical_partial_rows,
        "logical_final_rows": logical_final_rows,
        "gather_tile": 1,
        "gather_q_row": 1,
        "store_tile": 2,
        "store_slack_q": 2,
        "fout_slack_seed": fout_slack_seed,
    }
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
        meta,
    )


def make_runner(
    po,
    pl,
    indptr,
    pmap,
    fmap,
    fout,
    flse,
    H,
    Dv,
    out_dtype_str,
    output_lse,
    M=1,
    *,
    tier=None,
    disable_guards=False,
    num_partial_rows=None,
    num_final_rows=None,
):
    """Precompile + bind args; return a zero-overhead closure for the timed loop.

    tier: compile-time tier override for isolated tests. None (default) uses
    Tier.ALL (production path with device-side runtime tier selection).
    """
    num_tiles = fmap.shape[0]
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    if tier is None:
        compile_tier = Tier.ALL
    else:
        compile_tier = tier
    if num_partial_rows is None:
        num_partial_rows = int(po.size(0))
    if num_final_rows is None:
        num_final_rows = int(fout.size(0))
    # Split-K (opt-in): cooperative multi-block reduction for the low-tile /
    # high-split decode case. Metadata is inspected at setup (outside any
    # CUDA-graph capture); the scratch is pre-allocated + reused, so the
    # captured run() only launches the two kernels (capture-safe).
    if splitk_enabled() and tier is None and not disable_guards:
        diffs = indptr[1:] - indptr[:-1]
        active_tiles = int((diffs > 1).sum().item())
        max_splits_val = int(diffs.max().item()) if diffs.numel() else 0
        engage, K, num_slots = plan_splitk(
            active_tiles=active_tiles,
            H=H,
            max_seqlen_q=M,
            max_splits=max_splits_val,
            num_cu=num_cu,
        )
        if engage:
            lp, lc = compile_mla_reduce_splitk(
                H=H,
                Dv=Dv,
                out_dtype=out_dtype_str,
                K=K,
                output_lse=output_lse,
                waves_per_eu=waves_per_eu_from_env(),
            )
            sk_acc, sk_ml = _get_splitk_scratch(num_slots, K, Dv, fout.device.index)
            _npr = int(num_partial_rows)
            _nfr = int(num_final_rows)
            _ss = int(fout.stride(0))
            _sh = int(fout.stride(1))
            _grid_p = int(num_slots * K)
            _grid_c = int(num_slots)

            def run():
                st = torch.cuda.current_stream()
                lp(po, pl, indptr, pmap, sk_acc, sk_ml, _npr, _grid_p, st)
                lc(fmap, sk_acc, sk_ml, fout, flse, _ss, _sh, _nfr, _grid_c, st)

            return run

    use_persistent = should_use_persistent_launch(
        H=H,
        max_seqlen_q=M,
        num_reduce_tile=num_tiles,
        num_cu=num_cu,
    )
    kernel = compile_mla_reduce(
        H=H,
        Dv=Dv,
        out_dtype=out_dtype_str,
        tier=compile_tier,
        persistent=use_persistent,
        output_lse=output_lse,
        use_reduce_final_map=True,
        disable_guards=disable_guards,
        waves_per_eu=waves_per_eu_from_env(),
    )
    head = (
        po,
        pl,
        indptr,
        pmap,
        fmap,
        fout,
        flse,
        int(fout.stride(0)),
        int(fout.stride(1)),
        int(num_cu),
        int(num_tiles),
        int(M),
        int(num_partial_rows),
        int(num_final_rows),
    )

    def run():
        kernel(*head, torch.cuda.current_stream())

    return run


def bench_cudagraph(fn, num_warmup=25, num_iters=100):
    """CUDA-graph replay timing; returns ms/iter."""
    for _ in range(max(1, num_warmup)):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        fn()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(num_iters):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / num_iters


def run_cudagraph_replay(fn, num_warmup=3, num_replays=3):
    """Capture ``fn`` into a CUDA graph and replay it, to surface replay-only
    faults (the failure mode reported under real serving). ``fn`` writes into its
    bound output tensors; the caller inspects those after this returns."""
    for _ in range(max(1, num_warmup)):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        fn()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    for _ in range(max(1, num_replays)):
        graph.replay()
    torch.cuda.synchronize()
