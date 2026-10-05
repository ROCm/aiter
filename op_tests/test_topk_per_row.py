import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import topk
from aiter.ops.flydsl.kernels.tensor_shim import wave_size_of
from aiter.ops.flydsl.kernels.topk import (
    topk_per_row_decode_adaptive as adaptive_kernel,
)
from aiter.ops.flydsl.topk import topk_per_row as flydsl_decode_host
from aiter.ops.flydsl.topk.topk_per_row import _FLYDSL_TOPK_ONE_BLOCK_ARCHES
from aiter.ops.topk import _FLYDSL_TOPK_DECODE_GATES
from aiter.ops.topk_select import _choose as topk_select_backend_for
from aiter.ops.topk_select import topk_select
from aiter.test_common import benchmark, perftest


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    data_generation: str = "random",
    physical_width: int | None = None,
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    width = physical_width if physical_width is not None else max(row_ends)
    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        logits = torch.randn(row_starts.shape[0], width, dtype=dtype, device="cuda")
    elif data_generation == "10LSBits" or data_generation == "mixed":
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        # Generate random bits for the last 10 bits
        random_bottom_bits = torch.randint(
            0,
            2**10,
            (row_starts.shape[0], width),
            dtype=torch.int32,
            device="cuda",
        )
        # Combine: fixed top 22 bits with random last 10 bits
        logits_bits = (fixed_top_22_bits & top_22_bits_mask) | (
            random_bottom_bits & last_10_bits_mask
        )
        logits = logits_bits.view(dtype)

    if data_generation == "mixed":
        logits_random = torch.randn(
            row_starts.shape[0], width, dtype=dtype, device="cuda"
        )
        # Mix the two logits tensors randomly
        mask = torch.randint(0, 2, (row_starts.shape[0], 1), device="cuda").bool()
        logits = torch.where(mask, logits, logits_random)

    for i, end in enumerate(row_ends):
        logits[i, end:] = float("-inf")
    return logits


def create_row_boundaries(
    num_rows: int, num_prefix: int = 0, top_k: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row start and end indices for testing."""
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(
        num_prefix + 1, num_prefix + num_rows + 1, device="cuda", dtype=torch.int32
    )
    return row_starts, row_ends


def compare_topk_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
    stable: bool = False,
    values: torch.Tensor | None = None,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Both results should be sorted and contain the same top-k elements.
    """
    num_rows = cuda_indices.shape[0]

    for row_idx in range(num_rows):
        # Get valid elements using row boundaries
        row_start = row_starts[row_idx].item()
        row_end = row_ends[row_idx].item()
        row_length = row_end - row_start
        num_valid = min(top_k, row_length)
        cuda_row_indices = cuda_indices[row_idx][:num_valid].cpu()
        torch_row_indices = torch_indices[row_idx][:num_valid].cpu()

        # Compare the sets of indices first
        cuda_set = set(cuda_row_indices.tolist())
        torch_set = set(torch_row_indices.tolist())
        if cuda_set == torch_set:
            continue

        # Any difference in elements, compare the values
        logits_row = logits[row_idx]
        cuda_row_values = [logits_row[i] for i in cuda_row_indices]
        torch_row_values = [logits_row[i] for i in torch_row_indices]

        cuda_only_values, torch_only_values = [], []
        for idx in cuda_set - torch_set:
            cuda_pos = (cuda_row_indices == idx).nonzero(as_tuple=True)[0]
            cuda_only_values.append(cuda_row_values[cuda_pos[0]])

        for idx in torch_set - cuda_set:
            torch_pos = (torch_row_indices == idx).nonzero(as_tuple=True)[0]
            torch_only_values.append(torch_row_values[torch_pos[0]])

        if len(cuda_only_values) != len(torch_only_values):
            return False
        if not torch.allclose(
            torch.tensor(cuda_only_values),
            torch.tensor(torch_only_values),
            rtol=tolerance,
            atol=tolerance,
        ):
            return False

    if stable:
        for row, row_end in enumerate(row_ends.tolist()):
            valid_count = min(top_k, row_end)
            if valid_count > 1 and not bool(
                torch.all(
                    cuda_indices[row, 1:valid_count]
                    >= cuda_indices[row, : valid_count - 1]
                )
            ):
                return False

    if values is not None:
        valid = cuda_indices >= 0
        gathered = torch.gather(
            logits,
            1,
            cuda_indices.clamp_min(0).to(torch.int64),
        )
        if not torch.equal(values[valid], gathered[valid]):
            return False
        if not bool(torch.all(torch.isneginf(values[~valid]))):
            return False

    return True


@perftest()
def run_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    num_rows: int,
    stride_row: int,
    stride_col: int,
    k: int = 2048,
    flydsl: bool = False,
    stable: bool = False,
) -> None:
    """
    Run the top_k_per_row kernel. `flydsl=True` bypasses dispatch and calls the
    one-block radix kernel directly.
    """
    if flydsl:
        return aiter.flydsl_radix_topk_one_block_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            values,
            num_rows,
            stride_row,
            stride_col,
            k,
            stable,
        )
    return aiter.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride_row,
        stride_col,
        k=k,
        stable=stable,
    )


@perftest()
def run_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    fast: bool,
    k: int = 2048,
    flydsl: bool = False,
    stable: bool = False,
    values: torch.Tensor | None = None,
    max_row_len: int | None = None,
) -> None:
    """
    Run the top_k_per_row kernel.

    Note: the `_fast` ASM-kernel variant has `kTopK=2048` baked into its
    precompiled `.co`; it ignores any caller-supplied `k`. The dispatch
    here only allows `_fast` when k == 2048.
    """
    if flydsl:
        assert not fast, "fast and flydsl cannot both be enabled"
        return aiter.flydsl_top_k_per_row_decode(
            logits,
            next_n,
            seqLens,
            indices,
            numRows,
            stride0,
            stride1,
            k,
            stable,
            values,
            max_row_len=max_row_len,
        )
    elif fast:
        assert k == 2048, "top_k_per_row_decode_fast only supports k=2048"
        assert not stable and values is None
        return aiter.top_k_per_row_decode_fast(
            logits,
            next_n,
            seqLens,
            indices,
            numRows,
            stride0,
            stride1,
        )
    else:
        return aiter.top_k_per_row_decode(
            logits,
            next_n,
            seqLens,
            indices,
            numRows,
            stride0,
            stride1,
            k=k,
            stable=stable,
            values=values,
            max_row_len=max_row_len,
        )


@benchmark()
def test_top_k_per_row_prefill(
    num_rows: int,
    num_prefix: int,
    top_k: int,
    data_generation: str = "random",
    flydsl: bool = False,
    stable: bool = False,
    write_values: bool = False,
) -> dict:
    """
    Test topk_per_row_prefill.
    """
    ret = {}
    torch.set_default_device("cuda:0")

    # Create test data
    row_starts, row_ends = create_row_boundaries(num_rows, num_prefix)
    logits = create_random_logits(
        row_starts,
        row_ends,
        torch.float32,
        42,
        data_generation,
        physical_width=max(int(max(row_ends)), top_k) if flydsl else None,
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    values = (
        torch.empty((num_rows, top_k), dtype=torch.float32, device="cuda")
        if write_values
        else None
    )

    # Run the kernel
    _, us = run_top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        k=top_k,
        flydsl=flydsl,
        stable=stable,
    )

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    all_close = compare_topk_results(
        logits,
        indices,
        torch_indices,
        row_starts,
        row_ends,
        top_k,
        stable=stable,
        values=values,
    )

    # measure performance
    ret["context_len"] = logits.shape[1]
    ret["all_close"] = all_close
    ret["us"] = us
    return ret


@benchmark()
def test_top_k_per_row_decode(
    batch_size: int,
    context_len: int,
    top_k: int,
    next_n: int,
    data_generation: str = "random",
    fast: bool = False,
    flydsl: bool = False,
    stable: bool = False,
    write_values: bool = False,
) -> dict:
    """
    Test top_k_per_row_decode with seq_lens tensor.
    """
    torch.set_default_device("cuda:0")
    ret = {}
    # Create test data
    num_rows = batch_size * next_n
    seq_lens = torch.empty(batch_size, dtype=torch.int32, device="cuda").fill_(
        context_len
    )
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_rows, device="cuda") // next_n
    next_n_offset = torch.arange(num_rows, device="cuda") % next_n
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    logits = create_random_logits(
        row_starts,
        row_ends,
        torch.float32,
        42,
        data_generation,
        physical_width=max(context_len, top_k) if flydsl else None,
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    values = (
        torch.empty((num_rows, top_k), dtype=torch.float32, device="cuda")
        if write_values
        else None
    )

    # Run the kernel
    _, us = run_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        fast,
        k=top_k,
        flydsl=flydsl,
        stable=stable,
        values=values,
    )

    torch.cuda.synchronize()

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    all_close = compare_topk_results(
        logits,
        indices,
        torch_indices,
        row_starts,
        row_ends,
        top_k,
        stable=stable,
        values=values,
    )

    # measure performance
    ret["width"] = logits.shape[1]
    ret["all_close"] = all_close
    ret["us"] = us
    ret["fast"] = fast
    return ret


# A context-sized decode buffer that each call only partly fills, which is what
# `max_row_len` is for. Every carried card's adaptive bands take these shapes;
# test_decode_bound_gate() fails if a re-fitted table stops taking them.
_BOUNDED_BATCH = (4, 16)
_BOUNDED_CONTEXT = 131072
_BOUNDED_KS = (512, 1024, 2048)


def create_planted_logits(seq_lens: torch.Tensor, width: int, top_k: int):
    """Distinct logits whose top k sit at known positions inside each row's live
    length, with padding that outranks all of them, so a read past a row's end
    selects padding."""
    rows = seq_lens.shape[0]
    logits = -torch.arange(width, dtype=torch.float32, device="cuda").repeat(rows, 1)
    for r, n in enumerate(seq_lens.tolist()):
        stride = n // top_k
        pos = torch.arange(top_k, device="cuda") * stride + r % stride
        logits[r, pos] = 1000.0 + torch.arange(top_k, device="cuda")
        logits[r, n:] = 1e4
    return logits


@benchmark()
def test_top_k_per_row_decode_bounded(
    batch_size: int,
    context_len: int,
    top_k: int,
    stable: bool = False,
    next_n: int = 1,
) -> dict:
    """Decode rows of ragged length in a buffer 4x the bound, the longest row on
    the bound, called with `max_row_len` as a serving stack would. With
    `next_n > 1` each sequence owns `next_n` rows, slot `s` ending at
    `seq_len - next_n + s + 1`."""
    seq_lens = torch.randint(
        top_k + next_n - 1,
        context_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens[0] = context_len
    rows = batch_size * next_n
    slot = torch.arange(rows, dtype=torch.int32, device="cuda") % next_n
    row_ends = seq_lens.repeat_interleave(next_n) - next_n + slot + 1
    logits = create_planted_logits(row_ends, 4 * context_len, top_k)
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    args = (logits, next_n, seq_lens, indices, rows, *logits.stride())

    _, us = run_top_k_per_row_decode(
        *args, False, k=top_k, stable=stable, max_row_len=context_len
    )
    torch.cuda.synchronize()

    # The reference sees each row's live part only.
    row_starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    live = torch.arange(logits.shape[1], device="cuda")[None, :] < row_ends[:, None]
    masked = torch.where(live, logits, float("-inf"))
    torch_indices = masked.topk(top_k, dim=-1)[1]

    return {
        "backend": topk.decode_backend_for_call(
            *args, top_k, stable, max_row_len=context_len
        ),
        "all_close": compare_topk_results(
            masked, indices, torch_indices, row_starts, row_ends, top_k, stable=stable
        ),
        "us": us,
    }


@benchmark()
def test_top_k_per_row_decode_bounded_graph(
    batch_size: int,
    context_len: int,
    top_k: int,
    stable: bool = False,
    replays: int = 3,
) -> dict:
    """The bounded decode captured once in a CUDAGraph, then replayed on logits and
    `seq_lens` rewritten in place, as a serving stack replays its decode step."""
    width = 4 * context_len
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device="cuda")
    logits = create_planted_logits(seq_lens, width, top_k)
    indices = torch.empty((batch_size, top_k), dtype=torch.int32, device="cuda")
    args = (logits, 1, seq_lens, indices, batch_size, *logits.stride())

    def decode():
        aiter.top_k_per_row_decode(
            *args, k=top_k, stable=stable, max_row_len=context_len
        )

    # Compile outside the capture, which cannot record a JIT build.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        decode()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()

    row_starts = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    all_close = True
    for _ in range(replays):
        fresh = torch.randint(
            top_k, context_len + 1, (batch_size,), dtype=torch.int32, device="cuda"
        )
        seq_lens.copy_(fresh)
        logits.copy_(create_planted_logits(fresh, width, top_k))
        indices.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        live = torch.arange(width, device="cuda")[None, :] < seq_lens[:, None]
        masked = torch.where(live, logits, float("-inf"))
        torch_indices = masked.topk(top_k, dim=-1)[1]
        all_close &= compare_topk_results(
            masked, indices, torch_indices, row_starts, seq_lens, top_k, stable=stable
        )

    return {
        "backend": topk.decode_backend_for_call(
            *args, top_k, stable, max_row_len=context_len
        ),
        "replays": replays,
        "all_close": all_close,
    }


def short_tier_exit_pass(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Per row, the radix pass whose boundary bucket holds exactly the count still
    needed: 1 or 2, or 3 when only the last pass settles the row."""
    bits = logits.view(torch.int32).long() & 0xFFFFFFFF
    key = torch.where(bits >= 1 << 31, 0xFFFFFFFF - bits, bits | 1 << 31)
    kth = key.topk(top_k, dim=-1).values[:, -1:]
    exit_pass = torch.full((key.shape[0],), 3, device=key.device)
    for radix_pass, shift in ((2, 10), (1, 21)):
        exact = ((key >> shift) >= (kth >> shift)).sum(-1) == top_k
        exit_pass[exact] = radix_pass
    return exit_pass


@benchmark()
def test_top_k_per_row_decode_short_rows(
    batch_size: int, context_len: int, top_k: int, data: str
) -> dict:
    """Bounded decode whose rows all fit the single-workgroup short tier, on data
    that settles a row after the first radix pass (`planted`), mostly after the
    second (`random`), or only after the last (`ties`)."""
    short_max = adaptive_kernel.decode_adaptive_short_max(batch_size)
    seq_lens = torch.randint(
        top_k + 1,
        min(short_max, context_len) + 1,
        (batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    width = 4 * context_len
    if data == "planted":
        logits = create_planted_logits(seq_lens, width, top_k)
    else:
        shape = (batch_size, width)
        if data == "random":
            logits = torch.randn(shape, device="cuda")
        else:
            logits = torch.randint(0, 50, shape, device="cuda").float()
        live = torch.arange(width, device="cuda")[None, :] < seq_lens[:, None]
        logits = torch.where(live, logits, 1e4)
    indices = torch.empty((batch_size, top_k), dtype=torch.int32, device="cuda")
    args = (logits, 1, seq_lens, indices, batch_size, *logits.stride())

    _, us = run_top_k_per_row_decode(
        *args, False, k=top_k, stable=False, max_row_len=context_len
    )
    torch.cuda.synchronize()

    row_starts = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    live = torch.arange(width, device="cuda")[None, :] < seq_lens[:, None]
    masked = torch.where(live, logits, float("-inf"))
    torch_indices = masked.topk(top_k, dim=-1)[1]
    exits = short_tier_exit_pass(masked[:, : int(seq_lens.max())], top_k)

    return {
        "backend": topk.decode_backend_for_call(
            *args, top_k, False, max_row_len=context_len
        ),
        "data": data,
        "exit_passes": "".join(str(p) for p in sorted(set(exits.tolist()))),
        "all_close": compare_topk_results(
            masked, indices, torch_indices, row_starts, seq_lens, top_k
        ),
        "us": us,
    }


def adaptive_band_cells(card):
    """One bounded cell per band `card` ships, at the band's smallest corner, with
    each k group's members taken in turn across its bands."""
    for stable, per_group in topk._ADAPTIVE_BANDS_BY_K_GROUP[card].items():
        for ks, bands in per_group.items():
            for i, (min_width, _, min_rows, _) in enumerate(bands):
                yield min_rows, min_width, ks[i % len(ks)], stable


def test_decode_bound_gate():
    """Host-side rules of the decode gate, checked for every card the adaptive
    table carries; no kernel runs."""
    table = topk._ADAPTIVE_BANDS_BY_K_GROUP
    for (arch, cu), per_emit in table.items():
        unmeasured = next(c for c in range(1, 1024) if (arch, c) not in table)
        for stable, per_group in per_emit.items():
            for bands in per_group.values():
                # Ascending and disjoint: a malformed band admits an unmeasured
                # shape rather than failing.
                seen_to = 0
                for min_w, max_w, min_r, max_r in bands:
                    assert seen_to < min_w <= max_w and 0 < min_r <= max_r, bands
                    seen_to = max_w
            for batch, k in itertools.product(_BOUNDED_BATCH, _BOUNDED_KS):
                call = (stable, 4 * _BOUNDED_CONTEXT, batch, k, True)
                cell = (arch, cu, *call)
                bounded = topk._decode_backend(arch, cu, *call, _BOUNDED_CONTEXT)
                assert bounded == topk.BACKEND_ADAPTIVE, cell
                # A CU count the table does not carry never routes to the
                # adaptive kernel, and `None` declines its bands and nothing else.
                plain = topk._decode_backend(arch, unmeasured, *call, _BOUNDED_CONTEXT)
                assert plain != topk.BACKEND_ADAPTIVE, cell
                assert topk._decode_backend(arch, cu, *call, None) == plain, cell
    print(f"[decode_bound_gate] PASS: {len(table)} cards")


def test_decode_bound_entry_points(card):
    """`topk_select` and the public FlyDSL wrapper each carry `max_row_len` to the
    adaptive kernel. A dropped bound still returns correct indices, so the launch
    itself is checked, not only the result."""
    wave = wave_size_of(torch.cuda.current_device())
    rows, context_len, top_k, _ = next(
        (m, n, k, s)
        for m, n, k, s in adaptive_band_cells(card)
        if s
        and topk_select_backend_for(m, 4 * n, k, wave, True, "low", False, True)
        == "decode"
    )
    width = 4 * context_len
    seq_lens = torch.randint(
        top_k, context_len + 1, (rows,), dtype=torch.int32, device="cuda"
    )
    seq_lens[0] = context_len
    logits = create_planted_logits(seq_lens, width, top_k)
    live = torch.arange(width, device="cuda")[None, :] < seq_lens[:, None]
    masked = torch.where(live, logits, float("-inf"))
    torch_indices = masked.topk(top_k, dim=-1)[1]
    row_starts = torch.zeros(rows, dtype=torch.int32, device="cuda")

    def flydsl_wrapper():
        indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
        aiter.flydsl_top_k_per_row_decode(
            logits,
            1,
            seq_lens,
            indices,
            rows,
            *logits.stride(),
            top_k,
            True,
            max_row_len=context_len,
        )
        return indices

    entry_points = {
        "topk_select": lambda: topk_select(
            logits, top_k, end=seq_lens, tie="low", max_row_len=context_len
        )[1],
        "flydsl_top_k_per_row_decode": flydsl_wrapper,
    }
    run_adaptive = flydsl_decode_host._run_adaptive
    launches = []

    def record(*args, cfg_width, **kwargs):
        launches.append(cfg_width)
        return run_adaptive(*args, cfg_width=cfg_width, **kwargs)

    for name, call in entry_points.items():
        launches.clear()
        flydsl_decode_host._run_adaptive = record
        try:
            indices = call()
        finally:
            flydsl_decode_host._run_adaptive = run_adaptive
        torch.cuda.synchronize()
        assert launches == [
            context_len
        ], f"{name} launched the adaptive kernel as {launches}, not [{context_len}]"
        assert compare_topk_results(
            masked, indices, torch_indices, row_starts, seq_lens, top_k, stable=True
        ), f"{name} mismatch at rows={rows} width={width} bound={context_len}"
    print(
        f"[decode_bound_entry_points] PASS: {', '.join(entry_points)} at "
        f"rows={rows} width={width} bound={context_len} k={top_k}"
    )


def test_adaptive_compact_garbage(card):
    """The adaptive launch clears only the leading slots of each workspace row, so
    run a compact cell twice on a workspace whose every slot is garbage first."""
    rows, width, top_k = next(
        (m, n, 2048)
        for m in (64, 128, 256)
        for n in (65536, 131072)
        if adaptive_kernel.decode_adaptive_config(
            m, n, 2048, ordered=False, cu_count=card[1]
        )["compact"]
    )
    cfg = adaptive_kernel.decode_adaptive_config(
        rows, width, top_k, ordered=False, cu_count=card[1]
    )
    slots = adaptive_kernel.topk_workspace_slots(
        rows, 11, compact=True, compact_cap=cfg["kw"]["compact_cap_mult"] * top_k
    )
    workspace = flydsl_decode_host._get_adaptive_workspace(
        torch.device("cuda", torch.cuda.current_device()),
        torch.cuda.current_stream().cuda_stream,
        slots,
    )
    seq_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    row_starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    for seed in (1, 2):
        torch.manual_seed(seed)
        logits = torch.randn(rows, width, device="cuda")
        indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
        workspace.fill_(-1)
        flydsl_decode_host._decode_with_backend(
            logits,
            1,
            seq_lens,
            indices,
            rows,
            *logits.stride(),
            top_k,
            False,
            None,
            topk.BACKEND_ADAPTIVE,
            width,
        )
        torch.cuda.synchronize()
        # The launch cleared this buffer's leading slots, so it is the one it used.
        lead = adaptive_kernel.workspace_zero_row_slots(11, compact=True)
        assert (workspace.view(rows, -1)[:, :lead] >= 0).all()
        torch_indices = logits.topk(top_k, dim=-1)[1]
        assert compare_topk_results(
            logits, indices, torch_indices, row_starts, seq_lens, top_k
        ), f"compact garbage mismatch at rows={rows} width={width} seed={seed}"
    print(f"[adaptive_compact_garbage] PASS: rows={rows} width={width} k={top_k}")


def stable_reference(masked: torch.Tensor, top_k: int) -> torch.Tensor:
    """The k largest per row in ascending column order, ties to the smallest column.

    Index for index, where `compare_topk_results` accepts any member of a tie.
    """
    order = torch.sort(masked, dim=-1, descending=True, stable=True).indices
    return order[:, :top_k].sort(dim=-1).values


def test_adaptive_ordered_early_stop(card):
    """Stable decode on rows that settle before the last pass and rows that do not,
    on a multi-part cell with the pass-0 certificate, one without it, and a
    one-part cell: index for index the stable reference, and the same with early
    stop off."""
    top_k = 2048

    def cfg(rows, width):
        return adaptive_kernel.decode_adaptive_config(
            rows, width, top_k, ordered=True, cu_count=card[1]
        )

    def multi_part(rows, width, certified):
        # A row no longer than short_max runs on one part whatever the grid.
        short_max = adaptive_kernel.decode_adaptive_short_max(rows)
        c = cfg(rows, width)
        return (
            c["parts"] > 1
            and width > short_max
            and c["kw"].get("histogram_certificate", False) == certified
        )

    candidates = [(m, n) for m in (4, 8, 32, 64, 256) for n in (8192, 49152, 65536)]
    picks = (
        lambda m, n: multi_part(m, n, True),
        lambda m, n: multi_part(m, n, False),
        lambda m, n: cfg(m, n)["grid"] == 1,
    )
    cells = [
        next((m, n) for m, n in candidates if not cfg(m, n)["compact"] and pick(m, n))
        for pick in picks
    ]

    def logits_for(data, seq_lens, width):
        rows = seq_lens.shape[0]
        if data == "planted":
            return create_planted_logits(seq_lens, width, top_k)
        if data == "random":
            logits = torch.randn(rows, width, device="cuda")
        elif data == "ties":
            logits = torch.randint(0, 50, (rows, width), device="cuda").float()
        else:
            # Duplicated top values that all fall in one second-pass bucket.
            logits = 900.0 + 90.0 * torch.rand(rows, width, device="cuda")
            top = 1000.0 + (torch.arange(top_k, device="cuda") // 8) / 1024.0
            for r, n in enumerate(seq_lens.tolist()):
                logits[r, torch.arange(top_k, device="cuda") * (n // top_k)] = top
        live = torch.arange(width, device="cuda")[None, :] < seq_lens[:, None]
        return torch.where(live, logits, 1e4)

    es_env = os.environ.get(adaptive_kernel.EARLY_STOP_ENV)
    try:
        for rows, width in cells:
            exits = set()
            for data in ("planted", "duplicated", "random", "ties"):
                torch.manual_seed(0)
                seq_lens = torch.randint(
                    top_k + 1, width + 1, (rows,), dtype=torch.int32, device="cuda"
                )
                seq_lens[0] = width
                logits = logits_for(data, seq_lens, width)
                live = torch.arange(width, device="cuda")[None, :] < seq_lens[:, None]
                masked = torch.where(live, logits, float("-inf"))
                exits |= set(short_tier_exit_pass(masked, top_k).tolist())
                out = {}
                for es in ("1", "0"):
                    os.environ[adaptive_kernel.EARLY_STOP_ENV] = es
                    out[es] = torch.full(
                        (rows, top_k), -7, dtype=torch.int32, device="cuda"
                    )
                    flydsl_decode_host._decode_with_backend(
                        logits,
                        1,
                        seq_lens,
                        out[es],
                        rows,
                        *logits.stride(),
                        top_k,
                        True,
                        None,
                        topk.BACKEND_ADAPTIVE,
                        width,
                    )
                torch.cuda.synchronize()
                where = f"{data} rows={rows} width={width}"
                assert torch.equal(
                    out["1"].long(), stable_reference(masked, top_k)
                ), f"ordered early stop mismatch: {where}"
                assert torch.equal(out["1"], out["0"]), f"early stop changed: {where}"
            assert {1, 2} & exits and 3 in exits, f"exits {exits} at rows={rows}"
    finally:
        if es_env is None:
            os.environ.pop(adaptive_kernel.EARLY_STOP_ENV, None)
        else:
            os.environ[adaptive_kernel.EARLY_STOP_ENV] = es_env
    print(f"[adaptive_ordered_early_stop] PASS: cells={cells} k={top_k}")


def test_mb_workspace_reuse():
    """Regression for the persistent multi-block workspace + kernel self-reset.

    The mb path now runs on a cached, zeroed-once buffer (no per-call memset);
    the kernel must reset its counters/histograms to zero on exit so the *next*
    call on the same buffer is correct. This drives 3 calls with DIFFERENT data
    on the same cached buffer -- if self-reset were broken, a later call would be
    corrupted by an earlier call's leftover atomic counters / histograms.
    """
    num_rows, num_prefix, top_k = 4, 131072, 2048
    row_starts, row_ends = create_row_boundaries(num_rows, num_prefix)
    probe = create_random_logits(row_starts, row_ends, torch.float32, 0)
    stride0 = probe.stride(0)
    if not aiter.topk_use_mulblocks(num_rows, stride0):
        print(
            f"[mb_workspace_reuse] mb path not selected on this HW "
            f"(num_rows={num_rows}, seq={stride0}); skipping"
        )
        return
    max_end = int(max(row_ends))
    for call_idx, seed in enumerate((11, 22, 33)):
        logits = create_random_logits(row_starts, row_ends, torch.float32, seed)
        indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
        aiter.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            None,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=top_k,
        )
        ref = logits.topk(min(top_k, max_end), dim=-1)[1]
        mask = (ref >= 0) & ((ref - (row_ends - row_starts)[:, None]) < 0)
        ref = ref.masked_fill(~mask, -1)
        assert compare_topk_results(
            logits, indices, ref, row_starts, row_ends, top_k
        ), f"mb workspace reuse mismatch on call #{call_idx} (seed={seed})"
    print("[mb_workspace_reuse] PASS: 3 reused-buffer mb calls matched torch.topk")


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-c",
    "--context_len",
    type=int,
    default=[8, 128, 1024, 3072, 4096, 8192, 16384, 32768, 65536, 90000, 128000],
    nargs="+",
    help="""number of kv.
    e.g.: -c 64""",
)

parser.add_argument(
    "-k",
    "--top_k",
    type=int,
    default=[512, 1024, 2048],
    nargs="+",
    help="""top-k elements per row. The radix backend supports any positive
    int; the `_fast` ASM-kernel path only supports 2048 and is skipped
    for other values.
    e.g.: -k 512 1024 2048""",
)

parser.add_argument(
    "--num_prefix",
    type=int,
    default=[0],
    nargs="+",
    help="""top-k elements per row.
    e.g.: --num_prefix 8000 16000 24000 32000 40000 48000 56000""",
)

parser.add_argument(
    "-b",
    "--decode_batch_size",
    type=int,
    default=[4, 8, 16, 24],
    nargs="+",
    help="""decode_batch_size batch size.
    e.g.: -b 4""",
)

parser.add_argument(
    "-n",
    "--next_n",
    type=int,
    default=[1, 2, 3, 4],
    nargs="+",
    help="""next_n elements per sequence in a row.
    e.g.: -n 4""",
)

parser.add_argument(
    "-d",
    "--data_generation",
    type=str,
    default=["random"],
    choices=["random", "10LSBits", "mixed"],
    nargs="+",
    help="""Specify method for generating logits.
    e.g.: -d random""",
)

args = parser.parse_args()

# Self-reset / persistent-workspace regression (runs in CI via `python3 <file>`).
test_mb_workspace_reuse()
test_decode_bound_gate()


# Ask each path which arches it serves rather than keeping a second copy
# here: a copy drifts, and a test that drives a kernel production never
# dispatches reports on something nobody runs.
one_block_available = get_gfx() in _FLYDSL_TOPK_ONE_BLOCK_ARCHES
flydsl_decode_available = get_gfx() in _FLYDSL_TOPK_DECODE_GATES

df = []
for data_generation in args.data_generation:
    for m in args.context_len:
        for k in args.top_k:
            for num_prefix in args.num_prefix:
                ret = test_top_k_per_row_prefill(m, num_prefix, k, data_generation)
                df.append(ret)
                # Cover the one-block radix kernel directly, so a dispatch
                # change cannot hide a kernel regression.
                if not one_block_available:
                    continue
                for stable in (False, True):
                    for write_values in (False, True):
                        ret = test_top_k_per_row_prefill(
                            m,
                            num_prefix,
                            k,
                            data_generation,
                            flydsl=True,
                            stable=stable,
                            write_values=write_values,
                        )
                        df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("topk_per_row_prefill summary (markdown):\n%s", df_md)
assert df["all_close"].all(), f"topk_per_row_prefill mismatch:\n{df_md}"


df = []
for data_generation in args.data_generation:
    for m in args.decode_batch_size:
        for ctx in args.context_len:
            for k in args.top_k:
                for n in args.next_n:
                    for stable in (False, True):
                        for write_values in (False, True):
                            ret = test_top_k_per_row_decode(
                                m,
                                ctx,
                                k,
                                n,
                                data_generation,
                                stable=stable,
                                write_values=write_values,
                            )
                            df.append(ret)
                            if flydsl_decode_available:
                                ret = test_top_k_per_row_decode(
                                    m,
                                    ctx,
                                    k,
                                    n,
                                    data_generation,
                                    flydsl=True,
                                    stable=stable,
                                    write_values=write_values,
                                )
                                df.append(ret)
                        # `_fast` ASM kernel hardcodes k=2048 and is not stable.
                        if get_gfx() == "gfx942" and k == 2048 and not stable:
                            ret = test_top_k_per_row_decode(
                                m,
                                ctx,
                                k,
                                n,
                                data_generation,
                                fast=True,
                            )
                            df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("topk_per_row_decode summary (markdown):\n%s", df_md)
assert df["all_close"].all(), f"topk_per_row_decode mismatch:\n{df_md}"


device = torch.cuda.current_device()
card = (topk._decode_arch(device), topk._decode_cu_count(device))
if card in topk._ADAPTIVE_BANDS_BY_K_GROUP:
    # The table picks these shapes and k, not the command line: every band this
    # card ships runs once, at a corner small enough to allocate here.
    for name, run in (
        ("bounded", test_top_k_per_row_decode_bounded),
        ("bounded graph", test_top_k_per_row_decode_bounded_graph),
        (
            "bounded next_n=2",
            lambda m, n, k, s: test_top_k_per_row_decode_bounded(
                max(1, m // 2), n, k, s, next_n=2
            ),
        ),
    ):
        torch.manual_seed(0)
        df = pd.DataFrame([run(*c) for c in adaptive_band_cells(card)])
        df_md = df.to_markdown(index=False)
        aiter.logger.info("topk_per_row_decode %s summary (markdown):\n%s", name, df_md)
        assert df["all_close"].all(), f"topk_per_row_decode {name} mismatch:\n{df_md}"
        assert (
            df["backend"] == topk.BACKEND_ADAPTIVE
        ).all(), f"{name} decode left the adaptive kernel:\n{df_md}"
    # The short tier is unordered only, so it runs under the unstable bands.
    rows, context_len, top_k, _ = next(c for c in adaptive_band_cells(card) if not c[3])
    torch.manual_seed(0)
    df = pd.DataFrame(
        [
            test_top_k_per_row_decode_short_rows(rows, context_len, top_k, data)
            for data in ("planted", "random", "ties")
        ]
    )
    df_md = df.to_markdown(index=False)
    aiter.logger.info("topk_per_row_decode short rows summary (markdown):\n%s", df_md)
    assert df["all_close"].all(), f"short-row decode mismatch:\n{df_md}"
    assert (
        df["backend"] == topk.BACKEND_ADAPTIVE
    ).all(), f"short-row decode left the adaptive kernel:\n{df_md}"
    assert set("".join(df["exit_passes"])) == set(
        "123"
    ), f"short-row decode does not reach every radix pass exit:\n{df_md}"
    test_decode_bound_entry_points(card)
    test_adaptive_compact_garbage(card)
    test_adaptive_ordered_early_stop(card)
else:
    aiter.logger.warning(
        "%s at %d CU carries no adaptive decode bands; bounded decode skipped", *card
    )
