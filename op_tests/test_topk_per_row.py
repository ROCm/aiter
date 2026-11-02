import argparse
import contextlib
import os
from unittest import mock

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops import topk as T
from aiter.ops import topk_select as S
from aiter.ops.flydsl.topk.topk_per_row import _FLYDSL_TOPK_ONE_BLOCK_ARCHES
from aiter.ops.topk import _FLYDSL_TOPK_DECODE_GATES
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
def run_top_k_per_row_prefill_sampled(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    num_rows: int,
    stride_row: int,
    stride_col: int,
    k: int = 2048,
) -> None:
    """Run the sampled kernels through the same timing path.

    Same arguments as run_top_k_per_row_prefill so the two are measured under
    one decorator rather than two, which is the only way the numbers are
    comparable: @perftest() sizes its argument rotation from the input, so a
    separately written loop would not defeat L2 the same way.
    """
    return aiter.top_k_per_row_prefill_sampled(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride_row,
        stride_col,
        k=k,
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
        )


@benchmark()
def test_top_k_per_row_prefill(
    num_rows: int,
    num_prefix: int,
    top_k: int,
    data_generation: str = "random",
    flydsl: bool = False,
    stable: bool = False,
    backend: str = "aiter",
    write_values: bool = False,
) -> dict:
    """
    Test topk_per_row_prefill.

    `backend` picks which selector runs: "aiter" for top_k_per_row_prefill,
    "sampled" for top_k_per_row_prefill_sampled. Both go through the same data,
    the same torch.topk reference and the same @perftest timing, so the `us`
    column is comparable across the two rows of the summary table.

    `write_values` also requests the selected scores. They are checked by
    gathering the logits at the indices the SAME call returned, so a score that
    is self-consistently wrong cannot pass.
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

    # The sampled kernels honour rowEnds per row and emit indices only, so record
    # the shortest row: it is what exercises the per-row extent and the -1
    # padding, and it is the column to look at when a result disagrees.
    min_row_len = int((row_ends - row_starts).min())
    ret["backend"] = backend
    ret["min_row_len"] = min_row_len
    ret["write_values"] = write_values
    if backend == "sampled":
        if not aiter.topk_sampled_supports(num_rows, logits.stride(0), top_k):
            ret["context_len"] = logits.shape[1]
            ret["all_close"] = None
            ret["us"] = float("nan")
            ret["note"] = "unsupported geometry"
            return ret
        _, us = run_top_k_per_row_prefill_sampled(
            logits,
            row_starts,
            row_ends,
            indices,
            values,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=top_k,
        )
        ret["note"] = "" if min_row_len >= top_k else "ragged rows"
    else:
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
        ret["note"] = ""

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
    ragged: bool = False,
) -> dict:
    """
    Test top_k_per_row_decode with seq_lens tensor.

    ragged gives the requests different lengths in one buffer, so a launch
    mixes rows that one CTA finishes with rows split across CTAs.
    """
    torch.set_default_device("cuda:0")
    ret = {}
    # Create test data
    num_rows = batch_size * next_n
    if ragged:
        lengths = [context_len, context_len // 8, context_len // 64, context_len // 2]
        seq_lens = torch.tensor(
            [max(next_n, lengths[i % len(lengths)]) for i in range(batch_size)],
            dtype=torch.int32,
            device="cuda",
        )
    else:
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


def _decode_reference(logits, row_ends, top_k):
    ref = logits.topk(min(top_k, int(max(row_ends))), dim=-1)[1]
    mask = (ref >= 0) & ((ref - row_ends[:, None]) < 0)
    return ref.masked_fill(~mask, -1)


def test_flydsl_decode_split_reuse():
    """Long rows split across CTAs and merge in the same launch.

    The per-row arrival counters rearm themselves; this drives repeated eager
    calls, same-stream CUDA graph warmup/replay and two streams, stable and not,
    over a buffer that mixes split rows with rows one CTA finishes.
    """
    if get_gfx() not in _FLYDSL_TOPK_DECODE_GATES:
        print("[flydsl_decode_split_reuse] FlyDSL decode not served here; skipping")
        return
    torch.set_default_device("cuda:0")
    num_rows, width, top_k = 16, 1 << 19, 2048
    seq_lens = torch.tensor(
        [[width, 4096, 300_001, 70_003][i % 4] for i in range(num_rows)],
        dtype=torch.int32,
    )
    row_starts = torch.zeros(num_rows, dtype=torch.int32)

    def run(logits, indices, stable, values=None):
        aiter.flydsl_top_k_per_row_decode(
            logits, 1, seq_lens, indices, num_rows, width, 1, top_k, stable, values
        )

    def check(logits, indices, stable, what):
        ref = _decode_reference(logits, seq_lens, top_k)
        assert compare_topk_results(
            logits, indices, ref, row_starts, seq_lens, top_k, stable=stable
        ), f"flydsl decode split mismatch: {what}"

    def new_logits(seed):
        return create_random_logits(
            row_starts, seq_lens, torch.float32, seed, physical_width=width
        )

    for stable in (False, True):
        for call_idx, seed in enumerate((11, 22, 33)):
            logits = new_logits(seed)
            indices = torch.empty((num_rows, top_k), dtype=torch.int32)
            run(logits, indices, stable)
            check(logits, indices, stable, f"eager call #{call_idx} stable={stable}")

    logits = new_logits(34)
    indices = torch.empty((num_rows, top_k), dtype=torch.int32)
    values = torch.empty((num_rows, top_k), dtype=torch.float32)
    run(logits, indices, True, values)
    reference = _decode_reference(logits, seq_lens, top_k)
    assert compare_topk_results(
        logits,
        indices,
        reference,
        row_starts,
        seq_lens,
        top_k,
        stable=True,
        values=values,
    ), "stable merge values mismatch"

    graphs, buffers = [], []
    for seed in (44, 55):
        logits = new_logits(seed)
        indices = torch.empty((num_rows, top_k), dtype=torch.int32)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            run(logits, indices, False)
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            run(logits, indices, False)
        graphs.append(graph)
        buffers.append((logits, indices, capture_stream))
    graph_buffers = list(zip(graphs, buffers))
    for replay in range(20):
        # Replay the second graph first once: capture order must not initialize
        # or otherwise affect another graph's counter.
        replay_order = reversed(graph_buffers) if replay == 0 else graph_buffers
        for graph, (logits, indices, capture_stream) in replay_order:
            logits.copy_(new_logits(100 + replay))
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                graph.replay()
            capture_stream.synchronize()
            check(logits, indices, False, f"graph replay #{replay}")

    # Warm the stream-local counter, then capture with a temporary output view.
    logits = new_logits(58)
    output_owner = torch.empty((num_rows + 1, top_k), dtype=torch.int32)
    capture_stream = torch.cuda.Stream()
    warmup_indices = torch.empty((num_rows, top_k), dtype=torch.int32)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        run(logits, warmup_indices, False)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run(logits, output_owner[:num_rows], False)
    victims = (
        torch.full((num_rows, 16 * top_k), 17, dtype=torch.int32),
        torch.full((num_rows, 16 * top_k), 19.0, dtype=torch.float32),
        torch.full((num_rows,), 23, dtype=torch.int32),
    )
    for seed in (59, 60, 61):
        eager_logits = new_logits(seed)
        eager_indices = torch.empty((num_rows, top_k), dtype=torch.int32)
        run(eager_logits, eager_indices, False)
        check(eager_logits, eager_indices, False, "eager after capture")
    for replay in range(3):
        logits.copy_(new_logits(80 + replay))
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.replay()
        capture_stream.synchronize()
        check(logits, output_owner[:num_rows], False, "temporary-view graph replay")
        assert bool((victims[0] == 17).all())
        assert bool((victims[1] == 19).all())
        assert bool((victims[2] == 23).all())

    streams = [torch.cuda.Stream() for _ in range(2)]
    outs = []
    for seed, stream in zip((66, 77), streams):
        logits = new_logits(seed)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            calls = [
                torch.empty((num_rows, top_k), dtype=torch.int32) for _ in range(10)
            ]
            for indices in calls:
                run(logits, indices, False)
        outs.append((logits, calls))
    torch.cuda.synchronize()
    for logits, calls in outs:
        for indices in calls:
            check(logits, indices, False, "concurrent streams")
    print("[flydsl_decode_split_reuse] PASS: eager, graph replay and streams")


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


# `sampled` only offers a faster algorithm, so the checks below hold it to what
# the entries it serves already do.
_SAMPLED_N = 131072
_SAMPLED_K = 2048


@contextlib.contextmanager
def _use_sampled(on):
    """AITER_DISABLE_TOPK_SAMPLED set for the block and restored after it, with
    the memoised routing dropped on the way in and out."""
    flag = {"AITER_DISABLE_TOPK_SAMPLED": "0" if on else "1"}
    with mock.patch.dict(os.environ, flag):
        S._available.cache_clear()
        S._choose.cache_clear()
        try:
            yield
        finally:
            S._available.cache_clear()
            S._choose.cache_clear()


def _bounds(rows, width, device="cuda"):
    starts = torch.zeros(rows, dtype=torch.int32, device=device)
    ends = torch.full((rows,), width, dtype=torch.int32, device=device)
    return starts, ends


def _sign_key(v):
    """fp32 bits made monotone: a negative NaN below -inf, a positive one above +inf."""
    b = v.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(b >= 0x80000000, (~b) & 0xFFFFFFFF, b | 0x80000000)


def _nan_high_key(v):
    """topk_select's order: every NaN above +inf, tied with the others."""
    return torch.where(
        torch.isnan(v), torch.full_like(_sign_key(v), 0xFFFFFFFF), _sign_key(v)
    )


def _selected_keys_match(x, idx, k, key):
    ref = torch.topk(key(x), k, dim=1).values.sort(dim=1).values
    got = key(x.gather(1, idx.long())).sort(dim=1).values
    return torch.equal(got, ref)


def test_sampled_layout_contract():
    """The sampled entry refuses tensors its kernels would address wrongly, and
    the router leaves such calls on the path that served them before."""
    rows, N, K = 2, _SAMPLED_N, _SAMPLED_K
    if not aiter.topk_sampled_supports(rows, N, K):
        print("[sampled_layout] SKIP: `sampled` does not serve this shape here")
        return
    torch.manual_seed(0)
    x = torch.randn(rows, N, device="cuda")
    rs, re = _bounds(rows, N)

    def refused(**over):
        a = {
            "logits": x,
            "rowStarts": rs,
            "rowEnds": re,
            "indices": torch.empty(rows, K, dtype=torch.int32, device="cuda"),
            "values": None,
            "stride0": N,
            "workspace": None,
        }
        a.update(over)
        try:
            aiter.top_k_per_row_prefill_sampled(
                a["logits"],
                a["rowStarts"],
                a["rowEnds"],
                a["indices"],
                a["values"],
                rows,
                a["stride0"],
                1,
                K,
                workspace=a["workspace"],
            )
        except ValueError:
            return True
        return False

    guard = torch.full((rows, K + 17), -7, dtype=torch.int32, device="cuda")
    assert refused(indices=guard[:, :K]), "a row-strided indices view must be refused"
    assert bool((guard == -7).all()), "a refused call must not write"
    vguard = torch.zeros(rows, K + 17, device="cuda")
    assert refused(values=vguard[:, :K]), "a row-strided values view must be refused"
    assert bool((vguard == 0).all()), "a refused call must not write values"
    assert refused(rowStarts=rs.long()), "int64 rowStarts must be refused"
    assert refused(
        rowEnds=torch.full((rows, 2), N, dtype=torch.int32, device="cuda")[:, 0]
    )
    cpu_out = torch.empty(rows, K, dtype=torch.int32, device="cpu")
    assert refused(indices=cpu_out), "a CPU output"
    assert refused(logits=x[:1]), "fewer logits rows than numRows"
    assert refused(logits=x.t().contiguous().t()), "logits with an inner stride"
    wide = torch.zeros(rows, N + 64, device="cuda")[:, :N]
    assert refused(logits=wide), "a row stride other than stride0"
    two_k = torch.empty(rows, 2 * K, dtype=torch.int32, device="cuda")
    assert refused(indices=two_k), "indices rows wider than k"
    size = T._sampled_workspace_size_cached(rows, N, K)
    ws = torch.empty(size + 512, dtype=torch.uint8, device="cuda")[1 : 1 + size]
    assert refused(workspace=ws), "a workspace off the 256-byte grid"

    # Through the router an output whose rows are not k wide is what the
    # pre-`sampled` paths make of it: rows of k from its data pointer, and
    # nothing past them. Each row is checked against its own logits row, and
    # the slots past rows * k on their own, so a row written in another row's
    # place or a write past the end cannot hide in a sorted whole buffer.
    for shape, label in (
        ((rows, K + 17), "row-strided"),
        ((rows, 2 * K), "[rows, 2k]"),
    ):
        got = {}
        for on in (True, False):
            with _use_sampled(on):
                g = torch.full(shape, -7, dtype=torch.int32, device="cuda")
                T.top_k_per_row_prefill(x, rs, re, g[:, :K], None, rows, N, 1, K)
                torch.cuda.synchronize()
            flat = g.view(-1)
            tag = f"{label} output, sampled {'on' if on else 'off'}"
            assert bool((flat[rows * K :] == -7).all()), f"{tag}: wrote past rows*k"
            got[on] = flat[: rows * K].view(rows, K).sort(dim=1).values
            for r in range(rows):
                assert _selected_keys_match(
                    x[r : r + 1], got[on][r : r + 1], K, _sign_key
                ), f"{tag}: row {r} is not its own top-k"
        same = (got[True] == got[False]).all(dim=1)
        assert bool(same.all()), (
            f"the router changed a {label} call in rows "
            f"{(~same).nonzero().flatten().tolist()}"
        )
    print("[sampled_layout] PASS")


def test_sampled_nan_order():
    """topk_select ranks a NaN of either sign above +inf and returns the input's
    own bits; top_k_per_row_prefill keeps the sign order its radix paths use."""
    rows, N, k = T._SAMPLED_PREFILL_MAX_ROWS, _SAMPLED_N, 512
    if not aiter.topk_sampled_supports(rows, N, k):
        print("[sampled_nan] SKIP: `sampled` does not serve this shape here")
        return
    torch.manual_seed(1)
    x = torch.rand(rows, N, device="cuda") + 1.0
    bits = x.view(torch.int32)
    payloads = (-4194304, -4194303, 0x7FC00001, 0x7F800001, -8388607)
    for j, p in enumerate(payloads):
        bits[:, 1000 * j + 7] = p
    nan_cols = torch.tensor([1000 * j + 7 for j in range(len(payloads))], device="cuda")

    with _use_sampled(True):
        avail = S._available(
            N, k, 64, False, True, T._sampled_on_device(0), T._device_arch(0)
        )
        assert (
            S.topk_select_backend(rows, N, k, avail, device=0) == "sampled"
        ), "this shape must exercise `sampled`"
        vals, idx = aiter.topk_select(x, k, return_value=True)
        torch.cuda.synchronize()
    assert _selected_keys_match(x, idx, k, _nan_high_key), "topk_select NaN order"
    assert bool(torch.isin(nan_cols, idx).all()), "every NaN must be selected"
    assert torch.equal(
        vals.view(torch.int32), x.gather(1, idx.long()).view(torch.int32)
    ), "returned values must be the input's bits"

    rs, re = _bounds(rows, N)
    pidx = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    pval = torch.empty(rows, k, device="cuda")
    with _use_sampled(True):
        assert T._should_use_sampled_prefill(
            rows, N, 1, k, False, 0
        ), "this shape must exercise `sampled`"
        T.top_k_per_row_prefill(x, rs, re, pidx, pval, rows, N, 1, k)
        torch.cuda.synchronize()
    assert _selected_keys_match(x, pidx, k, _sign_key), "prefill sign order"
    assert torch.equal(
        pval.view(torch.int32), x.gather(1, pidx.long()).view(torch.int32)
    ), "prefill values must be the input's bits"
    print("[sampled_nan] PASS")


def test_sampled_overflow_all_equal():
    """Every score equal: the threshold admits the whole row, every wave
    overflows its reservation, and the row is answered by the exact fallback."""
    rows, N, K = 128, _SAMPLED_N, _SAMPLED_K
    if not aiter.topk_sampled_supports(rows, N, K):
        print("[sampled_overflow] SKIP: `sampled` does not serve this shape here")
        return
    x = torch.ones(rows, N, device="cuda")
    rs, re = _bounds(rows, N)
    idx = torch.empty(rows, K, dtype=torch.int32, device="cuda")
    val = torch.empty(rows, K, device="cuda")
    aiter.top_k_per_row_prefill_sampled(x, rs, re, idx, val, rows, N, 1, K)
    torch.cuda.synchronize()
    i = idx.long()
    assert bool(((i >= 0) & (i < N)).all()), "index out of range"
    s = i.sort(dim=1).values
    assert not bool((s[:, 1:] == s[:, :-1]).any()), "duplicate index"
    assert bool((val == 1.0).all()), "values"
    print("[sampled_overflow] PASS")


def test_sampled_eligibility_per_device():
    """Eligibility belongs to the GPU a call runs on, whichever GPU was asked
    about first. One process can drive gfx942 and gfx950 side by side; on a host
    with one kind, device 1 is made to read as gfx942 so both orders can run."""
    if torch.cuda.device_count() < 2 or not all(
        T._device_arch(d) == "gfx950" for d in (0, 1)
    ):
        print("[sampled_per_device] SKIP: needs two gfx950 GPUs")
        return
    rows, N, K = 4, _SAMPLED_N, _SAMPLED_K
    real_arch = T._device_arch

    def fake_arch(d):
        return "gfx942" if d == 1 else real_arch(d)

    def reset():
        T._sampled_supports_cached.cache_clear()
        S._available.cache_clear()
        S._choose.cache_clear()

    def routes_to_sampled(dev):
        x = torch.randn(rows, N, device=f"cuda:{dev}")
        rs, re = _bounds(rows, N, x.device)
        idx = torch.empty(rows, K, dtype=torch.int32, device=x.device)
        with mock.patch.object(T, "_top_k_per_row_prefill_sampled") as fn:
            T.top_k_per_row_prefill(x, rs, re, idx, None, rows, N, 1, K)
            prefill = fn.called
        select = S._choose(rows, N, N, K, 64, False, None, False, True, dev, True)
        return prefill, select == "sampled"

    for order in ((0, 1), (1, 0)):
        reset()
        T._device_arch = S._device_arch = fake_arch
        try:
            with _use_sampled(True):
                got = {dev: routes_to_sampled(dev) for dev in order}
        finally:
            T._device_arch = S._device_arch = real_arch
            reset()
        assert got[0] == (True, True), f"order {order}: gfx950 must route to sampled"
        assert got[1] == (False, False), f"order {order}: gfx942 must not"

    # The capability check answers for the device it is given, not the current one.
    for current, asked in ((0, 1), (1, 0)):
        with torch.cuda.device(current):
            assert aiter.topk_sampled_supports(rows, N, K, asked)

    # A tensor on a GPU that is not the current one is served on its own GPU.
    with torch.cuda.device(0):
        x = torch.randn(rows, N, device="cuda:1")
        rs, re = _bounds(rows, N, x.device)
        idx = torch.empty(rows, K, dtype=torch.int32, device="cuda:1")
        aiter.top_k_per_row_prefill_sampled(x, rs, re, idx, None, rows, N, 1, K)
        torch.cuda.synchronize(1)
    assert _selected_keys_match(x, idx, K, _sign_key), "cross-device call"
    print("[sampled_per_device] PASS")


def test_prefill_sampled_routing():
    """Which prefill calls top_k_per_row_prefill hands to `sampled`.

    Asserts on whether `sampled` ran and nothing else: which of the other paths
    takes a shape this routing declined is their dispatch's business.
    """
    floor = T._SAMPLED_MIN_STRIDE0.get(get_gfx_runtime())
    if floor is None:
        print("[prefill_sampled_routing] SKIP: `sampled` does not route on this arch")
        return
    rows, k = 4, _SAMPLED_K

    def routed(width, stable=False, k=k, rows=rows):
        torch.manual_seed(0)
        x = torch.randn(rows, width, device="cuda")
        rs, re = _bounds(rows, width)
        idx = torch.empty(rows, k, dtype=torch.int32, device="cuda")
        with mock.patch.object(
            T, "_top_k_per_row_prefill_sampled", autospec=True
        ) as fn:
            fn.side_effect = lambda *a, **kw: None
            T.top_k_per_row_prefill(
                x, rs, re, idx, None, rows, x.stride(0), 1, k=k, stable=stable
            )
            return fn.called

    assert not routed(floor // 2), f"stride0={floor // 2} is below the floor"
    assert routed(floor), f"stride0={floor} is at the floor and must use `sampled`"
    assert not routed(floor, stable=True), "stable=True must not use `sampled`"
    assert not routed(floor, k=T._SAMPLED_MIN_K - 1), "k below the k floor"
    assert not routed(floor, k=2 * T._SAMPLED_MAX_K), "k past the k bound"
    many = T._SAMPLED_PREFILL_MAX_ROWS + 1
    assert not routed(floor, rows=many), f"{many} rows must not use `sampled`"
    with mock.patch.dict(os.environ, {"AITER_DISABLE_TOPK_SAMPLED": "1"}):
        assert not routed(floor), "AITER_DISABLE_TOPK_SAMPLED must withdraw it"
    print("[prefill_sampled_routing] PASS: by stride0 / rows / k / stable / env")


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
    "--prefill_backend",
    type=str,
    default=["aiter"],
    choices=["aiter", "sampled"],
    nargs="+",
    help="""which prefill selector to measure: `aiter` for
    top_k_per_row_prefill, `sampled` for top_k_per_row_prefill_sampled. Pass both to
    get one summary table with a row per backend per shape.
    e.g.: --prefill_backend aiter sampled""",
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

_REGRESSIONS = {
    "mb-workspace-reuse": test_mb_workspace_reuse,
    "flydsl-decode-split-reuse": test_flydsl_decode_split_reuse,
    "sampled-routing": test_prefill_sampled_routing,
    "sampled-layout": test_sampled_layout_contract,
    "sampled-nan": test_sampled_nan_order,
    "sampled-overflow": test_sampled_overflow_all_equal,
    "sampled-per-device": test_sampled_eligibility_per_device,
}

parser.add_argument(
    "--regressions",
    choices=list(_REGRESSIONS),
    nargs="*",
    default=list(_REGRESSIONS),
    help="""Regression checks to run before the configured sweep.
    Pass --regressions with no values to skip them all.""",
)

args = parser.parse_args()

# Regressions run in CI via `python3 <file>`.
for name in args.regressions:
    _REGRESSIONS[name]()


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
                for backend in args.prefill_backend:
                    # Both ways round, matching the decode loop: the value
                    # stores are a separate instantiation of the output
                    # kernels, so skipping one leaves half of them untested.
                    for write_values in (False, True):
                        ret = test_top_k_per_row_prefill(
                            m,
                            num_prefix,
                            k,
                            data_generation,
                            backend=backend,
                            write_values=write_values,
                        )
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
                                # Ragged lengths only matter once some rows
                                # are long enough to be split.
                                for ragged in (
                                    (False, True) if ctx >= 65536 else (False,)
                                ):
                                    ret = test_top_k_per_row_decode(
                                        m,
                                        ctx,
                                        k,
                                        n,
                                        data_generation,
                                        flydsl=True,
                                        stable=stable,
                                        write_values=write_values,
                                        ragged=ragged,
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
