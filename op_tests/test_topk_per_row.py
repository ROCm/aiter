import argparse

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.topk import (
    topk_mb_workspace_size,
    topk_ob_workspace_size,
    topk_use_mulblocks,
)
from aiter.ops.topk_plain import topk_plain
from aiter.test_common import benchmark, run_perftest

# Argument rotation deep-copies every input to defeat L2. Past this the working
# set already exceeds L2, so pin the rotation to one set instead.
_ROTATE_MAX_BYTES = 256 << 20

# torch.topk costs ~37 ms per G element here (measured, linear in M*N), so the
# largest swept cell is a few seconds of reference. Past this it is skipped and
# all_close reports "skipped" rather than a pass it never ran.
_REF_MAX_ELEMS = 1 << 34
_REF_ITERS = 5

_MEM_HEADROOM = 0.8

# Peak of create_random_logits in units of the fp32 logits: 10LSBits holds the
# randint, the masked temporary and the or'd result; mixed adds a second randn
# and the where() output. Measured 1.00 / 3.00 / 4.28 at M=64, N=1048576.
_GEN_PEAK_MULTIPLIER = {"random": 1.0, "10LSBits": 3.0, "mixed": 4.3}

# topk_plain asserts k <= MAX_CAPACITY (topk_plain_kernels.cu:980) and the
# assert aborts the process, so it cannot be called past it and recovered from.
_PLAIN_MAX_K = 2048


def _rotate_args(nbytes: int) -> int:
    return 1 if nbytes > _ROTATE_MAX_BYTES else 0


def _free_budget() -> float:
    free, _total = torch.cuda.mem_get_info()
    return free * _MEM_HEADROOM


def _fits_in_memory(nbytes: int) -> bool:
    """Whether the cell's own tensors fit. run_perftest's rotation allocates on
    top of this -- up to num_iters copies of the input below _ROTATE_MAX_BYTES --
    but sizes itself to free memory, so it cannot be the thing that OOMs."""
    return nbytes <= _free_budget()


def _fmt_bytes(nbytes: int) -> str:
    if nbytes >= 2**30:
        return f"{nbytes / 2**30:.1f} GiB"
    return f"{nbytes / 2**20:.1f} MiB"


def _logits_bytes(num_rows: int, width: int, data_generation: str, dense: bool) -> int:
    """Peak bytes create_random_logits needs, including its temporaries.

    Budgeting one tensor here would clear a cell that then dies allocating its
    own input. Staircase rows additionally build a [num_rows, width] bool to
    mask each row's tail; equal-length rows have no tail and skip it.
    """
    logits = num_rows * width * 4
    peak = logits * _GEN_PEAK_MULTIPLIER.get(data_generation, 1.0)
    if not dense:
        peak += num_rows * width + width * 4
    return int(peak) + num_rows * 8 + (1 << 20)


def _workspace_bytes(num_rows: int, width: int, top_k: int, decode: bool) -> int:
    """Device scratch the kernel itself claims -- 12.5% of the logits, so 8 GiB
    at the top of the sweep. Decode always takes the one-block path."""
    if not decode and topk_use_mulblocks(num_rows, width):
        return int(topk_mb_workspace_size(num_rows, width, top_k, False))
    return int(topk_ob_workspace_size(num_rows, width, top_k, False))


def _degenerate(width: int, top_k: int) -> bool:
    """k >= width selects the whole row, so the kernel short-circuits and the
    compulsory-traffic model stops describing it -- the reported TB/s exceeds
    what HBM can deliver. Flagged rather than dropped: a missing row reads as
    untested."""
    return top_k >= width


def _traffic_bytes(num_rows: int, width: int, top_k: int, write_values: bool) -> int:
    """Compulsory traffic: every logit read once, every output written once.
    A lower bound -- radix select makes several passes over the logits."""
    per_out = 8 if write_values else 4
    return num_rows * width * 4 + num_rows * top_k * per_out


def _perf_columns(
    ret: dict,
    us: float,
    num_rows: int,
    width: int,
    top_k: int,
    write_values: bool = False,
    torch_us: float | None = None,
) -> None:
    ret["us"] = us
    ret["TB/s"] = _traffic_bytes(num_rows, width, top_k, write_values) / us / 1e6
    ret["Gelem/s"] = num_rows * width / us / 1e3
    ret["degenerate"] = _degenerate(width, top_k)
    if torch_us is not None:
        ret["torch us"] = torch_us
        ret["speedup"] = torch_us / us


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
    # int(t.max()) is one sync; Python's max(t) is one per row.
    width = physical_width if physical_width is not None else int(row_ends.max())
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

    # Equal-length rows have no tail; the broadcast would otherwise build a bool
    # the size of the logits (16 GiB at M=16384, N=1048576) to fill nothing.
    row_ends = row_ends.to(logits.device)
    if int(row_ends.min()) < width:
        col = torch.arange(width, device=logits.device, dtype=torch.int32).unsqueeze(0)
        logits.masked_fill_(col >= row_ends.unsqueeze(1), float("-inf"))
    return logits


def create_row_boundaries(
    num_rows: int, num_prefix: int = 0, top_k: int = 2048, dense_n: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row start and end indices for testing.

    Default: the chunked-prefill staircase, row i covering [0, num_prefix+i+1),
    so the row length varies down the tensor.

    dense_n: every row covers the same [0, dense_n). This is the plain
    rectangular [M, N] top-k an indexer hands to its top-k stage, where M and N
    are independent axes rather than N being a function of M.
    """
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    if dense_n is not None:
        row_ends = torch.full((num_rows,), dense_n, device="cuda", dtype=torch.int32)
    else:
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
    """Compare results from the CUDA top_k_per_row kernels with torch.topk.

    Index sets are deliberately NOT compared directly: when logits tie, the
    kernel and torch may pick different -- equally correct -- indices. What has
    to match is the multiset of *selected values*, so that is what is compared,
    after both sides are sorted descending.

    Fully vectorised: a per-row Python loop here costs more than the kernel it
    checks once num_rows reaches the thousands, and it forces one device sync
    per row.
    """
    num_rows, width = logits.shape
    device = logits.device

    row_lens = (row_ends - row_starts).to(torch.int64).to(device)
    num_valid = row_lens.clamp(min=0, max=top_k).unsqueeze(1)
    pos = torch.arange(top_k, device=device).unsqueeze(0)
    valid = pos < num_valid

    cuda_idx = cuda_indices.to(torch.int64)[:, :top_k]
    # Out of range in a real slot is a bug, not a tie -- and indexing would raise.
    if bool((((cuda_idx < 0) | (cuda_idx >= width)) & valid).any()):
        return False

    torch_idx = torch_indices.to(torch.int64)
    if torch_idx.shape[1] < top_k:
        torch_idx = torch.cat(
            [
                torch_idx,
                torch.full(
                    (num_rows, top_k - torch_idx.shape[1]),
                    -1,
                    dtype=torch.int64,
                    device=device,
                ),
            ],
            dim=1,
        )
    else:
        torch_idx = torch_idx[:, :top_k]

    neg_inf = float("-inf")
    cuda_vals = torch.gather(logits, 1, cuda_idx.clamp(0, width - 1))
    cuda_vals = torch.where(valid, cuda_vals, neg_inf)
    torch_vals = torch.gather(logits, 1, torch_idx.clamp(0, width - 1))
    torch_vals = torch.where(valid & (torch_idx >= 0), torch_vals, neg_inf)

    # allclose treats -inf == -inf as equal, so padding lines up either side.
    if not torch.allclose(
        cuda_vals.sort(dim=1, descending=True).values,
        torch_vals.sort(dim=1, descending=True).values,
        rtol=tolerance,
        atol=tolerance,
    ):
        return False

    # A key selected twice slips past the value check when it ties what it displaced.
    if top_k > 1:
        packed = torch.where(valid, cuda_idx, -1).sort(dim=1).values
        if bool(((packed[:, 1:] == packed[:, :-1]) & (packed[:, 1:] >= 0)).any()):
            return False

    if stable:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        if bool(((cuda_idx[:, 1:] < cuda_idx[:, :-1]) & pair_valid).any()):
            return False

    if values is not None:
        written = cuda_indices >= 0
        gathered = torch.gather(
            logits,
            1,
            cuda_indices.clamp_min(0).to(torch.int64),
        )
        if not torch.equal(values[written], gathered[written]):
            return False
        if not bool(torch.all(torch.isneginf(values[~written]))):
            return False

    return True


def _prefill_kernel(
    logits, row_starts, row_ends, indices, values, num_rows, stride_row, stride_col, k
):
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
    )


def _decode_kernel(
    logits,
    next_n,
    seqLens,
    indices,
    numRows,
    stride0,
    stride1,
    fast,
    k=2048,
    flydsl=False,
    stable=False,
    values=None,
):
    """Dispatch one of the three decode top-k backends.

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


def _reference_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    time_it: bool,
) -> tuple[torch.Tensor | None, float | None]:
    """torch.topk reference, masked to each row's own window.

    Returns (indices, us). Either may be None: the caller decides whether the
    shape is worth a reference at all, and torch.topk itself can run out of
    memory on shapes where the kernel under test does not (it materialises a
    values tensor plus its own select workspace).
    """
    k = min(top_k, int(row_ends.max()))
    try:
        if time_it:
            out, us = run_perftest(
                torch.topk,
                logits,
                k,
                dim=-1,
                num_iters=_REF_ITERS,
                num_warmup=1,
                num_rotate_args=1,
            )
            torch_indices = out[1]
        else:
            torch_indices = logits.topk(k, dim=-1)[1]
            us = None
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None

    mask = (torch_indices >= 0) & (
        (torch_indices - (row_ends - row_starts)[:, None]) < 0
    )
    return torch_indices.masked_fill(~mask, -1), us


def _prefill_footprint(num_rows, width, top_k, data_generation, dense):
    # top_k * 12: this kernel's indices, plus topk_plain's own ids and values.
    return (
        _logits_bytes(num_rows, width, data_generation, dense)
        + num_rows * top_k * 12
        + _workspace_bytes(num_rows, width, top_k, decode=False)
    )


def _rows_that_fit(num_rows, width, top_k, data_generation):
    """Largest row count whose dense cell fits, 0 if even one row does not.

    The footprint is monotonic in rows, so bisect it rather than modelling it.
    """
    if _prefill_footprint(1, width, top_k, data_generation, True) > _free_budget():
        return 0
    lo, hi = 1, num_rows
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if (
            _prefill_footprint(mid, width, top_k, data_generation, True)
            <= _free_budget()
        ):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _time_topk_plain(logits, num_rows, width, top_k, footprint):
    """The other entry point for a plain [M, N] top-k. It needs a values buffer
    even when only indices are wanted, which is itself part of the comparison."""
    ids = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    vals = torch.empty((num_rows, top_k), dtype=torch.float32, device="cuda")
    empty = torch.tensor([], dtype=torch.int32, device="cuda")
    _, us = run_perftest(
        topk_plain,
        logits,
        ids,
        vals,
        top_k,
        True,
        empty,
        empty,
        -1,
        1,
        num_rotate_args=_rotate_args(footprint),
    )
    return us, ids


def _run_prefill_block(
    num_rows, num_prefix, top_k, data_generation, dense_n, width, with_plain=False
):
    """One [num_rows, width] cell: time the kernel, check it against torch.topk."""
    row_starts, row_ends = create_row_boundaries(num_rows, num_prefix, dense_n=dense_n)
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, data_generation
    )
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    footprint = _prefill_footprint(
        num_rows, width, top_k, data_generation, dense_n is not None
    )

    _, us = run_perftest(
        _prefill_kernel,
        logits,
        row_starts,
        row_ends,
        indices,
        None,  # values
        num_rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
        num_rotate_args=_rotate_args(footprint),
    )

    run_ref = num_rows * width <= _REF_MAX_ELEMS
    torch_indices, torch_us = (
        _reference_topk(logits, row_starts, row_ends, top_k, time_it=True)
        if run_ref
        else (None, None)
    )
    plain_us = None
    plain_ids = None
    if with_plain and top_k <= _PLAIN_MAX_K:
        plain_us, plain_ids = _time_topk_plain(
            logits, num_rows, width, top_k, footprint
        )

    if torch_indices is None:
        ok = "skipped"
        note = "ref oom" if run_ref else f"ref > {_REF_MAX_ELEMS} elems"
    else:
        ok = compare_topk_results(
            logits, indices, torch_indices, row_starts, row_ends, top_k
        )
        if plain_ids is not None and ok is True:
            ok = compare_topk_results(
                logits, plain_ids, torch_indices, row_starts, row_ends, top_k
            )
            if ok is not True:
                ok = "plain mismatch"
        note = None
    del logits, indices, torch_indices
    return us, ok, torch_us, note, plain_us


@benchmark()
def test_top_k_per_row_prefill(
    num_rows: int,
    num_prefix: int,
    top_k: int,
    data_generation: str = "random",
    dense_n: int | None = None,
    chunk_m: bool = False,
    with_plain: bool = False,
) -> dict:
    """
    Test topk_per_row_prefill.
    """
    ret = {}
    torch.set_default_device("cuda:0")

    width = dense_n if dense_n is not None else num_prefix + num_rows
    ret["context_len"] = width
    dense = dense_n is not None
    footprint = _prefill_footprint(num_rows, width, top_k, data_generation, dense)

    rows_per_chunk = num_rows
    if not _fits_in_memory(footprint):
        # The op's own answer to a cell that does not fit is to cut M and run it
        # in pieces, so offer that rather than only reporting the shape as lost.
        # Chunking is dense-only: staircase row bounds are a function of the row
        # index, so a chunk of them is a different problem.
        rows_per_chunk = (
            _rows_that_fit(num_rows, width, top_k, data_generation)
            if chunk_m and dense
            else 0
        )
        if rows_per_chunk == 0:
            ret["all_close"] = "skipped"
            ret["note"] = f"needs {_fmt_bytes(footprint)}"
            ret["degenerate"] = _degenerate(width, top_k)
            return ret

    total_us = 0.0
    total_torch_us = 0.0
    total_plain_us = 0.0
    timed_torch = True
    timed_plain = with_plain
    verdicts = []
    note = None
    done = 0
    while done < num_rows:
        rows = min(rows_per_chunk, num_rows - done)
        us, ok, torch_us, chunk_note, plain_us = _run_prefill_block(
            rows, num_prefix, top_k, data_generation, dense_n, width, with_plain
        )
        total_us += us
        if torch_us is None:
            timed_torch = False
        else:
            total_torch_us += torch_us
        if plain_us is None:
            timed_plain = False
        else:
            total_plain_us += plain_us
        verdicts.append(ok)
        note = note or chunk_note
        done += rows

    chunks = len(verdicts)
    if any(v == "skipped" for v in verdicts):
        ret["all_close"] = "skipped"
    else:
        ret["all_close"] = all(verdicts)
    if note:
        ret["note"] = note
    if chunks > 1:
        ret["chunks"] = chunks
        ret["note"] = f"M split {chunks}x{rows_per_chunk}"

    _perf_columns(
        ret,
        total_us,
        num_rows,
        width,
        top_k,
        torch_us=total_torch_us if timed_torch else None,
    )
    if with_plain and top_k > _PLAIN_MAX_K:
        ret["plain us"] = f"n/a k>{_PLAIN_MAX_K}"
    elif timed_plain:
        ret["plain us"] = total_plain_us
        ret["plain/aiter"] = total_plain_us / total_us
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
    num_rows = batch_size * next_n
    width = max(context_len, top_k) if flydsl else context_len
    ret["width"] = width

    footprint = (
        _logits_bytes(num_rows, width, data_generation, dense=False)
        + num_rows * top_k * (8 if write_values else 4)
        + _workspace_bytes(num_rows, width, top_k, decode=True)
    )
    if not _fits_in_memory(footprint):
        ret["all_close"] = "skipped"
        ret["note"] = f"needs {_fmt_bytes(footprint)}"
        ret["degenerate"] = _degenerate(width, top_k)
        ret["fast"] = fast
        return ret

    # Create test data
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
        physical_width=width if flydsl else None,
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    values = (
        torch.empty((num_rows, top_k), dtype=torch.float32, device="cuda")
        if write_values
        else None
    )

    # Run the kernel
    _, us = run_perftest(
        _decode_kernel,
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
        num_rotate_args=_rotate_args(footprint),
    )

    torch.cuda.synchronize()

    # Run reference implementation
    run_ref = num_rows * width <= _REF_MAX_ELEMS
    torch_indices, torch_us = (
        _reference_topk(logits, row_starts, row_ends, top_k, time_it=True)
        if run_ref
        else (None, None)
    )

    # Compare results
    if torch_indices is None:
        ret["all_close"] = "skipped"
        ret["note"] = "ref oom" if run_ref else f"ref > {_REF_MAX_ELEMS} elems"
    else:
        ret["all_close"] = compare_topk_results(
            logits,
            indices,
            torch_indices,
            row_starts,
            row_ends,
            top_k,
            stable=stable,
            values=values,
        )

    _perf_columns(
        ret, us, num_rows, width, top_k, write_values=write_values, torch_us=torch_us
    )
    ret["fast"] = fast
    return ret


def test_compare_topk_results():
    """Guard the comparator itself.

    It decides every `all_close` in both tables, and it has to accept a
    different-but-equally-correct tie break while still rejecting a genuinely
    wrong pick -- a comparator that answers True too easily makes the whole
    file report success it has not verified.
    """
    i32 = torch.int32
    dev = "cuda"

    def idx(rows):
        return torch.tensor(rows, dtype=i32, device=dev)

    def bounds(num_rows, length):
        return (
            torch.zeros(num_rows, dtype=i32, device=dev),
            torch.full((num_rows,), length, dtype=i32, device=dev),
        )

    descending = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], device=dev)
    tied_top = torch.tensor([[5.0, 5.0, 3.0, 2.0, 1.0]], device=dev)
    tied_kth = torch.tensor([[9.0, 5.0, 5.0, 2.0, 1.0]], device=dev)
    s1, e1 = bounds(1, 5)

    short = torch.tensor([[5.0, 4.0, float("-inf"), float("-inf")]], device=dev)
    s_short, e_short = bounds(1, 2)

    two_rows = torch.tensor([[5.0, 4.0, 3.0], [9.0, 8.0, 7.0]], device=dev)
    s2, e2 = bounds(2, 3)

    cases = [
        ("identical picks", descending, idx([[0, 1]]), idx([[0, 1]]), s1, e1, 2, True),
        # Ties: both picks are correct, so the comparator must not care which.
        ("tie at the top", tied_top, idx([[0, 1]]), idx([[1, 0]]), s1, e1, 2, True),
        ("tie at slot k", tied_kth, idx([[0, 1]]), idx([[0, 2]]), s1, e1, 2, True),
        (
            "short row, -1 padded",
            short,
            idx([[0, 1, -1, -1]]),
            idx([[0, 1, -1, -1]]),
            s_short,
            e_short,
            4,
            True,
        ),
        (
            "smaller value picked",
            descending,
            idx([[0, 3]]),
            idx([[0, 1]]),
            s1,
            e1,
            2,
            False,
        ),
        ("duplicate index", descending, idx([[0, 0]]), idx([[0, 1]]), s1, e1, 2, False),
        (
            "out-of-range index",
            descending,
            idx([[0, 99]]),
            idx([[0, 1]]),
            s1,
            e1,
            2,
            False,
        ),
        (
            "one row of two wrong",
            two_rows,
            idx([[0, 1], [0, 2]]),
            idx([[0, 1], [0, 1]]),
            s2,
            e2,
            2,
            False,
        ),
    ]

    for name, logits, got, want_idx, starts, ends, k, expected in cases:
        assert (
            compare_topk_results(logits, got, want_idx, starts, ends, k) is expected
        ), f"compare_topk_results: {name} should be {expected}"
    print(f"[compare_topk_results] PASS: {len(cases)} comparator cases")


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
    max_end = int(row_ends.max())
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


# CI runs this file bare, so the no-flag defaults stay small. --sweep swaps in
# the grid; an axis given on the command line wins over both.
CI_DEFAULTS = {
    "context_len": [8, 128, 1024, 3072, 4096, 8192, 16384, 32768, 65536, 90000, 128000],
    "top_k": [512, 1024, 2048],
    "num_prefix": [0],
    "decode_batch_size": [4, 8, 16, 24],
    "next_n": [1, 2, 3, 4],
    "prefill_rows": [1, 4, 16, 64, 256, 1024, 4096, 16384],
    "prefill_n": [512, 1024, 4096, 16384, 65536, 262144, 1048576],
}

# The grid is powers of two so the cells are comparable; these are not, which
# is the point. Odd M, a prime N, and N=5144 (not a multiple of 16) would all
# pass unexercised otherwise. Run as a correctness pass, not a perf surface.
IRREGULAR_SHAPES = [
    (3, 1000, 512),
    (5, 5144, 1024),
    (7, 100003, 2048),
    (1, 513, 512),
    (17, 65537, 4096),
    (100, 999983, 1024),
]

_POW2_N = [512 << i for i in range(12)]  # 512 .. 1048576
_POW2_M = [1 << i for i in range(15)]  # 1 .. 16384

SWEEP_DEFAULTS = {
    **CI_DEFAULTS,
    "context_len": _POW2_N,
    "top_k": [512, 1024, 2048, 4096],
    "decode_batch_size": _POW2_M,
    "next_n": [1],
    "prefill_rows": _POW2_M,
    "prefill_n": _POW2_N,
}


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-c",
    "--context_len",
    type=int,
    default=None,
    nargs="+",
    help="""number of kv. In --sweep this is the decode table's N axis.
    e.g.: -c 64""",
)

parser.add_argument(
    "-k",
    "--top_k",
    type=int,
    default=None,
    nargs="+",
    help="""top-k elements per row. The radix backend supports any positive
    int; the `_fast` ASM-kernel path only supports 2048 and is skipped
    for other values.
    e.g.: -k 512 1024 2048""",
)

parser.add_argument(
    "--num_prefix",
    type=int,
    default=None,
    nargs="+",
    help="""staircase prefill offset: row i covers [0, num_prefix+i+1).
    Ignored in --dense / --sweep, where rows are equal length.
    e.g.: --num_prefix 8000 16000 24000 32000 40000 48000 56000""",
)

parser.add_argument(
    "-b",
    "--decode_batch_size",
    type=int,
    default=None,
    nargs="+",
    help="""decode_batch_size batch size.
    e.g.: -b 4""",
)

parser.add_argument(
    "-n",
    "--next_n",
    type=int,
    default=None,
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

parser.add_argument(
    "-M",
    "--prefill_rows",
    type=int,
    default=None,
    nargs="+",
    help="""Dense prefill M axis (query rows); used with --dense / --sweep.
    e.g.: -M 1 1024 16384""",
)

parser.add_argument(
    "-N",
    "--prefill_n",
    type=int,
    default=None,
    nargs="+",
    help="""Dense prefill N axis (kv candidates per row); used with --dense /
    --sweep. e.g.: -N 4096 1048576""",
)

parser.add_argument(
    "--dense",
    action="store_true",
    help="""Prefill over equal-length rows -- the rectangular [M, N] an indexer
    hands its top-k stage -- instead of the chunked-prefill staircase, making M
    and N independent axes. Implied by --sweep.""",
)

parser.add_argument(
    "--chunk_m",
    action="store_true",
    help="""When a dense cell does not fit, split M into pieces that do and run
    them in sequence, rather than reporting the shape as skipped. The row is
    marked with its split, because the reported time is a sum over chunks and
    not a single launch.""",
)

parser.add_argument(
    "--with_plain",
    action="store_true",
    help="""Also time aiter.topk_plain on the same logits and check it against
    the same reference, as a second column in the prefill table.""",
)

parser.add_argument(
    "--skip_irregular",
    action="store_true",
    help="""Drop the non-power-of-two correctness pass that --sweep otherwise
    appends to the prefill table.""",
)

parser.add_argument(
    "--sweep",
    action="store_true",
    help="""Sweep the M x N x top_k grid instead of the small CI defaults, and
    drop the decode stable/values/flydsl cross-product so one cell is one
    kernel. Expect minutes, not seconds.""",
)

parser.add_argument(
    "--ref_max_elems",
    type=int,
    default=_REF_MAX_ELEMS,
    help="""Skip the torch.topk reference above this M*N (it is the slowest
    thing in the sweep); those rows report all_close=skipped.""",
)

parser.add_argument(
    "--mem_headroom",
    type=float,
    default=0.8,
    help="""Fraction of free HBM a case may claim. Cases over budget are
    reported as skipped rather than taking the run down with an OOM.""",
)

args = parser.parse_args()

_axis_defaults = SWEEP_DEFAULTS if args.sweep else CI_DEFAULTS
for _axis, _default in _axis_defaults.items():
    if getattr(args, _axis) is None:
        setattr(args, _axis, _default)

dense = args.dense or args.sweep
_REF_MAX_ELEMS = args.ref_max_elems
_MEM_HEADROOM = args.mem_headroom

# Regressions that run in CI via `python3 <file>`.
test_compare_topk_results()
test_mb_workspace_reuse()


df = []
for data_generation in args.data_generation:
    for k in args.top_k:
        if dense:
            for m in args.prefill_rows:
                for n in args.prefill_n:
                    if k > n:
                        continue
                    df.append(
                        test_top_k_per_row_prefill(
                            m,
                            0,
                            k,
                            data_generation,
                            dense_n=n,
                            chunk_m=args.chunk_m,
                            with_plain=args.with_plain,
                        )
                    )
        else:
            for m in args.context_len:
                for num_prefix in args.num_prefix:
                    df.append(
                        test_top_k_per_row_prefill(m, num_prefix, k, data_generation)
                    )

if args.sweep and not args.skip_irregular:
    for m, n, k in IRREGULAR_SHAPES:
        df.append(
            test_top_k_per_row_prefill(
                m,
                0,
                k,
                args.data_generation[0],
                dense_n=n,
                chunk_m=args.chunk_m,
                with_plain=args.with_plain,
            )
        )

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("topk_per_row_prefill summary (markdown):\n%s", df_md)


df = []
flydsl_available = get_gfx() in ("gfx942", "gfx950")
for data_generation in args.data_generation:
    for m in args.decode_batch_size:
        for ctx in args.context_len:
            for k in args.top_k:
                for n in args.next_n:
                    # k > ctx exercises the -1 padding; not a perf cell.
                    if args.sweep and k > ctx:
                        continue
                    if args.sweep:
                        df.append(
                            test_top_k_per_row_decode(m, ctx, k, n, data_generation)
                        )
                        continue
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
                            if flydsl_available:
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
