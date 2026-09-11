import argparse

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, run_perftest

# Above this footprint, run_perftest's automatic argument rotation (deep copies
# of every input, to defeat L2) costs GiBs and buys nothing: a multi-GiB logits
# tensor already blows past a 4 MB L2 on its first pass.
_ROTATE_MAX_BYTES = 256 << 20

# torch.topk is the correctness reference, but it is also the slowest thing in
# the sweep: past a couple of G elements a single call runs for seconds. Beyond
# this the reference is skipped and `all_close` reports "skipped" rather than
# silently claiming a pass.
_REF_MAX_ELEMS = 1 << 31

# The reference only needs a stable number, not a tight one.
_REF_ITERS = 5

# Fraction of free HBM one case may claim before it is reported as skipped.
_MEM_HEADROOM = 0.8

# Peak allocation of create_random_logits, in units of the fp32 logits itself.
# Budgeting 1x here would clear a cell that then dies allocating its own input,
# which is the failure the guard exists to prevent.
#   random    1x  -- one randn
#   10LSBits  3x  -- randint, the masked temporary, and the or'd result
#   mixed     4x  -- the above, plus a second randn and the torch.where result
# Measured at M=64, N=1048576 with torch.cuda.max_memory_allocated: 1.03x,
# 3.00x, 4.28x (the remainder is the boundary mask, accounted for separately).
_GEN_PEAK_MULTIPLIER = {"random": 1.0, "10LSBits": 3.0, "mixed": 4.3}


def _logits_bytes(num_rows: int, width: int, data_generation: str, dense: bool) -> int:
    """Peak bytes create_random_logits needs for a [num_rows, width] fp32 tensor.

    Staircase rows additionally materialise a [num_rows, width] bool mask to
    fill each row's tail; equal-length rows have no tail and skip it.
    """
    logits = num_rows * width * 4
    peak = logits * _GEN_PEAK_MULTIPLIER.get(data_generation, 1.0)
    if not dense:
        # The broadcast comparison materialises a bool the size of the logits,
        # over an int32 column index.
        peak += num_rows * width + width * 4
    # Row bounds, plus slack for the allocator's block rounding.
    return int(peak) + num_rows * 8 + (1 << 20)


def _rotate_args(nbytes: int) -> int:
    """0 lets run_perftest size the rotation itself; 1 pins it to a single set."""
    return 1 if nbytes > _ROTATE_MAX_BYTES else 0


def _fits_in_memory(nbytes: int, headroom: float) -> bool:
    """Whether an allocation of nbytes leaves the device some slack.

    The sweep reaches shapes that cannot fit ([16384, 1048576] fp32 alone is
    64 GiB), and a raw OOM would kill the whole run rather than the one cell.
    """
    free, _total = torch.cuda.mem_get_info()
    return nbytes <= free * headroom


def _fmt_bytes(nbytes: int) -> str:
    """GiB loses everything under ~50 MiB, which is most of the CI shapes."""
    if nbytes >= 2**30:
        return f"{nbytes / 2**30:.1f} GiB"
    return f"{nbytes / 2**20:.1f} MiB"


def _traffic_bytes(num_rows: int, width: int, top_k: int, write_values: bool) -> int:
    """Compulsory traffic: every logit read once, every output written once.

    The radix kernels make several passes over the logits, so achieved HBM
    traffic is higher than this. The column is therefore a lower bound, useful
    for comparing shapes against each other rather than against peak HBM.
    """
    per_out = 8 if write_values else 4
    return num_rows * width * 4 + num_rows * top_k * per_out


def _degenerate(width: int, top_k: int) -> bool:
    """k >= width selects the whole row, so the kernel short-circuits.

    These cells are legal input but not a meaningful bandwidth point: with no
    selection left to do, `us` collapses and the compulsory-traffic model stops
    describing what the kernel read, which shows up as a TB/s above what the
    HBM can deliver. Flag them rather than drop them -- silently missing rows
    read as untested.
    """
    return top_k >= width


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
    # int(t.max()) is one device sync; Python's max(t) iterates the tensor
    # and syncs once per row, which dominates the kernel at large num_rows.
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

    # Mask each row's tail past its own row_end in one shot. Equal-length rows
    # have no tail, and the broadcast would otherwise materialise a bool tensor
    # the size of the logits (16 GiB at M=16384, N=1048576) only to fill nothing.
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

    # A row contributes min(top_k, row_length) real entries; the rest is padding.
    row_lens = (row_ends - row_starts).to(torch.int64).to(device)
    num_valid = row_lens.clamp(min=0, max=top_k).unsqueeze(1)
    pos = torch.arange(top_k, device=device).unsqueeze(0)
    valid = pos < num_valid

    cuda_idx = cuda_indices.to(torch.int64)[:, :top_k]
    # Out-of-range in a real slot is a kernel bug, not a tie: report it rather
    # than indexing with it (which would raise).
    if bool((((cuda_idx < 0) | (cuda_idx >= width)) & valid).any()):
        return False

    # torch.topk is called with min(top_k, max_row_end) columns, so it can be
    # narrower than top_k; pad it out so both sides share the slot layout.
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

    # allclose treats -inf == -inf as equal, so padded slots line up on both
    # sides and a finite-vs-padding mismatch still fails.
    if not torch.allclose(
        cuda_vals.sort(dim=1, descending=True).values,
        torch_vals.sort(dim=1, descending=True).values,
        rtol=tolerance,
        atol=tolerance,
    ):
        return False

    # The same key selected twice is a bug the value comparison alone can miss
    # whenever the duplicated value also ties with the one it displaced.
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


@benchmark()
def test_top_k_per_row_prefill(
    num_rows: int,
    num_prefix: int,
    top_k: int,
    data_generation: str = "random",
    dense_n: int | None = None,
) -> dict:
    """
    Test topk_per_row_prefill.
    """
    ret = {}
    torch.set_default_device("cuda:0")

    width = dense_n if dense_n is not None else num_prefix + num_rows
    ret["context_len"] = width

    # [16384, 1048576] fp32 is 64 GiB of logits alone. Skip the cell instead of
    # taking the whole sweep down with an OOM.
    footprint = (
        _logits_bytes(num_rows, width, data_generation, dense_n is not None)
        + num_rows * top_k * 4
    )
    if not _fits_in_memory(footprint, _MEM_HEADROOM):
        ret["all_close"] = "skipped"
        ret["note"] = f"needs {_fmt_bytes(footprint)}"
        return ret

    # Create test data
    row_starts, row_ends = create_row_boundaries(num_rows, num_prefix, dense_n=dense_n)
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, data_generation
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    # Run the kernel
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
            logits, indices, torch_indices, row_starts, row_ends, top_k
        )

    _perf_columns(ret, us, num_rows, width, top_k, torch_us=torch_us)
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

    # Decode rows differ by the next_n offset, so the mask is always built.
    footprint = _logits_bytes(
        num_rows, width, data_generation, dense=False
    ) + num_rows * top_k * (8 if write_values else 4)
    if not _fits_in_memory(footprint, _MEM_HEADROOM):
        ret["all_close"] = "skipped"
        ret["note"] = f"needs {_fmt_bytes(footprint)}"
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
        # Real kernel bugs, each of which must be rejected.
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


# CI runs this file bare (`python3 <file>`), so the no-flag defaults must stay
# small. --sweep swaps in the M x N x top_k grid the indexer's top-k stage
# actually serves; any axis given explicitly on the command line wins over both.
CI_DEFAULTS = {
    "context_len": [8, 128, 1024, 3072, 4096, 8192, 16384, 32768, 65536, 90000, 128000],
    "top_k": [512, 1024, 2048],
    "num_prefix": [0],
    "decode_batch_size": [4, 8, 16, 24],
    "next_n": [1, 2, 3, 4],
    "prefill_rows": [1, 4, 16, 64, 256, 1024, 4096, 16384],
    "prefill_n": [512, 1024, 4096, 16384, 65536, 262144, 1048576],
}

SWEEP_DEFAULTS = {
    **CI_DEFAULTS,
    "context_len": [512, 1024, 4096, 16384, 65536, 262144, 1048576],
    "top_k": [512, 1024, 2048, 4096],
    "decode_batch_size": [1, 4, 16, 64, 256],
    "next_n": [1, 2],
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

# An axis the user gave explicitly wins; otherwise --sweep picks the grid.
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
                        test_top_k_per_row_prefill(m, 0, k, data_generation, dense_n=n)
                    )
        else:
            for m in args.context_len:
                for num_prefix in args.num_prefix:
                    df.append(
                        test_top_k_per_row_prefill(m, num_prefix, k, data_generation)
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
                    # k > ctx is a short-row case worth testing (it exercises
                    # the -1 padding), just not a perf cell worth a sweep slot.
                    if args.sweep and k > ctx:
                        continue
                    if args.sweep:
                        # One cell, one kernel: the stable/values/flydsl
                        # variants are correctness knobs, not shape axes.
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
