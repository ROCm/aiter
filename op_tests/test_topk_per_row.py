import argparse

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, perftest


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    data_generation: str = "random",
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        logits = torch.randn(
            row_starts.shape[0], max(row_ends), dtype=dtype, device="cuda"
        )
    elif data_generation == "10LSBits" or data_generation == "mixed":
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        # Generate random bits for the last 10 bits
        random_bottom_bits = torch.randint(
            0,
            2**10,
            (row_starts.shape[0], max(row_ends)),
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
            row_starts.shape[0], max(row_ends), dtype=dtype, device="cuda"
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
) -> None:
    """
    Run the top_k_per_row kernel.
    """
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
) -> None:
    """
    Run the top_k_per_row kernel.

    Note: the `_fast` ASM-kernel variant has `kTopK=2048` baked into its
    precompiled `.co`; it ignores any caller-supplied `k`. The dispatch
    here only allows `_fast` when k == 2048.
    """
    if fast:
        assert k == 2048, "top_k_per_row_decode_fast only supports k=2048"
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
        )


@benchmark()
def test_top_k_per_row_prefill(
    num_rows: int, num_prefix: int, top_k: int, data_generation: str = "random"
) -> dict:
    """
    Test topk_per_row_prefill.
    """
    ret = {}
    torch.set_default_device("cuda:0")

    # Create test data
    row_starts, row_ends = create_row_boundaries(num_rows, num_prefix)
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, data_generation
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    torch.empty((num_rows, top_k), dtype=torch.float32, device="cuda").fill_(0)

    # Run the kernel
    _, us = run_top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        None,  # values
        num_rows,
        logits.stride(0),
        logits.stride(1),
        k=top_k,
    )

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    all_close = compare_topk_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
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
        row_starts, row_ends, torch.float32, 42, data_generation
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

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
        logits, indices, torch_indices, row_starts, row_ends, top_k
    )

    # measure performance
    ret["context_len"] = logits.shape[1]
    ret["all_close"] = all_close
    ret["us"] = us
    ret["fast"] = fast
    return ret


def _decode_row_ends_logits(row_ends: torch.Tensor, width: int, seed: int):
    """Logits whose out-of-range tail is *attractive*, plus the -inf reference.

    Padding the tail with -inf (as create_random_logits does) makes a too-large
    row bound invisible: the kernel would scan the tail, pick nothing from it,
    and still match. Here the tail holds the largest values in the row, so a
    bound that overshoots pulls them in and the comparison fails. The reference
    is the same tensor with the tail masked to -inf, i.e. the answer for a
    kernel that respects the bound exactly.
    """
    torch.manual_seed(seed)
    logits = torch.randn(row_ends.shape[0], width, dtype=torch.float32, device="cuda")
    cols = torch.arange(width, device="cuda")
    out_of_range = cols[None, :] >= row_ends[:, None]
    logits = logits + out_of_range * 100.0
    return logits, logits.masked_fill(out_of_range, float("-inf"))


def test_decode_row_ends_per_row():
    """Regression for `top_k_per_row_decode(..., rowEndsPerRow=...)`.

    `seqLens` is per-SEQUENCE and bounds row r by
    `seqLens[r // next_n] - next_n + (r % next_n) + 1` -- one KV row per query
    token. A caller whose index cache packs several tokens into one entry
    (DeepSeek-V4 CSA) computes its own per-ROW ends and passes them here; the
    C++ side routes them into the rowEnds slot and forces next_n = 1, which
    collapses that expression to `rowEnds[r]`.

    So this asserts three things:
      1. the ends are honoured verbatim, for ends no per-sequence seqLens could
         produce (irregular, non-monotonic);
      2. `seqLens` and `next_n` are ignored -- both are passed deliberately
         wrong, so dropping the `next_n = 1` collapse silently reverts every
         bound to the one-KV-row-per-token rule and this fails;
      3. it selects the same elements as the seqLens path when the ends agree.
    """
    top_k, width = 512, 4096
    # Short rows (end <= top_k) and long rows straddle the kernel's two code
    # paths; the ends are deliberately non-monotonic and not a fixed stride
    # apart, so no (seqLens, next_n) pair could generate them.
    row_ends = torch.tensor(
        [1, 2048, 17, 513, 4096, 128, 3000, 511, 2049, 512, 4095, 1000],
        dtype=torch.int32,
        device="cuda",
    )
    num_rows = row_ends.numel()
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    logits, logits_ref = _decode_row_ends_logits(row_ends, width, seed=7)

    # Wrong on purpose, and in range: if a regression made the kernel read
    # these, the bound would be wrong rather than out of bounds -- a mismatch,
    # not a segfault.
    bogus_seq_lens = torch.full((num_rows,), width, dtype=torch.int32, device="cuda")
    bogus_next_n = 3

    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    aiter.top_k_per_row_decode(
        logits,
        bogus_next_n,
        bogus_seq_lens,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        k=top_k,
        rowEndsPerRow=row_ends,
    )
    torch.cuda.synchronize()

    # A row with end <= top_k takes the kernel's fill path, which emits the
    # identity 0..end-1 padded with -1 -- independent of the values, so it
    # pins the bound exactly. Comparing the whole row matters: an overshooting
    # bound keeps the same leading entries and only shows up in the padding.
    for row_idx in range(num_rows):
        end = int(row_ends[row_idx])
        if end > top_k:
            continue
        expected = torch.full((top_k,), -1, dtype=torch.int32, device="cuda")
        expected[:end] = torch.arange(end, dtype=torch.int32, device="cuda")
        assert torch.equal(indices[row_idx], expected), (
            f"[row_ends_per_row] row {row_idx} (end={end}) took the fill path "
            f"but did not emit 0..{end - 1} padded with -1; got "
            f"{indices[row_idx][: end + 4].tolist()}"
        )

    # Long rows run the real radix top-k, so compare against torch on the
    # masked logits.
    ref = logits_ref.topk(min(top_k, int(row_ends.max())), dim=-1)[1]
    assert compare_topk_results(
        logits_ref, indices, ref, row_starts, row_ends, top_k
    ), "[row_ends_per_row] per-row ends did not match torch.topk over the bounded region"

    # Same bound expressed both ways must select the same elements. Here the
    # ends DO follow the per-sequence rule, so the seqLens path is the oracle.
    # Compared as sets, not bytes: the kernel returns a row in index order
    # rather than value order, and its tail is not stable across launches, so
    # even two identical calls differ elementwise. That is what
    # compare_topk_results is set-based for.
    next_n, batch_size, context_len = 4, 3, 2600
    eq_rows = batch_size * next_n
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device="cuda")
    rows = torch.arange(eq_rows, device="cuda")
    eq_ends = (seq_lens[rows // next_n] - next_n + (rows % next_n) + 1).to(torch.int32)
    eq_starts = torch.zeros(eq_rows, dtype=torch.int32, device="cuda")
    eq_logits, eq_logits_ref = _decode_row_ends_logits(eq_ends, width, seed=13)

    idx_seq_lens = torch.empty((eq_rows, top_k), dtype=torch.int32, device="cuda")
    idx_per_row = torch.empty((eq_rows, top_k), dtype=torch.int32, device="cuda")
    # The per-row call still gets a wrong seqLens, so it can only agree by
    # actually reading eq_ends.
    wrong_seq_lens = torch.full((batch_size,), width, dtype=torch.int32, device="cuda")
    for out, lens, extra in (
        (idx_seq_lens, seq_lens, {}),
        (idx_per_row, wrong_seq_lens, {"rowEndsPerRow": eq_ends}),
    ):
        aiter.top_k_per_row_decode(
            eq_logits,
            next_n,
            lens,
            out,
            eq_rows,
            eq_logits.stride(0),
            eq_logits.stride(1),
            k=top_k,
            **extra,
        )
    torch.cuda.synchronize()
    assert compare_topk_results(
        eq_logits_ref, idx_per_row, idx_seq_lens, eq_starts, eq_ends, top_k
    ), (
        "[row_ends_per_row] per-row ends selected different elements than the "
        "equivalent seqLens path -- the next_n = 1 collapse no longer holds"
    )
    print(
        f"[row_ends_per_row] PASS: {num_rows} irregular per-row ends honoured "
        f"verbatim (bogus seqLens/next_n={bogus_next_n} ignored), and "
        f"{eq_rows} rows agreed with the seqLens path"
    )


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

# Correctness regressions (run in CI via `python3 <file>`).
test_mb_workspace_reuse()
test_decode_row_ends_per_row()


df = []
for data_generation in args.data_generation:
    for m in args.context_len:
        for k in args.top_k:
            for num_prefix in args.num_prefix:
                ret = test_top_k_per_row_prefill(m, num_prefix, k, data_generation)
                df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("topk_per_row_prefill summary (markdown):\n%s", df_md)


df = []
for data_generation in args.data_generation:
    for m in args.decode_batch_size:
        for ctx in args.context_len:
            for k in args.top_k:
                for n in args.next_n:
                    ret = test_top_k_per_row_decode(
                        m, ctx, k, n, data_generation, False
                    )
                    df.append(ret)
                    # `_fast` ASM kernel hardcodes k=2048; skip otherwise.
                    if get_gfx() == "gfx942" and k == 2048:
                        ret = test_top_k_per_row_decode(
                            m, ctx, k, n, data_generation, True
                        )
                        df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("topk_per_row_decode summary (markdown):\n%s", df_md)
