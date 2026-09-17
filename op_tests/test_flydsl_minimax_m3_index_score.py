"""Oracle and shape matrix for the MiniMax-M3 decode index-score kernel.

The oracle is deliberately written straight from the contract rather than
derived from any kernel source -- a reference that shares the implementation's
assumptions cannot catch the implementation's bugs. That is now the only
judge here: the Triton decode scorer this file once cross-checked against has
been deleted from ATOM, the FlyDSL kernel having replaced it outright, so
every candidate below is a FlyDSL config and the oracle is what they answer to.

Run:
    python op_tests/test_flydsl_minimax_m3_index_score.py
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "ATOM"))
sys.path.insert(0, str(ROOT))

from atom.model_ops.minimax_m3.index_topk import SPARSE_BLOCK_SIZE

LOG2E = 1.4426950409
P = SPARSE_BLOCK_SIZE  # 128
D = 128
# Score tensor tolerance, matching test_m3_indexer_context_parallel.py:80.
ATOL = RTOL = 3e-5

# Sentinel for "the kernel is contractually allowed to leave this untouched"
# Using NaN means an accidental write shows up as a diff.
UNWRITTEN = float("nan")


# ---------------------------------------------------------------------------
# Oracle -- transcribed from the contract, rule by rule.
# ---------------------------------------------------------------------------
def score_oracle(idx_q, cache, block_table, seq_lens, S, H, sm_scale, max_block):
    """Reference block scores, computed in fp32 on whatever device holds inputs.

    Returns [H, B*S, max_block] with NaN wherever the contract says the kernel
    need not write (blk >= ceil(seq_len/128)).
    """
    B = seq_lens.shape[0]
    dev = idx_q.device
    out = torch.full((H, B * S, max_block), UNWRITTEN, dtype=torch.float32, device=dev)
    scale = sm_scale * LOG2E

    # Column n = tok*H + head, so a reshape(S, H) recovers the two axes.
    tok_of_col = torch.arange(S, device=dev).repeat_interleave(H)
    pos_in_page = torch.arange(P, device=dev)

    for b in range(B):
        L = int(seq_lens[b])
        nblk = (L + P - 1) // P
        # Rule 1: accumulate in fp32. Rule 2: K is lifted to Q's dtype,
        # not the other way round -- do the cast before going to fp32 so the
        # fp8 -> bf16 rounding is reproduced, not skipped.
        q = idx_q[b * S : (b + 1) * S].reshape(S * H, D)
        # Rule 2: every column carries its own causal cutoff.
        cuts = L - S + tok_of_col + 1  # [S*H]

        for p in range(nblk):
            page = int(block_table[b, p])
            k = cache[page].to(idx_q.dtype).float()  # [P, D]
            z = (k @ q.float().T) * scale  # [P, S*H]
            mask = (p * P + pos_in_page)[:, None] >= cuts[None, :]
            z = z.masked_fill(mask, float("-inf"))
            # Rule 3: a fully masked page still gets written (as -inf).
            out[:, b * S : (b + 1) * S, p] = z.amax(0).reshape(S, H).T

    return out


class SkipCase(Exception):
    """This (shape, config) pair is not legal -- not a failure."""


# Every candidate is a FlyDSL config. The Triton arm that used to sit here is
# gone with the kernel it wrapped; `--impl triton` now reports an unknown impl
# rather than quietly measuring nothing.
CANDIDATES = {}

# Imported hard. This used to be wrapped in `try/except ImportError: pass`
# because the kernel did not exist yet; now it is the only kernel, and
# swallowing the import would leave an empty matrix printing "0/0 passed".
from aiter.ops.flydsl.kernels.minimax_m3_index_score import (  # noqa: E402
    IndexScoreConfig,
    score_flydsl,
    selection_filter,
    shuffle_cache,
)


def _flydsl_variant(cfg):
    """One candidate per tunable config.

    The shuffled layout is an input transform, not a kernel argument, so the
    variant owns it: a kernel compiled with shuffled=True reading a plain cache
    is silently wrong rather than an error, and that is exactly the pairing
    this matrix exists to police.
    """

    def run(idx_q, cache, bt, lens, S, H, sm_scale, mb):
        if not selection_filter(S, H, cfg):
            raise SkipCase(f"illegal for S={S} H={H}")
        k = shuffle_cache(cache) if cfg.shuffled else cache
        return score_flydsl(idx_q, k, bt, lens, S, H, sm_scale, mb, cfg=cfg)

    return run


# Default first, then one knob at a time, then the interactions. Named so a
# failure line says which knob broke.
for _tag, _cfg in [
    ("flydsl", {}),
    ("fly_shuf", {"shuffled": True}),
    ("fly_l2", {"pages_per_wave": 2}),
    ("fly_qlds", {"q_to_lds": True}),
    ("fly_fw2", {"feat_waves": 2}),
    ("fly_fw4", {"feat_waves": 4}),
    ("fly_fw2q", {"feat_waves": 2, "q_to_lds": True}),
    ("fly_tw2", {"token_waves": 2}),
    ("fly_tw4", {"token_waves": 4}),
    ("fly_tw2s", {"token_waves": 2, "shuffled": True}),
    ("fly_tw2l2", {"token_waves": 2, "pages_per_wave": 2}),
    ("fly_fw2tw2", {"feat_waves": 2, "token_waves": 2}),
    ("fly_allon", {"feat_waves": 2, "q_to_lds": True, "shuffled": True,
                   "pages_per_wave": 2}),  # fmt: skip
    ("fly_allon_tw", {"feat_waves": 2, "token_waves": 2, "q_to_lds": True,
                      "shuffled": True, "pages_per_wave": 2}),  # fmt: skip
]:
    CANDIDATES[_tag] = _flydsl_variant(IndexScoreConfig(**_cfg))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_case(B, S, H, lens, cache_dtype, seed=0, q_zero=False, q_slice=False):
    """Build one test case. `lens` is a per-request python list."""
    torch.manual_seed(seed)
    dev = "cuda"
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    max_block = max(1, max((L + P - 1) // P for L in lens))

    if q_slice:
        # A non-contiguous view whose head stride is still
        # that of the full tensor. Any kernel that recomputes strides as H*128
        # instead of reading them from the host breaks exactly here.
        full = torch.randn(B * S, 4, D, dtype=torch.bfloat16, device=dev)
        idx_q = full[:, 1 : 1 + H]
        assert idx_q.stride(0) == 4 * D
    elif q_zero:
        # Forced ties: every score identical, exercising top-k order stability.
        idx_q = torch.zeros(B * S, H, D, dtype=torch.bfloat16, device=dev)
    else:
        idx_q = torch.randn(B * S, H, D, dtype=torch.bfloat16, device=dev)

    # One distinct physical page per (request, logical block), shuffled, so a
    # kernel that confuses logical blk with physical page cannot pass.
    num_pages = B * max_block
    cache = torch.randn(num_pages, P, D, dtype=torch.bfloat16, device=dev)
    if cache_dtype != torch.bfloat16:
        cache = cache.to(cache_dtype)
    block_table = (
        torch.randperm(num_pages, device=dev, dtype=torch.int32)
        .view(B, max_block)
        .contiguous()
    )
    return idx_q, cache, block_table, seq_lens, max_block


def build_matrix(fp8_dtype):
    """The shape matrix. (name, B, S, H, lens, dtype, kwargs)."""
    cases = []
    for dt, tag in ((torch.bfloat16, "bf16"), (fp8_dtype, "fp8")):
        # F = S*H sweep: below one tile, exactly one tile, multiple tiles.
        for S, H in ((1, 1), (4, 1), (4, 4), (8, 4)):
            cases.append((f"F{S*H}_s4096_{tag}", 2, S, H, [4096, 4096], dt, {}))
        # Page boundaries and partial tails.
        for L in (4, 127, 128, 129, 130, 513, 4096, 8192):
            cases.append((f"L{L}_{tag}", 2, 4, 4, [L, L], dt, {}))
        # The case the contract calls out by name: four queries whose causal
        # cutoffs straddle a page boundary (127/128/129/130).
        cases.append((f"causal_straddle_{tag}", 1, 4, 4, [130], dt, {}))
        # Ragged batch: per-request loop bounds must be independent.
        cases.append((f"ragged_{tag}", 4, 4, 4, [130, 4096, 127, 513], dt, {}))
        # The block -> (request, chunk) map is a prefix sum over per-request
        # chunk counts, so these two are its edge cases: a request that
        # contributes zero chunks (its cum entry repeats, and the map must skip
        # it rather than hand it work), and a skew wide enough that the holes
        # outnumber the work by 7:1.
        cases.append((f"ragged_zero_{tag}", 4, 4, 4, [0, 4096, 1, 513], dt, {}))
        cases.append(
            (f"ragged_skew_{tag}", 8, 4, 4, [8192] + [128] * 7, dt, {})
        )  # fmt: skip
        cases.append((f"qzero_tie_{tag}", 2, 4, 4, [513, 513], dt, {"q_zero": True}))
        cases.append((f"qslice_{tag}", 2, 4, 2, [513, 513], dt, {"q_slice": True}))
    return cases


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare(got, ref):
    """Compare honouring the three-way contract: finite / -inf / unwritten."""
    ref_unwritten = torch.isnan(ref)
    ref_ninf = torch.isneginf(ref)
    ref_finite = ~ref_unwritten & ~ref_ninf

    # Rule 3: -inf must be reproduced as -inf, not as a large negative float.
    if not torch.equal(torch.isneginf(got) & ~ref_unwritten, ref_ninf):
        n = int((torch.isneginf(got) & ~ref_unwritten).ne(ref_ninf).sum())
        return False, f"-inf mismatch at {n} slots"

    g, r = got[ref_finite], ref[ref_finite]
    if g.numel() == 0:
        return True, "no finite slots"
    if not torch.isfinite(g).all():
        return False, f"{int((~torch.isfinite(g)).sum())} non-finite in finite region"
    err = (g - r).abs()
    tol = ATOL + RTOL * r.abs()
    if (err > tol).any():
        i = int((err - tol).argmax())
        return False, f"max_err={err.max():.3e} tol={tol.flatten()[i]:.3e}"
    return True, f"max_err={err.max():.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="all")
    ap.add_argument("--filter", default="")
    ap.add_argument("-v", "--verbose", action="store_true", help="list passes too")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    fp8_dtype = torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz

    impls = list(CANDIDATES) if args.impl == "all" else args.impl.split(",")
    print(f"# arch: {arch}")
    print(f"# candidates: {', '.join(impls)}")
    print(f"# tolerance: atol=rtol={ATOL}\n")

    rows, failed, skipped = [], 0, 0
    for name, B, S, H, lens, dt, kw in build_matrix(fp8_dtype):
        if args.filter and args.filter not in name:
            continue
        idx_q, cache, bt, seq_lens, mb = make_case(B, S, H, lens, dt, **kw)
        sm_scale = D**-0.5
        ref = score_oracle(idx_q, cache, bt, seq_lens, S, H, sm_scale, mb)
        for impl in impls:
            try:
                got = CANDIDATES[impl](idx_q, cache, bt, seq_lens, S, H, sm_scale, mb)
                ok, detail = compare(got, ref)
            except SkipCase:
                skipped += 1
                continue
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"EXC {type(e).__name__}: {e}"[:60]
            failed += not ok
            rows.append((name, impl, ok, detail))

    w = max(len(r[0]) for r in rows)
    iw = max(len(r[1]) for r in rows)
    print(f"| {'case':<{w}} | {'impl':<{iw}} | {'ok':<4} | detail |")
    print(f"|{'-' * (w + 2)}|{'-' * (iw + 2)}|{'-' * 6}|--------|")
    for name, impl, ok, detail in rows:
        if not ok or args.verbose:
            print(f"| {name:<{w}} | {impl:<{iw}} | "
                  f"{'PASS' if ok else 'FAIL':<4} | {detail} |")  # fmt: skip

    print(f"\n{len(rows) - failed}/{len(rows)} passed, {skipped} skipped as illegal")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
