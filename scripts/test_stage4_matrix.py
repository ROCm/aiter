"""Stage 4 correctness gate for shuffled 5D fp8 KV-cache support
(shuffled-cache-support-investigation, "Proposed staged plan" -> Stage 4).

Extends Stage 3's e2e harness (test_stage3_e2e.py, which proved
served=True + correct for causal single-pass and split-K, plus the linear
default, cos ~0.9998) to the issue's standing-gate matrix on the shuffled
path: causal x {single-pass, split-K} x {bf16, fp16 output} x
{equal-length varlen, unequal-length varlen, all-decode} x one large
production-like case, plus sinks handling. Then repeats the same matrix on
the LINEAR (shuffled_kv_cache=False) default to confirm it is unperturbed.

Goes through the public `aiter.ops.unified_attention` router exclusively (never
the builder directly) -- Stage 3 found the one landmine
(`stride_kv_n`/marshalling) that only surfaces there.

Run from the aiter worktree root so `import aiter` resolves to THIS worktree:
    cd ~/projects/aiter/flydsl-unified-attention
    ENABLE_CK=0 python3 -u <this file>

Clears ~/.flydsl/cache before running -- required before ANY run meant to
reflect current kernel source (aiter CLAUDE.md's noted trap: the cache key
does not track `self.foo()` method edits).
"""

import os
import shutil
import sys
from unittest import mock

sys.path.insert(0, os.getcwd())

import torch

import aiter.ops.flydsl.unified_attention_kernels as uak
import aiter.ops.unified_attention as ua
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)

DEV = "cuda"
HEAD_SIZE = 128
BLOCK_SIZE = 64  # structurally _PAGE_SIZE; the fp8 gate refuses anything else
NUM_HEADS = (64, 4)  # GQA 16:1, matches the production trace ratio
NUM_BLOCKS = 1024  # Stage 3's default; the large-production case overrides it


def _clear_flydsl_cache():
    cache_dir = os.path.expanduser("~/.flydsl/cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"cleared {cache_dir}")
    else:
        print(f"{cache_dir} did not exist, nothing to clear")


def _served_and_kv_splits(kw):
    """Run through the public API, reporting whether FlyDSL served the call and
    which num_kv_splits it built with (tier confirmation). Calls served by
    pa_decode never reach `_get_kernel`, so they report the default 1."""
    real_served, seen = uak.flydsl_unified_attention, {}
    real_get_kernel = uak._get_kernel.__wrapped__

    def served_spy(*a, **k):
        r = real_served(*a, **k)
        seen["served"] = r is not None
        return r

    def kernel_spy(*a, **k):
        seen["num_kv_splits"] = a[5] if len(a) > 5 else k.get("num_kv_splits", 1)
        return real_get_kernel(*a, **k)

    uak._get_kernel.cache_clear()
    with mock.patch.object(
        uak, "flydsl_unified_attention", served_spy
    ), mock.patch.object(uak, "_get_kernel", kernel_spy):
        ua.unified_attention(**kw)
    return seen.get("served", False), seen.get("num_kv_splits", 1)


def _run_case(
    label,
    seq_lens,
    shuffled_kv_cache,
    causal=True,
    out_dtype=torch.bfloat16,
    sinks_enabled=False,
    expect_split_k=None,
    num_blocks=NUM_BLOCKS,
):
    """Build a case via the production `generate_data` (which uses the real
    `shuffle_kv_cache` writer-layout helper), run it through the public
    `unified_attention()`, and check served + tier + correctness."""
    query_lens = [q for q, _ in seq_lens]
    (
        query,
        key_cache,
        value_cache,
        maybe_shuffled_key_cache,
        maybe_shuffled_value_cache,
        gen_sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        maybe_quant_query,
        query_scales,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        q_dtype=e4m3_dtype,
        kv_dtype=e4m3_dtype,
        out_dtype=out_dtype,
        shuffled_kv_cache=shuffled_kv_cache,
        device=DEV,
    )

    sinks = gen_sinks if sinks_enabled else None

    kw = {
        "q": maybe_quant_query,
        "k": maybe_shuffled_key_cache,
        "v": maybe_shuffled_value_cache,
        "out": output,
        "cu_seqlens_q": cu_query_lens,
        "seqused_k": kv_lens,
        "max_seqlen_q": max_query_len,
        "max_seqlen_k": max_kv_len,
        "softmax_scale": scale,
        "causal": causal,
        "window_size": window_size,
        "block_table": block_tables,
        "softcap": 0,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "q_scales": query_scales,
        "output_scale": output_scale,
        "sinks": sinks,
        "shuffled_kv_cache": shuffled_kv_cache,
    }

    served, num_kv_splits = _served_and_kv_splits(kw)

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens.tolist(),
        block_tables=block_tables,
        scale=scale,
        out_dtype=out_dtype,
        sinks=sinks,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        output_scale=output_scale,
        causal=int(causal),
    )

    of, rf = output.float(), ref.float()
    err = (of - rf).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), rf.flatten(), dim=0
    ).item()
    tier_ok = expect_split_k is None or ((num_kv_splits > 1) == expect_split_k)
    ok = served and err < 1e-1 and cos > 0.99 and tier_ok
    tier_note = f" splits={num_kv_splits}" if expect_split_k is not None else ""
    print(
        f"{label:<46} served={served!s:<5} err={err:.4g} cos={cos:.6f}"
        f"{tier_note}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _run_matrix(shuffled_kv_cache, tag):
    """The standing-gate matrix, run against either the shuffled 5D path or
    the linear 4D default."""
    all_ok = True

    # causal x tier: single-pass prefill.
    all_ok &= _run_case(
        f"{tag} causal single-pass",
        [(200, 200), (150, 150)],
        shuffled_kv_cache,
        causal=True,
        expect_split_k=False,
    )
    # causal x tier: non-causal single-pass.
    all_ok &= _run_case(
        f"{tag} non-causal single-pass",
        [(200, 200), (150, 150)],
        shuffled_kv_cache,
        causal=False,
        expect_split_k=False,
    )
    # causal x tier: split-K decode, deep max_seqlen_k, underfilled machine.
    all_ok &= _run_case(
        f"{tag} causal split-K",
        [(1, kv) for kv in (2000, 1800, 2200, 1900, 2100, 2050, 1950, 2150)],
        shuffled_kv_cache,
        causal=True,
        expect_split_k=True,
    )
    # non-causal split-K, same shape.
    all_ok &= _run_case(
        f"{tag} non-causal split-K",
        [(1, kv) for kv in (2000, 1800, 2200, 1900, 2100, 2050, 1950, 2150)],
        shuffled_kv_cache,
        causal=False,
        expect_split_k=True,
    )

    # output dtype: fp16, single-pass.
    all_ok &= _run_case(
        f"{tag} fp16 out single-pass",
        [(200, 200), (150, 150)],
        shuffled_kv_cache,
        out_dtype=torch.float16,
        expect_split_k=False,
    )
    # output dtype: fp16, split-K (combine's part_dtype store-packer differs
    # per output dtype -- cross with the tier that exercises it).
    all_ok &= _run_case(
        f"{tag} fp16 out split-K",
        [(1, kv) for kv in (2000, 1800, 2200, 1900, 2100, 2050, 1950, 2150)],
        shuffled_kv_cache,
        out_dtype=torch.float16,
        expect_split_k=True,
    )

    # batch: equal-length varlen -- multiple sequences, all the same query
    # and KV length (varlen machinery exercised via cu_seqlens_q/block_table,
    # but no raggedness).
    all_ok &= _run_case(
        f"{tag} equal-length varlen",
        [(256, 256)] * 4,
        shuffled_kv_cache,
        causal=True,
        expect_split_k=False,
    )
    # batch: UNEQUAL-length varlen (ragged query lens, the production case)
    # mixing a long prefill with short/decode-ish queries at a shared depth.
    all_ok &= _run_case(
        f"{tag} unequal-length varlen (ragged)",
        [(400, 400), (37, 400), (1, 400), (191, 400)],
        shuffled_kv_cache,
        causal=True,
    )

    # batch: all-decode (max_seqlen_q == 1), small batch. Served by pa_decode
    # or the prefill body depending on layout, so the split tier is not pinned.
    all_ok &= _run_case(
        f"{tag} all-decode small-batch",
        [(1, kv) for kv in (300, 250, 400, 180)],
        shuffled_kv_cache,
        causal=True,
        expect_split_k=None,
    )

    # large production-like case: total_q ~8k across many seqs, deep KV
    # context (up to 15000), to catch shape-dependent addressing bugs the
    # small cases miss. `generate_data`'s block-table width is
    # min(needed_pages*num_seqs, num_blocks)//num_seqs -- too small a
    # num_blocks silently truncates per-sequence page columns below what the
    # deepest sequence needs, corrupting the torch REFERENCE (not the
    # kernel), so num_blocks is sized up for this case:
    # needed_pages(15000/64=235) * num_seqs(85) = 19975 <= 32768.
    large_seq_lens = [(128, 4000)] * 65 + [(1, 15000)] * 20
    all_ok &= _run_case(
        f"{tag} large production-like (total_q~8.3k, kv<=15k)",
        large_seq_lens,
        shuffled_kv_cache,
        causal=True,
        num_blocks=32768,
    )

    # sinks: single-pass. Deliberately shallow max_seqlen_k so the dispatch's
    # own tier-selection (sinks forces single-pass whenever it *would* have
    # split) is not what keeps this single-pass -- exercises the plain
    # sinks-supported path.
    all_ok &= _run_case(
        f"{tag} sinks single-pass",
        [(200, 200), (150, 150)],
        shuffled_kv_cache,
        causal=True,
        sinks_enabled=True,
        expect_split_k=False,
    )

    # sinks + would-otherwise-split-K shape: `flydsl_unified_attention` never
    # selects split-K when sinks is set (gated on `sinks is None` before
    # entering the split-count branch), so this is structurally single-pass
    # via the public dispatch, not a declined/raised call -- confirms that
    # invariant holds under the shuffled/linear path under test.
    all_ok &= _run_case(
        f"{tag} sinks forced single-pass (deep ctx)",
        [(1, kv) for kv in (2000, 1800, 2200, 1900, 2100, 2050, 1950, 2150)],
        shuffled_kv_cache,
        causal=True,
        sinks_enabled=True,
        expect_split_k=False,
    )

    return all_ok


def main():
    _clear_flydsl_cache()
    print(f"device: {torch.cuda.get_device_name(0)}")

    all_ok = True

    print("\n--- SHUFFLED (5D, vectorized) ---")
    all_ok &= _run_matrix(shuffled_kv_cache=True, tag="shuffled")

    print("\n--- LINEAR (4D, default) -- must be unperturbed ---")
    all_ok &= _run_matrix(shuffled_kv_cache=False, tag="linear")

    if not all_ok:
        print(
            "\nSTAGE 4 MATRIX FAILED -- see per-case served/err/cos/tier above. "
            "A served=False case may have declined in the FlyDSL availability or "
            "support gates and fallen back to Triton/Gluon; check which gate "
            "declined before debugging loader math."
        )
        sys.exit(1)

    print("\nALL STAGE 4 MATRIX CASES PASSED")


if __name__ == "__main__":
    main()
