"""Ad-hoc probe: verify the gfx1250 v4-nm Python .co launcher is correct on a
non-default stream AND under CUDA-graph capture/replay.

Run: ENABLE_CK=0 python op_tests/_v4_nm_stream_graph_check.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiter  # noqa: E402
import aiter.mla  # noqa: E402
from aiter import dtypes  # noqa: E402
from test_mla_v4_kargpreld import (  # noqa: E402
    _build_bf16_inputs,
    _native_to_2buff_for_asm,
    V_HEAD_DIM,
    NUM_KV_HEADS,
)

torch.set_default_device("cuda")


def _run(inp, q_packed, q_rope, kv_packed, kv_rope, num_kv_splits, logits, lse, out):
    return aiter.mla.mla_decode_fwd_v4_nm(
        q=q_packed,
        qrope=q_rope.contiguous(),
        kv_buffer=kv_packed,
        kvrope=kv_rope.contiguous(),
        output=out,
        qo_indptr=inp["qo_indptr"],
        kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"],
        kv_last_page_lens=inp["kv_last_page_lens"],
        split_indptr=torch.tensor(
            [i * num_kv_splits for i in range(inp["qo_indptr"].numel())],
            dtype=torch.int32,
        ),
        max_seqlen_q=inp["max_seqlen_q"],
        sink=inp["sink"],
        sm_scale=1.0 / (512**0.5),
        out_16_nosplit=0,
        num_kv_splits=num_kv_splits,
        logits=logits,
        attn_lse=lse,
    )


def _alloc(total_q, num_heads, num_kv_splits):
    logits = torch.empty(
        (total_q, num_kv_splits, num_heads, V_HEAD_DIM), dtype=dtypes.fp32
    )
    lse = torch.empty((total_q, num_kv_splits, num_heads, 1), dtype=dtypes.fp32)
    out = torch.empty((total_q, num_heads, V_HEAD_DIM), dtype=dtypes.bf16)
    return logits, lse, out


def main():
    gqa, qlen, batch, kv = 64, 1, 8, 271
    inp = _build_bf16_inputs(
        batch=batch, kv_seq_lens=kv, q_seq_logical=qlen, gqa_ratio=gqa, attn_sink=True
    )
    q_packed, q_rope = _native_to_2buff_for_asm(inp["q_bf16"])
    kv_packed, kv_rope = _native_to_2buff_for_asm(inp["kv_bf16"])
    total_q = batch * qlen
    num_heads = NUM_KV_HEADS * gqa

    def result_for(splits, on_stream=None):
        logits, lse, out = _alloc(total_q, num_heads, splits)
        if on_stream is not None:
            with torch.cuda.stream(on_stream):
                lg, _ = _run(inp, q_packed, q_rope, kv_packed, kv_rope, splits, logits, lse, out)
            on_stream.synchronize()
        else:
            lg, _ = _run(inp, q_packed, q_rope, kv_packed, kv_rope, splits, logits, lse, out)
            torch.cuda.synchronize()
        return (out if splits > 1 else lg[:, 0].to(dtypes.bf16)).clone()

    print("== test 1: default vs non-default stream ==")
    side = torch.cuda.Stream()
    for splits in (1, 8):
        ref = result_for(splits)
        alt = result_for(splits, on_stream=side)
        # NaN-safe compare (fp8 kv pad can be nan on unused split tails).
        eq = torch.equal(torch.nan_to_num(ref), torch.nan_to_num(alt))
        md = (torch.nan_to_num(ref) - torch.nan_to_num(alt)).abs().max().item()
        print(f"  splits={splits}: side-stream bit_exact={eq} max_abs_diff={md:.3e}")

    print("== test 2: CUDA graph capture + replay ==")
    splits = 1
    logits, lse, out = _alloc(total_q, num_heads, splits)
    split_indptr = torch.tensor(
        [i * splits for i in range(batch + 1)], dtype=torch.int32
    )
    valid = torch.full((batch,), splits, dtype=dtypes.i32)

    def call():
        aiter.mla.mla_decode_fwd_v4_nm(
            q=q_packed,
            qrope=q_rope.contiguous(),
            kv_buffer=kv_packed,
            kvrope=kv_rope.contiguous(),
            output=out,
            qo_indptr=inp["qo_indptr"],
            kv_indptr=inp["kv_indptr"],
            kv_page_indices=inp["kv_page_indices"],
            kv_last_page_lens=inp["kv_last_page_lens"],
            split_indptr=split_indptr,
            max_seqlen_q=inp["max_seqlen_q"],
            sink=inp["sink"],
            sm_scale=1.0 / (512**0.5),
            out_16_nosplit=0,
            num_kv_splits=splits,
            logits=logits,
            attn_lse=lse,
        )

    # Warmup (loads the .co module OUTSIDE capture) on a side stream, per the
    # torch cuda-graph recipe.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    eager = logits[:, 0].to(dtypes.bf16).clone()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    # Perturb the output buffer, replay, and confirm the graph re-ran the kernel.
    logits.zero_()
    g.replay()
    torch.cuda.synchronize()
    graphed = logits[:, 0].to(dtypes.bf16).clone()

    eq = torch.equal(torch.nan_to_num(eager), torch.nan_to_num(graphed))
    md = (torch.nan_to_num(eager) - torch.nan_to_num(graphed)).abs().max().item()
    print(f"  graph replay bit_exact_vs_eager={eq} max_abs_diff={md:.3e}")
    print("\n>>> stream+graph probe done.")


if __name__ == "__main__":
    main()
