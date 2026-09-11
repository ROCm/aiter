# SPDX-License-Identifier: MIT
"""Compile-only check for the TDM dispatch kernel -- no GPU needed.

``COMPILE_ONLY=1`` makes FlyDSL trace, lower and codegen the kernel and then
return instead of launching, so this exercises every DSL construct in
``make_dispatch_tdm`` (and every compile-time shape assert) without touching a
device. It cannot say whether the kernel is *correct*, only whether it is
well-formed.

    COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250 python compile_dispatch_tdm.py
"""
import os
import sys
import time

os.environ.setdefault("COMPILE_ONLY", "1")
os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")

from aiter.ops.flydsl.dispatch_combine_v2.intranode_kernels_tdm import (  # noqa: E402
    make_dispatch_tdm,
    tdm_stage_capacity,
)

# (npes, topk, hidden, elem, block, warp): the shapes that pick different
# compile-time paths -- meta TDM on/off, one warp per peer vs a peer split,
# and the tokens-per-wave fallback when topk does not divide the wave.
CASES = [
    (4, 8, 7168, 2, 64, 8),
    (4, 8, 2048, 2, 64, 16),
    (1, 8, 2048, 2, 64, 8),
    (4, 6, 4096, 2, 64, 8),
    (8, 8, 512, 2, 64, 8),  # tile too small for a meta batch -> scalar metadata
]


def main():
    bad = 0
    for npes, topk, hidden, elem, block, warp in CASES:
        max_tok = 512
        cap, slots = tdm_stage_capacity(
            npes=npes,
            experts_per_token=topk,
            max_tok_per_rank=max_tok,
            block_num=block,
            warp_num_per_block=warp,
        )
        tag = f"npes={npes} topk={topk} hidden={hidden} elem={elem} {block}x{warp}"
        t0 = time.time()
        try:
            run = make_dispatch_tdm(
                rank=0,
                npes=npes,
                experts_per_rank=8,
                experts_per_token=topk,
                hidden_dim=hidden,
                hidden_elem_size=elem,
                max_tok_per_rank=max_tok,
                max_recv=npes * max_tok,
                block_num=block,
                warp_num_per_block=warp,
                off_tok_off=0,
                off_recv_num=256,
                off_tis=512,
                off_out_idx=4096,
                off_out_wts=8192,
                off_out_tok=16384,
            )
            run(*([0] * 11), 0, 8)
        except Exception as exc:  # noqa: BLE001 - report every case, not just the first
            bad += 1
            print(f"FAIL {tag}\n  {type(exc).__name__}: {exc}", flush=True)
            if os.environ.get("TB"):
                import traceback

                traceback.print_exc()
                return 1
        else:
            print(
                f"ok   {tag}  cap/blk={cap} slots={slots}  ({time.time() - t0:.1f}s)",
                flush=True,
            )
    print(f"\n{len(CASES) - bad}/{len(CASES)} configs compiled")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
