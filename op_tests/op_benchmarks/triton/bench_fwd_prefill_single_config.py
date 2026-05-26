# SPDX-License-Identifier: MIT
"""Bench a single (BLOCK_M, BLOCK_N, num_warps, ...) config of AITER's
Triton FA varlen_fwd at a fixed (B, S, H, D, dtype) shape.

Config picked up from FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON if set, else
the AITER default for the current arch (see fwd_prefill.get_fwd_prefill_configs).
Defaults target the Qwen3-Omni / Gemma3 SigLIP ViT attention call
(B=1, S=3200, H=16, D=72, fp16, non-causal). Use --seq / --heads / --head-dim
to repurpose for any varlen-or-batch-1 attention call.

Pairs with sweep_fwd_prefill_configs.py, which forks this script under
different config envs.

Example:
    # Default config + shape
    python bench_fwd_prefill_single_config.py

    # Specific Triton config
    FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":128,"BLOCK_N":32,"PRE_LOAD_V":false,"num_warps":8,"num_stages":1,"waves_per_eu":6}' \\
        python bench_fwd_prefill_single_config.py

    # Different shape (e.g. Gemma3 ViT)
    python bench_fwd_prefill_single_config.py --seq 4096 --dtype bf16
"""

import argparse
import json
import os
import site
import sys


# Without amd_smi importable, AITER / Triton arch detection falls back. The
# TheRock ROCm SDK ships amd_smi under a non-standard share/ path; expose it
# before importing aiter.
for _site in site.getsitepackages():
    _amdsmi = os.path.join(_site, "_rocm_sdk_core", "share", "amd_smi")
    if os.path.isdir(_amdsmi) and _amdsmi not in sys.path:
        sys.path.insert(0, _amdsmi)
        break

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

import torch  # noqa: E402
import triton  # noqa: E402

# Low-level entry — bypasses aiter.ops.triton.attention.mha so the bench
# measures the kernel itself, not the wrapper. Pairs with the production
# dispatch from Dao-AILab/flash-attention's `flash_attn_triton_amd` path.
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import (  # noqa: E402
    flash_attn_2 as fa_aiter,
)


_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=3200, help="seqlen_q == seqlen_k")
    p.add_argument("--heads", type=int, default=16, help="num query heads")
    p.add_argument(
        "--kv-heads", type=int, default=None,
        help="num kv heads (default: same as --heads, i.e. MHA)",
    )
    p.add_argument("--head-dim", type=int, default=72)
    p.add_argument("--dtype", choices=tuple(_DTYPES), default="fp16")
    p.add_argument("--warmup-ms", type=int, default=200)
    p.add_argument("--rep-ms", type=int, default=600)
    return p.parse_args()


def main() -> int:
    args = _parse()
    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    Hk = args.kv_heads if args.kv_heads is not None else H
    dtype = _DTYPES[args.dtype]
    device = "cuda"
    torch.manual_seed(0)

    # Single-sequence varlen layout: cu_seqlens = [0, S, 2S, ..., B*S].
    # Real varlen calls keep S const across batch here for repro stability.
    q = torch.randn(B * S, H, D, dtype=dtype, device=device)
    k = torch.randn(B * S, Hk, D, dtype=dtype, device=device)
    v = torch.randn(B * S, Hk, D, dtype=dtype, device=device)

    cu_seqlens_q = torch.arange(
        0, (B + 1) * S, S, dtype=torch.int32, device=device,
    )
    cu_seqlens_k = cu_seqlens_q
    out = torch.empty_like(q)

    sm_scale = D ** -0.5

    def run():
        fa_aiter.varlen_fwd(
            q, k, v, out,
            cu_seqlens_q, cu_seqlens_k,
            seqused_k=None, leftpad_k=None, block_table_=None,
            alibi_slopes=None,
            max_seqlen_q=S, max_seqlen_k=S,
            dropout_p=0.0, softmax_scale=sm_scale,
            zero_tensors=False, causal=False,
            window_size_left=-1, window_size_right=-1,
            softcap=0.0, return_softmax=False,
        )

    _ = run()
    torch.cuda.synchronize()

    # triton.testing.do_bench clears the L2 cache before each measurement
    # iteration, so timings approximate the kernel's behavior inside a real
    # transformer block (where surrounding qkv-proj / MLP / norm work evicts
    # q/k/v between layer calls).
    ts = sorted(
        triton.testing.do_bench(
            run, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="all",
        )
    )
    n = len(ts)
    result = {
        "config_json": os.environ.get(
            "FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON", "default",
        ),
        "shape": {
            "batch": B, "seq": S, "heads": H, "kv_heads": Hk,
            "head_dim": D, "dtype": args.dtype,
        },
        "samples": n,
        "median_ms": ts[n // 2],
        "min_ms": ts[0],
        "p10_ms": ts[max(0, int(n * 0.1))],
        "p90_ms": ts[min(n - 1, int(n * 0.9))],
    }
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
