"""gfx950 Gluon unified attention: one softmax rewrite decides whether triton 3.8 compiles.

    python repro.py --fixed     # scale the reduced max vector  -> compiles
    python repro.py --broken    # scale the whole S tile        -> LLVM abort on triton 3.8

triton 3.8  --broken -> SIGABRT (rc 134) in the AMDGPU backend during codegen
triton 3.7  both paths compile and run
"""

import os
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--fixed"
assert mode in ("--fixed", "--broken"), __doc__
os.environ["UA_SOFTMAX_BROKEN"] = "1" if mode == "--broken" else "0"

import torch  # noqa: E402
import triton  # noqa: E402

from wrapper import unified_attention  # noqa: E402

# sinks + a prefill sequence + head_size <= 128 are all required to trigger it
HEAD_SIZE, PAGE, NUM_Q_HEADS, NUM_KV_HEADS = 128, 64, 8, 8
SEQ_LEN, NUM_BLOCKS = 777, 64
dev = "cuda"

torch.manual_seed(0)
q = torch.randn(SEQ_LEN, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=dev)
k = torch.randn(NUM_BLOCKS, PAGE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=dev)
v = torch.randn_like(k)
out = torch.empty_like(q)

print(f"triton {triton.__version__}, softmax {mode[2:]}: compiling...", flush=True)
unified_attention(
    q=q,
    k=k,
    v=v,
    out=out,
    cu_seqlens_q=torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=dev),
    seqused_k=torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev),
    max_seqlen_q=SEQ_LEN,
    softmax_scale=HEAD_SIZE**-0.5,
    causal=True,
    sinks=torch.randn(NUM_Q_HEADS, dtype=torch.float32, device=dev),
    block_table=torch.arange(NUM_BLOCKS, dtype=torch.int32, device=dev)[None, :],
)
torch.cuda.synchronize()
print("compiled and ran -- no abort")
