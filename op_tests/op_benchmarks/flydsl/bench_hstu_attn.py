from aiter.ops.flydsl import flydsl_hstu_attention_fwd
from dataclasses import dataclass
import argparse
import triton
import torch
from op_tests.triton_tests.attention.test_hstu_attn import (
    generate_sparse_seq_len,
    get_flops,
    get_bytes,
)


@dataclass
class FwdShape:
    batch_size: int
    max_seq_len: int
    sparsity: float
    dtype: str = "bf16"
    num_heads: int = 4
    head_dim: int = 128
    hidden_dim: int = 128
    max_attn_len: int = 0
    contextual_seq_len: int = 0
    target_size: int = 20

    def display_string(self) -> str:
        parts: list[str] = [
            f"batch_size={self.batch_size}",
            f"max_seq_len={self.max_seq_len}",
            f"sparsity={self.sparsity}",
            f"dtype={self.dtype}",
            f"num_heads={self.num_heads}",
            f"head_dim={self.head_dim}",
            f"hidden_dim={self.hidden_dim}",
            f"max_attn_len={self.max_attn_len}",
            f"contextual_seq_len={self.contextual_seq_len}",
            f"target_size={self.target_size}",
        ]
        return ",".join(parts)


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def run_benchmark(
    shape: FwdShape,
    device: torch.device = torch.device("cuda"),
    seed: int = 1001,
):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    dtype = _DTYPES[shape.dtype]

    alpha = 1.0 / shape.head_dim * 10000
    causal = True

    # generate inputs
    lengths = generate_sparse_seq_len(
        size=shape.batch_size,
        max_seq_len=shape.max_seq_len,
        sparsity=shape.sparsity,
        device=device,
    )

    num_targets = None
    if shape.target_size > 0:
        num_targets = torch.randint(
            1,
            shape.target_size + 1,
            (shape.batch_size,),
            device=lengths.device,
            dtype=lengths.dtype,
        )
        num_targets = torch.where(num_targets > lengths, lengths, num_targets)

    seq_offsets = torch.zeros((shape.batch_size + 1,), dtype=torch.int64, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, shape.num_heads, shape.head_dim * 2 + shape.hidden_dim),
        dtype=dtype,
        device=device,
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [shape.head_dim, shape.head_dim, shape.hidden_dim], dim=-1)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    def flydsl_attn():
        return flydsl_hstu_attention_fwd(
            shape.max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            num_targets,
            shape.max_attn_len,
            shape.contextual_seq_len,
        )

    ms = triton.testing.do_bench(flydsl_attn, warmup=2000, rep=2000)

    flops = get_flops(
        seq_offsets,
        shape.num_heads,
        shape.head_dim,
        shape.hidden_dim,
    )
    tflops = flops / ms / 1e9

    elem_size = q.element_size()
    bytes = get_bytes(
        seq_offsets.cpu().numpy(),
        shape.num_heads,
        shape.head_dim,
        shape.hidden_dim,
        elem_size,
    )
    bandwidth = bytes / (ms * 1e-3) * 1e-9  # GB/s

    return ms, tflops, bandwidth


def main():
    p = argparse.ArgumentParser(description="FlyDSL HSTU Attention benchmark")
    p.add_argument("--batch_size", type=int, default=120)
    p.add_argument("--max_seq_len", type=int, default=16384)
    p.add_argument("--sparsity", type=float, default=0.475)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=0, help="defaults to --head_dim")
    p.add_argument("--max_attn_len", type=int, default=0)
    p.add_argument("--contextual_seq_len", type=int, default=0)
    p.add_argument("--target_size", type=int, default=300)
    p.add_argument("--dtype", type=str, default="bf16", choices=list(_DTYPES))
    args = p.parse_args()

    args.hidden_dim = args.hidden_dim or args.head_dim

    shape = FwdShape(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        sparsity=args.sparsity,
        dtype=args.dtype,
        max_attn_len=args.max_attn_len,
        contextual_seq_len=args.contextual_seq_len,
        target_size=args.target_size,
        head_dim=args.head_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )
    ms, tflops, bandwidth = run_benchmark(
        shape=shape,
        device=torch.device("cuda"),
        seed=1001,
    )

    print(
        f"[FlyDSL HSTU Attention Forward] {shape.display_string()},ms={ms:.3f},tflops={tflops:.1f},gbps={bandwidth:.1f}"
    )


if __name__ == "__main__":
    main()
