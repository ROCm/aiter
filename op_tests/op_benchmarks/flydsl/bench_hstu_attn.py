from aiter.ops.flydsl import flydsl_hstu_attention_fwd
from dataclasses import dataclass
import argparse
import triton
import torch
from op_tests.flydsl_tests.test_flydsl_hstu_attention import (
    generate_hstu_attn_inputs,
)


# lower triangular mask, so no need to multiply by 2 for flops
def get_flops(seq_offsets: torch.Tensor, heads: int, attn_dim: int, hidden_dim: int):
    total_flops = 0.0
    seq_num = seq_offsets.shape[0] - 1
    for i in range(seq_num):
        length = seq_offsets[i + 1] - seq_offsets[i]
        total_flops += length * length * (attn_dim + hidden_dim) * heads
    return total_flops


def get_bytes(
    seq_offsets: torch.Tensor,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    elem_size: int,
):
    seq_num = seq_offsets.shape[0] - 1
    total_bytes = 0
    for i in range(seq_num):
        length = seq_offsets[i + 1] - seq_offsets[i]
        total_bytes += length * (attn_dim + length + hidden_dim) * heads * elem_size
    return total_bytes


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
    dtype = _DTYPES[shape.dtype]

    alpha = 1.0 / shape.head_dim * 10000
    causal = True

    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=shape.batch_size,
        max_seq_len=shape.max_seq_len,
        sparsity=shape.sparsity,
        heads=shape.num_heads,
        attn_dim=shape.head_dim,
        hidden_dim=shape.hidden_dim,
        target_size=shape.target_size,
        dtype=dtype,
        device=device,
        seed=seed,
    )

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
