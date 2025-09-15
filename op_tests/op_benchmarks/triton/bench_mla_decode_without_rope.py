import triton
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
)
import torch
import sys
import argparse
import itertools

import aiter

from aiter.ops.triton.mla_decode import (
    decode_attention_fwd_grouped,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.triton_tests.test_mla_decode import input_helper, ref_preprocess
from aiter.test_common import checkAllclose

arg_to_torch_dtype = {
    "fp8": torch.float8_e4m3fnuz,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

torch.set_default_device("cuda")

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8=False,
    q_scale=None,
    kv_scale=None,
) -> torch.Tensor:

    if is_fp8:
        scale *= q_scale * kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)

    m = attn_weights.max(-1).values

    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1)) 

    l = attn_weights_exp.sum(-1)

    if is_fp8:
        attn_weights_fp8 = attn_weights_exp.to(torch.float8_e4m3fnuz)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())

    out = out / l.transpose(0,1).unsqueeze(-1)
    
    if is_fp8:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    is_fp8 = q.dtype == torch.float8_e4m3fnuz

    if is_fp8:
        q = q.to(torch.float)
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(q,
                                      k,
                                      v,
                                      sm_scale,
                                      dtype,
                                      is_causal=is_causal,
                                      is_fp8=is_fp8,
                                      q_scale=q_scale,
                                      kv_scale=kv_scale)
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse

def nonvarlen_benchmark_configs(args: argparse.Namespace):
    batch_sizes = [1, 4, 16] if args.B == 0 else [args.B]
    N_HEADS = [16, 48] if args.hq == 0 else [args.hq]
    seq_len_k = [163, 8192] if args.sk == 0 else [args.sk]

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    mtp = args.mtp if args.mtp else 1

    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_k))
    configs = [
        (batch_size, N_HEAD, seq_len_k, kv_lora_rank, qk_rope_head_dim, mtp)
        for batch_size, N_HEAD, seq_len_k in configs
    ]
    return configs


def model_benchmark_configs(args: argparse.Namespace):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    fa_configs = []
    batch_size = args.B if args.B else 4
    mtp = args.mtp if args.mtp else 1

    for model_name, config in configs.items():
        num_q_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        assert (
            num_q_heads == num_kv_heads
        ), """Grouped Query Attention benchmarking not yet supported - try using a model
            with the same number of query and key/value heads (e.g deepseek-V3)"""
        qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        kv_lora_rank = config.get("kv_lora_rank", 512)

        N_CTX_K = args.sk if args.sk else [2**i for i in range(1, 14)]
        if isinstance(N_CTX_K, list):
            for seq_len in N_CTX_K:
                fa_configs.append(
                    (
                        batch_size,
                        num_q_heads,
                        seq_len,
                        kv_lora_rank,
                        qk_rope_head_dim,
                        mtp,
                    )
                )
        else:
            fa_configs.append(
                (
                    batch_size,
                    num_q_heads,
                    N_CTX_K,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    mtp,
                )
            )

    return fa_configs


def create_benchmark_configs(args: argparse.Namespace):
    dtype = arg_to_torch_dtype[args.dtype]
    x_names = ["BATCH", "H", "S", "kv_lora_rank", "qk_rope_head_dim", "mtp"]

    configs = []
    extra_args = {
        "dtype": dtype,
    }

    if args.model:
        x_vals_list = model_benchmark_configs(args)
    else:
        x_vals_list = nonvarlen_benchmark_configs(args)

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "throughput":
        unit = "TFLOPS"
    elif args.metric == "bandwidth":
        unit = "GB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    line_vals = [f"{unit}"]
    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-")],
            ylabel=unit,
            plot_name="mla",
            args=extra_args,
        )
    )
    return configs


def run_benchmark(args: argparse.Namespace):
    torch.manual_seed(0)

    @triton.testing.perf_report(create_benchmark_configs(args))
    def bench_mla(
        BATCH: int,
        H: int,  # number of query heads, equal to the number of k/v heads
        S: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        mtp: int,
        dtype: torch.dtype,
        num_kv_splits: int = 5,
        sm_scale: float = 1.0,
        logit_cap: float = 0.0,
        device="cuda",
        metric: str = "throughput",
        **kwargs,
    ):
        """
        Benchmarks our multi-head latent attention decode kernel.

        Todo:
        - Support variable length sequences (by writing new data generation fns that generate the
        appropriate paged kv cache).
        - Support GQA benchmarking (e.g generate inputs where q_heads = kv_heads * N.
        Right now q_heads == kv_heads).
        """

        # mtp = 1
        #
        kv_indptr, kv_indices, q, kv_cache, attn_logits, attn_lse, out_tri = (
            input_helper(
                BATCH,
                H,
                S,
                kv_lora_rank,
                qk_rope_head_dim,
                num_kv_splits,
                dtype,
                device,
                varlen=False,
                mtp=mtp,
            )
        )

        k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)


        qo_indptr = torch.zeros(BATCH + 1, dtype=torch.int, device=device)

        # out_tri = torch.empty(BATCH * (mtp + 1), H, kv_lora_rank, dtype=kv_cache.dtype, device=device)

        seq_lens_qo = torch.empty(BATCH, dtype=torch.int, device=device)

        seq_lens_qo.fill_(mtp + 1)
        max_seqlen_qo = seq_lens_qo.max().item()
        qo_indptr[1 : BATCH + 1] = torch.cumsum(seq_lens_qo, dim=0)

        out_ref, lse_ref = torch_mla_extend(
            q.reshape(-1, H, 576),
            kv_cache.reshape(-1, 1, 576),
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            is_causal=False,
            dtype=dtype,
        )

        # FLOPS calculation
        num_q_heads = H
        num_kv_heads = k_input.shape[1]
        assert num_q_heads >= num_kv_heads

        # per batch:
        attn_nope_flops = num_q_heads * kv_lora_rank * S * 2
        attn_rope_flops = num_q_heads * qk_rope_head_dim * S * 2
        av_flops = (
            num_q_heads * kv_lora_rank * S * 2
        )  # multiplying attention map with v

        total_flops = BATCH * (mtp + 1) * (
            attn_nope_flops + attn_rope_flops + av_flops
        )

        # Memory transfer calculations (per batch)
        # bytes read:
        q_elems_read = num_q_heads * (kv_lora_rank + qk_rope_head_dim)
        k_rope_elems_read = num_kv_heads * qk_rope_head_dim * S
        kv_nope_elems_read = num_kv_heads * kv_lora_rank * S

        # total indices read (across the full batch)
        kv_indptrs_read = BATCH + 1
        kv_indices_read = BATCH * S

        bytes_read = (
            BATCH * q_elems_read * q.element_size()
            + BATCH * k_rope_elems_read * k_input.element_size()
            + BATCH * kv_nope_elems_read * k_input.element_size()
            + kv_indptrs_read
            + kv_indices_read
        )

        # bytes written:
        out_elems = num_q_heads * kv_lora_rank
        new_k_pe_elems = qk_rope_head_dim  # to add to kv cache

        bytes_written = (
            BATCH * (mtp + 1) * out_elems * out_tri.element_size()
        )

        mem = bytes_read + bytes_written

        q_fp8 = q.to(torch.float8_e4m3fnuz)
        k_input_fp8 = k_input.to(torch.float8_e4m3fnuz)
        v_input_fp8 = v_input.to(torch.float8_e4m3fnuz)
        ms = triton.testing.do_bench(
            lambda: decode_attention_fwd_grouped(
                # q_fp8.reshape(-1, H * 2, 576),
                # k_input_fp8,
                # v_input_fp8,
                q.reshape(-1, H * (mtp + 1), 576),
                k_input,
                v_input,
                out_tri,
                kv_indptr,
                kv_indices,
                kv_lora_rank,
                attn_logits,
                attn_lse,
                num_kv_splits,
                sm_scale,
                logit_cap,
                mtp,
            ),
            warmup=25,
            rep=100,
        )
        # import pdb;pdb.set_trace()

        checkAllclose(out_ref, out_tri,
            msg=f"mla_decode-absorb    [golden vs triton]: {ms * 1000} us......",
        )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            tflops = total_flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_mla.run(save_path=".", print_data=True, show_plots=False)


# argparse lacks support for boolean argument type (sigh...)
def str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = get_parser(kernel_name="MLA Decode without RoPE")
    parser.add_argument("-B", type=int, default=0, help="Batch size.")
    parser.add_argument(
        "-hq",
        type=int,
        default=0,
        help="Number of query heads (equal to number of key/value heads)",
    )
    parser.add_argument(
        "-sk",
        type=int,
        default=0,
        help="Sequence length (since this is decode, this is the length of the key/value sequence)",
    )
    parser.add_argument(
        "-mtp",
        type=int,
        default=1,
        help="Q sequence length (mtp + 1 == qo_len) in MTP mode",
    )
    parser.add_argument(
        "--no-rope",
        action="store_true",
        default=False,
        help="Disable rotary positional embeddings.",
    )
    parser.add_argument(
        "--use-neox-style-rope",
        action="store_true",
        default=False,
        help="Use Neox style rotary positional embeddings over vanilla RoPE. This is incompatible with the --no-rope flag.",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model:
        assert not (
            args.hq
        ), "The -hq flag is unsupported when using --model (as the model config specifies hq)"
    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    # print(args)

    run_benchmark(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())

