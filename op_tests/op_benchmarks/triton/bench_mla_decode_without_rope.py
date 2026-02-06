import triton
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
)
import torch
import sys
import os
import argparse
import itertools
import random

import aiter

# from aiter.ops.triton.mla_decode import (
#     decode_attention_fwd_grouped,
# )
# from aiter.ops.triton.gluon.mla_decode import (
#     decode_attention_fwd_grouped,
# )
# from aiter.ops.triton.gluon.mla_decode_mi355 import (
#     decode_attention_fwd_grouped,
# )
# from aiter.ops.triton.gluon.mla_decode_fp8 import (
#     decode_attention_fwd_grouped as decode_attention_fwd_grouped_fp8,
# )
from aiter.ops.triton.mla_decode_dispatch import (
    decode_attention_fwd_grouped
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.triton_tests.test_mla_decode import ref_preprocess
from aiter.test_common import checkAllclose, run_perftest
from aiter import dtypes

arg_to_torch_dtype = {
    "fp8": torch.float8_e4m3fnuz,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

torch.set_default_device("cuda")

def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def kv_cache_cast_to_fp8(x: torch.Tensor, padding=True) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fnuz)

    padding_size = 0 if not padding else (16 - (block_size * 4) % 16) % 16
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4 + padding_size)),
        device=x.device,
        dtype=torch.float8_e4m3fnuz,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.float8_e4m3fnuz)
    x_fp8[:, block_size * head_dim : block_size * head_dim + 4 * block_size] = sf.view(
        num_blocks, block_size
    ).view(dtype=torch.float8_e4m3fnuz)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4 + padding_size)


def input_helper(
    B,
    H,
    S,
    kv_lora_rank,
    qk_rope_head_dim,
    num_kv_splits,
    page_block_size,
    dtype,
    device,
    rope_base=10,
    rope_max_seq_len=16324,
    rope_scaling=1.0,
    varlen=False,
    mtp=0,
):
    if varlen:
        seqlens = torch.randint(1, S + 1, (B,), dtype=torch.int32, device=device)
    else:
        seqlens = torch.full((B,), S, dtype=torch.int32, device=device)

    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            seqlens.cumsum(dim=0, dtype=torch.int32),
        ]
    )

    max_model_len = seqlens.max().item()
    num_blocks = max_model_len

    total_seqlen = cu_seqlens[-1]

    HK = 1
    SQ = mtp + 1

    q = torch.randn(B, SQ, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(
        num_blocks * B, page_block_size, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )
    # q = torch.randn(B, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    # kv_cache = torch.randn(
    #     total_seqlen, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    # )

    # interlancing [batch_start_off, batch_seq_len, batch_start_off, batch_seq_len, ...,]
    kv_indptr = cu_seqlens

    if page_block_size == 1:
        kv_indices = torch.arange(total_seqlen, device=device).to(torch.int32)
        block_tables = kv_indices
    else:
        max_block_len = (
            (max_model_len + page_block_size - 1) // page_block_size * page_block_size
        )
        block_tables = torch.zeros(
            (B, max_block_len), device="cuda", dtype=torch.int32
        )
        counter = 0
        block_idx_pool = list(range(num_blocks))
        random.shuffle(block_idx_pool)
        indices_list = []
        for i in range(B):
            ctx_len = seqlens[i].item()
            indices = []
            for j in range((ctx_len + page_block_size - 1) // page_block_size):
                block_tables[i][j] = block_idx_pool[counter % num_blocks]
                counter += 1
                car_len = min(ctx_len - j * page_block_size, page_block_size) - 1
                car_begin = block_tables[i][j] * page_block_size
                indices.append(torch.range(car_begin, car_begin + car_len, dtype=torch.int32))
            indice = torch.cat(indices)
            indices_list.append(indice[:max_model_len])
        kv_indices = torch.cat(indices_list).cuda()

    attn_logits = torch.zeros(
        B, H * SQ, num_kv_splits, kv_lora_rank, dtype=torch.float, device=device
    )
    attn_lse = torch.zeros(
        B, H * SQ, num_kv_splits, 1, dtype=torch.float, device=device
    )

    o = torch.zeros(B * SQ, H, kv_lora_rank, dtype=torch.bfloat16, device=device)

    return kv_indptr, block_tables, kv_indices, q, kv_cache, attn_logits, attn_lse, o


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
    # print("!@!!!!!!!", attn_weights)
    # import pdb;pdb.set_trace()
    # print(attn_weights[:,:, :27])
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
    q_scale=1,
    kv_scale=1,
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
        # print(i)
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
    mtp = args.mtp
    dtype = args.dtype

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
    x_names = ["BATCH", "H", "S", "kv_lora_rank", "qk_rope_head_dim", "mtp"]

    configs = []
    extra_args = {
        "dtype": args.dtype,
        "page_block_size": args.page_block_size,
        "varlen": args.varlen,
        "save_aot": args.aot,
        "metric": args.metric,
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
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(10123)

    @triton.testing.perf_report(create_benchmark_configs(args))
    def bench_mla(
        BATCH: int,
        H: int,  # number of query heads, equal to the number of k/v heads
        S: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        mtp: int,
        dtype: str,
        num_kv_splits: int = 8,
        sm_scale: float = 1.0,
        logit_cap: float = 0.0,
        device="cuda",
        page_block_size: int = 64,
        varlen: bool = False,
        save_aot: bool = False,
        metric: str = "bandwidth",
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

        torch_dtype = dtypes.d_dtypes[dtype]
        # mtp = 1
        #
        kv_indptr, block_tables, kv_indices, q, kv_cache, attn_logits, attn_lse, out_tri = (
            input_helper(
                BATCH,
                H,
                S,
                kv_lora_rank,
                qk_rope_head_dim,
                num_kv_splits,
                page_block_size,
                torch.bfloat16,
                device,
                varlen=varlen,
                mtp=mtp,
            )
        )
  
        # q = torch.ones_like(q)
        # kv_cache = torch.ones_like(kv_cache)

        k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)

        qo_indptr = torch.zeros(BATCH + 1, dtype=torch.int, device=device)

        # out_tri = torch.empty(BATCH * (mtp + 1), H, kv_lora_rank, dtype=kv_cache.dtype, device=device)

        seq_lens_qo = torch.empty(BATCH, dtype=torch.int, device=device)

        seq_lens_qo.fill_(mtp + 1)
        max_seqlen_qo = seq_lens_qo.max().item()
        qo_indptr[1 : BATCH + 1] = torch.cumsum(seq_lens_qo, dim=0)
        q = q.reshape(-1, H, 576)

        q_fp8 = q.to(dtypes.fp8)
        kv_cache_fp8 = kv_cache.to(dtypes.fp8)

        out_ref, lse_ref = torch_mla_extend(
            q_fp8 if dtype == "fp8" else q,
            kv_cache_fp8.reshape(-1, 1, 576) if dtype == "fp8" else kv_cache.reshape(-1, 1, 576),
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            is_causal=False,
            dtype=torch.bfloat16,
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

        bytes_per_elem = torch.finfo(torch_dtype).bits // 8
        bytes_read = (
            BATCH * q_elems_read * bytes_per_elem
            + BATCH * k_rope_elems_read * bytes_per_elem
            + BATCH * kv_nope_elems_read * bytes_per_elem
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
        q = q.reshape(-1, H * (mtp + 1), 576)
        if dtype == "fp8":
            q_fp8 = q_fp8.reshape(-1, H * (mtp + 1), 576)

        _, us = run_perftest(
            decode_attention_fwd_grouped,
            q_fp8 if dtype == "fp8" else q,
            kv_cache_fp8 if dtype == "fp8" else kv_cache,
            v_input,
            out_tri,
            kv_indptr,
            kv_indices,
            block_tables,
            kv_lora_rank,
            attn_logits,
            attn_lse,
            num_kv_splits,
            sm_scale,
            logit_cap,
            mtp,
        )
        cache_key = decode_attention_fwd_grouped(
            q_fp8 if dtype == "fp8" else q,
            kv_cache_fp8 if dtype == "fp8" else kv_cache,
            v_input,
            out_tri,
            kv_indptr,
            kv_indices,
            block_tables,
            kv_lora_rank,
            attn_logits,
            attn_lse,
            num_kv_splits,
            sm_scale,
            logit_cap,
            mtp,
        )
        # import pdb;pdb.set_trace()

        print(">>> ", cache_key)
        ms = us / 1000
        checkAllclose(out_ref, out_tri,
            msg=f"mla_decode-absorb    [golden vs triton]: {ms * 1000} us......",
        )

        tflops = total_flops / ms * 1e-9
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        print(f"{tflops=}")
        print(f"{bandwidth=}")

        # import pdb;pdb.set_trace()
        cal_diff(out_ref, out_tri, "out", True)

        if save_aot:
            from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
            import aiter.ops.triton.utils._triton.arch_info as arch_info

            dev = arch_info.get_device()
            triton_cache_dir = str(triton.knobs.cache.dir)
            aot_kernel_dir = f"{AITER_TRITON_CONFIGS_PATH}/mla/aot/"

            os.makedirs(aot_kernel_dir, exist_ok=True)
            aot_name = f"mla_n16x4_prefetch_k_paged_64_{dtype}_{dev}"

            src = os.path.join(triton_cache_dir, cache_key)
            dst = os.path.join(aot_kernel_dir, aot_name)
            if os.path.exists(dst):
                os.system(f"rm -rf {dst}")
            os.system(f"mv {src} {dst}")
            print(f"Moved cache from {src} to {dst}")

            os.system(f"zip -r mla_aot_kernel mla")
        return bandwidth

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
        "-pbs",
        "--page_block_size",
        type=int,
        default=64,
        help="kv cache page block size",
    )
    parser.add_argument(
        "--aot",
        action="store_true",
        default=False,
        help="Enable aot load.",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        default=False,
        help="KV variable length .",
    )
    parser.add_argument("--dtype", default="fp8")
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

