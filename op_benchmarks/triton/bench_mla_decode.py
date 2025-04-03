# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import os
import sys

from utils.benchmark_utils import get_model_configs, get_available_models, print_vgpr

# Add two parent directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


from aiter.ops.triton import mla_decode


import triton
import triton.language as tl
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def get_benchmark_configs():
    x_names = [
        "max_ctx_len",
        "batch_size",
        "nhead",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "dtype",
        "kvtype",
        "page_size",
        "num_kv_splits",
    ]
    x_vals_list = [
        (1024, 16, 16, 512, 128, 64, 128, torch.bfloat16, torch.bfloat16, 1, 32),
        (8192, 16, 16, 512, 128, 64, 128, torch.bfloat16, torch.bfloat16, 1, 32),
        (16324, 16, 16, 512, 128, 64, 128, torch.bfloat16, torch.bfloat16, 1, 32),
        # 163840 is the max positional embedding in deepseek-V3 model
        (163840, 1, 16, 512, 128, 64, 128, torch.bfloat16, torch.bfloat16, 1, 32),
    ]
    return x_names, x_vals_list


def model_benchmark_configs(args):
    config_file = args.model_configs
    if args.model == "all":
        configs = get_model_configs(config_path=config_file, models="deepseek")
    else:
        assert "deepseek" in args.model, "Only deepseek models are supported for this benchmark."
        configs = get_model_configs(config_path=config_file, models=args.model)
    batch_size = args.b if args.b else 128

    x_names = [
        "model_name",
        "max_ctx_len",
        "batch_size",
        "nhead",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "dtype",
        "kvtype",
        "page_size",
        "num_kv_splits",
    ]

    x_vals_list = []

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        max_ctx_len = args.sk if args.max_ctx_len else 163840 # max positional embedding in deepseek-V3 model
        x_vals_list.append((model_name, max_ctx_len, batch_size, HQ, 512, 128, 64, 128, torch.bfloat16, torch.bfloat16, 1, 32))

    return x_names, x_vals_list


def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []
    
    if args.model:
        x_names, x_vals_list = model_benchmark_configs(args)
    else:
        x_names, x_vals_list = get_benchmark_configs()

    line_vals = ["mla_decode"]

    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={}))

    @triton.testing.perf_report(configs)
    def bench_mla_decode(
        max_ctx_len,
        batch_size,
        nhead,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        dtype,
        kvtype,
        page_size,
        num_kv_splits,
        provider=None,
        model_name=None,
    ):
        kv_max_sz = 65536  # calculated by rest of mem after weight loaded in frameworks
        num_page = (kv_max_sz + page_size - 1) // page_size

        # for decode (mqa)
        qk_head_dim = kv_lora_rank + qk_rope_head_dim
        nhead_kv = 1
        v_head_dim = kv_lora_rank  # for attn_mqa in sglang

        q = torch.randn((batch_size, nhead, qk_head_dim), dtype=dtype)
        kv_buffer = torch.randn(
            (num_page * page_size, nhead_kv, qk_head_dim),  # decode kv head
            dtype=kvtype,
        )

        if qk_head_dim != v_head_dim:
            out_ref = q.new_empty((q.shape[0], nhead, v_head_dim)).fill_(-1)
        else:
            out_ref = torch.empty_like(q)

        sm_scale = 1.0 / (qk_head_dim**0.5)

        if args.equal_seqlens:
            seq_lens = torch.tensor([max_ctx_len for _ in range(batch_size)], dtype=torch.int)
        else:
            seq_lens = torch.tensor([torch.randint(1, max_ctx_len, (1,)).item() for _ in range(batch_size)], dtype=torch.int)
        
        kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int)
        kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens, dim=0)
        kv_indices = torch.randint(
            0, num_page, (kv_indptr[-1].item() + 1,), dtype=torch.int
        )
        attn_logits = torch.empty(
            (batch_size, nhead, num_kv_splits, v_head_dim + 1),
            dtype=torch.float32,
        )

        fn = lambda: mla_decode.decode_attention_fwd(
            q,
            kv_buffer,
            kv_buffer[..., :kv_lora_rank],
            out_ref,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
        )
        
        ms = triton.testing.do_bench_cudagraph(fn, rep=20)
        
        return ms

    bench_mla_decode.run(save_path=None, print_data=True, show_plots=False)
    return x_vals_list, x_names, line_vals

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA Prefill",
        allow_abbrev=False,
    )
    parser.add_argument('-plot_name', type=str, default="MLA-decode", help="Name of the results plot|table.")
    parser.add_argument('-model_configs', type=str, default="utils/model_configs.json", help="Model config json file.")
    available_models = get_available_models(filter="deepseek")  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: [" + ", ".join(available_models) +
        "]. Use 'all' to benchmark all models. Provide model family (the part before -) to benchmark all models in that family. One can provide multiple as -model \"llama3,mistral_7B\""
    )
    parser.add_argument('-model', type=str, default="", help=model_help)
    parser.add_argument('-b', type=int, default=0, help="Custom batch size.")
    parser.add_argument('-max_ctx_len', type=int, default=0, help="Custom max context length. Equal to the actual context lens if -equal_seqlens, otherwise max.")
    parser.add_argument("-dtype", default='bf16')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("-equal_seqlens", action="store_true", default=False,
                         help="Equal sequence lengths, i.e. total (prefix|extend) tokens = B * (prefix|extend). Otherwise we have randint(1, (prefix|extend), (B,)) as sequence lengths.")
    parser.add_argument("-include_gemms", action="store_true", default=False, help="Measure the w_kc and w_vc projection gemms (2 x torch.bmm calls) as part of the benchmark run.")
    return parser.parse_args()

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)


def main():
    args = parse_args()
    if args.print_vgpr: # print the vgpr usage of the kernel
        print_vgpr(lambda: run_bench(args), table_start=args.plot_name)
        return 0
    run_bench(args)


if __name__ == "__main__":
    main()