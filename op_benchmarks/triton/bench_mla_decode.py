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
        "ctx_lens",
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
        (16324, 16, 16, 512, 128, 64, 128, torch.bfloat16, torch.bfloat16, 1, 32),
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
        "ctx_lens",
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
        HK = HQ if config["num_key_value_heads"] is None else config["num_key_value_heads"]
        N_CTX_K = args.sk if args.sk else 163840 # max positional embedding in deepseek-V3 model
        HEAD_DIM = config["hidden_size"] // HQ
        x_vals_list.append((model_name, N_CTX_K, batch_size, HQ, 512, 128, 64, 128, torch.bfloat16, torch.bfloat16, 1, 32))

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
        ctx_lens,
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

        seq_lens = torch.tensor([torch.randint(1, ctx_lens, (1,)).item() for _ in range(batch_size)], dtype=torch.int)
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
    parser.add_argument('-model_configs', type=str, default="utils/model_configs.json", help="Model config json file.")
    available_models = get_available_models(filter="deepseek")  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: [" + ", ".join(available_models) +
        "]. Use 'all' to benchmark all models. Provide model family (the part before -) to benchmark all models in that family. One can provide multiple as -model \"llama3,mistral_7B\""
    )
    parser.add_argument('-model', type=str, default="", help=model_help)
    parser.add_argument('-b', type=int, default=0, help="Custom batch size.")
    parser.add_argument('-sk', type=int, default=0, help="Custom context length.")
    parser.add_argument("-dtype", default='bf16')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("-equal_seqlens", action="store_true", default=False,
                         help="Equal sequence lengths, i.e. total (prefix|extend) tokens = B * (prefix|extend). Otherwise we have randint(1, (prefix|extend), (B,)) as sequence lengths.")
    parser.add_argument("-include_gemms", action="store_true", default=False, help="Measure the w_kc and w_vc projection gemms (2 x torch.bmm calls) as part of the benchmark run.")
    return parser.parse_args()

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

import re
from prettytable import PrettyTable

def parse_vgpr_usage(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Extract VGPR-related information
    vgpr_info = []
    table_lines = []
    in_table = False

    for line in lines:
        # Parse autotuning outputs
        if re.search(r"Autotuning kernel", line):
            vgpr_info.append(line.strip())
        if re.search(r"Triton autotuning for function", line):
            vgpr_info.append(line.strip())

        if re.search(r"\.name:", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.vgpr_count:", line) or re.search(r"\.vgpr_spill_count:", line):
            vgpr_info.append(line.strip())
        # Detect start of table
        if re.match(r"^\s*MLA-decode:", line):
            in_table = True
            # table_lines.append(line.strip())
        elif in_table:
            table_lines.append(line.strip())

    # Print extracted information
    print("\n".join(vgpr_info))

    table = PrettyTable()
    table.field_names = table_lines[0].split()
    [table.add_row(line.split()[1:]) for line in table_lines[1:]]

    print(table)


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)

import sys
import time
import re
import os
import tempfile

def print_vgpr(args):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        output_file = temp_file.name

        # Redirect stdout and stderr to the temporary file
        sys.stdout = temp_file
        sys.stderr = temp_file
        
        os.environ["AMDGCN_ENABLE_DUMP"] = "1"
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        run_bench(args)  # Run the benchmark
        
        sys.stdout.flush()
        sys.stderr.flush()

    # Restore stdout and stderr to normal
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    time.sleep(0.5)  # Ensure everything is written before reading

    # Parse and print relevant output
    parse_vgpr_usage(output_file)

    # Remove the temporary file
    os.unlink(output_file)

def main():
    args = parse_args()
    if args.print_vgpr:
        print_vgpr(args)
        return 0
    run_bench(args)


if __name__ == "__main__":
    main()