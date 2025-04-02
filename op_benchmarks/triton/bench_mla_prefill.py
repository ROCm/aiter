import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from aiter.ops.triton.mla_prefill import extend_attention_fwd
# from aiter.ops.triton.pa_prefill import extend_attention_fwd

import logging
import time

import triton
import triton.language as tl

from utils.benchmark_utils import get_model_configs, get_available_models, print_vgpr

import sys
import torch
import pytest

import argparse

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

is_hip_ = is_hip()



def input_helper(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device,  equal_seqlens=False, requires_grad=False,):
    torch.manual_seed(0)
    
    if not equal_seqlens:
        max_extend_length = extend_length
        max_prefix_length = prefix_length
        
        seqlens_extend = torch.randint(1, max_extend_length + 1, (B, ), dtype=torch.int32)
        seqlens_prefix = torch.randint(1, max_prefix_length + 1, (B, ), dtype=torch.int32)
       
    else:
        seqlens_extend = torch.full((B, ), extend_length)
        seqlens_prefix = torch.full((B, ), prefix_length)
    
    
    cu_seqlens_extend = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_extend.cumsum(dim=0, dtype=torch.int32)])
    cu_seqlens_prefix = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_prefix.cumsum(dim=0, dtype=torch.int32)])
    
    cu_seqlens_extend = cu_seqlens_extend.to(device="cuda")
    cu_seqlens_prefix = cu_seqlens_prefix.to(device="cuda")

    total_extend = cu_seqlens_extend[-1].item()
    total_prefix = cu_seqlens_prefix[-1].item()
    
    q_extend = torch.randn(total_extend, H, v_head_dim + qk_rope_head_dim, dtype=dtype, device=device).requires_grad_(requires_grad)

    # extend parts
    k_extend = torch.randn(total_extend, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device).requires_grad_(requires_grad)
    v_extend = k_extend[..., :kv_lora_rank]
    o_extend = torch.empty(total_extend, H, v_head_dim, dtype=dtype, device=device)

    # extend indexing
    qo_indptr = cu_seqlens_extend # torch.arange(B + 1, device=device) * (extend_length) # 0, extend_length, extend_length*2
    
    # prefix parts
    k_buffer = torch.randn(total_prefix, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device).requires_grad_(requires_grad)
    v_buffer = k_buffer[..., :kv_lora_rank]

    # prefix indexing
    kv_indptr = cu_seqlens_prefix # torch.arange(B + 1, device=device) * prefix_length # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(total_prefix, device=device)

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    w_kc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device).requires_grad_(requires_grad)
    w_vc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device).requires_grad_(requires_grad)

    return q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc


def mla_forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, 
                w_kc, w_vc, attn_impl, kv_lora_rank, qk_rope_head_dim, v_head_dim, H, sm_scale=1.0, logit_cap=0.0):
    if attn_impl == "absorb":
        q_input = torch.empty((*q_extend.shape[:-1],kv_lora_rank+qk_rope_head_dim), dtype=q_extend.dtype, device=q_extend.device)
        q_input[..., kv_lora_rank:] = q_extend[..., v_head_dim:]
        q_nope = q_extend[..., :v_head_dim]
        q_nope = torch.bmm(q_nope.transpose(0, 1), w_kc.transpose(1,2))
        q_input[..., :kv_lora_rank] = q_nope.transpose(0, 1)
        
        tmp_out = torch.empty((*q_extend.shape[:-1], kv_lora_rank), dtype=q_extend.dtype, device=q_extend.device)
    else: # non-absorbed
        q_input = q_extend
    
        k_extend_c = torch.einsum('zc,hcd->zhd', k_extend[..., :kv_lora_rank], w_kc)
        k_extend_r = k_extend[..., kv_lora_rank:].unsqueeze(1).repeat(1, H, 1)
        k_extend = torch.cat((k_extend_c, k_extend_r), dim=-1)
        
        k_buffer_c = torch.einsum('zc,hcd->zhd', k_buffer[..., :kv_lora_rank], w_kc)
        k_buffer_r = k_buffer[..., kv_lora_rank:].unsqueeze(1).repeat(1, H, 1)
        k_buffer = torch.cat((k_buffer_c, k_buffer_r), dim=-1)
        
        v_extend = torch.einsum('zc,hcd->zhd', v_extend, w_vc)
        v_buffer = torch.einsum('zc,hcd->zhd', v_buffer, w_vc)
        
        tmp_out = torch.empty((*q_extend.shape[:-1], v_head_dim), dtype=q_extend.dtype, device=q_extend.device)
 
    triton_kernel = lambda: extend_attention_fwd(q_input, k_extend, v_extend, tmp_out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
    
    triton_kernel()
    
    if attn_impl == "absorbed":
        attn_bmm_output = torch.bmm(tmp_out.transpose(0, 1), w_vc)
        attn_output = attn_bmm_output.transpose(0, 1)
        ref_out = attn_output
    else:
        ref_out = tmp_out

    return ref_out, triton_kernel

def get_benchmark_configs():
    x_names = ["B", "H", "prefix", "extend", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim", "attn_impl"]
    x_vals_list = [
                    (2, 16, 1024, 1024, 512, 64, 128, "non-absorb"),
                    (2, 16, 4096, 4096, 512, 64, 128, "non-absorb"),
                    (2, 16, 8192, 4096, 512, 64, 128, "non-absorb"),
                    (2, 16, 8192, 4096, 512, 64, 128, "absorb"),
                    (2, 16, 16324, 8192, 512, 64, 128, "absorb"),
                  ]
    return x_names, x_vals_list

def model_benchmark_configs(args):
    config_file = args.model_configs
    # Only deepseek models are supported for this benchmark.
    if args.model == "all":
        configs = get_model_configs(config_path=config_file, models="deepseek")
    else:
        assert "deepseek" in args.model, "Only deepseek models are supported for this benchmark."
        configs = get_model_configs(config_path=config_file, models=args.model)
    
    batch_size = args.b if args.b else 1

    x_names = ["model", "B", "H", "prefix", "extend", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim", "attn_impl"]

    x_vals_list = []

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        # HK = HQ if config["num_key_value_heads"] is None else config["num_key_value_heads"]
        prefix = args.prefix if args.prefix else 16324 # max positional embedding in deepseek-V3 model
        extend = args.extend if args.extend else 8192 # max positional embedding in deepseek-V3 model
        attn_impl = args.attn_impl if args.attn_impl else "non-absorb"
        x_vals_list.append((model_name, batch_size, HQ, prefix,  extend, 512, 64, 128, attn_impl))

    return x_names, x_vals_list


def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []

    if args.model:
        x_names, x_vals_list = model_benchmark_configs(args)
    else:
        x_names, x_vals_list = get_benchmark_configs()

    line_vals = ["mla_extend"]

    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, attn_impl, sm_scale, logit_cap, device,
                  provider=None, model=None):
                
        warmup = 25
        rep = 100

        q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc = input_helper(
            B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)
       
        if args.include_gemms:
            # measure also the w_kc and w_vc projection gemms (bmms)
            fn = lambda: mla_forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend,
                            w_kc=w_kc, w_vc=w_vc, attn_impl=attn_impl, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim, H=H, sm_scale=sm_scale, logit_cap=logit_cap)
            ms = triton.testing.do_bench_cudagraph(fn, rep=rep)
        else:
            # only measure the triton kernel call time
            _, fn = mla_forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend,
                            w_kc=w_kc, w_vc=w_vc, attn_impl=attn_impl, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim, H=H, sm_scale=sm_scale, logit_cap=logit_cap)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        
        return ms

    bench_MLA.run(save_path=None, print_data=True, show_plots=False)
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
    parser.add_argument('-b', type=int, default=0, help="Batch size")
    parser.add_argument('-prefix', type=int, default=0, help="Prefix length")
    parser.add_argument('-extend', type=int, default=0, help="Extend length")
    parser.add_argument('-attn_impl', type=str, default="non-absorb", help="Whether to use absorbed or non-absorbed attention. Options: absorb, non-absorb")
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