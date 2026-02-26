import pandas as pd
import argparse
import os
import shutil
from collections import defaultdict
import numpy as np
import math
import triton

def calculate_mem_bw(batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, block_size, window_size, time_us):
    if window_size > 0:
        seq_kv_l = min(seq_kv_l, window_size)    
    Q = seq_q_l * num_heads_q * head_size * 2
    K = V = seq_kv_l * num_heads_k * head_size * 2
    mem = (Q + K + V + Q) * batch_size
    return (mem / 1e9) / (time_us * 1e-6)



def calculate_tflops(batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, window_size, time_us):
    if window_size > 0:
        seq_kv_l = min(seq_kv_l, window_size)
    # FLOPs for QK^T (multiply + add)
    flops_qk = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // 2

    # FLOPs for A x V (multiply + add)
    flops_av = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // 2
    flops_softmax = (5.0 * batch_size * num_heads_q * seq_q_l * seq_kv_l) // 2
    # Total FLOPs
    total_flops = flops_qk + flops_av + flops_softmax

    time_s = time_us * 1e-6

    # TFLOPs = total FLOPs / (time in seconds * 1e12)
    tflops = total_flops / (time_s * 1e12)
    return tflops

def match_name(candidate, names):
    for n in names:
        if candidate in n or n in candidate:
            return True, n
    return False, None

parser = argparse.ArgumentParser(description="")
parser.add_argument('--num_heads_q', type=int, default=16, help='')
parser.add_argument('--num_heads_k', type=int, default=2, help='')
parser.add_argument('--head_size', type=int, default=64, help='')
parser.add_argument('--seq_q_l', type=int, default=1, help='')
parser.add_argument('--seq_kv_l', type=int, default=1024, help='')
parser.add_argument('--prefill_cnt', type=int, default=-1, help='')
parser.add_argument('--decode_cnt', type=int, default=-1, help='')
parser.add_argument('--bs', type=int, default=1, help='')
parser.add_argument('--window_size', type=int, default=0, help='')
parser.add_argument('--block_size', type=int, default=16, help='')
parser.add_argument('--path', type=str, default="res", help='')
parser.add_argument('--repeat', type=int, default=1000, help='')
parser.add_argument('--kernel_names', type=str, nargs='*', help='')

args = parser.parse_args()

print(args.kernel_names)
repeat = args.repeat
path = args.path
kernel_names = args.kernel_names

data = pd.read_csv("results_kernel_trace.csv")
kernel_data = defaultdict(list)
name_map = dict()
for i, f_name in enumerate(data['Kernel_Name']):
    match, matched_name = match_name(f_name, kernel_names)
    if match:
        time = data['End_Timestamp'][i] - data['Start_Timestamp'][i]
        kernel_data[f_name].append(time * 10**-3)
        name_map[f_name] = matched_name

res_dict = dict()
for k, vals in kernel_data.items():
    # get only the actual runs, not tuning ones
    vals = vals[-repeat:]
    # remove warmup runs
    warm_cnt = len(vals) // 10
    vals = vals[warm_cnt:]
    vals = np.sort(vals)
    outliers = len(vals) // 5
    vals = vals[outliers:-outliers]
    if len(vals) == 0:
        continue
    results = [np.mean(vals), np.min(vals), np.max(vals), np.std(vals),np.median(vals)]
    mem_BW = calculate_mem_bw(args.bs, args.seq_q_l, args.seq_kv_l, args.num_heads_q, args.num_heads_k, args.head_size, args.block_size, args.window_size, np.mean(vals))
    tflops = calculate_tflops(args.bs, args.seq_q_l, args.seq_kv_l, args.num_heads_q, args.num_heads_k, args.head_size, args.window_size, np.mean(vals))
    print(f"{k}:{args.bs},{args.prefill_cnt},{args.decode_cnt},{args.window_size},{args.seq_q_l},{args.seq_kv_l},{args.num_heads_q},{args.num_heads_k},{args.head_size}, {results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW}, {tflops}")
    k = name_map[k]
    file_path = f"{path}/{k}_data.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w") as fptr:
            print("batch size,prefill_cnt,decode_cnt,window_size,seq q len, seq kv len,num_heads_q,num_heads_k,head size,avg,min,max,std,median,BW(GB/s), TFLOPs", file=fptr)
    with open(file_path, "a") as fptr:
        print(f"{args.bs},{args.prefill_cnt},{args.decode_cnt},{args.window_size},{args.seq_q_l},{args.seq_kv_l},{args.num_heads_q},{args.num_heads_k},{args.head_size},{results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW}, {tflops}", file=fptr)