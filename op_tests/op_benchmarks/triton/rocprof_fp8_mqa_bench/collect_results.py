import pandas as pd
import argparse
import os
import shutil
from collections import defaultdict
import numpy as np
import math
import triton

def calculate_mem_bw(seq_q_l, seq_kv_l, num_heads_q, head_dim, time_us):   
    Q = seq_q_l * num_heads_q * head_dim * 1.0 # fp8
    kv = seq_kv_l * 1 * head_dim * 1.0
    w = seq_q_l * num_heads_q * 4.0 # fp32 weights
    out = seq_q_l * seq_kv_l * 4.0
    mem = (Q + kv + w + out)
    return (mem / 1e9) / (time_us * 1e-6)


def calculate_tflops(seq_q_l, seq_kv_l, num_heads_q, head_dim, time_us):
    time_s = time_us * 1e-6
    causal = True
    if causal:

        valid_out_elements = (
            ((seq_kv_l**2 + seq_kv_l) / 2)
            if seq_q_l > seq_kv_l
            else (seq_q_l * seq_kv_l - ((seq_q_l**2 - seq_q_l) / 2))
        )
        total_flops = (
            2.0 * 1 * num_heads_q * valid_out_elements * head_dim
        )
    else:
        total_flops = (
            2.0 * 1 * num_heads_q * seq_q_l * seq_kv_l * head_dim
        )

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
parser.add_argument('--head_dim', type=int, default=512, help='')
parser.add_argument('--seq_q_l', type=int, default=1, help='')
parser.add_argument('--seq_kv_l', type=int, default=1024, help='')
parser.add_argument('--repeat', type=int, default=1000, help='')
parser.add_argument('--path', type=str, default="res", help='')
parser.add_argument('--kernel_names', type=str, nargs='*', help='')

args = parser.parse_args()
print(args.kernel_names)
repeat = args.repeat
path = args.path
kernel_names = args.kernel_names

if not os.path.exists(args.path):
    os.makedirs(args.path)

data = pd.read_csv("results_results.csv")
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
    mem_BW = calculate_mem_bw(args.seq_q_l, args.seq_kv_l, args.num_heads_q, args.head_dim, np.mean(vals))
    tflops = calculate_tflops(args.seq_q_l, args.seq_kv_l, args.num_heads_q, args.head_dim, np.mean(vals))
    print(f"{k}:{args.seq_q_l},{args.seq_kv_l},{args.num_heads_q},{args.head_dim},{results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW}, {tflops}")
    k = name_map[k]
    file_path = f"{path}/{k}_data.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w") as fptr:
            print("seq q len, seq kv len,num_heads_q,head_dim,avg,min,max,std,median,BW(GB/s), TFLOPs", file=fptr)
    with open(file_path, "a") as fptr:
        print(f"{args.seq_q_l},{args.seq_kv_l},{args.num_heads_q},{args.head_dim},{results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW}, {tflops}", file=fptr)