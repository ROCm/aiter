import pandas as pd
import argparse
import os
import sys
import re
import shutil
from collections import defaultdict
import numpy as np
import math
import triton

def calculate_mem_bw(causal, batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, block_size, window_size, time_us, q_fp8=0, kv_fp8=0):
    if window_size > 0:
        seq_kv_l = min(seq_kv_l, window_size)
    q_bytes = 1 if q_fp8 else 2
    kv_bytes = 1 if kv_fp8 else 2
    Q = seq_q_l * num_heads_q * head_size * q_bytes
    K = V = seq_kv_l * num_heads_k * head_size * kv_bytes
    out = seq_q_l * num_heads_q * head_size * 2  # output is always bf16
    mem = (Q + K + V + out) * batch_size
    return (mem / 1e9) / (time_us * 1e-6)



def calculate_tflops(causal, batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, window_size, time_us):
    if window_size > 0:
        seq_kv_l = min(seq_kv_l, window_size)
    # FLOPs for QK^T (multiply + add)
    div = 2 if causal else 1
    flops_qk = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // div

    # FLOPs for A x V (multiply + add)
    flops_av = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // div
    flops_softmax = (5.0 * batch_size * num_heads_q * seq_q_l * seq_kv_l) // div
    # Total FLOPs
    total_flops = flops_qk + flops_av# + flops_softmax

    time_s = time_us * 1e-6

    # TFLOPs = total FLOPs / (time in seconds * 1e12)
    tflops = total_flops / (time_s * 1e12)
    return tflops

# batch_size=1
# seq_q_l=4096
# seq_kv_l=4096
# num_heads_q=8
# num_heads_k=1
# head_size=64
# window_size=0
# time_us=110
# block_size=128
# causal=1
# mem_BW = calculate_mem_bw(causal, batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, block_size, window_size, time_us)
# tflops = calculate_tflops(causal, batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, window_size, time_us)
# print(f"mem_BW: {mem_BW}, tflops: {tflops}")
# sys.exit()

def compute_metrics(args, time_us):
    """BW (GB/s) and TFLOP/s for a given per-call kernel time in microseconds."""
    mem_BW = calculate_mem_bw(args.causal, args.bs, args.seq_q_l, args.seq_kv_l,
                              args.num_heads_q, args.num_heads_k, args.head_size,
                              args.block_size, args.window_size, time_us,
                              args.q_fp8, args.kv_fp8)
    tflops = calculate_tflops(args.causal, args.bs, args.seq_q_l, args.seq_kv_l,
                              args.num_heads_q, args.num_heads_k, args.head_size,
                              args.window_size, time_us)
    return mem_BW, tflops


def report(args, time_us, source):
    """Print a human-readable summary for a single time measurement."""
    mem_BW, tflops = compute_metrics(args, time_us)
    print(f"source: {source}")
    print(f"causal={args.causal} q_fp8={args.q_fp8} kv_fp8={args.kv_fp8} bs={args.bs} "
          f"seq_q={args.seq_q_l} seq_kv={args.seq_kv_l} nh_q={args.num_heads_q} "
          f"nh_k={args.num_heads_k} head={args.head_size}")
    print(f"time(us): {time_us}")
    print(f"BW(GB/s): {mem_BW}")
    print(f"TFLOPs: {tflops}")
    return mem_BW, tflops


def time_from_draw_log(path):
    """Per-dispatch time (us) from an FFM draw.log.

    Timestamps are emitted as 'Time:<ps>' lines (picoseconds); the first is the
    dispatch start and the last is the final DrawDone, so their difference is the
    elapsed kernel time."""
    times = [int(m.group(1)) for line in open(path)
             for m in [re.match(r"Time:(\d+)", line.strip())] if m]
    if not times:
        raise RuntimeError(f"No 'Time:<ps>' entries found in {path}")
    return (times[-1] - times[0]) / 1e6


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
parser.add_argument('--causal', type=int, default=1, help='')
parser.add_argument('--q_fp8', type=int, default=0, help='1: Q is fp8 (1 byte), 0: bf16 (2 bytes)')
parser.add_argument('--kv_fp8', type=int, default=0, help='1: K/V are fp8 (1 byte), 0: bf16 (2 bytes)')
parser.add_argument('--time_us', type=float, default=None, help='kernel time in microseconds supplied directly (highest precedence; skips draw_log/trace parsing)')
parser.add_argument('--draw_log', type=str, default=None, help='parse runtime from a draw.log instead of results_kernel_trace.csv')
parser.add_argument('--path', type=str, default="res", help='')
parser.add_argument('--repeat', type=int, default=1000, help='')
parser.add_argument('--kernel_names', type=str, nargs='*', help='')

args = parser.parse_args()

# Three ways to obtain the per-call kernel time, in precedence order:
#   1. --time_us : user supplies the time directly
#   2. --draw_log: parse the FFM emulator draw.log (AM trace-collection path)
#   3. (default) : parse the rocprofv3 kernel trace in results_kernel_trace.csv
if args.time_us is not None:
    report(args, args.time_us, "user (--time_us)")
    sys.exit(0)

if args.draw_log:
    report(args, time_from_draw_log(args.draw_log), f"draw_log ({args.draw_log})")
    sys.exit(0)

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
    mem_BW, tflops = compute_metrics(args, np.mean(vals))
    print(f"{k}:{args.bs},{args.prefill_cnt},{args.decode_cnt},{args.window_size},{args.seq_q_l},{args.seq_kv_l},{args.num_heads_q},{args.num_heads_k},{args.head_size}, {results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW}, {tflops}")
    k = name_map[k]
    file_path = f"{path}/{k}_data.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w") as fptr:
            print("batch size,prefill_cnt,decode_cnt,window_size,seq q len, seq kv len,num_heads_q,num_heads_k,head size,avg,min,max,std,median,BW(GB/s), TFLOPs", file=fptr)
    with open(file_path, "a") as fptr:
        print(f"{args.bs},{args.prefill_cnt},{args.decode_cnt},{args.window_size},{args.seq_q_l},{args.seq_kv_l},{args.num_heads_q},{args.num_heads_k},{args.head_size},{results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW}, {tflops}", file=fptr)