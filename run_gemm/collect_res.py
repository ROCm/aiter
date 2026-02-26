import pandas as pd
import argparse
import os
import shutil
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from aiter.ops.triton.gemm_afp4wfp4 import (
    _get_config,
)
def match_name(candidate, names):
    for n in names:
        if candidate in n or n in candidate:
            return True, n
    return False, None

def calculate_gemm_bw(M, N, K, time, out_dtype, K_split):
    total_in_elements = ((M * K) + (N * K)) * (34.0 / 32.0) # 1 uint8 per 32 fp4 for scales
    input = total_in_elements * 0.5
    output = (M * N * K_split) * 2.0 if "16" in out_dtype else (M * N * K_split) * 4.0
    total = input + output
    GBs = (total / 10**9) / (time * 10**-6)
    return GBs

def calculate_tflops(M, N, K, time):
    flops = 2 * M * N * K
    TFLOPS = (flops / 10**12) / (time * 10**-6)
    return TFLOPS

parser = argparse.ArgumentParser(description="Collect benchmark for layer-norm.")
parser.add_argument('--M', type=int, default=8192, help='size of the vectors')
parser.add_argument('--N', type=int, default=8192, help='size of the vectors')
parser.add_argument('--K', type=int, default=8192, help='size of the vectors')
parser.add_argument('--path', type=str, default="res", help='')
parser.add_argument('--repeat', type=int, default=1000, help='')
parser.add_argument('--kernel_names', type=str, nargs='*', help='')
parser.add_argument('--out_dtype', type=str, default="float32", help='')
parser.add_argument('--shuffle_weight_scales', type=int, default=0, help='')
parser.add_argument('--override_config', type=int, default=0, help='override the config with the one in the script')
args = parser.parse_args()
path = args.path
print(args.kernel_names)
size_name = f"{args.M}_{args.N}_{args.K}"
repeat = args.repeat
kernel_names = args.kernel_names
out_dtype = args.out_dtype
shuffle_weight_scales = args.shuffle_weight_scales
if not os.path.exists(path):
    os.makedirs(path)

shutil.copyfile("results.stats.csv", f"{path}/results.stats_{size_name}.csv")
shutil.copyfile("results.csv", f"{path}/results_{size_name}.csv")
data = pd.read_csv(f"{path}/results_{size_name}.csv")
#data = pd.read_csv("results.csv")
kernel_data = defaultdict(list)
name_map = dict()
has_reduce = False
has_gemm = False
for i, f_name in enumerate(data['KernelName']):
    match, matched_name = match_name(f_name, kernel_names)
    if match:
        kernel_data[f_name].append(data['DurationNs'][i] * 10**-3)
        name_map[f_name] = matched_name

config = _get_config(args.M, args.N, args.K // 2, shuffle=shuffle_weight_scales)[0]
K_split = config["NUM_KSPLIT"]
if K_split > 0:
    out_dtype = "float32"
if args.override_config:
    K_split = 1
    out_dtype = "bfloat16"
res_dict = dict()
for k, vals in kernel_data.items():
    vals = vals[-repeat:]
    vals = np.sort(vals)
    outliers = len(vals) // 10
    vals = vals[outliers:-outliers]
    if len(vals) == 0:
        results = [0.0] * 7
    else:
        bw = calculate_gemm_bw(args.M, args.N, args.K, np.mean(vals), out_dtype, K_split)
        tflops = calculate_tflops(args.M, args.N, args.K, np.mean(vals))
        results = [np.mean(vals), np.min(vals), np.max(vals), np.std(vals),np.median(vals),bw, tflops]
    print(k)
    print("----------------------------------------------------------------")
    print("M,N,K,avg,min,max,std,median, BW (GB/s), TFLOPs")    
    print(f"{args.M},{args.N},{args.K},{results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}, {results[6]}")
    print("----------------------------------------------------------------")
    k = name_map[k]
    file_path = f"{path}/{k}_data.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w") as fptr:
            print("M,N,K,avg,min,max,std,median,BW (GB/s),TFLOPs", file=fptr)
    with open(file_path, "a") as fptr:
        print(f"{args.M},{args.N},{args.K},{results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}, {results[6]}", file=fptr)