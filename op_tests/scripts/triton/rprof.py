import pandas as pd
import numpy as np
import argparse
import os

"""
rm -rf $HOME/.triton/cache

rocprofv2 --kernel-trace -o res \
python3 $HOME/aiter/op_tests/op_benchmarks/triton/bench_rope.py \
    -l thd \
    -T 8192 \
    -H 128  \
    -Q 1 \
    -D 64 \
    --cached=true \
    --rotate_style=gptj \
    --pos=true \
    --offs=false \
    --two_inputs=true \
    --inplace=true \
    --dtype=bf16 \
    --nope=false \
    --nope_first=false \
    --reuse_freqs_front_part=true \
    --repeat=1000 \
    > terminal_out
mem=$(grep "Total memory" terminal_out | awk '{print $4}' | head -n1)
flop=$(grep "Total flops" terminal_out | awk '{print $4}' | head -n1)
python rocprof-results-median.py results_res.csv -k rope -m $mem -f $flop

"""

parser = argparse.ArgumentParser(description="")
parser.add_argument("filename", type=str, help="")
parser.add_argument('-k', type = str, nargs = '+', help = 'list of keywords for kernel names')
parser.add_argument("-m", type = float, help="Memory (GB)", default=0)
parser.add_argument("-f", type = float, help="Operations (TFLOPS)", default=0)
args = parser.parse_args()

filename = args.filename
kernel_name_key_list = args.k
mem = args.m
flops = args.f
runtime_list = None

df = pd.read_csv(filename)
all_kernel_name=df['Kernel_Name']
unique_kernel_name_list = list({v: 1 for v in all_kernel_name}.keys())
target_kernel_list = []
for a_unique_kernel_name in unique_kernel_name_list:
    for a_kernel_name_key in kernel_name_key_list:
        if a_kernel_name_key in a_unique_kernel_name:
            target_kernel_list.append(a_unique_kernel_name)
            break

kernel_runtime_list_list = []
print("Kernel detected:")
for a_target_kernel_name in target_kernel_list:
    print(f"\t{a_target_kernel_name}")
    df_tmp = df[df['Kernel_Name'] == a_target_kernel_name]
    duration = (df_tmp['End_Timestamp'] - df_tmp['Start_Timestamp']).to_numpy()
    kernel_runtime_list_list.append(duration)

runtime_list = kernel_runtime_list_list[0]
for i in range(1, len(kernel_runtime_list_list)):
    runtime_list = runtime_list + kernel_runtime_list_list[i]

runtime_list = runtime_list/1e3
sort_idx = np.argsort(runtime_list)
p50_idx = sort_idx[len(sort_idx)//2  ]
p25_idx = sort_idx[len(sort_idx)//4  ]
p75_idx = sort_idx[len(sort_idx)//4*3]
runtime    = runtime_list[p50_idx]
runtime_25 = runtime_list[p25_idx]
runtime_75 = runtime_list[p75_idx]

# print(f"{runtime : .3f}, {runtime_25 : .3f}, {runtime_75 : .3f}")

print(f"{runtime : .3f} (us)")
if mem > 0:
    print(f"{(mem) : .6e} (GB)")
    print(f"{(mem/runtime*1e6) : .2f} (GB/s)")

if flops > 0:
    print(f"{(flops) : .6e} (TLOPS)")
    print(f"{(flops/runtime*1e6) : .2f} (TLOPS/s)")

