import os
import sys
import ater
import pandas as pd
import torch
import torch.nn.functional as F
import ater
from ater.test_common import checkAllclose, perftest
from gemm_a8w8_common import kernelInstance, kernels_list
import argparse



def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)

def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(untuned_gemm_file), f"Not exist a8w8_untuned_gemm.csv file: {untuned_gemm_file}"
    untunedf = pd.read_csv(untuned_gemm_file)
    return untunedf

def get_tuned_gemm_list(tuned_gemm_file):
    if os.path.exists(tuned_gemm_file):
        tunedf = pd.read_csv(tuned_gemm_file)
    else:
        tunedf = pd.DataFrame(columns=["M", "N", "K", "kernelId", "us", "kernelName"])
    return tunedf

@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id):
    ater.gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernel_id)
    return out


def tune_gemm(m, n, k):
    dim = (m, n, k)
    x = torch.randint(-20, 20, (m, k), dtype=torch.int8, device="cuda")
    weight = torch.randint(-20, 20, (n, k), dtype=torch.int8, device="cuda")
    x_scale = torch.rand([m, 1], dtype=torch.bfloat16, device="cuda")
    w_scale = torch.rand([1, n], dtype=torch.bfloat16, device="cuda")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

    ref_out = run_torch(x, weight, x_scale, w_scale)

    print(f"*******************M:{m} X N:{n} X K:{k}**************************")
    print(f"Start tuning a8w8 gemm kernel for M:{m}, N:{n}, K{k}:")
    kernels_num = len(kernels_list)
    best_kernelId = -1
    best_time = -1
    for i in range(kernels_num):
        try:
            (out), avg_t = kernel_instance_test(x, weight, x_scale, w_scale, out, i)
            isClosed = torch.isclose(out, ref_out, rtol=1e-3, atol=1000)
            if isClosed.all():
                print(f"{str(dim):<20} kernelid:{i:<3d}\t avg: {avg_t:<8.2f} us, {kernels_list[i].name}")
                if best_time < 0 or avg_t < best_time:
                    best_kernelId = i
                    best_time = avg_t
            else:
                print(f"{str(dim):<20} kernelid:{i:<3d}\t No pass, {kernels_list[i].name}") 
        except RuntimeError as e:
            print(f"{str(dim):<20} kernelid:{i:<3d}\t No support, {kernels_list[i].name}") 

    if best_kernelId == -1:
        print(f"No kernel can be used for M:{m}, N:{n}, K:{k}")
    else:
        best_time = round(best_time, 4)
        print(f"Tuning result for M:{m}, N:{n}, K:{k} is {best_kernelId} {best_time}")
    print(f"*******************M:{m} X N:{n} X K{k}**************************")
    
    return best_kernelId, best_time


def tune_gemm_list(untunedf, tunedf, issorted = False):
    for i in range(len(untunedf)):
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]
        
        if tunedf[(tunedf["M"]==M) & (tunedf["N"]==N) & (tunedf["K"]==K)].empty:
            kernelId, time = tune_gemm(M, N, K)
            temp = pd.DataFrame({"M":[M], "N":[N], "K":[K], "kernelId":[kernelId], 
                           "us":[time], "kernelName":[kernels_list[kernelId].name]})
            tunedf = pd.concat([tunedf, temp], ignore_index=True)

        else:
            print(f"M:{M}, N:{N}, K{K} is in tuned gemm, skip!!!")
        print()
        print()
    if issorted:
        tunedf = tunedf.sort_values(by=["M", "N", "K"])
    print("Totall tuning result:")
    print(tunedf)
    return tunedf



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a8w8 kernel",
    )

    parser.add_argument(
        "-i",
        "--untune_file",
        default="ater/configs/a8w8_untuned_gemm.csv",
        required=False,
        help="input"
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="ater/configs/a8w8_tuned_gemm.csv",
        required=False,
        help="output: tuning result store this file"
    )

    parser.add_argument(
        "--sort",
        action='store_true',
        required=False,
        help="Arranged according to the M N K size"
    )

    args = parser.parse_args()
    untunedf = get_untuned_gemm_list(args.untune_file)
    tunedf = get_tuned_gemm_list(args.tune_file)
    tunedf = tune_gemm_list(untunedf, tunedf, args.sort)
    tunedf.to_csv(args.tune_file, index=False)
