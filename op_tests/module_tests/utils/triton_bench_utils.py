import pandas as pd
import numpy as np


import subprocess


def get_avg_latency(filename, kernel_name_key_list):
    df = pd.read_csv(filename)
    all_kernel_name = df["Kernel_Name"]
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
        df_tmp = df[df["Kernel_Name"] == a_target_kernel_name]
        duration = (df_tmp["End_Timestamp"] - df_tmp["Start_Timestamp"]).to_numpy()
        kernel_runtime_list_list.append(duration)

    runtime_list = kernel_runtime_list_list[0]
    for i in range(1, len(kernel_runtime_list_list)):
        runtime_list = runtime_list + kernel_runtime_list_list[i]

    runtime_list = runtime_list / 1e3
    sort_idx = np.argsort(runtime_list)
    p50_idx = sort_idx[len(sort_idx) // 2]
    # p25_idx = sort_idx[len(sort_idx) // 4]
    # p75_idx = sort_idx[len(sort_idx) // 4 * 3]
    runtime = runtime_list[p50_idx]
    # runtime_25 = runtime_list[p25_idx]
    # runtime_75 = runtime_list[p75_idx]

    print(f"{runtime : .3f} (us)")
    return runtime


def run_triton_a4w4(M, N, K):
    cmd = [
        "rocprofv2",
        "--kernel-trace",
        "-o",
        "res",
        "python3",
        "../../op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py",
        "--shape",
        str(M),
        str(N),
        str(K),
    ]
    command_str = " ".join(cmd)
    print("command_string={}".format(command_str))
    subprocess.run(cmd, stderr=subprocess.STDOUT, text=True)

    filename = "results_res.csv"
    kernel_name_key_list = ["gemm"]
    return get_avg_latency(filename, kernel_name_key_list)


def run_triton_a8w8_blockscale(M, N, K):
    cmd = [
        "rocprofv2",
        "--kernel-trace",
        "-o",
        "res",
        "python3",
        "../../op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py",
        "--shape",
        str(M),
        str(N),
        str(K),
    ]
    command_str = " ".join(cmd)
    print("command_string={}".format(command_str))
    subprocess.run(cmd, stderr=subprocess.STDOUT, text=True)

    filename = "results_res.csv"
    kernel_name_key_list = ["gemm"]
    return get_avg_latency(filename, kernel_name_key_list)


def run_triton_a8w8_per_token(M, N, K):
    cmd = [
        "rocprofv2",
        "--kernel-trace",
        "-o",
        "res",
        "python3",
        "../../op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py",
        "--shape",
        str(M),
        str(N),
        str(K),
    ]
    command_str = " ".join(cmd)
    print("command_string={}".format(command_str))
    subprocess.run(cmd, stderr=subprocess.STDOUT, text=True)

    filename = "results_res.csv"
    kernel_name_key_list = ["gemm"]
    return get_avg_latency(filename, kernel_name_key_list)


def run_triton_a8w8_per_tensor(M, N, K):
    cmd = [
        "rocprofv2",
        "--kernel-trace",
        "-o",
        "res",
        "python3",
        "../../op_tests/op_benchmarks/triton/bench_gemm_a8w8.py",
        "--shape",
        str(M),
        str(N),
        str(K),
    ]
    command_str = " ".join(cmd)
    print("command_string={}".format(command_str))
    subprocess.run(cmd, stderr=subprocess.STDOUT, text=True)

    filename = "results_res.csv"
    kernel_name_key_list = ["gemm"]
    return get_avg_latency(filename, kernel_name_key_list)


def run_triton_a16w16(M, N, K):
    cmd = [
        "rocprofv2",
        "--kernel-trace",
        "-o",
        "res",
        "python3",
        "../../op_tests/op_benchmarks/triton/bench_gemm_a16w16.py",
        "--shape",
        str(M),
        str(N),
        str(K),
    ]
    command_str = " ".join(cmd)
    print("command_string={}".format(command_str))
    subprocess.run(cmd, stderr=subprocess.STDOUT, text=True)

    filename = "results_res.csv"
    kernel_name_key_list = ["gemm"]
    return get_avg_latency(filename, kernel_name_key_list)
