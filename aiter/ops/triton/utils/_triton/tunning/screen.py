from itertools import product
import os
import sys
import triton
import argparse
import subprocess
from aiter.ops.triton.utils._triton.tunning._utils import get_config_list


def echo_to_file(msg: str, filename: str, clear: bool = False):
    if clear:
        os.popen(f"echo '{msg}' > {filename}").read()
    else:
        os.popen(f"echo '{msg}' >> {filename}").read()


def date_to_file(filename: str):
    os.popen(f"date >> {filename}").read()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, help="M dim")
    parser.add_argument("N", type=int, help="N dim")
    parser.add_argument("K", type=int, help="K dim")
    parser.add_argument("G", type=int, help="GPU card ID")
    parser.add_argument("F", type=str, help="Unit test filename")
    parser.add_argument(
        "--m-range", nargs="+", type=int, help="BLOCK_SIZE_M range", default=[]
    )
    parser.add_argument(
        "--n-range", nargs="+", type=int, help="BLOCK_SIZE_N range", default=[]
    )
    parser.add_argument(
        "--k-range", nargs="+", type=int, help="BLOCK_SIZE_K range", default=[]
    )
    parser.add_argument(
        "--k-split",
        nargs="+",
        type=int,
        help="NUM_KSPLIT range",
        default=[3, 4, 7, 8, 14, 16, 28],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite log files",
        default=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    M = args.M
    N = args.N
    K = args.K
    G = args.G
    ut_filename = args.F
    m_range = args.m_range
    n_range = args.n_range
    k_range = args.k_range
    k_split_range = args.k_split
    force_overwrite = args.overwrite

    assert M == triton.next_power_of_2(M), "M has to be power of 2"
    assert os.path.isfile(ut_filename), f"{ut_filename} not found"
    assert all(
        [v == triton.next_power_of_2(v) for v in m_range]
    ), "All possible BLOCK_SIZE_M must be power of 2"
    assert all(
        [v == triton.next_power_of_2(v) for v in n_range]
    ), "All possible BLOCK_SIZE_N must be power of 2"
    assert all(
        [v == triton.next_power_of_2(v) for v in k_range]
    ), "All possible BLOCK_SIZE_K must be power of 2"

    # default m, n, k, split-k range
    if len(m_range) == 0:
        m_range = [4, 8]
        possible_ms = [16, 32, 64, 128, 256, 512]
        m_range += [v for v in possible_ms if v <= M]

    if len(n_range) == 0:
        n_range = [16]
        possible_ns = [32, 64, 128, 256]
        n_range += [v for v in possible_ns if v <= N]

    if len(k_range) == 0:
        k_range = [128]
        possible_ks = [256, 512, 1024]
        k_range += [v for v in possible_ks if v <= K]

    spk_range = [1]
    for spk in k_split_range:
        if K % spk == 0 and spk not in spk_range:
            spk_range.append(spk)

    ############################################################
    # # for AFP4WFP4_GEMM_preshuffe please use this
    # if M >= 256:
    #     Ms = [32, 64, 128, 256]
    # elif M >= 128:
    #     Ms = [32, 64, 128]
    # elif M >= 64:
    #     Ms = [32, 64]
    # elif M >= 32:
    #     Ms = [32]
    # else:
    #     Ms = [4, 8, 16]
    # Ns = [32, 64, 128]
    # Ks = [256, 512, 1024]
    ############################################################

    ############################################################
    # # for a8w8_GEMM_blockscale/a8w8_GEMM_blockscale_preshuffe/a16w8_GEMM_blockscale/a16w8_GEMM_blockscale_preshuffe, Ks can only be 128
    # k_range = [128]
    ############################################################

    parms = {
        "BLOCK_SIZE_M": m_range,
        "BLOCK_SIZE_N": n_range,
        "BLOCK_SIZE_K": k_range,
        "GROUP_SIZE_M": [1, 4, 8],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [1, 2, 4, 6, 8],
        "matrix_instr_nonkdim": [16],
        "cache_modifier": [0, 1],
        "NUM_KSPLIT": spk_range,
    }

    comb = list(product(*parms.values()))
    comb_p = []
    for a_comb in comb:
        (
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            waves_per_eu,
            matrix_instr_nonkdim,
            cache_modifier,
            NUM_KSPLIT,
        ) = a_comb
        # skip cases
        if NUM_KSPLIT > 1 and GROUP_SIZE_M > 1:
            continue
        if BLOCK_SIZE_K > K // NUM_KSPLIT:
            continue
        if BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages != 1:  # k_itr == 1 case
            continue
        if BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1:  # k_itr > 1 case
            continue
        comb_p.append(a_comb)
    comb = comb_p
    file_tag = f"{ut_filename}-{M}-{N}-{K}"
    log_filename = f"screen-{file_tag}.log"
    assert (
        force_overwrite == True or os.path.isfile(log_filename) == False
    ), f"{log_filename} exists, please save your file somewhere else or use --overwrite to force overwrite log files"
    s = " ".join([str(v) for v in parms.keys()])
    echo_to_file(f"Number of combinations = {len(comb)}", log_filename, True)
    echo_to_file(f"{s}", log_filename)
    i_comb_start = 0
    comb_max_batch = 100
    date_to_file(log_filename)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = f"{G}"
    exclude_mnk = {}
    while i_comb_start < len(comb):
        skip_i_comb_start = i_comb_start
        skip_i_comb_end = i_comb_start
        while (
            i_comb_start < len(comb) and tuple(comb[i_comb_start][0:3]) in exclude_mnk
        ):
            skip_i_comb_end = i_comb_start
            i_comb_start += 1
        if skip_i_comb_end > skip_i_comb_start:
            mnk_str = f"(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) = {comb[skip_i_comb_start][:3]}"
            print(
                f"Skipping case {skip_i_comb_start} ~ {skip_i_comb_end}: {mnk_str}",
                flush=True,
            )
        if i_comb_start >= len(comb):
            break
        i_comb_end = i_comb_start + 1
        while (
            i_comb_end < len(comb)
            and i_comb_end - i_comb_start < comb_max_batch
            and comb[i_comb_start][0:3] == comb[i_comb_end][0:3]
        ):
            i_comb_end += 1

        mnk_str = (
            f"(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) = {comb[i_comb_start][:3]}"
        )
        print(f"Running case {i_comb_start} ~ {i_comb_end - 1}: {mnk_str}", flush=True)
        echo_to_file(
            f"Running case {i_comb_start} ~ {i_comb_end - 1}: {mnk_str}", log_filename
        )
        comb_str = ""
        for a_comb in comb[i_comb_start:i_comb_end]:
            comb_str += " ".join([str(v) for v in a_comb])
            comb_str += " "
        comb_str = comb_str.strip()

        cmd = f"""rocprofv3 --kernel-trace -f csv -o res-{file_tag} -- python3 {ut_filename} {M} {N} {K} {comb_str}"""
        cmd = cmd.split(" ")

        rocprof_filename = f"res-{file_tag}_kernel_trace.csv"

        if os.path.isfile(rocprof_filename) == True:
            process = subprocess.Popen(
                ["rm", rocprof_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            process.communicate()

        # print(cmd_list)env={"HIP_VISIBLE_DEVICES": f"{G}"}
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )  # .read()
        stdout_data, stderr_data = process.communicate()

        if process.returncode == 0:
            if os.path.isfile(rocprof_filename) == True:
                cmd_rprof = f"""python3 rprof.py {rocprof_filename} -k gemm"""
                cmd_rprof = cmd_rprof.split(" ")
                process = subprocess.Popen(
                    cmd_rprof, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout_data, stderr_data = process.communicate()
                if process.returncode == 0:
                    prof_output = stdout_data.split("\n")
                    if prof_output[-1].strip() == "":
                        prof_output.pop()
                    number_of_kernel_runtime = prof_output.count("Kernel detected:")
                    assert (i_comb_end - i_comb_start) == number_of_kernel_runtime

                    prof_output_i = 0

                    for a_comb in comb[i_comb_start:i_comb_end]:
                        s = " ".join([str(v) for v in a_comb])
                        echo_to_file(f"screencase {s}", log_filename)
                        assert prof_output[prof_output_i] == "Kernel detected:"
                        prof_output_i += 1
                        while (
                            prof_output_i < len(prof_output)
                            and prof_output[prof_output_i] != "Kernel detected:"
                        ):
                            echo_to_file(prof_output[prof_output_i], log_filename)
                            prof_output_i += 1
                else:
                    echo_to_file(
                        f"[Error]: {rocprof_filename} reading error:", log_filename
                    )
                    for l in stderr_data:
                        echo_to_file(f"\t{l}", log_filename)
            else:
                echo_to_file(f"[Error]: {rocprof_filename} not found", log_filename)
        else:
            stderr_data = stderr_data.split("\n")
            echo_to_file(f"[Error]: when running rocprof, error message:", log_filename)
            for i_line, aline in enumerate(stderr_data):
                if (
                    "exceeds triton maximum tensor numel" in aline
                    or "OutOfResources" in aline
                    or "AssertionError" in aline
                ):
                    echo_to_file(f"\t...", log_filename)
                    for j_line in range(
                        max(0, i_line - 5), min(len(stderr_data), i_line + 5)
                    ):
                        echo_to_file(f"\t{stderr_data[j_line]}", log_filename)
                    echo_to_file(f"\t...", log_filename)
                    break
            else:
                echo_to_file(f"\tUn-identified error:", log_filename)
                for l in stderr_data:
                    echo_to_file(f"\t{l}", log_filename)
            exclude_mnk[tuple(comb[i_comb_start][:3])] = 1
            echo_to_file(f"Excluding all {mnk_str} cases", log_filename)
            echo_to_file(f"", log_filename)

        i_comb_start = i_comb_end
        date_to_file(log_filename)
    echo_to_file("Screen complete", log_filename)


if __name__ == "__main__":
    sys.exit(main())
