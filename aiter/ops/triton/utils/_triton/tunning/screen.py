from itertools import product
import os
import sys
import triton
import argparse
import subprocess
from _utils import pre_pruning_rules


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
        "--block-size-m-range",
        nargs="+",
        type=int,
        help="BLOCK_SIZE_M range",
        default=[],
    )
    parser.add_argument(
        "--block-size-n-range",
        nargs="+",
        type=int,
        help="BLOCK_SIZE_N range",
        default=[],
    )
    parser.add_argument(
        "--block-size-k-range",
        nargs="+",
        type=int,
        help="BLOCK_SIZE_K range",
        default=[],
    )
    parser.add_argument(
        "--num-ksplit-range",
        nargs="+",
        type=int,
        help="NUM_KSPLIT range (only included the elements by which K is divisible)",
        default=[3, 4, 7, 8, 14, 16, 28],
    )
    parser.add_argument(
        "--group-size-m-range",
        nargs="+",
        type=int,
        help="GROUP_SIZE_M range",
        default=[1, 4, 8],
    )
    parser.add_argument(
        "--num-warps-range",
        nargs="+",
        type=int,
        help="GROUP_SIZE_M range",
        default=[1, 4, 8],
    )
    parser.add_argument(
        "--num-stages-range",
        nargs="+",
        type=int,
        help="GROUP_SIZE_M range",
        default=[1, 2],
    )
    parser.add_argument(
        "--waves-per-eu-range",
        nargs="+",
        type=int,
        help="GROUP_SIZE_M range",
        default=[1, 2, 4, 6, 8],
    )
    parser.add_argument(
        "--matrix-instr-nonkdim-range",
        nargs="+",
        type=int,
        help="matrix_instr_nonkdim range",
        default=[16],
    )
    parser.add_argument(
        "--cache-modifier-range",
        nargs="+",
        type=int,
        help="cache_modifier range (0 = '.cg', 1 = null)",
        default=[0, 1],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite log files",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose print",
        default=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    M = getattr(args, "M")
    N = getattr(args, "N")
    K = getattr(args, "K")
    G = getattr(args, "G")
    ut_filename = getattr(args, "F")
    block_size_m_range = getattr(args, "block_size_m_range")
    block_size_n_range = getattr(args, "block_size_n_range")
    block_size_k_range = getattr(args, "block_size_k_range")
    num_ksplit_range = getattr(args, "num_ksplit_range")
    group_size_m_range = getattr(args, "group_size_m_range")
    num_warps_range = getattr(args, "num_warps_range")
    num_stages_range = getattr(args, "num_stages_range")
    waves_per_eu_range = getattr(args, "waves_per_eu_range")
    matrix_instr_nonkdim_range = getattr(args, "matrix_instr_nonkdim_range")
    cache_modifier_range = getattr(args, "cache_modifier_range")

    force_overwrite = getattr(args, "overwrite")
    verbose = getattr(args, "verbose")

    assert M == triton.next_power_of_2(M), "M has to be power of 2"
    assert os.path.isfile(ut_filename), f"{ut_filename} not found"
    assert all(
        [v == triton.next_power_of_2(v) for v in block_size_m_range]
    ), "All possible BLOCK_SIZE_M must be power of 2"
    assert all(
        [v == triton.next_power_of_2(v) for v in block_size_n_range]
    ), "All possible BLOCK_SIZE_N must be power of 2"
    assert all(
        [v == triton.next_power_of_2(v) for v in block_size_k_range]
    ), "All possible BLOCK_SIZE_K must be power of 2"

    # default m, n, k, split-k range
    if len(block_size_m_range) == 0:
        block_size_m_range = [4, 8]
        possible_ms = [16, 32, 64, 128, 256, 512]
        block_size_m_range += [v for v in possible_ms if v <= M]

    if len(block_size_n_range) == 0:
        block_size_n_range = [16]
        possible_ns = [32, 64, 128, 256]
        block_size_n_range += [v for v in possible_ns if v <= N]

    if len(block_size_k_range) == 0:
        block_size_k_range = [128]
        possible_ks = [256, 512, 1024]
        block_size_k_range += [v for v in possible_ks if v <= K]

    spk_range = [1]
    for spk in num_ksplit_range:
        if K % spk == 0 and spk not in spk_range:
            spk_range.append(spk)

    ############################################################
    # # for AFP4WFP4_GEMM_preshuffe
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
        "BLOCK_SIZE_M": block_size_m_range,
        "BLOCK_SIZE_N": block_size_n_range,
        "BLOCK_SIZE_K": block_size_k_range,
        "GROUP_SIZE_M": group_size_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stages_range,
        "waves_per_eu": waves_per_eu_range,
        "matrix_instr_nonkdim": matrix_instr_nonkdim_range,
        "cache_modifier": cache_modifier_range,
        "NUM_KSPLIT": spk_range,
    }
    print("Raw tunning space:", flush=True)
    for k, v in parms.items():
        print(f"\t{k} = {v}", flush=True)

    parms_comb_list = list(product(*parms.values()))
    parms_comb_list_pruned = []
    print()
    print(f"Pre-pruning cases...", flush=True)
    n_case_remove = 0
    for config_list in parms_comb_list:
        if pre_pruning_rules(M, N, K, config_list, verbose=verbose):
            n_case_remove += 1
            continue
        parms_comb_list_pruned.append(config_list)
    print(f"{n_case_remove} cases are removed during pre-pruning", flush=True)
    print(f"Total number of cases to run: {len(parms_comb_list_pruned)}", flush=True)
    print()
    parms_comb_list = parms_comb_list_pruned
    file_tag = f"{ut_filename}-{M}-{N}-{K}"
    log_filename = f"screen-{file_tag}.log"
    print(f"Screening results will be output to {log_filename}", flush=True)
    print()
    assert (
        force_overwrite == True or os.path.isfile(log_filename) == False
    ), f"{log_filename} exists, please save your file somewhere else or use --overwrite to force overwrite log files"
    s = " ".join([str(v) for v in parms.keys()])
    echo_to_file(f"Number of combinations = {len(parms_comb_list)}", log_filename, True)
    echo_to_file(f"{s}", log_filename)
    i_comb_start = 0
    comb_max_batch = 100
    date_to_file(log_filename)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = f"{G}"
    exclude_mnk = {}
    while i_comb_start < len(parms_comb_list):
        skip_i_comb_start = i_comb_start
        skip_i_comb_end = i_comb_start
        while (
            i_comb_start < len(parms_comb_list)
            and tuple(parms_comb_list[i_comb_start][0:3]) in exclude_mnk
        ):
            skip_i_comb_end = i_comb_start
            i_comb_start += 1
        if skip_i_comb_end > skip_i_comb_start:
            mnk_str = f"(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) = {parms_comb_list[skip_i_comb_start][:3]}"
            print(
                f"Skipping case {skip_i_comb_start} ~ {skip_i_comb_end}: {mnk_str}",
                flush=True,
            )
        if i_comb_start >= len(parms_comb_list):
            break
        i_comb_end = i_comb_start + 1
        while (
            i_comb_end < len(parms_comb_list)
            and i_comb_end - i_comb_start < comb_max_batch
            and parms_comb_list[i_comb_start][0:3] == parms_comb_list[i_comb_end][0:3]
        ):
            i_comb_end += 1

        mnk_str = f"(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) = {parms_comb_list[i_comb_start][:3]}"
        print(f"Running case {i_comb_start} ~ {i_comb_end - 1}: {mnk_str}", flush=True)
        echo_to_file(
            f"Running case {i_comb_start} ~ {i_comb_end - 1}: {mnk_str}", log_filename
        )
        comb_str = ""
        for a_comb in parms_comb_list[i_comb_start:i_comb_end]:
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

        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
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

                    for a_comb in parms_comb_list[i_comb_start:i_comb_end]:
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
                    if verbose:
                        print(f"[Error]: {rocprof_filename} reading error:", flush=True)
                        for l in stderr_data:
                            print(f"\t{l}", flush=True)
            else:
                if verbose:
                    print(f"[Error]: {rocprof_filename} not found", flush=True)
        else:
            stderr_data = stderr_data.split("\n")
            if verbose:
                print(f"[Error]: when running rocprof, error message:", flush=True)
            for i_line, aline in enumerate(stderr_data):
                if (
                    "exceeds triton maximum tensor numel" in aline
                    or "OutOfResources" in aline
                    or "AssertionError" in aline
                ):
                    if verbose:
                        print(f"\t...", flush=True)
                        for j_line in range(
                            max(0, i_line - 5), min(len(stderr_data), i_line + 5)
                        ):
                            print(f"\t{stderr_data[j_line]}", flush=True)
                        print(f"\t...", flush=True)
                    break
            else:
                if verbose:
                    print(f"\tUn-identified error:", flush=True)
                    for l in stderr_data:
                        print(f"\t{l}", flush=True)

            exclude_mnk[tuple(parms_comb_list[i_comb_start][:3])] = 1

            if verbose:
                print(f"Excluding all {mnk_str} cases", flush=True)
                print(f"", flush=True)

        i_comb_start = i_comb_end
        date_to_file(log_filename)
    echo_to_file("Screen complete", log_filename)


if __name__ == "__main__":
    sys.exit(main())
