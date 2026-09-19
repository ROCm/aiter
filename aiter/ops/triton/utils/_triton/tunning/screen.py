import argparse
import os
import subprocess
import sys
from itertools import product

import triton
from _utils import (
    BACKEND_ENV,
    check_backend_allowed,
    get_arch,
    get_backend,
    get_schema,
    pre_pruning_rules,
    schema_name_for,
)

# The gluon schemas spell the block sizes without SIZE; the legacy
# --block-size-*-range flags still drive them so existing invocations keep
# working across both backends.
_BLOCK_ALIASES = {
    "BLOCK_M": "BLOCK_SIZE_M",
    "BLOCK_N": "BLOCK_SIZE_N",
    "BLOCK_K": "BLOCK_SIZE_K",
}


def default_range(key: str, backend: str, M: int, N: int, K: int, args):
    """Default search range for a schema key not pinned by a flag."""
    if key in ("BLOCK_SIZE_M", "BLOCK_M"):
        if args.block_size_m_range:
            return list(args.block_size_m_range)
        rng = [4, 8] if backend == "triton" else [16]
        return rng + [v for v in (16, 32, 64, 128, 256, 512) if v <= M and v not in rng]
    if key in ("BLOCK_SIZE_N", "BLOCK_N"):
        if args.block_size_n_range:
            return list(args.block_size_n_range)
        return [16] + [v for v in (32, 64, 128, 256) if v <= N]
    if key in ("BLOCK_SIZE_K", "BLOCK_K"):
        if args.block_size_k_range:
            return list(args.block_size_k_range)
        return [128] + [v for v in (256, 512, 1024) if v <= K]
    if key == "NUM_KSPLIT":
        spk = [1]
        for v in args.num_ksplit_range:
            if K % v == 0 and v not in spk:
                spk.append(v)
        return spk
    if key == "GROUP_SIZE_M":
        return list(args.group_size_m_range)
    if key == "num_warps":
        if args.num_warps_range:
            return list(args.num_warps_range)
        # gfx1250 gluon kernels are built for 1/2/4 warps; 8 does not lower.
        return [1, 2, 4] if backend == "gluon" else [1, 4, 8]
    if key == "num_stages":
        return list(args.num_stages_range)
    if key == "waves_per_eu":
        return list(args.waves_per_eu_range)
    if key == "matrix_instr_nonkdim":
        return list(args.matrix_instr_nonkdim_range)
    if key == "cache_modifier":
        return list(args.cache_modifier_range)
    if key == "NUM_BUFFERS":
        return [2, 3, 4, 5, 6, 8]
    if key == "kernel_type":
        return [0, 1]
    if key in ("CTAS_M", "CTAS_N"):
        return [1]
    if key == "B_SCALE_TDM":
        return [0, 1]
    if key == "LOOP_UNROLL_FACTOR":
        return [1, 2]
    raise ValueError(
        f"No default range for schema key '{key}'. Pass one with --param {key} ..."
    )


def build_parms(schema, backend: str, M: int, N: int, K: int, args):
    """Search space for every key in the active schema, in schema order."""
    overrides = {}
    for entry in args.param:
        assert (
            len(entry) >= 2
        ), f"--param needs a key and at least one value, got {entry}"
        overrides[entry[0]] = [int(v) for v in entry[1:]]

    schema_keys = [k for k, _ in schema]
    for key in overrides:
        if key not in schema_keys and _BLOCK_ALIASES.get(key) not in schema_keys:
            raise AssertionError(
                f"--param {key} is not in this schema. Keys: {schema_keys}"
            )

    parms = {}
    for key in schema_keys:
        if key in overrides:
            parms[key] = overrides[key]
        elif _BLOCK_ALIASES.get(key) in overrides:
            parms[key] = overrides[_BLOCK_ALIASES[key]]
        else:
            parms[key] = default_range(key, backend, M, N, K, args)
        assert len(parms[key]) > 0, f"Empty range for {key}"
    return parms


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
        "--backend",
        type=str,
        choices=["triton", "gluon"],
        default=None,
        help="Kernel backend to tune. Default: gluon on gfx1250, triton elsewhere. "
        "Selects the config schema, so it also changes which params are searched.",
    )
    parser.add_argument(
        "--param",
        action="append",
        nargs="+",
        metavar=("KEY", "VALUE"),
        default=[],
        help="Set any schema param range, e.g. --param NUM_BUFFERS 2 4 6. Encoded "
        "params take ints: kernel_type 0='bandwidth_bound' 1='compute_bound', "
        "cache_modifier 0='.cg' 1=null, B_SCALE_TDM 0/1. Repeatable; overrides "
        "the matching legacy flag.",
    )
    parser.add_argument(
        "--block-size-m-range",
        nargs="+",
        type=int,
        help="BLOCK_SIZE_M (gluon: BLOCK_M) range",
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
        help="num_warps range (default: 1 2 4 for gluon, 1 4 8 for triton)",
        default=[],
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
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds per batch of rocprofv3 (default: 900)",
        default=900,
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
    block_size_m_range = args.block_size_m_range
    block_size_n_range = args.block_size_n_range
    block_size_k_range = args.block_size_k_range
    block_size_m_range = args.block_size_m_range
    block_size_n_range = args.block_size_n_range
    block_size_k_range = args.block_size_k_range

    force_overwrite = args.overwrite
    verbose = args.verbose
    batch_timeout = args.timeout

    assert M == triton.next_power_of_2(M), "M has to be power of 2"
    assert os.path.isfile(ut_filename), f"{ut_filename} not found"
    assert all(
        v == triton.next_power_of_2(v) for v in block_size_m_range
    ), "All possible BLOCK_SIZE_M must be power of 2"
    assert all(
        v == triton.next_power_of_2(v) for v in block_size_n_range
    ), "All possible BLOCK_SIZE_N must be power of 2"
    assert all(
        v == triton.next_power_of_2(v) for v in block_size_k_range
    ), "All possible BLOCK_SIZE_K must be power of 2"

    backend = args.backend if args.backend is not None else get_backend()
    arch = get_arch()
    check_backend_allowed(ut_filename, backend, arch)
    schema = get_schema(ut_filename, backend)
    schema_name = schema_name_for(ut_filename, backend)
    print(f"Arch: {arch}", flush=True)
    print(f"Backend: {backend} (schema: {schema_name})", flush=True)
    print()

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

    # First three schema keys are the block sizes; batching and block-level
    # exclusion below both key off config_list[0:3].
    block_label = ", ".join(k for k, _ in schema[:3])

    parms = build_parms(schema, backend, M, N, K, args)
    print("Raw tunning space:", flush=True)
    for k, v in parms.items():
        print(f"\t{k} = {v}", flush=True)

    parms_comb_list = list(product(*parms.values()))
    parms_comb_list_pruned = []
    print()
    print("Pre-pruning cases...", flush=True)
    n_case_remove = 0
    for config_list in parms_comb_list:
        if pre_pruning_rules(M, N, K, config_list, verbose=verbose, schema=schema):
            n_case_remove += 1
            continue
        parms_comb_list_pruned.append(config_list)
    print(f"{n_case_remove} cases are removed during pre-pruning", flush=True)
    print(f"Total number of cases to run: {len(parms_comb_list_pruned)}", flush=True)
    print()
    parms_comb_list = parms_comb_list_pruned
    assert len(parms_comb_list) > 0, (
        "Every config was pruned. Widen the ranges, or check that the block "
        "sizes fit the LDS budget at this NUM_BUFFERS."
    )
    # Log names are backend-qualified so a gluon sweep never overwrites a triton
    # one for the same shape, and view-screen.py can tell them apart.
    file_tag = f"{ut_filename}-{backend}-{M}-{N}-{K}"
    log_filename = f"screen-{file_tag}.log"
    print(f"Screening results will be output to {log_filename}", flush=True)
    print()
    assert force_overwrite or not os.path.isfile(
        log_filename
    ), f"{log_filename} exists, please save your file somewhere else or use --overwrite to force overwrite log files"
    s = " ".join([str(v) for v in parms])
    echo_to_file(f"Number of combinations = {len(parms_comb_list)}", log_filename, True)
    # view-screen.py reads these two lines to decode the screencase columns.
    echo_to_file(f"backend = {backend}", log_filename)
    echo_to_file(f"schema = {schema_name}", log_filename)
    echo_to_file(f"{s}", log_filename)
    i_comb_start = 0
    comb_max_batch = int(os.environ.get("SCREEN_MAX_BATCH", "100"))
    date_to_file(log_filename)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = f"{G}"
    # The ut script reads the backend from the environment: argv is a positional
    # stream of config ints chunked by schema length, so an extra positional
    # there would desync parsing.
    env[BACKEND_ENV] = backend
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
            mnk_str = f"({block_label}) = {parms_comb_list[skip_i_comb_start][:3]}"
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

        mnk_str = f"({block_label}) = {parms_comb_list[i_comb_start][:3]}"
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

        if os.path.isfile(rocprof_filename):
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
        try:
            stdout_data, stderr_data = process.communicate(timeout=batch_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_data, stderr_data = process.communicate()
            if verbose:
                print(
                    f"[Error]: rocprofv3 timed out after {batch_timeout}s for {mnk_str}",
                    flush=True,
                )
            stderr_data = "TimeoutExpired"

        if process.returncode == 0:
            if os.path.isfile(rocprof_filename):
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
                        for stderr_str in stderr_data:
                            print(f"\t{stderr_str}", flush=True)
            else:
                if verbose:
                    print(f"[Error]: {rocprof_filename} not found", flush=True)
        else:
            stderr_data = stderr_data.split("\n")
            if verbose:
                print("[Error]: when running rocprof, error message:", flush=True)
            # Determine if this is a block-size-dependent error (exclude block)
            # or a param-specific error (skip batch, don't exclude block)
            is_block_size_error = False
            for i_line, aline in enumerate(stderr_data):
                if (
                    "exceeds triton maximum tensor numel" in aline
                    or "OutOfResources" in aline
                ):
                    is_block_size_error = True
                    if verbose:
                        print("\t...", flush=True)
                        for j_line in range(
                            max(0, i_line - 5), min(len(stderr_data), i_line + 5)
                        ):
                            print(f"\t{stderr_data[j_line]}", flush=True)
                        print("\t...", flush=True)
                    break
                elif (
                    "PassManager::run failed" in aline
                    or "RuntimeError" in aline
                    or "AssertionError" in aline
                    or "TimeoutExpired" in aline
                ):
                    # Compilation or runtime error for specific param combo,
                    # not necessarily all configs with this block size
                    if verbose:
                        print(
                            "\tParam-specific error (not excluding block size):",
                            flush=True,
                        )
                        for j_line in range(
                            max(0, i_line - 5), min(len(stderr_data), i_line + 5)
                        ):
                            print(f"\t{stderr_data[j_line]}", flush=True)
                    break
            else:
                if verbose:
                    print("\tUn-identified error:", flush=True)
                    for stderr_str in stderr_data:
                        print(f"\t{stderr_str}", flush=True)

            if is_block_size_error:
                exclude_mnk[tuple(parms_comb_list[i_comb_start][:3])] = 1
                if verbose:
                    print(f"Excluding all {mnk_str} cases", flush=True)
                    print()
            else:
                if verbose:
                    print(
                        f"Skipping batch {i_comb_start}~{i_comb_end-1} (block size NOT excluded)",
                        flush=True,
                    )
                    print()

        i_comb_start = i_comb_end
        date_to_file(log_filename)
    echo_to_file("Screen complete", log_filename)


if __name__ == "__main__":
    sys.exit(main())
