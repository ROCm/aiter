import os
import triton
from triton.testing import runtime
import torch
import triton.language as tl


@triton.jit
def split_dummy(d_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(d_ptr + pid)
    x = x + 1
    tl.store(d_ptr + pid, x)
    return


def run_profile(fn: callable, n_run: int = 250):
    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    for _ in range(n_run):
        cache.zero_()
        di.synchronize()
        fn()
        di.synchronize()
    d = torch.empty(128, dtype=torch.float32, device="cuda")
    cache.zero_()
    di.synchronize()
    split_dummy[(128,)](d)
    di.synchronize()


config_parms_key = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "cache_modifier",
    "NUM_KSPLIT",
]


def get_config_list(argv: list[str]) -> list[dict | None]:
    config_argv = argv
    num_config_parms_key = len(config_parms_key)
    config_list = []
    while len(config_argv) >= num_config_parms_key:
        config_list.append(
            {
                config_parms_key[i]: int(config_argv[i])
                for i in range(num_config_parms_key)
            }
        )
        config_list[-1]["cache_modifier"] = (
            ".cg" if config_list[-1]["cache_modifier"] == 0 else None
        )
        config_argv = config_argv[num_config_parms_key:]

    if len(config_list) == 0:
        config_list = [None]

    return config_list


def get_input_shape(argv: list[str]) -> list[int]:
    return [int(v) for v in argv]


def get_input_shape_and_config_list(
    argv: list[str], shape_size: int = 3
) -> tuple[list[int], list[dict | None]]:
    input_shape = get_input_shape(argv[1 : shape_size + 1])
    config_list = get_config_list(argv[shape_size + 1 :])
    return input_shape, config_list


def read_screen_file(filename, case_data):
    err_lines_limit = 100000
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            for newline in f:
                try:
                    err_lines = 0
                    while not newline.startswith("screencase"):
                        newline = f.readline()
                        err_lines += 1
                        if err_lines >= err_lines_limit:
                            break
                    screencaseline = newline[:]
                    err_lines = 0
                    while not newline.strip().endswith("(us)"):
                        if newline.startswith("screencase"):
                            screencaseline = newline[:]
                        newline = f.readline()
                        err_lines += 1
                        if err_lines >= err_lines_limit:
                            break
                    r = float(newline.strip().split()[0])
                    # if int(screencaseline[len("screencase")+1:].strip().split()[-1]) == 1: continue # remove this comment to consider only split-k case
                    # if int(screencaseline[len("screencase")+1:].strip().split()[-1]) != 1: continue # remove this comment to consider only split-k case
                    # if int(screencaseline[len("screencase")+1:].strip().split()[2]) != 128: continue # remove this comment to consider only BK=128
                    case_data.append(
                        [r, screencaseline[len("screencase") + 1 :].strip()]
                    )
                except:
                    break


def pre_pruning_rules(M: int, N: int, K: int, config_list: list[int], verbose: bool):
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
    ) = config_list
    # remove cases
    if BLOCK_SIZE_K >= 2 * (K // NUM_KSPLIT):
        if verbose:
            print(
                f"Remove case {config_list} because BLOCK_SIZE_K >= 2 * (K // NUM_KSPLIT)"
            )
        return True
    if NUM_KSPLIT > 1 and GROUP_SIZE_M > 1:
        if verbose:
            print(
                f"Remove case {config_list} because NUM_KSPLIT > 1 and GROUP_SIZE_M > 1"
            )
        return True
    if BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages > 1:  # k_itr == 1 case
        if verbose:
            print(
                f"Remove case {config_list} because BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages > 1"
            )
        return True
    if BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1:  # k_itr > 1 case
        if verbose:
            print(
                f"Remove case {config_list} because BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1"
            )
        return True
    return False
