import os
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.testing import runtime


@triton.jit
def split_dummy(d_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(d_ptr + pid)
    x = x + 1
    tl.store(d_ptr + pid, x)


def run_profile(fn: Callable, n_run: int = 250):
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
                except IndexError:
                    break


def pre_pruning_rules(M: int, N: int, K: int, config_list: list[int], verbose: bool):
    (
        _BLOCK_SIZE_M,
        _BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        _num_warps,
        num_stages,
        _waves_per_eu,
        _matrix_instr_nonkdim,
        _cache_modifier,
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
    if _extra_pruning_rules(
        M, N, _BLOCK_SIZE_M, _BLOCK_SIZE_N, _num_warps, config_list, verbose
    ):
        return True
    return False


def _extra_pruning_rules(
    M: int,
    N: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    num_warps: int,
    config_list: list[int],
    verbose: bool,
):
    """Opt-in cross-dimension rules, enabled only via $SCREEN_EXTRA_PRUNE.

    Disabled by default, so every existing tuning flow is unaffected. The spec
    is a comma-separated key=value list:

      B=<batch>   batch dim of a batched GEMM (grid multiplier); default 1
      CU=<count>  compute units to fill; 0 disables the occupancy rule

    Example: SCREEN_EXTRA_PRUNE="B=128,CU=256"
    """
    spec = os.environ.get("SCREEN_EXTRA_PRUNE", "")
    if not spec:
        return False
    kv = dict(p.split("=", 1) for p in spec.split(",") if "=" in p)
    B = int(kv.get("B", 1))
    CU = int(kv.get("CU", 0))

    # A workgroup grid smaller than the CU count leaves compute units idle for
    # the whole kernel; no other knob can recover that.
    if CU:
        grid = B * -(-M // BLOCK_SIZE_M) * -(-N // BLOCK_SIZE_N)
        if grid < CU:
            if verbose:
                print(f"Remove case {config_list} because grid {grid} < {CU} CUs")
            return True

    # Warp/tile balance: every warp should own at least one 16x16 mfma tile,
    # and no warp should be handed an oversized slab.
    tile = BLOCK_SIZE_M * BLOCK_SIZE_N
    if not (tile / 8192 <= num_warps <= tile / 256):
        if verbose:
            print(
                f"Remove case {config_list} because num_warps={num_warps} "
                f"is unbalanced for a {BLOCK_SIZE_M}x{BLOCK_SIZE_N} tile"
            )
        return True
    return False
