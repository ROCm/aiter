import sys
import argparse
import itertools
import subprocess
import torch
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from op_tests.triton_tests.test_gemm_a16w16 import generate_gemm_a16w16_inputs

def run_bash_command(commandstring, capture=True):
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None

#configs
def get_full_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    split_k_range = [1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 2, 4, 8, 16, 32]
    # For now we see better perf with num_stages=2 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [2]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32]
    cache_modifier = [""]

    space = itertools.product(block_mn_range, block_mn_range, block_k_range, num_warps_range, group_m_range,
                              num_stage_range, waves_per_eu_range, matrix_instr_nonkdim_range, cache_modifier)

    for instance in space:
        block_m, block_n, block_k, num_warps, group_m, num_stages, waves_per_eu, matrix_instr_nonkdim, cache_modifier = instance
        configs.append({
            'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_m,
            'num_warps': num_warps, 'num_stages': num_stages, 'waves_per_eu': waves_per_eu,
            'matrix_instr_nonkdim': matrix_instr_nonkdim, 'cache_modifier': cache_modifier})

    return configs

#prune configs
def prune_configs(M, N, K, configs, elemBytes_a, elemBytes_b):
    pruned_configs = []

    if M < 32 or N < 32:
        mfma = 16
    else:
        mfma = 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")
        num_stages = config.get("num_stages")
        matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
        if matrix_instr_nonkdim > mfma:
            continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elemens per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
            continue
        if M <= matrix_instr_nonkdim and BLOCK_SIZE_M != matrix_instr_nonkdim:
            continue
        if N <= matrix_instr_nonkdim and BLOCK_SIZE_N != matrix_instr_nonkdim:
            continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if BLOCK_SIZE_M > M * 2 and BLOCK_SIZE_M != 16:
            continue
        if BLOCK_SIZE_N > N * 2 and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0 and SPLIT_K != 1:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDSA = BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
        LDSB = BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        if num_stages <= 1:
            # No pipeline, buffer A and buffer B can re-use each other
            LDS = max(LDSA, LDSB)
        else:
            # Pipeline, we need (num_stages - 1) buffers for both A and B at the same time
            LDS = (LDSA + LDSB) * (num_stages - 1)
        driver = triton.runtime.driver.active
        max_shared = driver.utils.get_device_properties(driver.get_current_device())["max_shared_mem"]
        if LDS > max_shared:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue
            # check if tiling is integer multiple of GEMM size because we have no boundary check
            if M % BLOCK_SIZE_M != 0 or N % BLOCK_SIZE_N != 0:
                continue

        pruned_configs.append(config)

    return pruned_configs

#tune
    #loop over all shapes
        #get valid configs for this shape
        #loop over all configs
            #call the wrapper for the gemm op. Do we need to run a few times
            #
            #collect time from rocprof
            #append times and config
        #determine best config for this shape
        #store the best config for this shape in a file

def get_shapes(args):
    return [(args.M, args.N, args.K, args.layout)]

#arg parser
def parse_args():
    parser = argparse.ArgumentParser(description="Tune AITER Triton GEMM Kernels")

    parser.add_argument("-M", type=int, default=0)
    parser.add_argument("-N", type=int, default=0)
    parser.add_argument("-K", type=int, default=0)
    parser.add_argument("-layout", type=str,choices=["TT", "TN", "NT", "NN"], default='TN')
    parser.add_argument("-op", type=str, default='all')
    parser.add_argument("-shapes_file", type=str, default="", help='JSON with GEMM shapes to tune')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    shapes = get_shapes(args)

    configs_full = get_full_tuning_space()
    #print(configs_full[0])

    run_bash_command("rm -rf ~/.triton/cache")


    for (M, N, K, layout) in shapes:
        for cfg in [configs_full[0]]:
            print(f"cfg={cfg}")
            x, w, out_dtype, y = generate_gemm_a16w16_inputs(M, N, K, torch.bfloat16, layout=layout, output=True)
            gemm_a16w16(x, w, out_dtype, y, cfg)
#main
if __name__ == "__main__":
    sys.exit(main())