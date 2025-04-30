import argparse
import json
import torch
import triton
import aiter.ops.triton
from op_tests.triton_tests.test_gemm_a16w16 import generate_gemm_a16w16_inputs

import aiter.ops.triton.gemm_a16w16 

##############################
#NOTE: hardcoded for now, but will be provided by user of tuning script 
#either through command line or json
##############################
#User args
#   op
#   set of shapes
#   configs for each shape
op = "gemm_a16w16"
shapes = [(1024,1024,256), (32,32,32)]
c_dtype = torch.bfloat16
aconfigs = [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
    ]

def tune_op(shapes, configs):

    cfgs = []
    for _ , cfg in configs.items():
        cdict = {}
        kwargs = {}
        for k, v in cfg.items():
            if k in ["num_warps", "num_stages", "num_ctas=1", "maxnreg"]:
                kwargs[k] = v
            else:
                cdict[k] = v
            #print(f"cdict={cdict}")
            cfgs.append(triton.Config(cdict, **kwargs))

    #print(f"cfgs={cfgs}")
    #Get decorated op function with autotuner
    aiter.ops.triton.gemm_a16w16._gemm_a16_w16_kernel = triton.autotune(configs= cfgs, key= ['M', 'N', 'K'])(aiter.ops.triton.gemm_a16w16._gemm_a16_w16_kernel)

    #Loop over shapes
    best_config = {}
    for shape in shapes:
        M, N, K = shape
        x, w = generate_gemm_a16w16_inputs(M, N, K, c_dtype)

        #Triton Autotuner complains of duplicate config, so make to set it to NULL in the actual call
        aiter.ops.triton.gemm_a16w16.gemm_a16w16(x, w, c_dtype, config={})

        bc = aiter.ops.triton.gemm_a16w16._gemm_a16_w16_kernel.best_config
        best_config[f"{M}-{N}-{K}"] = f"{bc}"

    print(f"best_configs={best_config}")
    with open("GEMM_A16W16.json", "w") as f:
        json.dump(best_config, f, indent=4)

def read_cfg_sh(fname):
    with open(fname, 'r') as f:
        cfg_sh = json.load(f)

        shapes = cfg_sh["shapes"]
        configs = cfg_sh["configs"]
        print("Reading Config Shape File")
        print(f"shapes={shapes}")
        print(f"configs={configs}")
        print("\n")

        return (shapes, configs)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Triton AITER TunerGEMM",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cfg_sh_file', type=str)

    args = parser.parse_args()

    return args
    
def main():
    args = parse_args()

    shapes, configs = read_cfg_sh(args.cfg_sh_file)

    tune_op(shapes, configs)


if __name__ == "__main__":
    main()