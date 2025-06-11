import argparse
import json
import torch
import triton
import aiter.ops.triton

import aiter.ops.triton.gemm_a16w16
import aiter.ops.triton.gemm_a8w8
import aiter.ops.triton.gemm_afp4wfp4
from aiter.ops.triton.utils.tuning_util import AUTOTUNE_OPS
from op_tests.triton_tests.test_gemm_a16w16 import generate_gemm_a16w16_inputs
from op_tests.triton_tests.test_gemm_a8w8 import generate_gemm_a8w8_inputs
from op_tests.triton_tests.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs


def tune_op(op, shapes_params, configs):
    cfgs = []
    for _, cfg in configs.items():
        cdict = {}
        kwargs = {}
        for k, v in cfg.items():
            if k in ["num_warps", "num_stages", "num_ctas=1", "maxnreg"]:
                kwargs[k] = v
            else:
                cdict[k] = v
            print(f"cdict={cdict}")
            cfgs.append(triton.Config(cdict, **kwargs))

    kernel_func = getattr(AUTOTUNE_OPS[op]["module"], AUTOTUNE_OPS[op]["kernels"][0])
    kernel_func = triton.autotune(configs=cfgs, key=["M", "N", "K"])(kernel_func)
    setattr(AUTOTUNE_OPS[op]["module"], AUTOTUNE_OPS[op]["kernels"][0], kernel_func)
    # print(kernel_func)

    # Loop over shapes_params
    best_config = {}
    for sp in shapes_params:
        inputs = AUTOTUNE_OPS[op]["input_generator_func"](**sp)
        print(f"AFPWFP4 len(inputs)={len(inputs)}")
        # Triton Autotuner complains of duplicate config, so set it to NULL in the actual call
        AUTOTUNE_OPS[op]["op_func"](*inputs, config={})

        bc = kernel_func.best_config
        best_config[f"{sp}"] = f"{bc}"

    print(f"best_configs={best_config}")
    with open(f"{op}.json", "w") as f:
        json.dump(best_config, f, indent=4)


def read_cfg_sh(fname):
    with open(fname, "r") as f:
        cfg_sh = json.load(f)

        shapes_params = cfg_sh["shapes_params"]
        configs = cfg_sh["configs"]
        print("Reading Config Shape File")
        print(f"shapes_params={shapes_params}")
        print(f"configs={configs}")
        print("\n")

        return (shapes_params, configs)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Triton AITER TunerGEMM",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--op", type=str)
    parser.add_argument("--cfg_sh_file", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.op not in AUTOTUNE_OPS.keys():
        print(f"Op {args.op} not available in Triton AITER")
    print(f"AUTOTUNE_OPS={AUTOTUNE_OPS}")

    shapes, configs = read_cfg_sh(args.cfg_sh_file)

    tune_op(args.op, shapes, configs)


if __name__ == "__main__":
    main()
