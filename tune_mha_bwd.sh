#!/usr/bin/env bash

os=$(uname --kernel-name)
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
tune_configs_dir="${script_dir}/tune_mha_bwd_configs"

# MI300X config:
aiter_config_file="${script_dir}/aiter/ops/triton/configs/MI300X-MHA-DEFAULT.json"
# MI350X config:
# aiter_config_file="${script_dir}/aiter/ops/triton/configs/MI350X-MHA-DEFAULT.json"


os_path() {
    local path="${1}"
    if [ "${os}" != "Linux" ]; then
        path=$(cygpath --windows "${path}")
    fi
    echo "${path}"
}


# Stolen from Pure Bash Bible:
# https://github.com/dylanaraps/pure-bash-bible?tab=readme-ov-file#get-the-base-name-of-a-file-path
basename() {
    # Usage: basename "path" ["suffix"]
    local tmp

    tmp=${1%"${1##*[!/]}"}
    tmp=${tmp##*/}
    tmp=${tmp%"${2/"$tmp"}"}

    printf '%s\n' "${tmp:-/}"
}


gen_tune_configs() {
    rm --recursive --force "${tune_configs_dir}"
    mkdir --parents "${tune_configs_dir}"
    local os_tune_configs_dir
    os_tune_configs_dir=$(os_path "${tune_configs_dir}")
    python <<EOF
import itertools
import json
import os

debug = False

if debug:
    print("Generating configs....")

block_sizes = [2**n for n in range(5, 8)]
num_warps_range = [2**n for n in range(1, 4)]
num_stages_range = list(range(1, 4))
waves_per_eu_range = [0, 1]

for i, config in enumerate(
    itertools.product(
        block_sizes,  # block_m1
        block_sizes,  # block_n1
        block_sizes,  # block_m2
        block_sizes,  # block_n2
        num_warps_range,  # num_warps
        num_stages_range,  # num_stages
        waves_per_eu_range,  # waves_per_eu
    )
):
    block_m1, block_n1, block_m2, block_n2, num_warps, num_stages, waves_per_eu = config
    config_name = f"{i:04d}__BM1_{block_m1:03d}__BN1_{block_n1:03d}__BM2_{block_m2:03d}__BN2_{block_n2:03d}__NW_{num_warps}__NS_{num_stages}__WE_{waves_per_eu}"
    config_dict = {
        "BLOCK_M1": block_m1,
        "BLOCK_N1": block_n1,
        "BLOCK_M2": block_m2,
        "BLOCK_N2": block_n2,
        "BLK_SLICE_FACTOR": 2,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": 16,
        "num_warps": num_warps,
        "num_ctas": 1,
        "num_stages": num_stages,
    }

    config_filename = os.path.join(r"${os_tune_configs_dir}", f"{config_name}.json")
    with open(config_filename, "w") as config_file:
        json.dump(config_dict, config_file, indent=2)

if debug:
    print(f"{i + 1} configs generated.")
EOF
}


replace_aiter_config() {
    local tune_config_file="${1}"
    git restore "${aiter_config_file}"
    local os_tune_config_file
    os_tune_config_file=$(os_path "${tune_config_file}")
    local os_aiter_config_file
    os_aiter_config_file=$(os_path "${aiter_config_file}")
python <<EOF
import json

debug = False

with open(r"${os_aiter_config_file}", "r") as aiter_config_file:
    aiter_config = json.load(aiter_config_file)
if debug:
    print("AITER config:")
    print(aiter_config["bkwd_onekernel"]["onekernel"])

with open(r"${os_tune_config_file}", "r") as tune_config_file:
    tune_config = json.load(tune_config_file)
if debug:
    print("Tune config")
    print(tune_config)

aiter_config["bkwd_onekernel"]["onekernel"] = tune_config
with open(r"${os_aiter_config_file}", "w") as aiter_config_file:
    json.dump(aiter_config, aiter_config_file, indent=2)
EOF
}


gen_tune_configs

for tune_config in "${tune_configs_dir}"/*.json; do
    tune_config_name=$(basename "${tune_config}" '.json')
    replace_aiter_config "${tune_config}"

    # Run unit test.
    if ! pytest -qqq --tb=no "${script_dir}/op_tests/triton_tests/test_mha.py::test_mha_backward_with_pe[True-0.0-192-128-128-128-4096-4096-1]" &> /dev/null; then
        # Test failed.
        result='fail,NA'
    else
        # Test passed, run benchmark.
        time=$(python "${script_dir}/op_tests/op_benchmarks/triton/bench_mha.py" \
            --dtype bf16 -mode bwd -b 1 -hq 128 -hk 128 -sq 4096 -sk 4096 -d 192 -dv 128 \
            -causal true --layout bshd -metric time 2> /dev/null \
            | tail -1 | tr --squeeze-repeats ' ' | cut --delimiter=' ' --fields=7)
        result="pass,${time}"
    fi

    echo "${tune_config_name},${result}"
done
