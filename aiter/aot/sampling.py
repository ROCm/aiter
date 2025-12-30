from collections import namedtuple
import os
import concurrent.futures
from csrc.cpp_itfs.sampling.top_k_renorm_probs import (
    compile as top_k_renorm_probs_compile,
)
from csrc.cpp_itfs.sampling.top_p_sampling_from_probs import (
    compile as top_p_sampling_from_probs_compile,
)
from csrc.cpp_itfs.sampling.top_k_top_p_sampling_from_probs import (
    compile as top_k_top_p_sampling_from_probs_compile,
)

TopKRenormConfig = namedtuple(
    "TopKRenormConfig",
    ["vec_size", "func_name"],
)

TopPSamplingConfig = namedtuple(
    "TopPSamplingConfig",
    ["vec_size", "deterministic", "func_name"],
)

TopKTopPSamplingConfig = namedtuple(
    "TopKTopPSamplingConfig",
    ["vec_size", "deterministic", "func_name"],
)


def process_top_k_renorm_config(config):
    return top_k_renorm_probs_compile(config.vec_size)


def process_top_p_sampling_config(config):
    return top_p_sampling_from_probs_compile(config.vec_size, config.deterministic)


def process_top_k_top_p_sampling_config(config):
    return top_k_top_p_sampling_from_probs_compile(
        config.vec_size, config.deterministic
    )


def main():
    # Generate configs for top_k_renorm_probs
    top_k_renorm_configs = []
    for vec_size in range(1, 5):
        top_k_renorm_configs.append(
            TopKRenormConfig(
                vec_size=vec_size,
                func_name="top_k_renorm_probs",
            )
        )

    # Generate configs for top_p_sampling_from_probs
    top_p_sampling_configs = []
    for vec_size in range(1, 5):
        for deterministic in [False, True]:
            top_p_sampling_configs.append(
                TopPSamplingConfig(
                    vec_size=vec_size,
                    deterministic=deterministic,
                    func_name="top_p_sampling_from_probs",
                )
            )

    # Generate configs for top_k_top_p_sampling_from_probs
    top_k_top_p_sampling_configs = []
    for vec_size in range(1, 5):
        for deterministic in [False, True]:
            top_k_top_p_sampling_configs.append(
                TopKTopPSamplingConfig(
                    vec_size=vec_size,
                    deterministic=deterministic,
                    func_name="top_k_top_p_sampling_from_probs",
                )
            )

    max_jobs = int(os.environ.get("MAX_JOBS", os.cpu_count() or 16))

    # Process all configs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        executor.map(process_top_k_renorm_config, top_k_renorm_configs)
        executor.map(process_top_p_sampling_config, top_p_sampling_configs)
        executor.map(process_top_k_top_p_sampling_config, top_k_top_p_sampling_configs)


if __name__ == "__main__":
    main()
