import triton
from triton.testing import runtime


@triton.jit
def split_dummy():
    return


def run_profile(fn: callable, n_run: int = 250):
    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    for _ in range(n_run):
        cache.zero_()
        di.synchronize()
        fn()
        di.synchronize()
    cache.zero_()
    di.synchronize()
    split_dummy[(1,)]()
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
