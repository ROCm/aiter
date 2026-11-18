from aiter.aot.pa_common import build_configs
from aiter_worker_limits import adopt_legacy_max_jobs, run_configs
from csrc.cpp_itfs.pa.pa import compile


def process_config(config):
    return compile(
        config.gqa_ratio,
        config.head_size,
        config.npar_loops,
        config.dtype,
        config.kv_dtype,
        config.fp8_kv_dtype,
        config.out_dtype,
        config.block_size,
        config.alibi_enabled,
    )


def main():
    run_configs(build_configs(), process_config)


if __name__ == "__main__":
    adopt_legacy_max_jobs()
    main()
