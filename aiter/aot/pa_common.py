"""Shared page-attention AOT config space for the pa and pa_ragged drivers."""

from collections import namedtuple

PAConfig = namedtuple(
    "PAConfig",
    [
        "gqa_ratio",
        "head_size",
        "npar_loops",
        "block_size",
        "dtype",
        "kv_dtype",
        "fp8_kv_dtype",
        "out_dtype",
        "alibi_enabled",
    ],
)

# (dtype, kv_dtype, fp8_kv_dtype); out_dtype mirrors dtype.
_DTYPE_VARIANTS = (
    ("_Float16", "_Float16", "auto"),
    ("__hip_bfloat16", "__hip_bfloat16", "auto"),
    ("_Float16", "uint8_t", "fp8"),
    ("__hip_bfloat16", "uint8_t", "fp8"),
)


def build_configs() -> list[PAConfig]:
    """Enumerate the full pa/pa_ragged config space in a stable order."""
    configs = []
    for gqa_ratio in range(1, 17):
        for alibi_enabled in ["false", "true"]:
            for block_size in [1, 16, 32]:
                for npar_loops in range(1, 9):
                    for head_size in [64, 128]:
                        for dtype, kv_dtype, fp8_kv_dtype in _DTYPE_VARIANTS:
                            configs.append(
                                PAConfig(
                                    gqa_ratio=gqa_ratio,
                                    head_size=head_size,
                                    npar_loops=npar_loops,
                                    dtype=dtype,
                                    kv_dtype=kv_dtype,
                                    fp8_kv_dtype=fp8_kv_dtype,
                                    out_dtype=dtype,
                                    block_size=block_size,
                                    alibi_enabled=alibi_enabled,
                                )
                            )
    return configs
