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


############################################################
# Backend resolution
#
# The tuner passes the backend to the ut_*.py subprocess through an env var
# rather than argv, because argv is a positional stream of config ints that is
# chunked by schema length -- an extra positional there would desync parsing.
############################################################

BACKEND_ENV = "AITER_TUNE_BACKEND"

# Archs where a kernel family defaults to gluon. Kept separate from "a gluon
# config dir exists": a dir can exist for an op whose python entry point has no
# backend= parameter (see BACKEND_KWARG_OPS below).
_GLUON_DEFAULT_ARCHS = ("gfx1250",)


def get_arch() -> str:
    from aiter.ops.triton.utils._triton import arch_info

    return arch_info.get_arch()


def get_backend(default: str | None = None) -> str:
    """Backend for this tuning run: env var wins, else arch default."""
    backend = os.environ.get(BACKEND_ENV, "").strip().lower()
    if backend:
        assert backend in (
            "triton",
            "gluon",
        ), f"{BACKEND_ENV}='{backend}' is invalid, must be 'triton' or 'gluon'"
        return backend
    if default is not None:
        return default
    return "gluon" if any(a in get_arch() for a in _GLUON_DEFAULT_ARCHS) else "triton"


############################################################
# Config schemas
#
# Each entry is an ordered list of (key, encoding). Order is the argv order and
# the JSON emit order. The first three entries MUST be the block sizes: screen.py
# batches rocprofv3 invocations by config_list[0:3] and prunes whole block-size
# groups when one of them fails to compile.
#
# Encodings map the int on argv to the value the kernel wants:
#   INT            -> int as-is
#   CACHE_MODIFIER -> 0: ".cg",             1: None
#   KERNEL_TYPE    -> 0: "bandwidth_bound", 1: "compute_bound"
#   BOOL           -> 0: False,             1: True
############################################################

INT = "int"
CACHE_MODIFIER = "cache_modifier"
KERNEL_TYPE = "kernel_type"
BOOL = "bool"


def decode_param(value: int, encoding: str):
    if encoding == INT:
        return int(value)
    if encoding == CACHE_MODIFIER:
        return ".cg" if int(value) == 0 else None
    if encoding == KERNEL_TYPE:
        return "bandwidth_bound" if int(value) == 0 else "compute_bound"
    if encoding == BOOL:
        return bool(int(value))
    raise ValueError(f"Unknown encoding '{encoding}'")


def encode_param_json(value: int, encoding: str) -> str:
    """Render an argv int as the JSON literal for a config file."""
    if encoding == INT:
        return str(int(value))
    if encoding == CACHE_MODIFIER:
        return '".cg"' if int(value) == 0 else "null"
    if encoding == KERNEL_TYPE:
        return '"bandwidth_bound"' if int(value) == 0 else '"compute_bound"'
    if encoding == BOOL:
        return "true" if int(value) else "false"
    raise ValueError(f"Unknown encoding '{encoding}'")


# The historical triton GEMM schema. Order preserved exactly so screen*.log files
# and JSON configs produced before schemas existed still parse.
TRITON_GEMM = [
    ("BLOCK_SIZE_M", INT),
    ("BLOCK_SIZE_N", INT),
    ("BLOCK_SIZE_K", INT),
    ("GROUP_SIZE_M", INT),
    ("num_warps", INT),
    ("num_stages", INT),
    ("waves_per_eu", INT),
    ("matrix_instr_nonkdim", INT),
    ("cache_modifier", CACHE_MODIFIER),
    ("NUM_KSPLIT", INT),
]

# gfx1250 gluon a16w16: the kernel reads exactly these. Note there is no
# GROUP_SIZE_M -- the grid is a flat cdiv(M,BM)*cdiv(N,BN), so tuning it would
# search a dimension the kernel ignores.
GLUON_A16W16 = [
    ("BLOCK_M", INT),
    ("BLOCK_N", INT),
    ("BLOCK_K", INT),
    ("num_warps", INT),
    ("NUM_BUFFERS", INT),
    ("kernel_type", KERNEL_TYPE),
]

GLUON_A16W16_PERSISTENT = [
    ("BLOCK_M", INT),
    ("BLOCK_N", INT),
    ("BLOCK_K", INT),
    ("GROUP_SIZE_M", INT),
    ("num_warps", INT),
    ("NUM_BUFFERS", INT),
]

# gluon mxfp4 preshuffle keeps the BLOCK_SIZE_* spelling.
GLUON_MXFP4_PRESHUFFLE = [
    ("BLOCK_SIZE_M", INT),
    ("BLOCK_SIZE_N", INT),
    ("BLOCK_SIZE_K", INT),
    ("num_warps", INT),
    ("NUM_BUFFERS", INT),
]

GLUON_BATCHED_A16W16 = [
    ("BLOCK_SIZE_M", INT),
    ("BLOCK_SIZE_N", INT),
    ("BLOCK_SIZE_K", INT),
    ("GROUP_SIZE_M", INT),
    ("num_warps", INT),
    ("waves_per_eu", INT),
    ("matrix_instr_nonkdim", INT),
    ("cache_modifier", CACHE_MODIFIER),
    ("NUM_KSPLIT", INT),
    ("NUM_BUFFERS", INT),
    ("kernel_type", KERNEL_TYPE),
]

# gfx1250 gluon blockscale takes the triton key set with the pipeline depth
# spelled NUM_BUFFERS (it was num_stages before the configs were renamed; gfx950
# gluon still reads num_stages, hence the arch split in UT_SCHEMA).
GLUON_A8W8_BLOCKSCALE = [
    ("BLOCK_SIZE_M", INT),
    ("BLOCK_SIZE_N", INT),
    ("BLOCK_SIZE_K", INT),
    ("GROUP_SIZE_M", INT),
    ("num_warps", INT),
    ("NUM_BUFFERS", INT),
    ("waves_per_eu", INT),
    ("matrix_instr_nonkdim", INT),
    ("cache_modifier", CACHE_MODIFIER),
    ("NUM_KSPLIT", INT),
]

GLUON_AFP8WFP8_PRESHUFFLE = [
    ("BLOCK_SIZE_M", INT),
    ("BLOCK_SIZE_N", INT),
    ("BLOCK_SIZE_K", INT),
    ("GROUP_SIZE_M", INT),
    ("num_warps", INT),
    ("waves_per_eu", INT),
    ("cache_modifier", CACHE_MODIFIER),
    ("NUM_KSPLIT", INT),
    ("NUM_BUFFERS", INT),
    ("kernel_type", KERNEL_TYPE),
    ("CTAS_M", INT),
    ("CTAS_N", INT),
    ("B_SCALE_TDM", BOOL),
    ("LOOP_UNROLL_FACTOR", INT),
]

SCHEMAS = {
    "triton_gemm": TRITON_GEMM,
    "gluon_a16w16": GLUON_A16W16,
    "gluon_a16w16_persistent": GLUON_A16W16_PERSISTENT,
    "gluon_mxfp4_preshuffle": GLUON_MXFP4_PRESHUFFLE,
    "gluon_a8w8_blockscale": GLUON_A8W8_BLOCKSCALE,
    "gluon_batched_a16w16": GLUON_BATCHED_A16W16,
    "gluon_afp8wfp8_preshuffle": GLUON_AFP8WFP8_PRESHUFFLE,
}

# ut_*.py -> schema per backend. A None gluon entry means "this op has no gluon
# kernel"; the tuner refuses --backend gluon for it rather than silently tuning
# triton params that the gluon path would ignore.
#
# Gluon support is arch-dependent, not just op-dependent: gemm_a8w8 has a gluon
# kernel on gfx950 only, so its gluon entry carries a "gluon_archs" restriction
# and asking for gluon on gfx1250 is refused. Where an entry has no
# "gluon_archs", the gluon kernel is available on every arch that has one.
#
# Several gfx1250 gluon config dirs (gemm_a8w8_blockscale,
# gemm_a8w8_blockscale_preshuffled, gemm_afp4wfp4) hold triton-shaped configs --
# the gluon kernel there consumes the same keys -- so they map to "triton_gemm"
# under both backends. That is deliberate, not a copy/paste slip.
UT_SCHEMA = {
    "ut_a16w16_gemm.py": {"triton": "triton_gemm", "gluon": "gluon_a16w16"},
    "ut_a16w16_gemm_atomic.py": {"triton": "triton_gemm", "gluon": None},
    "ut_a16w16_gemm_gated.py": {"triton": "triton_gemm", "gluon": None},
    "ut_a16w8_gemm_blockscale.py": {"triton": "triton_gemm", "gluon": None},
    "ut_a16w8_gemm_blockscale_preshuffle.py": {"triton": "triton_gemm", "gluon": None},
    "ut_a16wfp4_gemm.py": {"triton": "triton_gemm", "gluon": None},
    "ut_a8w8_gemm.py": {
        "triton": "triton_gemm",
        "gluon": "triton_gemm",
        "gluon_archs": ("gfx950",),
    },
    # gfx1250 gluon spells the pipeline depth NUM_BUFFERS; gfx950 gluon still
    # reads num_stages, which is what "triton_gemm" carries.
    "ut_a8w8_gemm_blockscale.py": {
        "triton": "triton_gemm",
        "gluon": {"gfx1250": "gluon_a8w8_blockscale", "default": "triton_gemm"},
    },
    "ut_a8w8_gemm_blockscale_preshuffle.py": {
        "triton": "triton_gemm",
        "gluon": {"gfx1250": "gluon_a8w8_blockscale", "default": "triton_gemm"},
    },
    "ut_a8w8_gemm_per_token_scale.py": {"triton": "triton_gemm", "gluon": None},
    "ut_a8wfp4_gemm.py": {"triton": "triton_gemm", "gluon": None},
    "ut_afp4wfp4_gemm.py": {"triton": "triton_gemm", "gluon": "triton_gemm"},
    "ut_afp4wfp4_gemm_preshuffle.py": {
        "triton": "triton_gemm",
        "gluon": "gluon_mxfp4_preshuffle",
    },
    "ut_afp4wfp4_gemm_pre_quant_atomic.py": {"triton": "triton_gemm", "gluon": None},
    "ut_afp8wfp8_gemm_preshuffle.py": {
        "triton": "triton_gemm",
        "gluon": "gluon_afp8wfp8_preshuffle",
    },
    "ut_batched_gemm_a16w16.py": {
        "triton": "triton_gemm",
        "gluon": "gluon_batched_a16w16",
    },
    "ut_template.py": {"triton": "triton_gemm", "gluon": None},
}

# Ops whose python entry point takes backend=. Everything else either has no
# gluon path at all, or picks the backend itself from the arch -- passing
# backend= to those is a TypeError.
BACKEND_KWARG_UTS = {
    "ut_a16w16_gemm.py",
    "ut_a8w8_gemm.py",
    "ut_a8w8_gemm_blockscale.py",
    "ut_a8w8_gemm_blockscale_preshuffle.py",
    "ut_afp4wfp4_gemm.py",
    "ut_afp8wfp8_gemm_preshuffle.py",
    "ut_batched_gemm_a16w16.py",
}

# Ops that hard-select gluon from the arch with no way to override. Tuning these
# on gfx1250 always measures the gluon kernel regardless of --backend, so the
# tuner refuses --backend triton for them instead of writing configs into a
# triton dir that the gluon kernel will never read.
ARCH_FORCED_GLUON_UTS = {
    "ut_afp4wfp4_gemm_preshuffle.py": _GLUON_DEFAULT_ARCHS,
}

# Backward compat: the old module-level name, still the triton key order.
config_parms_key = [k for k, _ in TRITON_GEMM]


def schema_name_for(ut_filename: str, backend: str, arch: str | None = None) -> str:
    ut = os.path.basename(ut_filename)
    assert ut in UT_SCHEMA, (
        f"{ut} has no schema registered in _utils.UT_SCHEMA. Add one (and its "
        f"gluon entry, or None if the op has no gluon kernel) before tuning it."
    )
    name = UT_SCHEMA[ut].get(backend)
    assert name is not None, (
        f"{ut} has no '{backend}' kernel, so there is nothing to tune for that "
        f"backend. Available: "
        f"{[b for b, s in UT_SCHEMA[ut].items() if s is not None and b != 'gluon_archs']}"
    )
    # An entry may vary by arch (the same gluon op can spell a param differently
    # on gfx950 and gfx1250); a plain string applies everywhere.
    if isinstance(name, dict):
        arch = arch if arch is not None else get_arch()
        for arch_key, schema_name in name.items():
            if arch_key != "default" and arch_key in arch:
                return schema_name
        name = name["default"]
    return name


def get_schema(ut_filename: str, backend: str, arch: str | None = None):
    return SCHEMAS[schema_name_for(ut_filename, backend, arch)]


def check_backend_allowed(ut_filename: str, backend: str, arch: str | None = None):
    """Raise if this ut/backend/arch combination cannot actually be measured."""
    ut = os.path.basename(ut_filename)
    schema_name_for(ut, backend)  # raises if the op has no such kernel
    arch = arch if arch is not None else get_arch()

    if backend == "gluon":
        gluon_archs = UT_SCHEMA[ut].get("gluon_archs")
        if gluon_archs is not None and not any(a in arch for a in gluon_archs):
            raise AssertionError(
                f"{ut} has a gluon kernel only on {list(gluon_archs)}, but this "
                f"machine is {arch}. Tune it with --backend triton here."
            )

    forced = ARCH_FORCED_GLUON_UTS.get(ut)
    if forced is not None and any(a in arch for a in forced) and backend != "gluon":
        raise AssertionError(
            f"{ut} selects the gluon kernel from the arch on {arch} and takes "
            f"no backend= override, so --backend {backend} would still run "
            f"gluon and write configs the kernel never reads. Use "
            f"--backend gluon."
        )


############################################################
# argv parsing
############################################################


def get_config_list(argv: list[str], schema=None) -> list[dict | None]:
    schema = schema if schema is not None else TRITON_GEMM
    config_argv = argv
    n = len(schema)
    config_list = []
    while len(config_argv) >= n:
        chunk = config_argv[:n]
        config_list.append(
            {key: decode_param(chunk[i], enc) for i, (key, enc) in enumerate(schema)}
        )
        config_argv = config_argv[n:]

    if len(config_list) == 0:
        config_list = [None]

    return config_list


def get_input_shape(argv: list[str]) -> list[int]:
    return [int(v) for v in argv]


def get_input_shape_and_config_list(
    argv: list[str],
    shape_size: int = 3,
    ut_filename: str | None = None,
    backend: str | None = None,
) -> tuple[list[int], list[dict | None]]:
    """
    Parse ``sys.argv`` into (shape, configs).

    ``ut_filename`` selects the config schema; pass ``__file__`` from the ut
    script. Omitting it keeps the historical triton schema, so old call sites
    behave exactly as before.
    """
    input_shape = get_input_shape(argv[1 : shape_size + 1])
    if ut_filename is None:
        schema = TRITON_GEMM
    else:
        backend = backend if backend is not None else get_backend()
        schema = get_schema(ut_filename, backend)
    config_list = get_config_list(argv[shape_size + 1 :], schema=schema)
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
                    case_data.append(
                        [r, screencaseline[len("screencase") + 1 :].strip()]
                    )
                except IndexError:
                    break


############################################################
# Pre-pruning
############################################################

# gfx1250 workgroup LDS budget. The gluon pipeline stages A and B tiles through
# NUM_BUFFERS shared-memory slots, so a tile that cannot hold the requested depth
# gets silently clamped to a shallower pipeline (or fails to compile). Estimated
# rather than exact -- swizzle padding adds a few percent -- so this only prunes
# configs that are already over budget before padding, never borderline ones.
LDS_LIMIT_BYTES = 327680


def _gluon_lds_bytes(block_m, block_n, block_k, num_buffers, elem_bytes=2) -> int:
    return num_buffers * (block_m + block_n) * block_k * elem_bytes


def pre_pruning_rules(
    M: int,
    N: int,
    K: int,
    config_list,
    verbose: bool = False,
    schema=None,
) -> bool:
    """True if this config should be dropped before it costs a rocprofv3 run."""
    schema = schema if schema is not None else TRITON_GEMM
    cfg = {key: config_list[i] for i, (key, _) in enumerate(schema)}

    def drop(reason: str) -> bool:
        if verbose:
            print(f"Remove case {config_list} because {reason}")
        return True

    block_m = cfg.get("BLOCK_SIZE_M", cfg.get("BLOCK_M"))
    block_n = cfg.get("BLOCK_SIZE_N", cfg.get("BLOCK_N"))
    block_k = cfg.get("BLOCK_SIZE_K", cfg.get("BLOCK_K"))
    num_ksplit = cfg.get("NUM_KSPLIT", 1)
    group_size_m = cfg.get("GROUP_SIZE_M", 1)
    num_stages = cfg.get("num_stages")
    num_buffers = cfg.get("NUM_BUFFERS")

    k_per_split = K // num_ksplit

    if block_k >= 2 * k_per_split:
        return drop("BLOCK_SIZE_K >= 2 * (K // NUM_KSPLIT)")
    if num_ksplit > 1 and group_size_m > 1:
        return drop("NUM_KSPLIT > 1 and GROUP_SIZE_M > 1")

    if num_stages is not None:
        if block_k == k_per_split and num_stages > 1:  # k_itr == 1 case
            return drop("BLOCK_SIZE_K == K // NUM_KSPLIT and num_stages > 1")
        if block_k < k_per_split and num_stages == 1:  # k_itr > 1 case
            return drop("BLOCK_SIZE_K < K // NUM_KSPLIT and num_stages == 1")

    if num_buffers is not None:
        # A pipeline deeper than the number of k tiles just idles the extra slots.
        num_k_tiles = triton.cdiv(k_per_split, block_k)
        if num_buffers > num_k_tiles + 1:
            return drop(f"NUM_BUFFERS > num_k_tiles + 1 ({num_k_tiles} k tiles)")
        lds = _gluon_lds_bytes(block_m, block_n, block_k, num_buffers)
        if lds > LDS_LIMIT_BYTES:
            return drop(
                f"estimated LDS {lds} B > {LDS_LIMIT_BYTES} B budget "
                f"(tile {block_m}x{block_n}x{block_k}, NUM_BUFFERS={num_buffers})"
            )

    return False
