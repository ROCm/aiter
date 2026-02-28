# SPDX-License-Identifier: MIT
# FlyDSL A8W8 bpreshuffle kernel instance definitions.
# Mirrors the CKTile pattern in gemm_a8w8_bpreshuffle_cktile_common.py
# so the tuning framework can treat FlyDSL as another "libtype".
from dataclasses import dataclass
import os


def get_gfx():
    """Detect GPU arch: honour GPU_ARCHS env, fall back to chip_info, default gfx942."""
    env = os.environ.get("GPU_ARCHS", "")
    if env and env != "native":
        return env.split(";")[-1].strip()
    try:
        import sys
        this_dir = os.path.dirname(os.path.abspath(__file__))
        jit_utils = os.path.abspath(os.path.join(this_dir, "..", "aiter", "jit", "utils"))
        if jit_utils not in sys.path:
            sys.path.insert(0, jit_utils)
        from chip_info import get_gfx as _get_gfx
        return _get_gfx()
    except Exception:
        return "gfx942"

_DTYPE_SHORT = {
    "fp8": "F8",
    "int8": "I8",
    "bf16": "B16",
    "fp16": "F16",
}


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    q_dtype_a: str          # "fp8" | "int8"
    q_dtype_w: str          # "fp8" | "int8"
    dtype: str              # output dtype: "bf16" | "fp16"
    lds_stage: int          # 1 or 2
    use_cshuffle_epilog: int  # 0 or 1
    use_async_copy: int       # 0 or 1
    sScheduler: str           # "Default"

    @property
    def name(self) -> str:
        qa = _DTYPE_SHORT.get(self.q_dtype_a, self.q_dtype_a.upper())
        qw = _DTYPE_SHORT.get(self.q_dtype_w, self.q_dtype_w.upper())
        dt = _DTYPE_SHORT.get(self.dtype, self.dtype.upper())
        return "_".join([
            "flydsl",
            "bpreshuflle",
            "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
            qa,
            qw,
            dt,
            "x".join(map(str, [
                self.lds_stage,
                self.use_cshuffle_epilog,
                self.use_async_copy,
            ])),
            self.sScheduler.lower(),
        ])


def _ki(tile_m, tile_n, tile_k, lds_stage,
        cshuffle=0, async_copy=0, scheduler="Default",
        q_dtype_a="fp8", q_dtype_w="fp8", dtype="bf16"):
    return kernelInstance(
        tile_m, tile_n, tile_k,
        q_dtype_a, q_dtype_w, dtype,
        lds_stage, cshuffle, async_copy, scheduler,
    )


# fmt: off
# ---------------------------------------------------------------------------
# gfx942 (MI300): MFMA fp8 uses float8_e4m3fnuz, KWTile=64
# Tile search space matches CKTile tile dimensions for fair tuning comparison.
# Two lds_stage variants (2=ping-pong, 1=single) per tile.
# ---------------------------------------------------------------------------
kernels_list_942 = {
    #      tile_m  tile_n  tile_k  lds  csh  acp  scheduler
    # ---- small M (decode / token-gen) ----
    0:  _ki(16,     64,     256,    2, 0, 0, "Default"),
    1:  _ki(16,     64,     512,    2, 0, 0, "Default"),
    2:  _ki(16,     128,    256,    2, 0, 0, "Default"),
    3:  _ki(16,     128,    512,    2, 0, 0, "Default"),
    4:  _ki(16,     256,    256,    2, 0, 0, "Default"),
    5:  _ki(16,     256,    512,    2, 0, 0, "Default"),
    6:  _ki(16,     512,    256,    2, 0, 0, "Default"),
    7:  _ki(16,     192,    256,    2, 0, 0, "Default"),
    # ---- M=32 ----
    8:  _ki(32,     64,     128,    2, 0, 0, "Default"),
    9:  _ki(32,     64,     256,    2, 0, 0, "Default"),
    10: _ki(32,     64,     512,    2, 0, 0, "Default"),
    11: _ki(32,     128,    128,    2, 0, 0, "Default"),
    12: _ki(32,     128,    256,    2, 0, 0, "Default"),
    13: _ki(32,     192,    128,    2, 0, 0, "Default"),
    14: _ki(32,     192,    256,    2, 0, 0, "Default"),
    15: _ki(32,     256,    128,    2, 0, 0, "Default"),
    16: _ki(32,     256,    256,    2, 0, 0, "Default"),
    # ---- M=48 ----
    17: _ki(48,     64,     256,    2, 0, 0, "Default"),
    18: _ki(48,     128,    256,    2, 0, 0, "Default"),
    19: _ki(48,     192,    256,    2, 0, 0, "Default"),
    20: _ki(48,     256,    256,    2, 0, 0, "Default"),
    # ---- M=64 ----
    21: _ki(64,     64,     128,    2, 0, 0, "Default"),
    22: _ki(64,     64,     256,    2, 0, 0, "Default"),
    23: _ki(64,     128,    128,    2, 0, 0, "Default"),
    24: _ki(64,     128,    256,    2, 0, 0, "Default"),
    25: _ki(64,     192,    128,    2, 0, 0, "Default"),
    26: _ki(64,     192,    256,    2, 0, 0, "Default"),
    27: _ki(64,     256,    64,     2, 0, 0, "Default"),
    28: _ki(64,     256,    128,    2, 0, 0, "Default"),
    29: _ki(64,     256,    256,    2, 0, 0, "Default"),
    # ---- M=80 ----
    30: _ki(80,     64,     256,    2, 0, 0, "Default"),
    31: _ki(80,     128,    256,    2, 0, 0, "Default"),
    32: _ki(80,     192,    256,    2, 0, 0, "Default"),
    33: _ki(80,     256,    256,    2, 0, 0, "Default"),
    # ---- M=96 ----
    34: _ki(96,     64,     128,    2, 0, 0, "Default"),
    35: _ki(96,     64,     256,    2, 0, 0, "Default"),
    36: _ki(96,     128,    128,    2, 0, 0, "Default"),
    37: _ki(96,     128,    256,    2, 0, 0, "Default"),
    38: _ki(96,     192,    128,    2, 0, 0, "Default"),
    39: _ki(96,     192,    256,    2, 0, 0, "Default"),
    40: _ki(96,     256,    128,    2, 0, 0, "Default"),
    41: _ki(96,     256,    256,    2, 0, 0, "Default"),
    # ---- M=112 ----
    42: _ki(112,    64,     256,    2, 0, 0, "Default"),
    43: _ki(112,    128,    256,    2, 0, 0, "Default"),
    44: _ki(112,    192,    256,    2, 0, 0, "Default"),
    45: _ki(112,    256,    256,    2, 0, 0, "Default"),
    # ---- M=128 ----
    46: _ki(128,    64,     128,    2, 0, 0, "Default"),
    47: _ki(128,    64,     256,    2, 0, 0, "Default"),
    48: _ki(128,    128,    64,     2, 0, 0, "Default"),
    49: _ki(128,    128,    128,    2, 0, 0, "Default"),
    50: _ki(128,    128,    256,    2, 0, 0, "Default"),
    51: _ki(128,    192,    128,    2, 0, 0, "Default"),
    52: _ki(128,    192,    256,    2, 0, 0, "Default"),
    53: _ki(128,    256,    128,    2, 0, 0, "Default"),
    # ---- M=160/192/224/256 ----
    54: _ki(160,    192,    128,    2, 0, 0, "Default"),
    55: _ki(192,    64,     128,    2, 0, 0, "Default"),
    56: _ki(192,    128,    128,    2, 0, 0, "Default"),
    57: _ki(224,    64,     128,    2, 0, 0, "Default"),
    58: _ki(224,    128,    128,    2, 0, 0, "Default"),
    59: _ki(224,    192,    128,    2, 0, 0, "Default"),
    60: _ki(256,    64,     128,    2, 0, 0, "Default"),
    61: _ki(256,    128,    128,    2, 0, 0, "Default"),
    62: _ki(256,    192,    128,    2, 0, 0, "Default"),

    # ---- lds_stage=1 variants (single LDS buffer) ----
    63: _ki(16,     64,     256,    1, 0, 0, "Default"),
    64: _ki(16,     64,     512,    1, 0, 0, "Default"),
    65: _ki(16,     128,    256,    1, 0, 0, "Default"),
    66: _ki(16,     128,    512,    1, 0, 0, "Default"),
    67: _ki(16,     256,    256,    1, 0, 0, "Default"),
    68: _ki(16,     256,    512,    1, 0, 0, "Default"),
    69: _ki(16,     512,    256,    1, 0, 0, "Default"),
    70: _ki(32,     64,     128,    1, 0, 0, "Default"),
    71: _ki(32,     64,     256,    1, 0, 0, "Default"),
    72: _ki(32,     64,     512,    1, 0, 0, "Default"),
    73: _ki(32,     128,    128,    1, 0, 0, "Default"),
    74: _ki(32,     128,    256,    1, 0, 0, "Default"),
    75: _ki(64,     64,     128,    1, 0, 0, "Default"),
    76: _ki(64,     64,     256,    1, 0, 0, "Default"),
    77: _ki(64,     128,    128,    1, 0, 0, "Default"),
    78: _ki(64,     128,    256,    1, 0, 0, "Default"),
    79: _ki(64,     256,    128,    1, 0, 0, "Default"),
    80: _ki(128,    64,     128,    1, 0, 0, "Default"),
    81: _ki(128,    128,    128,    1, 0, 0, "Default"),
    82: _ki(128,    128,    256,    1, 0, 0, "Default"),
    83: _ki(128,    256,    128,    1, 0, 0, "Default"),
}

default_kernels_dict_942 = {
    (-1): _ki(128,  128,    128,    2, 0, 0, "Default"),
    (-2): _ki(16,   64,     512,    2, 0, 0, "Default"),
    (-3): _ki(32,   64,     512,    2, 0, 0, "Default"),
    (-4): _ki(64,   256,    64,     2, 0, 0, "Default"),
    (-5): _ki(128,  128,    64,     2, 0, 0, "Default"),
    (-6): _ki(128,  64,     128,    2, 0, 0, "Default"),
    (-7): _ki(64,   256,    128,    2, 0, 0, "Default"),
}

# ---------------------------------------------------------------------------
# gfx950 (MI350): MFMA fp8 uses float8_e4m3fn, KWTile=128
# Same tile search space, just different arch defaults.
# ---------------------------------------------------------------------------
kernels_list_950 = {
    0:  _ki(16,     64,     256,    2, 0, 0, "Default"),
    1:  _ki(16,     64,     512,    2, 0, 0, "Default"),
    2:  _ki(16,     128,    256,    2, 0, 0, "Default"),
    3:  _ki(16,     128,    512,    2, 0, 0, "Default"),
    4:  _ki(16,     256,    256,    2, 0, 0, "Default"),
    5:  _ki(16,     256,    512,    2, 0, 0, "Default"),
    6:  _ki(16,     512,    256,    2, 0, 0, "Default"),
    7:  _ki(16,     192,    256,    2, 0, 0, "Default"),
    8:  _ki(32,     64,     128,    2, 0, 0, "Default"),
    9:  _ki(32,     64,     256,    2, 0, 0, "Default"),
    10: _ki(32,     64,     512,    2, 0, 0, "Default"),
    11: _ki(32,     128,    128,    2, 0, 0, "Default"),
    12: _ki(32,     128,    256,    2, 0, 0, "Default"),
    13: _ki(32,     192,    128,    2, 0, 0, "Default"),
    14: _ki(32,     192,    256,    2, 0, 0, "Default"),
    15: _ki(32,     256,    128,    2, 0, 0, "Default"),
    16: _ki(32,     256,    256,    2, 0, 0, "Default"),
    17: _ki(48,     64,     256,    2, 0, 0, "Default"),
    18: _ki(48,     128,    256,    2, 0, 0, "Default"),
    19: _ki(48,     192,    256,    2, 0, 0, "Default"),
    20: _ki(48,     256,    256,    2, 0, 0, "Default"),
    21: _ki(64,     64,     128,    2, 0, 0, "Default"),
    22: _ki(64,     64,     256,    2, 0, 0, "Default"),
    23: _ki(64,     128,    128,    2, 0, 0, "Default"),
    24: _ki(64,     128,    256,    2, 0, 0, "Default"),
    25: _ki(64,     192,    128,    2, 0, 0, "Default"),
    26: _ki(64,     192,    256,    2, 0, 0, "Default"),
    27: _ki(64,     256,    64,     2, 0, 0, "Default"),
    28: _ki(64,     256,    128,    2, 0, 0, "Default"),
    29: _ki(64,     256,    256,    2, 0, 0, "Default"),
    30: _ki(80,     64,     256,    2, 0, 0, "Default"),
    31: _ki(80,     128,    256,    2, 0, 0, "Default"),
    32: _ki(80,     192,    256,    2, 0, 0, "Default"),
    33: _ki(80,     256,    256,    2, 0, 0, "Default"),
    34: _ki(96,     64,     128,    2, 0, 0, "Default"),
    35: _ki(96,     64,     256,    2, 0, 0, "Default"),
    36: _ki(96,     128,    128,    2, 0, 0, "Default"),
    37: _ki(96,     128,    256,    2, 0, 0, "Default"),
    38: _ki(96,     192,    128,    2, 0, 0, "Default"),
    39: _ki(96,     192,    256,    2, 0, 0, "Default"),
    40: _ki(96,     256,    128,    2, 0, 0, "Default"),
    41: _ki(96,     256,    256,    2, 0, 0, "Default"),
    42: _ki(112,    64,     256,    2, 0, 0, "Default"),
    43: _ki(112,    128,    256,    2, 0, 0, "Default"),
    44: _ki(112,    192,    256,    2, 0, 0, "Default"),
    45: _ki(112,    256,    256,    2, 0, 0, "Default"),
    46: _ki(128,    64,     128,    2, 0, 0, "Default"),
    47: _ki(128,    64,     256,    2, 0, 0, "Default"),
    48: _ki(128,    128,    64,     2, 0, 0, "Default"),
    49: _ki(128,    128,    128,    2, 0, 0, "Default"),
    50: _ki(128,    128,    256,    2, 0, 0, "Default"),
    51: _ki(128,    192,    128,    2, 0, 0, "Default"),
    52: _ki(128,    192,    256,    2, 0, 0, "Default"),
    53: _ki(128,    256,    128,    2, 0, 0, "Default"),
    54: _ki(160,    192,    128,    2, 0, 0, "Default"),
    55: _ki(192,    64,     128,    2, 0, 0, "Default"),
    56: _ki(192,    128,    128,    2, 0, 0, "Default"),
    57: _ki(224,    64,     128,    2, 0, 0, "Default"),
    58: _ki(224,    128,    128,    2, 0, 0, "Default"),
    59: _ki(224,    192,    128,    2, 0, 0, "Default"),
    60: _ki(256,    64,     128,    2, 0, 0, "Default"),
    61: _ki(256,    128,    128,    2, 0, 0, "Default"),
    62: _ki(256,    192,    128,    2, 0, 0, "Default"),
    63: _ki(256,    256,    128,    2, 0, 0, "Default"),

    64: _ki(16,     64,     256,    1, 0, 0, "Default"),
    65: _ki(16,     64,     512,    1, 0, 0, "Default"),
    66: _ki(16,     128,    256,    1, 0, 0, "Default"),
    67: _ki(16,     128,    512,    1, 0, 0, "Default"),
    68: _ki(16,     256,    256,    1, 0, 0, "Default"),
    69: _ki(16,     256,    512,    1, 0, 0, "Default"),
    70: _ki(16,     512,    256,    1, 0, 0, "Default"),
    71: _ki(32,     64,     128,    1, 0, 0, "Default"),
    72: _ki(32,     64,     256,    1, 0, 0, "Default"),
    73: _ki(32,     64,     512,    1, 0, 0, "Default"),
    74: _ki(32,     128,    128,    1, 0, 0, "Default"),
    75: _ki(32,     128,    256,    1, 0, 0, "Default"),
    76: _ki(64,     64,     128,    1, 0, 0, "Default"),
    77: _ki(64,     64,     256,    1, 0, 0, "Default"),
    78: _ki(64,     128,    128,    1, 0, 0, "Default"),
    79: _ki(64,     128,    256,    1, 0, 0, "Default"),
    80: _ki(64,     256,    128,    1, 0, 0, "Default"),
    81: _ki(128,    64,     128,    1, 0, 0, "Default"),
    82: _ki(128,    128,    128,    1, 0, 0, "Default"),
    83: _ki(128,    128,    256,    1, 0, 0, "Default"),
    84: _ki(128,    256,    128,    1, 0, 0, "Default"),
}

default_kernels_dict_950 = {
    (-1): _ki(128,  256,    256,    2, 0, 0, "Default"),
    (-2): _ki(16,   64,     512,    2, 0, 0, "Default"),
    (-3): _ki(32,   64,     512,    2, 0, 0, "Default"),
    (-4): _ki(128,  128,    128,    2, 0, 0, "Default"),
}
# fmt: on

arch = get_gfx()
if arch == "gfx942":
    kernels_list = kernels_list_942
    default_kernels_dict = default_kernels_dict_942
else:
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950
