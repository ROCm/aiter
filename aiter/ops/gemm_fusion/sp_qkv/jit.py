# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""JIT build of the sp_qkv small-M kernels (Phase-1 recipe, byte-identical sources).

Mirrors candidates/SM-T1-r{17,18}/build_device.sh + host/build_host.sh with explicit -D flags derived from the profile
row params. Device: hipcc -> .so -> .hip_fatbin -> clang-offload-bundler -> sm_gemm.hsaco / sm_ran.hsaco; the expected
code-object hashes are generated into gen/sm_expected_hashes.h (as the Phase-1 recipe does) and the host driver is built
against it. One build directory per cache key (sha256 of flags + source manifest + wrapper sources + toolchain).

Build RECIPES (cache keys are unchanged for "legacy"; the others fold the recipe name into the key because their artifacts differ):
  legacy  one build per call, serial: device .so x2 -> unbundle -> hash header -> host driver (the Phase-1 recipe, kept for A/B and as the reference).
  aot1    DEFAULT. Same device compile commands (so the code objects are byte-identical to legacy) but run by prebuild.py as ONE parallel task graph over every unit a profile
          needs; the host drivers no longer wait for the device builds (the expected hashes are linked in as a tiny extern-string object, sm_hashes.o).
  aot2    aot1 + device TUs compiled straight to a code object (--offload-device-only --no-gpu-bundle-output): no host pass, no objcopy, no bundler.
          Opt-in until `prebuild --verify-device-only` shows the .text/.rodata of every kernel equal to aot1 on the node.
See prebuild.py (profile enumeration, parallel scheduler, rank coordination, per-task wall report).

Environment (MORI is consumed read-only):
  MORI_ROOT      include root containing mori/cco/cco.hpp      (default /reference-mori)
  MORI_CCO_SO    libmori_cco.so                                (default $MORI_ROOT/build/src/cco/libmori_cco.so)
  ROCM_PATH      default /opt/rocm
  AITER_JIT_DIR  cache root (default ~/.aiter/jit); builds go to <root>/gemm_fusion_sp_qkv/<key>/
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "..", "csrc", "gemm_fusion", "sp_qkv")
)
SRCS = ("r17", "r18", "r19", "r20", "r21", "r22", "r23")
LARGE_SRCS = ("r11", "r12", "r13", "r18")
R18_FAMILY = (
    "r18",
    "r19",
    "r20",
    "r21",
    "r22",
    "r23",
)  # r19 is derived from r18 (design card r2 D-1 path A), r20 from r19: they accept the r18-only flags (r20: see small_defs for what it refuses)
# kernel ABI generation of a source: 2 = the shipped drivers (poison words, pointers baked in the device block); 3 = R12 (per-launch kernargs, per-comm ERR/STEP page,
# call_id tags). Plans of abi 3 take `call_id` in run(), carry the global page request and register no per-plan poison hooks.
SMALL_ABI = {
    "r17": 2,
    "r18": 2,
    "r19": 3,
    "r20": 3,
    "r21": 3,
    "r22": 3,
    "r23": 3,
}
# padded rows per source (SM_MP): r17-r19 cover local m 1..32 (T <= 128); r21 / r22 (MP128) cover m 1..128, r23 (MP256) m 1..256; r20 (R13 MP64) covers m 1..64 (decode T up to 256: the profile uses it for m 33..64)
SMALL_MP = {
    "r17": 32,
    "r18": 32,
    "r19": 32,
    "r20": 64,
    "r21": 128,
    "r22": 128,
    "r23": 256,
}
# large: 2 = r11 (per-plan poison word), 3 = r12 (LG_COMM_ABI_VERSION 4: per-comm ERR/STEP page, call_id tags), 4 = r13 (LG_COMM_ABI_VERSION 5: + the real m_real per launch, lg_launch_v3)
LARGE_ABI = {
    "r11": 2,
    "r12": 3,
    "r13": 4,
    "r18": 4,
}  # r18 = large/r18-put: the r13 ABI (LG_COMM_ABI_VERSION 5, LgCommArgsV4 360 B) + lg_put_v1
# R16 pad-M (scheme 2: "the kernel only gets the valid m; the padded m only selects the kernel / plan"): sources whose LAUNCH accepts any real local m <= the bucket M the
# binding / plan was made for. small: sm_launch_blk_v3m(..., m, ...) (r19 / r20 / r22 / r23; r21 and the abi 2 sources only launch their bound M); large: lg_launch_v3(..., m_real, ...) (r13, ABI 5)
SMALL_PAD_M = ("r19", "r20", "r22", "r23")
# Sources that can be BUILT but have NO plan (none at the moment).
SMALL_BUILD_ONLY: tuple[str, ...] = ()
LARGE_PAD_M = ("r13", "r18")
# R18 route B (src "r18"): the plan launches the large/r17 FlyDSL GEMM (destination-major BF16 epilogue into S) and then large/r18-put's lg_put_launch_v1 (put kernel + r13's quiet_ready). The comm
# image / driver tree of the source named "r18" is large/r18-put; the GEMM source (a FlyDSL Python kernel, compiled by FlyDSL at first use, not by hipcc) is large/r17.
LARGE_SRC_DIR = {"r18": "r18-put"}
R17_SRC = "r17"


def large_src_dir(src: str) -> str:
    """The directory name under csrc/gemm_fusion/sp_qkv/large/ of a large source (r18 = large/r18-put)."""
    return LARGE_SRC_DIR.get(src, src)


def src_pad_m(kind: str, src: str) -> bool:
    """True when a plan of this source can launch with a real local m below its bucket M (R16 pad-M)."""
    if kind not in ("small", "large"):
        raise ValueError(kind)
    return src in (SMALL_PAD_M if kind == "small" else LARGE_PAD_M)


def src_abi(kind: str, src: str) -> int:
    """Kernel ABI generation (2 or 3) of a source; kind = "small" | "large"."""
    table = SMALL_ABI if kind == "small" else LARGE_ABI
    if src not in table:
        raise ValueError(f"{kind} src {src!r} not in {tuple(table)}")
    return table[src]


def src_mp(src: str) -> int:
    """Padded rows per source (SM_MP): the largest local m a small plan of this source accepts."""
    if src not in SMALL_MP:
        raise ValueError(f"small src {src!r} not in {tuple(SMALL_MP)}")
    return SMALL_MP[src]


def _source_sha(kind: str, src: str) -> str:
    """sha256 of <kind>/<src>/SOURCE_SHA256.txt; a clear BuildError while a source (r19 / r12) has not been delivered yet."""
    if kind == "large":
        src = large_src_dir(src)
    path = os.path.join(CSRC, kind, src, "SOURCE_SHA256.txt")
    if not os.path.exists(path):
        raise BuildError(
            f"{kind} source {src!r} is not present ({path}): the R12 sources (small/r19, large/r12) are delivered by the implementors"
        )
    return _sha(path)


RECIPES = ("legacy", "aot1", "aot2")
DEFAULT_RECIPE = "aot1"


def recipe_key(key: str, recipe: str) -> str:
    """Cache key of a recipe: the legacy key itself, otherwise a hash of (legacy key, recipe) so that artifacts of different recipes never mix."""
    if recipe not in RECIPES:
        raise ValueError(f"recipe {recipe!r} not in {RECIPES}")
    return (
        key
        if recipe == "legacy"
        else hashlib.sha256(f"{key}|{recipe}".encode()).hexdigest()[:20]
    )


from aiter.gemm_fusion.prebuild import (
    BuildError,
)


@dataclass(frozen=True)
class SmallBuild:
    key: str
    dir: str
    gemm_hsaco: str
    ran_hsaco: str
    driver_so: str


# ---------------------------------------------------------------- flags
def small_defs(params: dict, mode: str) -> tuple[list[str], list[str]]:
    """(device_defs, host_defs) for one profile row. mode in {dense, indexer, reuse}.
    params (all optional, ints): src ("r17"|"r18"), sk, nh, no_credit, push8, push_halves, b_rowsplit, rx_bypass,
    reuse_no_idx. NT stores are never enabled (rejected in ledger r160)."""
    if mode not in ("dense", "indexer", "reuse"):
        raise ValueError(f"mode {mode!r}")
    src = params.get("src", "r17")
    if src not in SRCS:
        raise ValueError(f"src {src!r}")
    allowed = {
        "src",
        "sk",
        "nh",
        "no_credit",
        "push8",
        "push_halves",
        "b_rowsplit",
        "rx_bypass",
        "reuse_no_idx",
        "nt",
        "b_quad",
        "gemm_bn128",
        "rx3",
        "pfk",
        "p_nt",
        "ik_reload_p",
        "ik_seq",
        "push_passtrip",
        "push_ws_late",
        "tile",
    }
    extra = set(params) - allowed
    if extra:
        raise ValueError(f"unknown small-M params {sorted(extra)}")
    if params.get("nt", 0):
        raise ValueError("NT stores are rejected (ledger r160)")
    idx = 1 if mode == "indexer" else 0
    dense = 1 if mode == "dense" else 0
    nh = int(params.get("nh", 1))
    if (
        src in ("r19", "r20", "r21", "r22", "r23")
        and int(params.get("no_credit", 1)) != 1
    ):
        raise ValueError(
            f"{src}: SM_NO_CREDIT must be 1 (the credit slow path used the retired poison word)"
        )
    if src in ("r20", "r21", "r22", "r23"):
        # R13 MP64 / MP128: ONE push shape, the B row split is intrinsic, B_HEADS = 1 (the 216 / 240 B WG counts assume it) and the bypass loads are REQUIRED in every mode
        # (sm_abi_v3.h #errors on SM_PUSH8 / SM_PUSH_HALVES / SM_B_ROWSPLIT / !SM_RX_BYPASS): the flags that no longer exist are refused here, before a compile
        gone = [k for k in ("push8", "push_halves", "b_rowsplit") if params.get(k)]
        if gone:
            raise ValueError(
                f"{src} has no {gone} (one push shape, intrinsic B row split)"
            )
        if int(params.get("rx_bypass", 0)) != 1:
            raise ValueError(
                f"{src} requires rx_bypass=1 in every mode (R13 card R2.4 D-10; gfx950 only)"
            )
        # B_HEADS: r20 supports NH1 only (its 216 / 240 B WG counts); r21 supports NH1 (default: 432 / 480 B WGs, grid 552 / 608) and NH2 (216 / 240 B WGs, grid 336 / 368:
        # the A/B for the occupancy caveat of the 1024-slot launch)
        allowed_nh = (1, 2) if src in ("r21", "r22", "r23") else (1,)
        if nh not in allowed_nh:
            raise ValueError(f"{src} requires nh in {allowed_nh} (B_HEADS), got {nh}")
    abi = [
        f"-DSM_R5X_INDEXER={idx}",
        f"-DSM_DENSE={dense}",
        f"-DSM_T1_B_HEADS={nh}",
        f"-DSM_SK_CFG={int(params.get('sk', 3))}",
        f"-DSM_PUSH8={int(params.get('push8', 0))}",
        f"-DSM_PUSH_HALVES={int(params.get('push_halves', 0))}",
        "-DSM_NT_STORES=0",
    ]
    if src in R18_FAMILY:
        abi += [
            f"-DSM_B_ROWSPLIT={int(params.get('b_rowsplit', 0))}",
            f"-DSM_RX_BYPASS={int(params.get('rx_bypass', 0))}",
            f"-DSM_REUSE_NO_IDX={int(params.get('reuse_no_idx', 0))}",
        ]
    elif any(params.get(k) for k in ("b_rowsplit", "rx_bypass", "reuse_no_idx")):
        raise ValueError("b_rowsplit/rx_bypass/reuse_no_idx require src=r18 or r19")
    # r22 (tuned MP128, R14 F1) / r23 (MP256) option switches. Source defaults: B_QUAD 1, RX3 1, GEMM_BN128 1 (r22) / 0 (r23: BN128 is a net loss on GPU, BM256 x BN64 PFK4 is the
    # default). r22: 0 = the r21 behaviour for the A/B. r23: REQUIRES rx3=1 and b_quad=1 (`#error` otherwise); gemm_bn128 and the GEMM prefetch depth `pfk` are optional. They change
    # the GRID (B-quad, BN) and the Rx size (RX3), all compile-time in sm_abi_v3.h, so they go to the host build too and are passed EXPLICITLY (a header default change cannot move a
    # build silently).
    knobs = [k for k in ("b_quad", "gemm_bn128", "rx3") if k in params]
    if knobs and src not in ("r22", "r23"):
        raise ValueError(f"{knobs} exist only in src r22 and r23")
    if "pfk" in params and src != "r23":
        raise ValueError("['pfk'] exists only in src r23")
    if "p_nt" in params and src not in ("r22", "r23"):
        raise ValueError("['p_nt'] exists only in src r22 and r23")
    dev_extra: list[str] = []
    ik_opts = [k for k in ("ik_reload_p", "ik_seq") if k in params]
    if ik_opts and src != "r23":
        raise ValueError(f"{ik_opts} exist only in src r23")
    push_opts = [k for k in ("push_passtrip", "push_ws_late") if k in params]
    if push_opts and src not in ("r22", "r23"):
        raise ValueError(f"{push_opts} exist only in src r22 and r23")
    if src == "r23":
        # r23 indexer fallbacks for the RAN spill (opt-in, default 0): ik_reload_p = SM_IK_RELOAD_P (the IK owner re-loads P instead of holding the pushed values; a device body
        # switch, no geometry), ik_seq = SM_IK_SEQ (the two IK iterations become two IK WGs per owner: SM_T1_IK_WGS 8, grid 252, K 24 per (src, dst)). ik_seq changes the RAN GRID,
        # which the HOST computes at compile time from the same header macros (RAN_GRID = SM_T1_GRID: nothing hard-codes IK_WGS), so it goes to device AND host; ik_reload_p is device-only.
        # Both act on the IK owners, which exist only in the INDEXER build: set to 1 in dense / reuse they would only duplicate a build, so they are refused there.
        ik = {k: int(params.get(k, 0)) for k in ("ik_reload_p", "ik_seq")}
        bad = [k for k, v in ik.items() if v not in (0, 1)]
        if bad:
            raise ValueError(f"r23 {bad} must be 0 or 1")
        if mode != "indexer" and any(ik.values()):
            raise ValueError(
                f"r23 {[k for k, v in ik.items() if v]} apply to the indexer build only (IK owners), not mode {mode!r}"
            )
        dev_extra.append(f"-DSM_IK_RELOAD_P={ik['ik_reload_p']}")
        ik_abi = [f"-DSM_IK_SEQ={ik['ik_seq']}"]
    else:
        ik_abi = []
    if src in ("r22", "r23"):
        # SM_SK_CFG (split-K factor, device AND host: P [SK][MP][N]) is 2 or 3 in the source (static_assert); SM_P_NT (device only: the GEMM epilogue's store flavour for the P
        # partials, values and reduction order unchanged): 0 = default, 1 = non-temporal, 2 = sc0 sc1 nt (system write-through). Both passed explicitly.
        sk = int(params.get("sk", 3))
        if sk not in (2, 3):
            raise ValueError(f"{src} sk must be 2 or 3 (SM_SK_CFG), got {sk}")
        p_nt = int(params.get("p_nt", 0))
        if p_nt not in (0, 1, 2):
            raise ValueError(f"{src} p_nt must be 0, 1 or 2, got {p_nt}")
        dev_extra.append(f"-DSM_P_NT={p_nt}")
        # push live-set options (r22 and r23; opt-in, default 0, DEVICE only, push role only, valid in every mode): push_passtrip = SM_PUSH_PASSTRIP (two one-pass sub-trips: pk 72
        # floats, +1 P-load round trip per iteration), push_ws_late = SM_PUSH_WS_LATE (Wscale loaded in its own short trip after the decision: -24 floats during the P loads). They
        # change the push body only (no grid, K or window), so the host build never sees them; passed explicitly.
        push = {k: int(params.get(k, 0)) for k in ("push_passtrip", "push_ws_late")}
        bad_push = [k for k, v in push.items() if v not in (0, 1)]
        if bad_push:
            raise ValueError(f"{src} {bad_push} must be 0 or 1")
        dev_extra.append(f"-DSM_PUSH_PASSTRIP={push['push_passtrip']}")
        dev_extra.append(f"-DSM_PUSH_WS_LATE={push['push_ws_late']}")
    if src in ("r22", "r23"):
        bn_default = 0 if src == "r23" else 1
        vals = {
            "b_quad": int(params.get("b_quad", 1)),
            "gemm_bn128": int(params.get("gemm_bn128", bn_default)),
            "rx3": int(params.get("rx3", 1)),
        }
        bad = [k for k, v in vals.items() if v not in (0, 1)]
        if bad:
            raise ValueError(f"{src} {bad} must be 0 or 1")
        if src == "r23" and (vals["rx3"] != 1 or vals["b_quad"] != 1):
            raise ValueError(
                f"{src} (MP256) requires rx3=1 and b_quad=1 (4 Rx slots would need 9.4 MB > 8 MiB; sm_abi_v3.h #error); gemm_bn128 is optional (default 0)"
            )
        abi += [
            f"-DSM_B_QUAD={vals['b_quad']}",
            f"-DSM_GEMM_BN128={vals['gemm_bn128']}",
            f"-DSM_RX3={vals['rx3']}",
        ]
        if src == "r23":
            # BM256 GEMM prefetch depth: LDS = PFK x (A 32 KiB + B 8 KiB (BN64) / 16 KiB (BN128)) <= 160 KiB: BN64 PFK 3 (120 KiB) or 4 (160 KiB, the default), BN128 PFK 3 (144 KiB, default)
            pfk = int(params.get("pfk", 3 if vals["gemm_bn128"] else 4))
            if pfk not in (3, 4):
                raise ValueError(f"{src} pfk must be 3 or 4, got {pfk}")
            if vals["gemm_bn128"] and pfk == 4:
                raise ValueError(
                    f"{src} gemm_bn128=1 with pfk=4 needs 4 x 48 KiB = 192 KiB of LDS > the 160 KiB per CU"
                )
            abi.append(f"-DSM_GEMM_PFK={pfk}")
    abi += ik_abi  # r23 SM_IK_SEQ: device AND host (the RAN grid is a compile-time host constant)
    # device-only (Phase-1 build_device.sh defaults P0=1 P1=1 P2=0 STAMPS=0 R1=1 R2=0 R3=1 R4=1 U2=1)
    dev = abi + [
        "-DSM_U2_HOST_PTRS=1",
        "-DSM_P0=1",
        "-DSM_P1=1",
        "-DSM_P2=0",
        "-DSM_STAMPS=0",
        "-DSM_R1_RELAXED_SPIN=1",
        "-DSM_R2_LIGHT_RELEASE=0",
        "-DSM_R3_SRC_CREDIT=1",
        "-DSM_R4_LAST_WG_FENCE=1",
        f"-DSM_NO_CREDIT={int(params.get('no_credit', 1))}",
        *dev_extra,
    ]
    # host gets every ABI-affecting switch (grid / thread count / ntiles are compile-time in sm_abi_v2.h)
    return dev, abi


# FlyDSL GEMM tile configs of the r18 plans (large/r17 compile_preshuffle_gemm options; ids = the aiter a8w8 bpreshuffle tuned names of microbench_r17.py): 477 = flydsl_bpreshuflle_112x192x256_F8_F8_B16_1x1x4x2,
# 236 = flydsl_bpreshuflle_128x192x256_F8_F8_B16_1x0x4x2, 238 = 160x192x128, 237 = 128x256x128 (BN 256: the wave rule uses each tile's OWN tile_n). More tiles are added here when Evidence's sweep names them (the wave rule is r18_gemm.tile_for).
R18_TILES = {
    477: {
        "tile_m": 112,
        "tile_n": 192,
        "tile_k": 256,
        "use_async_copy": True,
        "waves_per_eu": 1,
        "xcd_swizzle": 4,
        "lds_stage": 2,
    },
    236: {
        "tile_m": 128,
        "tile_n": 192,
        "tile_k": 256,
        "use_async_copy": True,
        "waves_per_eu": None,
        "xcd_swizzle": 4,
        "lds_stage": 2,
    },
    # Evidence's sweep winners (flydsl a8w8 bpreshuffle kernels_list options: async copy 1, waves_per_eu 0 = unset, xcd_swizzle 4, lds_stage 2, scheduler on); tile_k 128, K 6144 = 48 steps
    238: {
        "tile_m": 160,
        "tile_n": 192,
        "tile_k": 128,
        "use_async_copy": True,
        "waves_per_eu": None,
        "xcd_swizzle": 4,
        "lds_stage": 2,
        "enable_scheduler": True,
    },
    237: {
        "tile_m": 128,
        "tile_n": 256,
        "tile_k": 128,
        "use_async_copy": True,
        "waves_per_eu": None,
        "xcd_swizzle": 4,
        "lds_stage": 2,
        "enable_scheduler": True,
    },
}
# MEASURED tile per (N, launch M) from Evidence's sweep (us in the comment): per N, ascending (m_max, tile id) pairs, the first with m <= m_max wins; outside the table (other N, m above the last bound)
# r18_gemm.tile_for falls back to the wave rule. 600 / 9216: 238 45.2; 700 / 9216: 238 46.3 vs 237 47.5; 600 / 9856: 238 49.6; 700 / 9856: 237 53.1.
R18_TILE_TABLE = {
    9216: ((512, 477), (768, 238)),
    9856: ((384, 477), (512, 236), (640, 238), (768, 237)),
}
# sha256 of large/r17/SOURCE_SHA256.txt (the FlyDSL dst-map GEMM the r18 plans were integrated against; the generated flydsl_dstmap_gemm.py itself is byte-compared by its own static check)
R17_SRC_SHA = "c6beda30ccc63b6b0fa6e78cd979920c6b5a798eef6ae7392228768d05bb26ff"


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def cache_key(src: str, dev: list[str], host: list[str], tool: dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(_source_sha("small", src).encode())
    for rel in ("host/sm_fusion_driver.cpp", "host/gf_comm.cpp"):
        h.update(_sha(os.path.join(CSRC, rel)).encode())
    h.update(repr((sorted(dev), sorted(host), sorted(tool.items()))).encode())
    return h.hexdigest()[:20]


def toolchain() -> dict[str, str]:
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    mori_root = os.environ.get("MORI_ROOT", "/reference-mori")
    so = os.environ.get(
        "MORI_CCO_SO", os.path.join(mori_root, "build", "src", "cco", "libmori_cco.so")
    )
    return {
        "hipcc": os.path.join(rocm, "bin", "hipcc"),
        "clangxx": os.path.join(rocm, "llvm", "bin", "clang++"),
        "bundler": os.path.join(rocm, "llvm", "bin", "clang-offload-bundler"),
        "rocm": rocm,
        "mori_root": mori_root,
        "mori_so": so,
    }


def jit_root() -> str:
    root = os.environ.get("AITER_JIT_DIR") or os.path.join(
        os.path.expanduser("~"), ".aiter", "jit"
    )
    return os.path.join(root, "gemm_fusion_sp_qkv")


# ---------------------------------------------------------------- command plan (pure; unit-tested)
SMALL_DEVICE_TUS = (("sm_gemm", "sk_gemm_fp8.cu"), ("sm_ran", "ran_kernel.cu"))


def device_cmd_one(
    src: str, dev_defs: list[str], tool: dict[str, str], out: str, name: str
) -> list[str]:
    """The legacy compile command of ONE small device TU (name in sm_gemm | sm_ran): hipcc -> <out>/img/<name>.so (device fatbin embedded)."""
    sdir = os.path.join(CSRC, "small", src)
    flags = dev_defs + [
        f"-I{os.path.join(sdir, 'device')}",
        f"-I{os.path.join(sdir, 'interface')}",
        f"-I{os.path.join(CSRC, 'third_party', 'opus_phase1')}",  # Phase-1 opus headers (aiter's own opus.hpp differs)
        f"-I{tool['mori_root']}/include",
        "-D__HIPCC_RTC__",
        "-DBUILD_CCO_SDMA=1",
        "-DOPUS_TILE_CONTAINER=0",
        "-std=c++20",
        "-O3",
        "-fno-fast-math",
        "--offload-arch=gfx950",
    ]
    cu = dict(SMALL_DEVICE_TUS)[name]
    return [
        tool["hipcc"],
        *flags,
        "-fPIC",
        "-shared",
        os.path.join(sdir, "device", cu),
        "-o",
        os.path.join(out, "img", name + ".so"),
    ]


def device_cmds(
    src: str, dev_defs: list[str], tool: dict[str, str], out: str
) -> list[list[str]]:
    return [device_cmd_one(src, dev_defs, tool, out, n) for n, _cu in SMALL_DEVICE_TUS]


DEVICE_ONLY_FLAGS = ("--offload-device-only", "--no-gpu-bundle-output")


def device_only_cmd(legacy_cmd: list[str], hsaco: str) -> list[str]:
    """Derive the aot2 command from a legacy device command: the SAME flags and source, minus the host-side `-shared` link, with the device-only switches and the
    code object as the output (no host pass, no objcopy, no clang-offload-bundler). Derived mechanically so nothing else can drift.
    """
    if "-shared" not in legacy_cmd or "-o" not in legacy_cmd:
        raise ValueError("not a legacy device command")
    cmd = [a for a in legacy_cmd if a != "-shared"]
    i = cmd.index("-o")
    cmd[i + 1] = hsaco
    return [cmd[0], *DEVICE_ONLY_FLAGS, *cmd[1:]]


def host_cmds(
    src: str, host_defs: list[str], tool: dict[str, str], out: str
) -> list[list[str]]:
    sdir = os.path.join(CSRC, "small", src)
    flags = host_defs + [
        "-x",
        "c++",
        "-D__HIP_PLATFORM_AMD__=1",
        f"-I{tool['rocm']}/include",
        f"-I{os.path.join(out, 'gen')}",
        f"-I{os.path.join(sdir, 'interface')}",
        f"-I{os.path.join(sdir, 'host')}",
        f"-I{tool['mori_root']}/include",
        "-std=c++20",
        "-O3",
        "-fno-fast-math",
        "-fPIC",
        "-pthread",
    ]
    libs = [
        "-x",
        "none",
        tool["mori_so"],
        f"-L{tool['rocm']}/lib",
        "-lamdhip64",
        "-ldl",
        "-Wl,--no-undefined",
    ]
    drv = [
        tool["clangxx"],
        *flags,
        "-shared",
        os.path.join(CSRC, "host", "sm_fusion_driver.cpp"),
        "-o",
        os.path.join(out, "sm_driver.so"),
        *libs,
    ]
    return [drv]


def host_compile_cmd(
    src: str, host_defs: list[str], tool: dict[str, str], out: str
) -> list[str]:
    """The host driver compiled to an object (same flags as host_cmds, `-c` instead of `-shared`, no link libs)."""
    sdir = os.path.join(CSRC, "small", src)
    flags = host_defs + [
        "-x",
        "c++",
        "-D__HIP_PLATFORM_AMD__=1",
        f"-I{tool['rocm']}/include",
        f"-I{os.path.join(out, 'gen')}",
        f"-I{os.path.join(sdir, 'interface')}",
        f"-I{os.path.join(sdir, 'host')}",
        f"-I{tool['mori_root']}/include",
        "-std=c++20",
        "-O3",
        "-fno-fast-math",
        "-fPIC",
        "-pthread",
    ]
    return [
        tool["clangxx"],
        *flags,
        "-c",
        os.path.join(CSRC, "host", "sm_fusion_driver.cpp"),
        "-o",
        os.path.join(out, "sm_driver.o"),
    ]


def hashes_compile_cmd(tool: dict[str, str], out: str) -> list[str]:
    return [
        tool["clangxx"],
        "-x",
        "c++",
        "-std=c++20",
        "-O1",
        "-fPIC",
        "-c",
        os.path.join(out, "gen", "sm_hashes.cpp"),
        "-o",
        os.path.join(out, "sm_hashes.o"),
    ]


def host_link_cmd(
    tool: dict[str, str], out: str, objs: list[str], so: str
) -> list[str]:
    libs = [
        "-x",
        "none",
        tool["mori_so"],
        f"-L{tool['rocm']}/lib",
        "-lamdhip64",
        "-ldl",
        "-Wl,--no-undefined",
    ]
    return [
        tool["clangxx"],
        "-shared",
        "-pthread",
        "-o",
        so,
        *[os.path.join(out, o) for o in objs],
        *libs,
    ]


def hashes_header_extern() -> str:
    """gen/sm_expected_hashes.h of the decoupled recipes: the vendored driver only passes SM_EXPECTED_*_SHA256 to sm_image_matches(const char*, const char*), so the macros can
    name extern strings that sm_hashes.o (generated after the device builds) defines: the host compile no longer depends on the device builds.
    """
    return (
        '#pragma once\nextern "C" const char sm_expected_gemm_sha256[];\nextern "C" const char sm_expected_ran_sha256[];\n'
        "#define SM_EXPECTED_GEMM_SHA256 sm_expected_gemm_sha256\n#define SM_EXPECTED_RAN_SHA256 sm_expected_ran_sha256\n"
    )


def hashes_source(gemm_sha: str, ran_sha: str) -> str:
    return (
        'extern "C" {\nextern const char sm_expected_gemm_sha256[];\nextern const char sm_expected_ran_sha256[];\n'
        f'const char sm_expected_gemm_sha256[] = "{gemm_sha}";\nconst char sm_expected_ran_sha256[] = "{ran_sha}";\n}}\n'
    )


def comm_cmd(tool: dict[str, str], out: str) -> list[str]:
    flags = [
        "-x",
        "c++",
        "-D__HIP_PLATFORM_AMD__=1",
        f"-I{tool['rocm']}/include",
        f"-I{tool['mori_root']}/include",
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-pthread",
    ]
    libs = [
        "-x",
        "none",
        tool["mori_so"],
        f"-L{tool['rocm']}/lib",
        "-lamdhip64",
        "-ldl",
        "-Wl,--no-undefined",
    ]
    return [
        tool["clangxx"],
        *flags,
        "-shared",
        os.path.join(CSRC, "host", "gf_comm.cpp"),
        "-o",
        os.path.join(out, "gf_comm.so"),
        *libs,
    ]


def _run(cmd: list[str], log: list[str]):
    log.append(" ".join(cmd))
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    if p.returncode != 0:
        raise BuildError(
            f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-4000:]}"
        )
    return p.stdout


def _unbundle(tool, so, fatbin, hsaco, log):
    _run(["objcopy", "-O", "binary", "--only-section=.hip_fatbin", so, fatbin], log)
    listing = _run([tool["bundler"], "--type=o", f"--input={fatbin}", "--list"], log)
    tgt = next((ln.strip() for ln in listing.splitlines() if "gfx950" in ln), None)
    if not tgt:
        raise BuildError(f"no gfx950 bundle entry in {fatbin}")
    _run(
        [
            tool["bundler"],
            "--type=o",
            "--unbundle",
            f"--targets={tgt}",
            f"--input={fatbin}",
            f"--output={hsaco}",
        ],
        log,
    )


def build_small(
    params: dict,
    mode: str,
    tool: dict[str, str] | None = None,
    recipe: str | None = None,
) -> SmallBuild:
    """Build (or reuse) the 2 HSACOs + host driver for one parameter set. Safe under concurrent ranks (atomic rename). recipe: see the module doc; the default
    (aot1) runs the unit's TUs in parallel and hits the artifacts prebuild_profile() produced.
    """
    recipe = recipe or DEFAULT_RECIPE
    if recipe != "legacy":
        from . import prebuild

        return prebuild.build_small(params, mode, tool, recipe)
    tool = tool or toolchain()
    src = params.get("src", "r17")
    dev, host = small_defs(params, mode)
    key = cache_key(src, dev, host, tool)
    final = os.path.join(jit_root(), key)
    sb = SmallBuild(
        key,
        final,
        os.path.join(final, "img", "sm_gemm.hsaco"),
        os.path.join(final, "img", "sm_ran.hsaco"),
        os.path.join(final, "sm_driver.so"),
    )
    if os.path.exists(os.path.join(final, "BUILD_OK")):
        return sb
    os.makedirs(jit_root(), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=key + ".", dir=jit_root())
    log: list[str] = []
    try:
        os.makedirs(os.path.join(tmp, "img"))
        os.makedirs(os.path.join(tmp, "gen"))
        for cmd, n in zip(device_cmds(src, dev, tool, tmp), ("sm_gemm", "sm_ran")):
            _run(cmd, log)
            so = os.path.join(tmp, "img", n + ".so")
            _unbundle(
                tool,
                so,
                os.path.join(tmp, "img", n + ".fatbin"),
                os.path.join(tmp, "img", n + ".hsaco"),
                log,
            )
        gh, rh = _sha(os.path.join(tmp, "img", "sm_gemm.hsaco")), _sha(
            os.path.join(tmp, "img", "sm_ran.hsaco")
        )
        with open(os.path.join(tmp, "gen", "sm_expected_hashes.h"), "w") as f:
            f.write(
                f'#pragma once\n#define SM_EXPECTED_GEMM_SHA256 "{gh}"\n#define SM_EXPECTED_RAN_SHA256 "{rh}"\n'
            )
        for cmd in host_cmds(src, host, tool, tmp):
            _run(cmd, log)
        with open(os.path.join(tmp, "build-command.txt"), "w") as f:
            f.write("\n".join(log) + "\n")
        with open(os.path.join(tmp, "BUILD_OK"), "w") as f:
            f.write(key + "\n")
        try:
            os.rename(tmp, final)
        except OSError:  # another rank finished first
            if not os.path.exists(os.path.join(final, "BUILD_OK")):
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return sb


def comm_key(tool: dict[str, str]) -> str:
    return hashlib.sha256(
        _sha(os.path.join(CSRC, "host", "gf_comm.cpp")).encode()
        + repr(sorted(tool.items())).encode()
    ).hexdigest()[:20]


def build_comm(tool: dict[str, str] | None = None, recipe: str | None = None) -> str:
    """Build (or reuse) gf_comm.so; returns its path (one TU: the recipe does not change the artifact or its key). The default recipes go through the rank-safe engine
    (one process compiles, the others wait on the unit lock); `legacy` is the old atomic-rename-only path.
    """
    recipe = recipe or DEFAULT_RECIPE
    if recipe != "legacy":
        from . import prebuild

        return prebuild.build_comm(tool)
    tool = tool or toolchain()
    h = hashlib.sha256(
        _sha(os.path.join(CSRC, "host", "gf_comm.cpp")).encode()
        + repr(sorted(tool.items())).encode()
    ).hexdigest()[:20]
    final = os.path.join(jit_root(), "comm_" + h)
    so = os.path.join(final, "gf_comm.so")
    if os.path.exists(so):
        return so
    os.makedirs(jit_root(), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="comm_" + h + ".", dir=jit_root())
    try:
        _run(comm_cmd(tool, tmp), [])
        try:
            os.rename(tmp, final)
        except OSError:
            if not os.path.exists(so):
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return so


# ======================================================================= large-M (LG-QSDMA-r11)
LARGE_SRC = "r11"


@dataclass(frozen=True)
class LargeBuild:
    key: str
    dir: str
    hsaco: str
    hsaco_sha256: str
    driver_so: str


def large_defs(params: dict, mode: str) -> list[str]:
    """-D flags (identical on device and host where the host reads them). mode in {dense, indexer, reuse}: reuse uses the DENSE build
    (LG_X_INDEXER=0, compact F=2304). params: no_gridread (0|1|2, default 1 = r11/r10 default). Everything else is the r11 recipe default.
    """
    if mode not in ("dense", "indexer", "reuse"):
        raise ValueError(f"mode {mode!r}")
    allowed = {
        "plan",
        "n",
        "half_first",
        "quarter_first",
        "no_gridread",
        "src",
        "tag_early",
        "poll_batch",
        "qr_wgs",
        "sk",
        "tile",
    }
    extra = set(params) - allowed
    if extra:
        raise ValueError(f"unknown large-M params {sorted(extra)}")
    if params.get("src", LARGE_SRC) not in LARGE_SRCS:
        raise ValueError(f"large src {params.get('src')!r} not in {LARGE_SRCS}")
    ng = int(params.get("no_gridread", 1))
    if ng not in (0, 1, 2):
        raise ValueError("no_gridread")
    # r18 (route B): `tile` pins the FlyDSL GEMM tile config (R18_TILES id: 477 = 112x192, 236 = 128x192) of the plan; absent = the wave rule picks it per launch M (r18_gemm.tile_for). It is a
    # property of the GEMM (compiled by FlyDSL), not a -D of the comm build: it does not change the build key.
    if "tile" in params:
        if params.get("src", LARGE_SRC) != "r18":
            raise ValueError("['tile'] exists only in large/r18 (src=r18)")
        if params["tile"] not in R18_TILES:
            raise ValueError(
                f"r18 tile must be one of {sorted(R18_TILES)}, got {params['tile']!r}"
            )
    # r12b knobs (large/r12 only; the header defaults are 1, build_device.sh passes 1): tag_early = LG_TAG_EARLY (0|1: arrival tag stored in the first-put batch),
    # poll_batch = LG_POLL_BATCH (0|1|2: target + ERR loaded back-to-back; 2 = ERR every 8th iteration), qr_wgs = LG_QR_WGS (1|64: quiet_ready workgroups, the NaN fill folded in). Passed EXPLICITLY for r12 so an A/B is a different build key and a
    # header default change cannot move a build silently; r11 has no such knobs.
    knobs = [k for k in ("tag_early", "poll_batch", "qr_wgs") if k in params]
    if knobs and params.get("src", LARGE_SRC) not in ("r12", "r13", "r18"):
        raise ValueError(
            f"{knobs} exist only in large/r12, r13 and r18 (src=r12 | r13 | r18)"
        )
    tag_early, poll_batch = int(params.get("tag_early", 1)), int(
        params.get("poll_batch", 1)
    )
    if tag_early not in (0, 1):
        raise ValueError("tag_early must be 0 or 1")
    if poll_batch not in (0, 1, 2):
        raise ValueError("poll_batch must be 0, 1 or 2")
    # r12c: the NaN fill is folded into quiet_ready, which runs LG_QR_WGS workgroups (64 = the default, 1 = the A1'' single-WG form identical to r11's healthy path). The HOST
    # launches quiet_ready with LG_QR_WGS workgroups, so the host build must get the SAME value (see large_host_cmd); the vendored build_host.sh does not pass it.
    qr_wgs = int(params.get("qr_wgs", 64))
    if qr_wgs not in (1, 64):
        raise ValueError("qr_wgs must be 1 or 64")
    defs = [
        f"-DLG_X_INDEXER={1 if mode == 'indexer' else 0}",
        f"-DLG_NO_GRIDREAD={ng}",
        "-DBUILD_CCO_SDMA=1",
        "-DLG_REMOTE_HALF=0",
        "-DLG_R6_POSTER=1",
        "-DLG_R6_ARRIVAL_ACQ=0",
        "-DLG_X_FORCE_TIMEOUT_G1=0",
        "-DLG_X_FORCE_TIMEOUT_G0=0",
        "-DLG_R8_FOLD=0",
        "-DLG_X_NEG_DOUBLE_PUT=0",
        "-DLG_X_POSTER_LAST=0",
        "-DLG_DEBUG_POSTCOUNT=0",
        "-DLG_NEGCTRL_NOCAS=0",
        "-DLG_L1C_SWIZZLE=0",
        "-DLG_L2A_BYPASS=0",
        "-DLG_L2B_ONE_RELEASE=1",
        "-DLG_STAMPS=0",
        "-DLG_NEGCTRL_NOWB=0",
        "-DLG_NEGCTRL_NOWAIT=0",
        "-DLG_NEGCTRL_ISO=0",
        "-DLG_ISO_DELAY_ITERS=0",
    ]
    if params.get("src", LARGE_SRC) in ("r12", "r13", "r18"):
        defs += [
            f"-DLG_TAG_EARLY={tag_early}",
            f"-DLG_POLL_BATCH={poll_batch}",
            f"-DLG_QR_WGS={qr_wgs}",
        ]
    return defs


def large_cache_key(
    defs: list[str], tool: dict[str, str], src: str | None = None
) -> str:
    h = hashlib.sha256()
    h.update(_source_sha("large", src or LARGE_SRC).encode())
    h.update(_sha(os.path.join(CSRC, "host", "lg_fusion_driver.cpp")).encode())
    h.update(repr((sorted(defs), sorted(tool.items()))).encode())
    return h.hexdigest()[:20]


def large_device_cmd(
    defs: list[str], tool: dict[str, str], out: str, src: str | None = None
) -> list[str]:
    ldir = os.path.join(CSRC, "large", large_src_dir(src or LARGE_SRC))
    flags = defs + [
        f"-I{os.path.join(ldir, 'device')}",
        f"-I{os.path.join(ldir, 'interface')}",
        f"-I{os.path.join(CSRC, 'third_party', 'opus_phase1')}",
        f"-I{tool['mori_root']}/include",
        "-D__HIPCC_RTC__",
        "-DOPUS_TILE_CONTAINER=0",
        "-std=c++20",
        "-O3",
        "-fno-fast-math",
        "--offload-arch=gfx950",
    ]
    return [
        tool["hipcc"],
        *flags,
        "-fPIC",
        "-shared",
        os.path.join(ldir, "device", "gemm_quad_fp8_comm.cu"),
        "-o",
        os.path.join(out, "img", "lg_gemm_quad_fp8_comm.so"),
    ]


def large_host_cmd(
    defs: list[str], tool: dict[str, str], out: str, src: str | None = None
) -> list[str]:
    ldir = os.path.join(CSRC, "large", large_src_dir(src or LARGE_SRC))
    hostdefs = [
        d
        for d in defs
        if d.split("=")[0]
        in (
            "-DLG_REMOTE_HALF",
            "-DLG_R6_POSTER",
            "-DLG_X_INDEXER",
            "-DLG_R8_FOLD",
            "-DLG_DEBUG_POSTCOUNT",
            "-DLG_QR_WGS",  # r12c: the host launches quiet_ready with this many WGs: it MUST equal the device build's value
        )
    ]
    flags = hostdefs + [
        "-x",
        "c++",
        "-D__HIP_PLATFORM_AMD__=1",
        f"-I{tool['rocm']}/include",
        f"-I{os.path.join(ldir, 'interface')}",
        f"-I{os.path.join(ldir, 'host')}",
        f"-I{tool['mori_root']}/include",
        "-std=c++20",
        "-O2",
        "-fno-fast-math",
        "-fPIC",
        "-pthread",
    ]
    libs = [
        "-x",
        "none",
        tool["mori_so"],
        f"-L{tool['rocm']}/lib",
        "-lamdhip64",
        "-ldl",
        "-Wl,--no-undefined",
    ]
    return [
        tool["clangxx"],
        *flags,
        "-shared",
        os.path.join(CSRC, "host", "lg_fusion_driver.cpp"),
        "-o",
        os.path.join(out, "liblg_comm.so"),
        *libs,
    ]


def large_device_only_cmd(
    defs: list[str], tool: dict[str, str], out: str, src: str | None = None
) -> list[str]:
    return device_only_cmd(
        large_device_cmd(defs, tool, out, src),
        os.path.join(out, "img", "lg_gemm_quad_fp8_comm.hsaco"),
    )


def build_large(
    params: dict,
    mode: str,
    tool: dict[str, str] | None = None,
    recipe: str | None = None,
) -> LargeBuild:
    """Build (or reuse) the large-M HSACO (all plan entries n1/n2/n4/n2h/n4h/n3q + quiet_ready + drain) and host driver for one mode."""
    recipe = recipe or DEFAULT_RECIPE
    if recipe != "legacy":
        from . import prebuild

        return prebuild.build_large(params, mode, tool, recipe)
    tool = tool or toolchain()
    defs = large_defs(params, mode)
    src = params.get("src", LARGE_SRC)
    key = large_cache_key(defs, tool, src)
    final = os.path.join(jit_root(), "lg_" + key)
    mk = lambda d, sha="": LargeBuild(
        key,
        d,
        os.path.join(d, "img", "lg_gemm_quad_fp8_comm.hsaco"),
        sha,
        os.path.join(d, "liblg_comm.so"),
    )
    ok = os.path.join(final, "BUILD_OK")
    if os.path.exists(ok):
        return mk(final, open(ok).read().split()[-1])
    os.makedirs(jit_root(), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="lg_" + key + ".", dir=jit_root())
    log: list[str] = []
    try:
        os.makedirs(os.path.join(tmp, "img"))
        _run(large_device_cmd(defs, tool, tmp, src), log)
        so = os.path.join(tmp, "img", "lg_gemm_quad_fp8_comm.so")
        _unbundle(
            tool,
            so,
            os.path.join(tmp, "img", "lg.fatbin"),
            os.path.join(tmp, "img", "lg_gemm_quad_fp8_comm.hsaco"),
            log,
        )
        sha = _sha(os.path.join(tmp, "img", "lg_gemm_quad_fp8_comm.hsaco"))
        _run(large_host_cmd(defs, tool, tmp, src), log)
        with open(os.path.join(tmp, "build-command.txt"), "w") as f:
            f.write("\n".join(log) + "\n")
        with open(os.path.join(tmp, "BUILD_OK"), "w") as f:
            f.write(key + " " + sha + "\n")
        try:
            os.rename(tmp, final)
        except OSError:
            if not os.path.exists(ok):
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return mk(final, open(ok).read().split()[-1])
