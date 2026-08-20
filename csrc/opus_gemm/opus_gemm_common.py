# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS kernel registrations shared by selection and code generation."""

import os
from dataclasses import dataclass, field

# Legacy cache policy = traits default for split-barrier & persistent a16w16 (see
# opus_gemm_traits_a16w16_gfx950.cuh).
_LEGACY_CACHECTL = (0, 17)

_GFX942_KERNEL_NAME_TAGS = {
    "a16w16_kbuf1_sk": "splitk_legacy",
    "a16w16_kbuf2v_sk": "splitk_p1",
    "a16w16_kbuf2v_bk128_sk": "splitk_p1_bk128",
    "a16w16_em3en4_lds1_pgr2_sk": "splitk_em3en4_lds1_pgr2",
    "a16w16_wave_k_coop": "wkc",
    "a16w16_wave_k_coop_accum": "wkc_accum",
    "a16w16_kbuf2v": "p1",
    "a16w16_kbuf2v_bk128": "p1_bk128",
    "a16w16_quad_mfma32_kbuf1": "quad_mfma32",
    "a16w16_quad_mfma32_kbuf1_sk": "splitk_quad_mfma32_bf16ws",
}


@dataclass
class OpusGemmInstance:
    BLOCK_SIZE: int
    B_M: int
    B_N: int
    B_K: int
    T_M: int
    T_N: int
    W_M: int
    W_N: int
    W_K: int
    VEC_A: int
    VEC_B: int
    VEC_C: int
    GROUP_M: int
    GROUP_N: int
    GROUP_K: int
    kernel_tag: str
    output_dtypes: list[str] = field(default_factory=lambda: ["fp32_t"])
    # Flatmm-only. Defaults to 2 (match existing behavior for non-flatmm kernels).
    # Only emitted in the generated instance name when kernel_tag == "a16w16_flatmm".
    WG_PER_CU: int = 2
    # Compile-time OOB (out-of-bounds) tail handling.
    has_oob: bool = True
    # Cache policy for A/B loads (CDNA4 ISA Table 49). -1 = use traits default.
    # 0=LRU, 1=SC0(LLC Evict), 17=SC0+SC1(L2 Bypass).
    cachectl_a: int = -1
    cachectl_b: int = -1
    # 4g_safe variant flag. True = use the *_4g_safe_gfx950.cuh pipeline
    # header (per-WG-tight buffer-resource sizing -- safe for tensors
    # whose full extent exceeds 4 GiB). False = use the legacy header
    # (full row/col-band BR sizing which wraps at 4 GiB). Same Traits
    # struct and kargs struct either way; only the pipeline body and
    # kernel symbol differ. The kid families that set this to True live
    # under SPLITK_4G_SAFE_KIDS / NON_SPLITK_4G_SAFE_KIDS.
    is_4g_safe: bool = False

    # Optional arch prefix (e.g.
    arch_prefix: str = ""
    # Optional generated name tag override for same-pipeline variants.
    name_tag: str = ""
    # Physical workspace storage dtype for this exact kid. External-workspace
    # kids must declare it explicitly; non-workspace kids leave it unset.
    # The launch-dispatch host specialization remains fp32 independently.
    splitk_workspace_dtype: str | None = None

    # gfx1250 cluster/TDM split-K consumer tiling: "tileN" (split N) or
    # "tileM" (split M). Only consumed by the a16w16_cluster_tdm_splitk_ws tag.
    ctdm_layout: str = "tileN"

    # gfx1250 cluster_tdm_splitk_ws prefetch depth P (== LDS slots == in-flight
    # TDM count; producer keeps exactly this many TDMs in flight). 2 or 3.
    num_slots: int = 3
    # gfx1250 cluster_tdm_splitk_ws target WG/CU co-residency (1 or 2). 1 is
    # enforced via LDS padding in the traits; chosen by _ctdm_pick_configs() so
    # two WGs never oversubscribe a SIMD-pair's 256-request direct-copy budget.
    wg_per_cu: int = 2

    # gfx1250 clusterlaunch (multicast) cluster geometry: WGs per cluster in M/N
    # (__cluster_dims__(cluster_wg_m, cluster_wg_n, 1)). Only consumed by the
    # a16w16_clusterlaunch_tdm_splitk_ws tag; ignored by every other pipeline.
    cluster_wg_m: int = 4
    cluster_wg_n: int = 4

    # gfx950 a8w8 MXFP8 BMM compile-time axes.  BMM instances live in the
    # canonical global kid registry, but their generated symbols share the
    # ``opus_bmm`` root and a uniform exact-kid launcher signature.
    direct_only: bool = False
    prefetch_scale: bool = False
    fused_reduce: bool = False
    preload_sf: bool = False
    skip_scale_wait: bool = False
    pack_scale_on_demand: bool = False
    k1024_only: bool = False
    k1024_lb1: bool = False
    preload_sf_lds: bool = False
    name_root: str = "opus_gemm"

    # gfx1250 fused single-kernel split-K. SplitK and the N-direction cluster
    # peer count are compile-time kernel properties. The partial storage dtype
    # deliberately uses the shared splitk_workspace_dtype field above so every
    # external-workspace kid has one exact-kid dtype source of truth.
    fuse_split_k: int = 0
    # Historical field name retained for #4246/tuned-config compatibility; in
    # the current N-direction pipeline this is the number of N-tile peers.
    fuse_m_cluster: int = 1

    @property
    def name(self) -> str:
        parts = [
            self.name_root,
            "x".join(map(str, [self.BLOCK_SIZE, self.B_M, self.B_N, self.B_K])),
            "x".join(map(str, [self.T_M, self.T_N])),
            "x".join(map(str, [self.W_M, self.W_N, self.W_K])),
            "x".join(map(str, [self.GROUP_M, self.GROUP_N, self.GROUP_K])),
        ]
        if self.arch_prefix:
            parts.insert(1, self.arch_prefix)
        # tag inserts shift right by one slot when arch_prefix is set
        tag_at = 1 + (1 if self.arch_prefix else 0)
        if self.kernel_tag == "a8w8_mxscale_bmm_flatmm_splitk":
            parts.insert(tag_at, "a8w8_mxscale_flatmm_splitk")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.direct_only:
                parts.append("selfload")
            if self.prefetch_scale:
                parts.append("scaleprefetch")
            if self.preload_sf:
                parts.append("sfpreload")
        elif self.kernel_tag == "a8w8_mxscale_bmm_minterleave":
            parts.insert(tag_at, "a8w8_mxscale_flatmm_minterleave")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.skip_scale_wait:
                parts.append("skip_scale_wait")
        elif self.kernel_tag == "a8w8_mxscale_bmm_fused":
            parts.insert(tag_at, "a8w8_mxscale_flatmm_fused")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a8w8_mxscale_bmm_pipeline":
            parts.insert(tag_at, "a8w8_mxscale_pipeline")
            if self.k1024_only:
                parts.append("k1024")
            elif self.k1024_lb1:
                parts.append("k1024lb1")
            elif self.preload_sf_lds:
                parts.append("preload_sf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_mouter":
            parts.insert(tag_at, "a8w8_mxscale_flatmm_mouter")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.skip_scale_wait:
                parts.append("ssw")
        elif self.kernel_tag == "a8w8_mxscale_bmm_mouter_tunable":
            parts.insert(tag_at, "a8w8_mxscale_flatmm_mouter_tunable")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.skip_scale_wait:
                parts.append("ssw")
        elif self.kernel_tag == "a8w8_mxscale_bmm_wave8n2":
            parts.insert(tag_at, "a8w8_mxscale_flatmm_wave8n2")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a8w8_mxscale_bmm_wave4m2_selfload":
            parts.insert(tag_at, "a8w8_mxscale_flatmm_wave4m2_selfload")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.skip_scale_wait:
                parts.append("ssw")
            if self.pack_scale_on_demand:
                parts.append("psod")
        elif self.kernel_tag == "a16w16_flatmm":
            parts.insert(tag_at, "flatmm")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a16w16_flatmm_splitk":
            parts.insert(tag_at, "flatmm_splitk")
            parts.append(f"wgpcu{self.WG_PER_CU}")
        elif self.kernel_tag == "a16w16_persistent":
            parts.insert(tag_at, "persistent")
        elif self.kernel_tag == "a16w16_mono_tile":
            parts.insert(tag_at, "mono_tile")
        elif self.kernel_tag == "a16w16_cluster_tdm_splitk_ws":
            # gfx1250 typed-workspace split-K with a separate reduce kernel.
            # Name it opus_gemm_gfx1250_splitk_* (note the "splitk_" segment) so
            # the reduce-TU arch detection in gen_instances.py -- which keys on
            # "opus_gemm_<arch>_splitk_" -- buckets it like the gfx942 splitk kids.
            # The T_M x T_N segment (1x2 for tileN, 2x1 for tileM) keeps the name
            # unique between the two consumer-tiling layouts.
            parts.insert(tag_at, "splitk_cluster_tdm_ws")
            # Prefetch depth P and WG/CU occupancy make each (tile, P, wg) symbol
            # unique (the producer + LDS-pad differ by these).
            parts.append(f"p{self.num_slots}w{self.wg_per_cu}")
        elif self.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_ws":
            # gfx1250 CLUSTER-LAUNCH (multicast) split-K. Same "splitk_" segment so
            # the reduce-TU arch detection (keys on "opus_gemm_<arch>_splitk_")
            # buckets it like the other gfx1250 splitk kids. The cluster geometry
            # cCWMxCWN plus pPwW keep each (tile, cluster, P, wg) symbol unique.
            parts.insert(tag_at, "splitk_clusterlaunch_tdm_ws")
            parts.append(f"c{self.cluster_wg_m}x{self.cluster_wg_n}")
            parts.append(f"p{self.num_slots}w{self.wg_per_cu}")
        elif self.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse":
            # Keep "splitk_" out of this visible segment: fused kids do not
            # need a separate reduce-kernel TU. The historical fuse_m_cluster
            # field is an N-peer count, hence the n{} spelling.
            parts.insert(tag_at, "skfuse")
            parts.append(f"n{self.fuse_m_cluster}s{self.fuse_split_k}")
            parts.append(
                "wsf32"
                if self.splitk_workspace_dtype == "fp32_t"
                else "wsbf16"
            )
            parts.append(f"p{self.num_slots}w{self.wg_per_cu}")
        elif self.name_tag:
            parts.insert(tag_at, self.name_tag)
        elif self.kernel_tag in _GFX942_KERNEL_NAME_TAGS:
            name_tag = _GFX942_KERNEL_NAME_TAGS[self.kernel_tag]
            parts.insert(tag_at, name_tag)
        if not self.has_oob:
            parts.append("nooob")
        if self.is_4g_safe:
            parts.append("4g_safe")
        # Legacy cache policy = traits default for split-barrier & persistent a16w16: CACHECTL_A=0
        # (LRU), CACHECTL_B=17 (BYPASS_L2).
        if (self.cachectl_a, self.cachectl_b) != _LEGACY_CACHECTL and (
            self.cachectl_a >= 0 or self.cachectl_b >= 0
        ):
            parts.append(f"cA{self.cachectl_a}cB{self.cachectl_b}")
        return "_".join(parts)
    @property
    def m_align(self) -> int:
        """M multiple enforced by the generated launcher (1 means tail-safe)."""
        mult = _BMM_M_ALIGN_TILES.get(self.kernel_tag)
        if mult is not None:
            return self.B_M * mult if mult else 1
        return 1 if self.has_oob else self.B_M


def a16w16_flatmm_prefetch_k_iter(instance: OpusGemmInstance) -> int:
    """Mirror gfx950 ``Traits::prefetch_k_iter`` for host-side planning.

    The exact launcher and both GEMM/BMM tuning paths must agree on the
    minimum number of K tiles a flatmm instance can consume. Keep this
    scalar-only calculation next to the canonical instance metadata so the
    runtime launch plan and tuner do not drift.
    """
    sizeof_da = 2  # BF16
    load_group_m = 64 if instance.W_M >= 32 else 32
    load_group_n = 64 if instance.W_N >= 32 else 32
    load_group_k = instance.W_K * 2
    num_m = instance.B_M // load_group_m
    num_n = instance.B_N // load_group_n
    num_k = instance.B_K // load_group_k
    smem_linear = 64 * 16 // sizeof_da  # WARP_SIZE=64
    smem_sub = smem_linear // load_group_k
    slots = load_group_m // smem_sub
    padding = 16 // sizeof_da if instance.W_M >= 32 else 2 * 16 // sizeof_da
    per_group_load = slots * (smem_linear + padding) * sizeof_da
    per_iter = (num_m + num_n) * num_k * per_group_load
    lds_total = 163840
    return max(
        1,
        (lds_total // max(instance.WG_PER_CU, 1)) // max(per_iter, 1),
    )


_BMM_M_ALIGN_TILES = {
    "a8w8_mxscale_bmm_flatmm_splitk": 0,
    "a8w8_mxscale_bmm_pipeline": 0,
    "a8w8_mxscale_bmm_fused": 0,
    "a8w8_mxscale_bmm_minterleave": 2,
    "a8w8_mxscale_bmm_wave4m2_selfload": 2,
    "a8w8_mxscale_bmm_wave8n2": 1,
    "a8w8_mxscale_bmm_mouter": 1,
    "a8w8_mxscale_bmm_mouter_tunable": 1,
}

# PR #4320 originally used a private, colliding BMM id namespace.  The current
# exact-kid router uses one canonical registry, so gfx950 BMM ids occupy the
# previously empty 8000 band.  The low digits intentionally preserve the
# upstream id for tuning/debug correlation.
BMM_MXSCALE_KID_OFFSET = 8000


def bmm_mxscale_global_kid(upstream_kid: int) -> int:
    return BMM_MXSCALE_KID_OFFSET + int(upstream_kid)


def _a16w16(bs, bm, bn, bk, tn, wm, wn, wk, has_oob=True, cachectl_a=0, cachectl_b=17):
    """Factory for a16w16 split-barrier kid instances.

    cachectl_a / cachectl_b default to (0, 17) = (LRU, BYPASS_L2), which
    matches the traits-default cache policy for the split-barrier pipeline
    (see opus_gemm_a16w16_traits_gfx950 in
    csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh).
    This is the "legacy" policy used by KID 4..9 and 1004..1009 -- the
    `_LEGACY_CACHECTL` special-case in OpusGemmInstance.name keeps these
    kids emitting the bare `..._0x0x0` symbol (no `_cA0cB17` suffix) so
    the Python policy and OPUS tuned CSV stay bit-compatible.
    """
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    inst = OpusGemmInstance(
        bs,
        bm,
        bn,
        bk,
        2,
        tn,
        wm,
        wn,
        wk,
        vec,
        vec,
        4,
        0,
        0,
        0,
        "a16w16",
        ["fp32_t", "bf16_t"],
        has_oob=has_oob,
    )
    inst.cachectl_a = cachectl_a
    inst.cachectl_b = cachectl_b
    return inst


def _a16w16_flatmm_splitk(bm, bn, bk, wg_per_cu, has_oob=True):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        256,
        bm,
        bn,
        bk,
        2,
        1,  # T_M, T_N
        16,
        16,
        32,  # MFMA 16x16x32
        vec,
        vec,
        4,  # VEC
        0,
        0,
        0,  # GROUP (unused)
        "a16w16_flatmm_splitk",
        ["fp32_t"],
        wg_per_cu,
        has_oob=has_oob,
        splitk_workspace_dtype="fp32_t",
    )


def _a16w16_flatmm(bm, bn, bk, wg_per_cu):
    # Flatmm locked config (per gcnasm/opus_fmm/INTEGRATION.md): BLOCK_SIZE=256, T_M=2, T_N=1,
    # MFMA=(16,16,32), VEC=(8,8,4), HAS_BIAS...
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        256,
        bm,
        bn,
        bk,
        2,
        1,  # T_M, T_N (T_N hardcoded to 1 for the warp-spec pipeline)
        16,
        16,
        32,  # MFMA 16x16x32
        vec,
        vec,
        4,  # VEC
        0,
        0,
        0,  # GROUP (unused)
        "a16w16_flatmm",
        ["bf16_t", "fp32_t"],
        wg_per_cu,
    )


# fmt: off
# --- per-pipeline kernel instance lists ---
a8w8_scale_kernels_list = {
    1: OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
}


def _a8w8_mxscale_bmm_flatmm_splitk(
    bm, bn, bk, wg_per_cu, direct_only=False, prefetch_scale=False, preload_sf=False
):
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256, bm, bn, bk, t_m, t_n, 16, 16, 128, 16, 16, 4,
        1, 128, 128, "a8w8_mxscale_bmm_flatmm_splitk", ["fp32_t"],
        wg_per_cu, splitk_workspace_dtype="fp32_t",
    )
    inst.name_root = "opus_bmm"
    inst.direct_only = direct_only
    inst.prefetch_scale = prefetch_scale
    inst.preload_sf = preload_sf
    return inst


_BMM_MXSCALE_SPLITK_TILES = {
    316: (16,  32, 256, 2, False, False),
    317: (16,  32, 256, 2, False, True),
    318: (16,  32, 128, 2, False, False),
    319: (16,  32, 256, 4, False, False),
    314: (16,  32, 512, 2, False, False),
    313: (16,  64, 256, 2, False, False),
    312: (16, 128, 256, 1, False, False),
    311: (16,  32, 512, 2, False, True),
    321: (32,  32, 256, 2, False, True),
    323: (32,  32, 128, 2, False, True),
    320: (64,  32, 256, 2, False, False),
    322: (64,  32, 256, 1, False, False),
    640: (32,  64, 256, 2, False, False),
    642: (32,  64, 256, 1, False, False),
    646: (32,  64, 256, 2, True,  False),
    650: (64,  64, 128, 2, False, False),
    653: (64,  64, 128, 2, False, True),
    128: (128, 128, 128, 1, False, False),
    137: (128, 128, 128, 1, False, True),
    138: (64,  128, 256, 1, False, False),
    139: (128, 64,  256, 1, False, False),
    256: (32, 256, 128, 1, False, False),
    64:  (64, 128, 128, 2, False, False),
    0:   (32, 128, 128, 2, False, False),
    32:  (32, 128, 128, 2, False, False),
}
_bmm_flatmm_local = {
    kid: _a8w8_mxscale_bmm_flatmm_splitk(bm, bn, bk, wg, direct, prefetch)
    for kid, (bm, bn, bk, wg, direct, prefetch) in _BMM_MXSCALE_SPLITK_TILES.items()
}
_BMM_MXSCALE_SPLITK_PRELOAD_TILES = {
    324: (64, 32, 256, 2),
    325: (128, 128, 128, 1),
    326: (128, 64, 256, 1),
    327: (64, 128, 256, 1),
}
_bmm_flatmm_local.update({
    kid: _a8w8_mxscale_bmm_flatmm_splitk(bm, bn, bk, wg, preload_sf=True)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_SPLITK_PRELOAD_TILES.items()
})


def _a8w8_mxscale_bmm_minterleave(bm, bn, bk, wg_per_cu, skip_scale_wait=False):
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256, bm, bn, bk, t_m, t_n, 16, 16, 128, 16, 16, 4,
        1, 128, 128, "a8w8_mxscale_bmm_minterleave", ["fp32_t"], wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.skip_scale_wait = skip_scale_wait
    return inst


_bmm_minterleave_local = {
    162: _a8w8_mxscale_bmm_minterleave(128, 128, 128, 1, False),
    163: _a8w8_mxscale_bmm_minterleave(128, 128, 128, 1, True),
}


def _a8w8_mxscale_bmm_spec(tag, bm, bn, bk, wg_per_cu, **flags):
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256, bm, bn, bk, t_m, t_n, 16, 16, 128, 16, 16, 4,
        1, 128, 128, tag, ["fp32_t"], wg_per_cu,
        splitk_workspace_dtype=("fp32_t" if tag == "a8w8_mxscale_bmm_fused" else None),
    )
    inst.name_root = "opus_bmm"
    for key, value in flags.items():
        setattr(inst, key, value)
    return inst


_bmm_fused_local = {
    100: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_fused", 32, 128, 128, 2),
}


def _a8w8_mxscale_bmm_pipeline(**flags):
    inst = OpusGemmInstance(
        512, 256, 256, 128, 2, 1, 16, 16, 128, 16, 16, 4,
        1, 128, 128, "a8w8_mxscale_bmm_pipeline", ["fp32_t"], 1,
    )
    inst.name_root = "opus_bmm"
    for key, value in flags.items():
        setattr(inst, key, value)
    return inst


_bmm_pipeline_local = {
    149: _a8w8_mxscale_bmm_pipeline(B_M=128),
    150: _a8w8_mxscale_bmm_pipeline(),
    151: _a8w8_mxscale_bmm_pipeline(k1024_only=True),
    152: _a8w8_mxscale_bmm_pipeline(k1024_lb1=True),
    158: _a8w8_mxscale_bmm_pipeline(preload_sf_lds=True),
}
_bmm_mouter_local = {
    131: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter", 128, 128, 128, 1),
    144: _a8w8_mxscale_bmm_spec(
        "a8w8_mxscale_bmm_mouter", 128, 128, 128, 1, skip_scale_wait=True
    ),
}
_bmm_mouter_tunable_local = {
    160: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter_tunable", 128, 128, 128, 1),
    161: _a8w8_mxscale_bmm_spec(
        "a8w8_mxscale_bmm_mouter_tunable", 128, 128, 128, 1,
        skip_scale_wait=True,
    ),
}
_bmm_wave8n2_local = {
    132: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_wave8n2", 128, 128, 128, 1),
}
_BMM_WAVE4M2_TILES = {
    134: (False, False),
    142: (True, False),
    148: (True, True),
}
_bmm_wave4m2_local = {
    kid: _a8w8_mxscale_bmm_spec(
        "a8w8_mxscale_bmm_wave4m2_selfload", 128, 128, 128, 1,
        skip_scale_wait=skip, pack_scale_on_demand=pack,
    )
    for kid, (skip, pack) in _BMM_WAVE4M2_TILES.items()
}


def _globalize_bmm_kids(kernels):
    return {bmm_mxscale_global_kid(kid): inst for kid, inst in kernels.items()}


a8w8_mxscale_bmm_flatmm_splitk_kernels_list = _globalize_bmm_kids(_bmm_flatmm_local)
a8w8_mxscale_bmm_fused_kernels_list = _globalize_bmm_kids(_bmm_fused_local)
a8w8_mxscale_bmm_minterleave_kernels_list = _globalize_bmm_kids(_bmm_minterleave_local)
a8w8_mxscale_bmm_mouter_kernels_list = _globalize_bmm_kids(_bmm_mouter_local)
a8w8_mxscale_bmm_mouter_tunable_kernels_list = _globalize_bmm_kids(
    _bmm_mouter_tunable_local
)
a8w8_mxscale_bmm_pipeline_kernels_list = _globalize_bmm_kids(_bmm_pipeline_local)
a8w8_mxscale_bmm_wave8n2_kernels_list = _globalize_bmm_kids(_bmm_wave8n2_local)
a8w8_mxscale_bmm_wave4m2_selfload_kernels_list = _globalize_bmm_kids(
    _bmm_wave4m2_local
)
a8w8_mxscale_bmm_kernel_lists = (
    a8w8_mxscale_bmm_flatmm_splitk_kernels_list,
    a8w8_mxscale_bmm_fused_kernels_list,
    a8w8_mxscale_bmm_minterleave_kernels_list,
    a8w8_mxscale_bmm_mouter_kernels_list,
    a8w8_mxscale_bmm_mouter_tunable_kernels_list,
    a8w8_mxscale_bmm_pipeline_kernels_list,
    a8w8_mxscale_bmm_wave8n2_kernels_list,
    a8w8_mxscale_bmm_wave4m2_selfload_kernels_list,
)
BMM_MXSCALE_KIDS = frozenset(
    kid for family in a8w8_mxscale_bmm_kernel_lists for kid in family
)
assert len(BMM_MXSCALE_KIDS) == sum(map(len, a8w8_mxscale_bmm_kernel_lists))


a8w8_kernels_list = {
    2: OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0, "a8w8", ["fp32_t"]),
}

a16w16_kernels_list = {
    # -- MFMA 16x16x32, T_N=2, BS=256 (2-block/CU capable) --
    # 3:  _a16w16(256, 128, 128, 32,  2, 16, 16, 32),  # disabled: intermittent accuracy (suspected compiler issue with VGPR=104/AGPR=64)
    4:  _a16w16(256, 128, 256, 32,  2, 16, 16, 32),
    5:  _a16w16(256, 256, 128, 32,  2, 16, 16, 32),
    # -- MFMA 16x16x32, T_N=4, BS=512 (1-block/CU) --
    6:  _a16w16(512, 128, 128, 64,  4, 16, 16, 32),
    7:  _a16w16(512, 256, 128, 64,  4, 16, 16, 32),
    8:  _a16w16(512, 128, 256, 64,  4, 16, 16, 32),
    9:  _a16w16(512, 256, 256, 64,  4, 16, 16, 32),  # existing / current default
}

# Removed (kids 100-115, a16w16_flatmm non-splitk): Rationale: the non-splitk a16w16_flatmm
# pipeline has two latent correctness b...
a16w16_flatmm_kernels_list = {}

# 11 splitk tiles mirroring gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp_splitk.cc -t 0..10 dispatch
# exactly: * 8 WG_PER_CU=2 tiles (...
a16w16_flatmm_splitk_kernels_list = {
    # WG_PER_CU=2, cc tile 0..7
    200: _a16w16_flatmm_splitk( 64,  64,  64, 2),   # cc tile 0: M>=128 sweet spot (default)
    201: _a16w16_flatmm_splitk( 32,  32,  64, 2),   # cc tile 1
    202: _a16w16_flatmm_splitk( 32,  32, 128, 2),   # cc tile 2
    203: _a16w16_flatmm_splitk( 32,  64,  64, 2),   # cc tile 3
    204: _a16w16_flatmm_splitk( 32, 128,  64, 2),   # cc tile 4
    205: _a16w16_flatmm_splitk( 64,  32,  64, 2),   # cc tile 5
    206: _a16w16_flatmm_splitk( 64,  32, 128, 2),   # cc tile 6: recommended for medium M
    207: _a16w16_flatmm_splitk(128,  32,  64, 2),   # cc tile 7
    # WG_PER_CU=1, cc tile 8..10 (160 KB/wg LDS; zero VGPR spill only)
    208: _a16w16_flatmm_splitk( 64,  64, 128, 1),   # cc tile 8: deep K, high compute/load ratio
    209: _a16w16_flatmm_splitk(256,  32,  64, 1),   # cc tile 9: very tall, narrow N
    210: _a16w16_flatmm_splitk( 32, 256,  64, 1),   # cc tile 10: very wide, narrow M
    # Tile coverage extension (kids 211..223): B_M=96 OR B_N=96 lanes for shapes whose M or N is a
    # multiple of 96.
    211: _a16w16_flatmm_splitk( 32,  96,  64, 1),   # pfk=9, VGPR=176/512, AGPR=24
    212: _a16w16_flatmm_splitk( 32,  96,  64, 2),   # pfk=4, VGPR=176/256, AGPR=24
    213: _a16w16_flatmm_splitk( 32,  96, 128, 1),   # pfk=4, VGPR=288/512, AGPR=24
    214: _a16w16_flatmm_splitk( 64,  96,  64, 1),   # pfk=7, VGPR=192/512, AGPR=48
    215: _a16w16_flatmm_splitk( 64,  96,  64, 2),   # pfk=3, VGPR=192/256, AGPR=48
    216: _a16w16_flatmm_splitk( 64,  96, 128, 1),   # pfk=3, VGPR=320/512, AGPR=48
    217: _a16w16_flatmm_splitk( 96,  32,  64, 1),   # pfk=9, VGPR=144/512, AGPR=24
    218: _a16w16_flatmm_splitk( 96,  32,  64, 2),   # pfk=4, VGPR=144/256, AGPR=24
    219: _a16w16_flatmm_splitk( 96,  32, 128, 1),   # pfk=4, VGPR=224/512, AGPR=24
    220: _a16w16_flatmm_splitk( 96,  64,  64, 1),   # pfk=7, VGPR=176/512, AGPR=48
    221: _a16w16_flatmm_splitk( 96,  64,  64, 2),   # pfk=3, VGPR=176/256, AGPR=48
    222: _a16w16_flatmm_splitk( 96,  64, 128, 1),   # pfk=3, VGPR=288/512, AGPR=48
    223: _a16w16_flatmm_splitk( 96,  96,  64, 2),   # pfk=3, VGPR=208/256, AGPR=72  (81% VGPR -- watch)
}

# non-OOB variants: kid + 1000, same tile but HAS_OOB=false.
a16w16_kernels_list_nooob = {
    kid + 1000: _a16w16(
        inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
        inst.T_N, inst.W_M, inst.W_N, inst.W_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_kernels_list.items()
}

# CPOL variants for a16w16: 3 policies per kid, tuner picks best per shape.
_CACHECTL_CONFIGS = [
    (2000, 1, 17, "Mheavy"),   # kid_offset, cachectl_a, cachectl_b
    (3000, 17, 1, "Nheavy"),
    (4000, 0,  0, "balanced"),
]
a16w16_kernels_list_cpol = {}
for offset, ca, cb, _tag in _CACHECTL_CONFIGS:
    for kid, inst in a16w16_kernels_list.items():
        new_inst = _a16w16(
            inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
            inst.T_N, inst.W_M, inst.W_N, inst.W_K,
        )
        new_inst.cachectl_a = ca
        new_inst.cachectl_b = cb
        a16w16_kernels_list_cpol[kid + offset] = new_inst

a16w16_kernels_list_cpol_nooob = {}
for offset, ca, cb, _tag in _CACHECTL_CONFIGS:
    for kid, inst in a16w16_kernels_list.items():
        new_inst = _a16w16(
            inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
            inst.T_N, inst.W_M, inst.W_N, inst.W_K, has_oob=False,
        )
        new_inst.cachectl_a = ca
        new_inst.cachectl_b = cb
        a16w16_kernels_list_cpol_nooob[kid + offset + 1000] = new_inst

a16w16_flatmm_splitk_kernels_list_nooob = {
    kid + 1000: _a16w16_flatmm_splitk(
        inst.B_M, inst.B_N, inst.B_K, inst.WG_PER_CU, has_oob=False,
    )
    for kid, inst in a16w16_flatmm_splitk_kernels_list.items()
}

# -- a16w16 persistent (M-outer + N-fast XCD swizzle) ---------------------- Pipeline:
# csrc/opus_gemm/include/gfx950/opus_gemm_pi...


def _a16w16_persistent(bm, bn, bk, has_oob=True,
                       cachectl_a=0, cachectl_b=17):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    inst = OpusGemmInstance(
        512,         # BLOCK_SIZE
        bm, bn, bk,  # BLOCK
        2, 4,        # T_M, T_N
        16, 16, 32,  # W_M, W_N, W_K  (MFMA 16x16x32)
        vec, vec, 4, # VEC
        0, 0, 0,     # GROUP (unused for persistent)
        "a16w16_persistent",
        ["bf16_t", "fp32_t"],
        has_oob=has_oob,
    )
    inst.cachectl_a = cachectl_a
    inst.cachectl_b = cachectl_b
    return inst


# 4-tile sweep, all B_K=64.
_PERSISTENT_TILES = [
    # (B_M, B_N, B_K)
    (256, 256, 64),  # tile 0: mouter default; 32Kx2Kx7K best 1208 TFLOPS
    (128, 256, 64),  # tile 1: narrow M
    (256, 128, 64),  # tile 2: narrow N
    (128, 128, 64),  # tile 3: small
]

# Legacy (300..303): cachectl == (0, 17).
a16w16_persistent_kernels_list = {
    300 + i: _a16w16_persistent(bm, bn, bk)
    for i, (bm, bn, bk) in enumerate(_PERSISTENT_TILES)
}

# Cpol variants (304..315): 3 groups x 4 tiles, mirroring _CACHECTL_CONFIGS but with a single
# compact base offset per cpol group.
_PERSISTENT_CPOL_GROUPS = [
    # (base_kid, cachectl_a, cachectl_b)
    (304,  1, 17),   # Mheavy
    (308, 17,  1),   # Nheavy
    (312,  0,  0),   # balanced
]
a16w16_persistent_kernels_list_cpol = {}
for _base, _ca, _cb in _PERSISTENT_CPOL_GROUPS:
    for i, (bm, bn, bk) in enumerate(_PERSISTENT_TILES):
        a16w16_persistent_kernels_list_cpol[_base + i] = _a16w16_persistent(
            bm, bn, bk, cachectl_a=_ca, cachectl_b=_cb
        )

# Nooob mirrors at +1000 for both legacy (1300..1305) and cpol (1306..1323).
# Explicit cachectl inheritance keeps name() consistent with parents.
a16w16_persistent_kernels_list_nooob = {
    kid + 1000: _a16w16_persistent(
        inst.B_M, inst.B_N, inst.B_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_persistent_kernels_list.items()
}
a16w16_persistent_kernels_list_cpol_nooob = {
    kid + 1000: _a16w16_persistent(
        inst.B_M, inst.B_N, inst.B_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_persistent_kernels_list_cpol.items()
}

# -- a16w16 mono-tile (single-MMA-per-K-iter, 8 waves) ---------------------
#
# Pipeline:
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh
# Traits:
#   csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh
#   :: opus_gemm_a16w16_mono_tile_traits_gfx950
#
# Locks: BLOCK_SIZE=512, T_M=2, T_N=4, T_K=1, W_M=W_N=16, W_K=32 (MFMA
# 16x16x32 BF16), VEC=8. Single v_c accumulator over the full B_M x B_N
# tile per K iter (no quad-subtile, no split barrier). Intrinsically
# non-OOB (launcher enforces M%B_M==N%B_N==K%B_K==0) and HAS_BIAS=false
# (launcher rejects non-empty bias up front). No splitK.
#
# B_M <= 192 hard cap. The 7 tiles below were picked to cover
# (M-bucket x N-bucket) combinations not already served well by the
# persistent / splitk families.


def _a16w16_mono_tile(bm, bn, bk):
    vec = 16 // 2  # VEC_A = VEC_B = 8 for bf16
    return OpusGemmInstance(
        512,         # BLOCK_SIZE (8 waves * 64)
        bm, bn, bk,  # BLOCK
        2, 4,        # T_M, T_N
        16, 16, 32,  # W_M, W_N, W_K  (MFMA 16x16x32)
        vec, vec, vec,  # VEC_A=VEC_B=VEC_C=8
        0, 0, 0,     # GROUP (unused)
        "a16w16_mono_tile",
        ["bf16_t", "fp32_t"],
        has_oob=False,
    )


# 5 mono-tile tiles, kids 1400..1404. Kid range deliberately starts at
# 1400 (above the persistent +1000 nooob mirror range that ends at 1323)
# and below the next reserved family slot. No "base/nooob" mirror split:
# mono-tile is non-OOB by construction, so kids land in the >=1000 band
# the way other families' nooob mirrors do.
#
# B_K=128 tiles (e.g. (64,256,128), (128,128,128)) are intentionally
# excluded: the pipeline uses 2x smem_a + 3x smem_b (A double-buffered,
# B triple-buffered as r0/r1/w), which pushes those tiles to 165-231 KiB
# of LDS -- over gfx950's 160 KiB budget. Re-enable only after the
# pipeline drops B to two slots.
_MONO_TILE_TILES = [
    # (B_M, B_N, B_K)
    (192, 256, 64),   # 1400
    (128, 256, 64),   # 1401
    (192, 128, 64),   # 1402
    (128, 128, 64),   # 1403
    ( 64, 128, 64),   # 1404
]
a16w16_mono_tile_kernels_list = {
    1400 + i: _a16w16_mono_tile(bm, bn, bk)
    for i, (bm, bn, bk) in enumerate(_MONO_TILE_TILES)
}

# -- 4g_safe variants (offset +5000) ---------------------------------------
#
# Per-WG-tight buffer-resource sizing pipelines that handle tensors whose
# full extent exceeds 4 GiB without buffer_inst num_records wrap. Same
# Traits / kargs as their legacy siblings; only the pipeline header and
# kernel symbol differ. See
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_4g_safe_gfx950.cuh
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_4g_safe_gfx950.cuh
#   csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_4g_safe_gfx950.cuh
#
# Offset choice: +5000 sits above the cpol band (which uses +2000/+3000/+4000)
# and well clear of the nooob mirror band (+1000). 4g_safe kids carry HAS_OOB
# from their parent (M/N tail is absorbed by the per-WG BR num_records, so
# the per-thread predicate is structurally a no-op for valid in-tile threads;
# we still emit both has_oob variants for consistency with the legacy axis).
_FOUR_G_SAFE_OFFSET = 5000


def _make_4g_safe(inst: "OpusGemmInstance") -> "OpusGemmInstance":
    """Clone an OpusGemmInstance with is_4g_safe=True; everything else
    (kernel_tag, traits, kargs, BLOCK/B_*/T_*/W_*/VEC_*, cachectl, has_oob)
    is inherited verbatim. The codegen dispatch in gen_instances.py reads
    is_4g_safe to pick the 4g_safe pipeline header + kernel symbol."""
    from dataclasses import replace
    return replace(inst, is_4g_safe=True)


a16w16_kernels_list_4g_safe = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_kernels_list.items()
}
a16w16_kernels_list_4g_safe_nooob = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_kernels_list_nooob.items()
}
a16w16_persistent_kernels_list_4g_safe = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_persistent_kernels_list.items()
}
a16w16_persistent_kernels_list_4g_safe_nooob = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_persistent_kernels_list_nooob.items()
}
a16w16_mono_tile_kernels_list_4g_safe = {
    kid + _FOUR_G_SAFE_OFFSET: _make_4g_safe(inst)
    for kid, inst in a16w16_mono_tile_kernels_list.items()
}


# -- gfx942 kernel lists ------------------------------------------------ Kid offset: gfx942
GFX942_KID_OFFSET = 10000

# Split-K launch policy is consumed by both the Python Torch-workspace planner
# and the generated host launcher.  Keep it beside the canonical instances so
# the two sides cannot silently drift and disagree about workspace capacity.
GFX942_MAX_AUTO_SPLIT_K = 16
GFX942_MIN_ITERS_PER_SPLIT = 2
GFX942_QUAD_MFMA32_SPLITK_TAG = "a16w16_quad_mfma32_kbuf1_sk"
GFX942_EVEN_LOOP_SPLITK_TAGS = frozenset(
    {
        "a16w16_kbuf2v_sk",
        "a16w16_kbuf2v_bk128_sk",
        GFX942_QUAD_MFMA32_SPLITK_TAG,
    }
)

# gfx942 bf16-workspace launchers can use the exact-N row-block reducer only
# for these output widths.  Keep the *set* here beside the instance source so
# runtime selection, tuning, and codegen can consume one value.  The detailed
# (VEC, N_VEC, ROWS_PER_BLOCK) reduce configurations remain codegen-owned.
GFX942_BF16WS_EXACT_N = frozenset({64, 128, 256, 384, 512, 1024, 2048})


def _a16w16_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Factory for gfx942 a16w16 kbuf1-large-tile kid instances (kid 10000,
    MFMA 16x16x16). Same algorithm family as kbuf1 (4-phase, 2 barriers/iter)
    but with a larger tile + BS=512 + inline LDS-staged epilogue.
    """
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk,
        2, tn,            # T_M, T_N
        wm, wn, wk,       # MFMA
        vec, vec, 4,      # VEC
        0, 0, 0,          # GROUP (unused)
        "a16w16_kbuf1_large_tile",
        ["fp32_t", "bf16_t"],
        arch_prefix="gfx942",
    )


def _a16w16_quad_mfma32_gfx942(bs, bm, bn, bk, tm, tn, wm, wn, wk):
    """gfx942 quad MFMA32 path."""
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk,
        tm, tn,
        wm, wn, wk,
        vec, vec, 4,
        0, 0, 0,
        "a16w16_quad_mfma32_kbuf1",
        ["bf16_t"],
        arch_prefix="gfx942",
    )


def _a16w16_quad_mfma32_sk_bf16ws_gfx942(bs, bm, bn, bk, tm, tn, wm, wn, wk, group_m=0):
    """gfx942 quad MFMA32 splitK path with bf16 workspace."""
    vec = 16 // 2  # bf16
    inst = OpusGemmInstance(
        bs, bm, bn, bk,
        tm, tn,
        wm, wn, wk,
        vec, vec, 4,
        group_m, 0, 0,
        "a16w16_quad_mfma32_kbuf1_sk",
        ["fp32_t"],
        arch_prefix="gfx942",
    )
    inst.splitk_workspace_dtype = "bf16_t"
    return inst


def _a16w16_splitk_tag_gfx942(bs, bm, bn, bk, tn, wm, wn, wk, tag):
    """Factory for gfx942 splitK kids that write fp32 workspace + reduce."""
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk,
        2, tn,
        wm, wn, wk,
        vec, vec, 4,
        0, 0, 0,
        tag,
        ["fp32_t"],
        arch_prefix="gfx942",
        splitk_workspace_dtype="fp32_t",
    )


def _a16w16_kbuf1_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK 4-phase split-barrier, E_M>=2 OK."""
    return _a16w16_splitk_tag_gfx942(
        bs, bm, bn, bk, tn, wm, wn, wk, "a16w16_kbuf1_sk"
    )


def _with_bf16_splitk_workspace(inst, name_tag):
    """Variant marker: same splitK pipeline, bf16 workspace + generated name tag."""
    inst.name_tag = name_tag
    inst.splitk_workspace_dtype = "bf16_t"
    return inst


def _a16w16_kbuf1_sk_bf16ws_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK 4-phase split-barrier with bf16 workspace."""
    inst = _a16w16_kbuf1_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk)
    return _with_bf16_splitk_workspace(inst, "splitk_legacy_bf16ws")


def _a16w16_kbuf2v_bk128_sk_bf16ws_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 B_K=128 with bf16 workspace."""
    inst = _a16w16_kbuf2v_bk128_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk)
    return _with_bf16_splitk_workspace(inst, "splitk_p1_bk128_bf16ws")


# gfx942 P1-family non-splitK factories (siblings of corresponding splitK kids).
def _a16w16_p1_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Non-splitK P1 (K-dbuf depth=2 + V-dbuf), sibling of 10201."""
    vec = 16 // 2  # bf16
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_kbuf2v_bk128_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Non-splitK P1 + B_K=128 sub-K decomp, sibling of 10203."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v_bk128", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_kbuf2v_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 (K-dbuf depth=2 + V-dbuf), fp32 workspace + reduce."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v_sk", ["fp32_t"], arch_prefix="gfx942",
        splitk_workspace_dtype="fp32_t",
    )


def _a16w16_kbuf2v_bk128_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 + B_K=128 sub-K decomp."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v_bk128_sk", ["fp32_t"], arch_prefix="gfx942",
        splitk_workspace_dtype="fp32_t",
    )


def _a16w16_wave_k_coop_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Wave-K-cooperative small-M/N kid; tn partitions waves over N."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 1, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_wave_k_coop", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_wave_k_coop_accum_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """Wave-K-cooperative splitK atomic accumulate path."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 1, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_wave_k_coop_accum", ["bf16_t"], arch_prefix="gfx942",
    )


def _a16w16_em3en4_lds1_pgr2_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK EM3EN4: host 128x96, device 96x128 LDSB1."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_em3en4_lds1_pgr2_sk", ["fp32_t"], arch_prefix="gfx942",
        splitk_workspace_dtype="fp32_t",
    )


def _a8w8_blockscale_bpreshuffle_singlebuf_gfx942(
    bs,
    bm,
    bn,
    bk,
    tm,
    tn,
    wm,
    wn,
    wk,
    vec=8,
):
    """Single-K-buffer gfx942 A8W8 blockscale bpreshuffle tune path."""
    name_tag = "a8w8_bs_bpreshuf_sb_tailm_v16" if vec == 16 else "a8w8_bs_bpreshuf_sb_tailm"
    return OpusGemmInstance(
        bs, bm, bn, bk,
        tm, tn,
        wm, wn, wk,
        vec, vec, 4,
        1, 128, 128,
        "a8w8_blockscale_bpreshuffle_singlebuf",
        ["bf16_t"],
        name_tag=name_tag,
        arch_prefix="gfx942",
        has_oob=True,
    )


# gfx942 kid registry -- per-family flat maps.

gfx942_nosplit_kernels_list = {
    10000: _a16w16_gfx942        (512, 128, 128,  64,    4, 16, 16, 16),   # kbuf1_large_tile (4-phase, big tile)
    10001: _a16w16_p1_gfx942     (256,  64,  64,  64,    2, 16, 16, 16),   # P1 depth=2 sibling of 10201
    10003: _a16w16_kbuf2v_bk128_gfx942(256, 64,  64, 128,    2, 16, 16, 16),   # P1 B_K=128 sibling of 10203
    10006: _a16w16_quad_mfma32_gfx942(256, 256, 256, 32, 2, 2, 32, 32, 8), # quad MFMA32 pipeline
    10300: _a16w16_wave_k_coop_gfx942(512, 16, 16, 64,    1, 16, 16, 16),  # wave-K-coop 16x16, T_K=8
    10301: _a16w16_wave_k_coop_gfx942(512, 16, 32, 32,    1, 16, 16, 16),  # WKC 16x32, B_K=32
    10302: _a16w16_wave_k_coop_gfx942(512, 32, 16, 64,    1, 16, 16, 16),  # WKC 32x16, aliased partial
    10303: _a16w16_wave_k_coop_gfx942(256, 32, 32, 64,    1, 16, 16, 16),  # WKC 32x32, T_K=4
    10305: _a16w16_wave_k_coop_gfx942(512, 16, 32, 64,    1, 16, 16, 16),  # WKC 16x32, B_K=64
    10310: _a16w16_wave_k_coop_accum_gfx942(256, 16, 16, 64, 1, 16, 16, 16),  # WKC 16x16 split8 atomic accumulate
    10311: _a16w16_wave_k_coop_accum_gfx942(512, 16, 32, 32, 1, 16, 16, 16),  # WKC 16x32 split8 atomic accumulate
    10312: _a16w16_wave_k_coop_accum_gfx942(512, 32, 16, 64, 1, 16, 16, 16),  # WKC 32x16 split8 atomic accumulate
    10313: _a16w16_wave_k_coop_accum_gfx942(256, 32, 32, 64, 1, 16, 16, 16),  # WKC 32x32 split8 atomic accumulate
    10314: _a16w16_wave_k_coop_accum_gfx942(256, 64, 16, 64, 1, 16, 16, 16),  # WKC 64x16 split8 atomic accumulate
}

gfx942_splitk_kernels_list = {
    10200: _a16w16_kbuf1_sk_gfx942      (512, 128, 128,  64,    4, 16, 16, 16),                # legacy 4-phase large tile
    10201: _a16w16_kbuf2v_sk_gfx942     (256,  64,  64,  64,    2, 16, 16, 16),                # P1 depth=2 + V-dbuf
    10203: _a16w16_kbuf2v_bk128_sk_gfx942(256, 64,  64, 128,    2, 16, 16, 16),                # P1 B_K=128 sub-K decomp
    10204: _a16w16_em3en4_lds1_pgr2_sk_gfx942 (256, 128,  96, 128,    2, 16, 16, 16),                # EM3EN4 LDS1/PGR2 hipb-orientation (host 128M x 96N)
    10205: _a16w16_kbuf1_sk_gfx942      (512,  64, 128,  64,    4, 16, 16, 16),                # legacy 4-phase M64 x N128
    10210: _a16w16_kbuf1_sk_bf16ws_gfx942(512, 128, 128,  64,    4, 16, 16, 16),                # legacy 4-phase large tile + bf16 workspace
    10213: _a16w16_kbuf2v_bk128_sk_bf16ws_gfx942(256, 64,  64, 128,    2, 16, 16, 16),           # P1 B_K=128 + bf16 workspace
    10216: _a16w16_quad_mfma32_sk_bf16ws_gfx942(256, 256, 256, 32, 2, 2, 32, 32, 8, group_m=6),  # 10006 bf16 splitK sibling, group-M=6
}

gfx942_a8w8_kernels_list = {
    11000: _a8w8_blockscale_bpreshuffle_singlebuf_gfx942(
        512, 128, 128, 128, 4, 2, 16, 16, 32, vec=16),
}
# NOTE: 10402 (a16w16_naive_64x64) was removed -- 32.85us never matched WKC's
# 11.88us on tuned shapes (bf16_tuned_ge...

gfx942_kernels_list = {
    **gfx942_nosplit_kernels_list,
    **gfx942_splitk_kernels_list,
    **gfx942_a8w8_kernels_list,
}

# -- gfx1250 kernel lists ----------------------------------------------------
# Kid offset: gfx1250 kids live in the 20000+ range, disjoint from gfx950
# (<10000) and gfx942. Both #4246 families are represented: two-stage
# cluster/TDM kids and fused in-cluster-reduce kids.
GFX1250_KID_OFFSET = 20000


def _a16w16_cluster_tdm_splitk_ws_gfx1250(bm, bn, bk, layout, num_slots=3, wg_per_cu=2):
    """Factory for the gfx1250 a16w16 cluster/TDM split-K (workspace + reduce) kid.

    Locked geometry from the kernel base
    (demon_gcn/wmma_opus_rdna4/gemm_a16w16_cluster_tdm_splitk_reduce_4wave.cc):
    BLOCK_SIZE=128 (4 waves x 32 = 2 producer + 2 consumer), MFMA 16x16x32,
    NO-CLUSTER (one WG per B_M x B_N tile). The main kernel WMMA-accumulates in
    fp32 and casts each split's partial into the exact kid's typed workspace; a
    separate reduce kernel sums the split slices in fp32, folds bias, and casts
    to the Y dtype. The #4246 two-stage contract uses bf16 workspace. The
    output_dtypes = ["fp32_t"] token selects the existing host launch-dispatch
    specialization; Y bf16/fp32 remains a runtime decision in the reducer.

    layout: "tileN" (consumers split N; B_N>=32) -> T_M=1, T_N=2;
            "tileM" (consumers split M; B_M>=32) -> T_M=2, T_N=1.
    """
    vec = 16 // 2  # bf16 -> VEC_A = VEC_B = 8
    t_m, t_n = (2, 1) if layout == "tileM" else (1, 2)
    return OpusGemmInstance(
        128,            # BLOCK_SIZE (4 waves x 32 lanes)
        bm, bn, bk,
        t_m, t_n,       # T_M, T_N (encodes the consumer tiling layout)
        16, 16, 32,     # MFMA 16x16x32
        vec, vec, 8,    # VEC_A, VEC_B, VEC_C
        0, 0, 0,        # GROUP (unused)
        "a16w16_cluster_tdm_splitk_ws",
        ["fp32_t"],
        arch_prefix="gfx1250",
        splitk_workspace_dtype="bf16_t",
        ctdm_layout=layout,
        num_slots=num_slots,
        wg_per_cu=wg_per_cu,
    )


def _ctdm_pick_configs(bm, bn, bk):
    """Resource-feasible (P, wg_per_cu) configs for a gfx1250 cluster_tdm tile.

    Hardware prerequisites (gfx1250, per CU):
      * Direct-copy TDM budget: 256 256-byte requests per SIMD-pair (A and B sit
        on separate pairs). The per-TDM (one B_K slot) request count is
            req = rows * B_K * 2 / 256        (rows = B_M for A, B_N for B)
        2 WG/CU share a pair UNCONTROLLED -> each operand must be < 128; a single
        WG must be < 256. (req == 256 deadlocks the TDM engine -- the original
        32x256x128 hang.)
      * LDS: 320 KB / CU. LDS(P) = P * (B_M + B_N) * (B_K + 8) * 2 bytes.
        2 WG/CU need LDS(P) <= 160 KB; 1 WG/CU needs <= 320 KB.
      * VGPR (1024/SIMD, 512/wave at 2 WG/CU) is not the binding constraint for
        the current tiles and is left to the compiler.

    Returns a list of (num_slots P, wg_per_cu) for P in {3, 2}, picking the max
    feasible wg per P. Empty if the tile cannot run at any P (req >= 256).
    """
    rpr = bk // 128                       # 256B-req rows-multiplier (B_K/128)
    req_a = bm * rpr                       # per-TDM A request count
    req_b = bn * rpr                       # per-TDM B request count
    pitch = bk + 8                         # bf16 padded row pitch
    out = []
    # Prefetch depth P in {3, 2}: the run-ahead producer supports both (lower P
    # = lower LDS, can enable 2 WG/CU when P=3 LDS > 160 KB).
    for P in (3, 2):
        lds = P * (bm + bn) * pitch * 2
        if lds > 320 * 1024:
            continue                       # won't fit even 1 WG/CU
        if req_a < 128 and req_b < 128 and lds <= 160 * 1024:
            out.append((P, 2))             # 2 WG/CU safe
        elif req_a < 256 and req_b < 256:
            out.append((P, 1))             # force 1 WG/CU (LDS-pad in traits)
        # else: req >= 256 on some operand -> not runnable at this P
    return out


# Initial tile set seeded from the feasible no-cluster sweep
# (demon_gcn/wmma_opus_rdna4/instances_full_nocluster_feasible.csv), curated to
# the gfx1250 untuned shapes (small M / large N / large K).
#
# SCOPE (on-hardware validated, see op_tests/test_opus_gfx1250_ws.py -- 156/156):
# both the small-M tileN tiles and the fully generalized M/N tiles are wired:
#   * tileN: B_M==16, B_N>=32 (kExpN = B_N/32; N-wave-split + register-expand).
#   * tileM: B_M>=32 (kExpM = B_M/32) with any B_N (kExpN = B_N/16).
# Two earlier generalization bugs have been FIXED (2026-06):
#   (a) kExpM>1 && kExpN>1 -> NaN at the software-pipeline tail (per-split
#       k_steps%3==2): the sched_group_barrier DS/WMMA counts were hard-coded
#       for the kExpM==kExpN==1 base; now scaled by the register expansion in
#       the traits header (kSchedDsCount / kSchedWmmaCount).
#   (b) tileN with kExpN>1 (B_N>32, kTileN=2) -> wrong values: the B-read
#       N-decomposition order (make_layout_rb_ctdm) disagreed with the C-store
#       order; B now mirrors A (kExpN outer, kTileN=wave_n inner).
# Candidate tiles (B_M, B_N, B_K, layout). Each is expanded across its
# resource-feasible (P, wg_per_cu) configs by _ctdm_pick_configs(); tiles whose
# per-TDM request count hits the 256 direct-copy limit on some operand (e.g.
# 32x256x128, 32x128x256) yield no config and are dropped automatically.
_GFX1250_CTDM_TILES = [
    # -- ORIGINAL 11 tiles: KEEP THIS ORDER (indices 0..10) -- tuned CSVs and
    #    the Python heuristic reference the stable kids derived from these
    #    indices. Do NOT reorder/insert.
    # tileN family (B_M=16)
    (16, 32, 128, "tileN"),
    (16, 32, 256, "tileN"),
    (16, 32, 512, "tileN"),
    (16, 64, 128, "tileN"),
    (16, 128, 128, "tileN"),
    # tileM family (B_M>=32)
    (32, 32, 128, "tileM"),
    (32, 64, 128, "tileM"),
    (32, 128, 128, "tileM"),
    (32, 64, 256, "tileM"),
    (64, 16, 128, "tileM"),
    (64, 64, 128, "tileM"),
    # -- APPENDED (idx 11+): the remaining no-spill tiles from the offline
    #    LDS/VGPR sweep, so the plain (no-cluster) pipeline covers the same tile
    #    set as the clusterlaunch sweep. Layout: B_M==16 -> tileN else tileM.
    #    _ctdm_pick_configs() still drops any tile whose per-TDM direct-copy
    #    request hits the 256-request limit (those deadlock the no-cluster TDM
    #    engine -- they remain available only via the multicast clusterlaunch
    #    variant). New kids are 20088+ (idx*8), clear of the clusterlaunch band.
    (16, 64, 256, "tileN"),
    (16, 128, 256, "tileN"),
    (16, 256, 128, "tileN"),
    (32, 32, 256, "tileM"),
    (32, 128, 256, "tileM"),
    (32, 256, 128, "tileM"),
    (64, 32, 128, "tileM"),
    (64, 32, 256, "tileM"),
    (64, 64, 256, "tileM"),
    (64, 128, 128, "tileM"),
    (64, 128, 256, "tileM"),
    (64, 256, 128, "tileM"),
    (128, 32, 128, "tileM"),
    (128, 32, 256, "tileM"),
    (128, 64, 128, "tileM"),
    (128, 64, 256, "tileM"),
    (128, 128, 128, "tileM"),
]

# Kid numbering is stable for tuned CSVs and the Python heuristic:
#   plain (no-cluster) kids occupy [20000, 20100), ONE P=3 kid per tile (P=2 is
#   dropped -- unvalidated). Tiles the picker rejects (>=256-request TDM
#   direct-copy, now FIXED) fall back to P=3, 1 WG/CU so every no-spill tile still
#   emits a plain kid (LDS(P=3) <= 320 KB for this set).
#
# The consumer kExpN stability guard (previously _GFX1250_MAX_KEXPN=8) is removed.
gfx1250_kernels_list = {}
GFX1250_PLAIN_KID_OF = {}  # (B_M,B_N,B_K) -> kid (P=3; tuner + Python heuristic)
_GFX1250_KID_BASE = 20000
_p_kid = _GFX1250_KID_BASE
for _bm, _bn, _bk, _layout in _GFX1250_CTDM_TILES:
    # P=3 only (P=2 kids removed -- not validated); fall back to (P=3, 1 WG/CU)
    # for the high-request tiles the picker drops.
    _cfgs = [c for c in _ctdm_pick_configs(_bm, _bn, _bk) if c[0] == 3] or [(3, 1)]
    for _P, _wg in _cfgs:
        gfx1250_kernels_list[_p_kid] = _a16w16_cluster_tdm_splitk_ws_gfx1250(
            _bm, _bn, _bk, _layout, num_slots=_P, wg_per_cu=_wg
        )
        GFX1250_PLAIN_KID_OF[(_bm, _bn, _bk)] = _p_kid
        _p_kid += 1
assert _p_kid <= 20100, f"plain gfx1250 kids overflow the [20000,20100) band: {_p_kid}"

GFX1250_BASE_KIDS = frozenset(gfx1250_kernels_list.keys())


# -- gfx1250 CLUSTER-LAUNCH (multicast) variant ------------------------------
# Same 4-wave TDM split-K + typed workspace + reduce kernel, but launched as a
# (cluster_wg_m x cluster_wg_n x 1) workgroup CLUSTER: peers co-reside and share
# A/B TDM loads via CLUSTER_LOAD_ASYNC multicast (named-barrier producer/consumer
# handshake, same as the plain base). The host launcher rounds the grid up to the
# cluster dims; surplus workgroups take the pipeline's uniform tile_oob exit.
# Logical workspace strides remain based on the unrounded tile counts. Distinct kid
# band (20500+) so it never collides with the no-cluster base kids (20000..20087).
def _a16w16_clusterlaunch_tdm_splitk_ws_gfx1250(
    bm, bn, bk, layout, cwm, cwn, num_slots=3, wg_per_cu=2
):
    from dataclasses import replace

    inst = _a16w16_cluster_tdm_splitk_ws_gfx1250(
        bm, bn, bk, layout, num_slots=num_slots, wg_per_cu=wg_per_cu
    )
    return replace(
        inst,
        kernel_tag="a16w16_clusterlaunch_tdm_splitk_ws",
        cluster_wg_m=cwm,
        cluster_wg_n=cwn,
    )


# Full clusterlaunch sweep = {no-spill tile} x {valid cluster dim}.
#
# Tiles: the LDS/VGPR-no-spill (B_M, B_N, B_K, wg_per_cu) set from the offline
# sweep (all P=3). wg_per_cu per tile = 2 WG/CU when the P=3 LDS <= 160 KB AND
# VGPR <= 512 (both 2-WG-co-residency limits hold), else 1 WG/CU. Layout follows
# the base rule: B_M==16 -> tileN (consumers split N), B_M>=32 -> tileM.
#   #  B_M  B_N  B_K  LDS(KB)  VGPR  -> wg
#   (see the agent-provided table; wg derived as above)
_GFX1250_CLUSTERLAUNCH_TILES = [
    # B_M=16 (tileN)
    (16, 32, 128, 2), (16, 32, 256, 2), (16, 64, 128, 2), (16, 64, 256, 2),
    (16, 128, 128, 2), (16, 128, 256, 1), (16, 256, 128, 1),
    # B_M=32 (tileM)
    (32, 32, 128, 2), (32, 32, 256, 2), (32, 64, 128, 2), (32, 64, 256, 2),
    (32, 128, 128, 2), (32, 128, 256, 1), (32, 256, 128, 1),
    # B_M=64 (tileM)
    (64, 32, 128, 2), (64, 32, 256, 2), (64, 64, 128, 2), (64, 64, 256, 1),
    (64, 128, 128, 2), (64, 128, 256, 1), (64, 256, 128, 1),
    # B_M=128 (tileM)
    (128, 32, 128, 2), (128, 32, 256, 1), (128, 64, 128, 2), (128, 64, 256, 1),
    (128, 128, 128, 1),
]
_GFX1250_CLUSTERLAUNCH_P = 3            # all tiles use prefetch depth P=3
# Clusterlaunch kids occupy [20100, 21000) (plain uses [20000, 20100)).
_GFX1250_CLUSTERLAUNCH_KID_BASE = 20100
_GFX1250_MAX_MULTICAST_WG = 5          # TDM multicast fan-out limit (WGs per group)


def _gfx1250_valid_cluster_dims():
    """Valid (cwm, cwn) cluster dims, shared across all clusterlaunch tiles.

    Constraints:
      * cwm in 1..4, cwn in 1..5 (the requested sweep range).
      * Each side <= 5: TDM multicast fans out to at most 5 WGs (A shared by the
        cwn column peers, B by the cwm row peers).
      * cwm*cwn <= 16: the per-cluster workgroup_mask is 16-bit (also drops the
        cwn==5 & cwm==4 corner -> "cwn==5 => cwm<=3").
      * exclude (1,1): a degenerate 1-WG cluster has no multicast peers (self
        mask) and hangs at runtime.
    """
    dims = []
    for cwn in range(1, 6):       # cwn = 1..5
        for cwm in range(1, 5):   # cwm = 1..4
            if cwm == 1 and cwn == 1:
                continue
            if cwm > _GFX1250_MAX_MULTICAST_WG or cwn > _GFX1250_MAX_MULTICAST_WG:
                continue
            if cwm * cwn > 16:
                continue
            dims.append((cwm, cwn))
    return dims


# Deterministic kid numbering: 20500 + running index over (tile outer, then
# cluster dim (cwn outer, cwm inner)). Kid numbers are provisional -- a global
# renumber is pending. The kExpN stability guard has been removed, so ALL 26
# no-spill tiles are expanded (incl. B_N=256 tileM -> kExpN=16). The multicast
# clusterlaunch path is not bound by the no-cluster 256-request TDM limit, so no
# per-TDM request drop is applied here.
gfx1250_clusterlaunch_kernels_list = {}
GFX1250_CLUSTERLAUNCH_KID_OF = {}   # (B_M,B_N,B_K,cwm,cwn) -> kid (for tuner)
_cl_kid = _GFX1250_CLUSTERLAUNCH_KID_BASE
for _bm, _bn, _bk, _wg in _GFX1250_CLUSTERLAUNCH_TILES:
    _layout = "tileN" if _bm == 16 else "tileM"
    for _cwm, _cwn in _gfx1250_valid_cluster_dims():
        gfx1250_clusterlaunch_kernels_list[_cl_kid] = (
            _a16w16_clusterlaunch_tdm_splitk_ws_gfx1250(
                _bm, _bn, _bk, _layout, _cwm, _cwn,
                num_slots=_GFX1250_CLUSTERLAUNCH_P, wg_per_cu=_wg,
            )
        )
        GFX1250_CLUSTERLAUNCH_KID_OF[(_bm, _bn, _bk, _cwm, _cwn)] = _cl_kid
        _cl_kid += 1

assert _cl_kid <= 21000, f"clusterlaunch gfx1250 kids overflow [20100,21000): {_cl_kid}"
GFX1250_CLUSTERLAUNCH_KIDS = frozenset(gfx1250_clusterlaunch_kernels_list.keys())


# -- gfx1250 FUSED single-kernel split-K -------------------------------
# The first SplitK-1 WGs publish typed partial tiles to caller-owned storage;
# the last WG consumes those tiles after the cluster barrier and writes Y in
# the same kernel. There is no separate reduce launch, but this remains an
# external-workspace family. Workspace is tile-major:
#   [num_tiles_m, num_tiles_n, SplitK-1, B_M, B_N]
# SplitK and the N-peer count are compile-time properties of each exact kid.
#
# The final #4246 decision leaves this family unregistered until its pipeline
# is fixed. The factory, emitter, and device source remain available, but no kid
# is visible to exact dispatch/capability queries and [21000, 30000) is free.
def _a16w16_splitk_fuse_gfx1250(
    bm,
    bn,
    bk,
    layout,
    split_k,
    n_cluster,
    ws_dtype="bf16_t",
    num_slots=3,
    wg_per_cu=2,
):
    from dataclasses import replace

    inst = _a16w16_cluster_tdm_splitk_ws_gfx1250(
        bm, bn, bk, layout, num_slots=num_slots, wg_per_cu=wg_per_cu
    )
    return replace(
        inst,
        kernel_tag="a16w16_clusterlaunch_tdm_splitk_fuse",
        # The <fp32_t> host token selects the workspace dispatch ABI. The
        # fused launcher chooses the real bf16/fp32 Y type at runtime.
        output_dtypes=["fp32_t"],
        splitk_workspace_dtype=ws_dtype,
        fuse_split_k=split_k,
        # Historical #4246 field name; physically this is an N-peer count.
        fuse_m_cluster=n_cluster,
    )


GFX1250_SPLITK_FUSE_ENABLED = False

gfx1250_splitk_fuse_kernels_list = {}
GFX1250_SPLITK_FUSE_KID_BASE = 21000
GFX1250_SPLITK_FUSE_KID_OF = {}
_sf_kid = GFX1250_SPLITK_FUSE_KID_BASE

# The bounded LDS ring used by the fused reducer must fit inside the traits'
# shared allocation. These constants mirror the fused pipeline.
_FUSE_REDUCE_RING = 3
_FUSE_NUM_SLOTS = 3


def _fuse_ring_lds_ok(bm, bn, bk, wg, ws_bytes):
    pitch = bk + 8
    seg_ab = _FUSE_NUM_SLOTS * (bm + bn) * pitch * 2
    lds_total = (
        160 * 1024 + 1024 if wg == 1 and seg_ab <= 160 * 1024 else seg_ab
    )
    return _FUSE_REDUCE_RING * bm * bn * ws_bytes <= lds_total


# BF16 storage covers SplitK 2..15; the conservative FP32 family covers 2..8.
# Cluster dims are (SplitK, n_cluster, 1), so SplitK*n_cluster must fit the
# 16-WG cluster budget and each axis stays within its hardware limit.
_FUSE_WS_SWEEP = (("bf16_t", 2, 15), ("fp32_t", 4, 8))
_FUSE_MAX_NCLUSTER = 5
_fuse_tiles_seen = set()
_fuse_tiles = _GFX1250_CLUSTERLAUNCH_TILES if GFX1250_SPLITK_FUSE_ENABLED else ()
for _bm, _bn, _bk, _wg in _fuse_tiles:
    if (_bm, _bn, _bk) in _fuse_tiles_seen:
        continue
    _fuse_tiles_seen.add((_bm, _bn, _bk))
    _layout = "tileN" if _bm == 16 else "tileM"
    for _ws, _ws_bytes, _sk_hi in _FUSE_WS_SWEEP:
        if not _fuse_ring_lds_ok(_bm, _bn, _bk, _wg, _ws_bytes):
            continue
        for _nc in range(1, _FUSE_MAX_NCLUSTER + 1):
            for _sk in range(2, _sk_hi + 1):
                if _sk * _nc > 16:
                    continue
                gfx1250_splitk_fuse_kernels_list[_sf_kid] = (
                    _a16w16_splitk_fuse_gfx1250(
                        _bm,
                        _bn,
                        _bk,
                        _layout,
                        split_k=_sk,
                        n_cluster=_nc,
                        ws_dtype=_ws,
                        wg_per_cu=_wg,
                    )
                )
                GFX1250_SPLITK_FUSE_KID_OF[
                    (_bm, _bn, _bk, _layout, _sk, _nc, _ws)
                ] = _sf_kid
                _sf_kid += 1

assert _sf_kid <= 30000, (
    "splitk_fuse gfx1250 kids overflow [21000,30000): "
    f"ended at {_sf_kid - 1}"
)
GFX1250_SPLITK_FUSE_KIDS = frozenset(gfx1250_splitk_fuse_kernels_list.keys())
assert bool(GFX1250_SPLITK_FUSE_KIDS) == GFX1250_SPLITK_FUSE_ENABLED

# Flatten the eight BMM launcher tags into the same canonical exact-kid
# registry used by every other OPUS family.
a8w8_mxscale_bmm_kernels_list = {
    kid: instance
    for family in a8w8_mxscale_bmm_kernel_lists
    for kid, instance in family.items()
}

# combined list (used by production gen_instances / dispatch)
kernels_list = {
    **a8w8_scale_kernels_list,
    **a8w8_kernels_list,
    **a8w8_mxscale_bmm_kernels_list,
    **a16w16_kernels_list,
    **a16w16_kernels_list_nooob,
    **a16w16_kernels_list_cpol,
    **a16w16_kernels_list_cpol_nooob,
    **a16w16_flatmm_kernels_list,
    **a16w16_flatmm_splitk_kernels_list,
    **a16w16_flatmm_splitk_kernels_list_nooob,
    **a16w16_persistent_kernels_list,
    **a16w16_persistent_kernels_list_cpol,
    **a16w16_persistent_kernels_list_nooob,
    **a16w16_persistent_kernels_list_cpol_nooob,
    **a16w16_mono_tile_kernels_list,
    **a16w16_kernels_list_4g_safe,
    **a16w16_kernels_list_4g_safe_nooob,
    **a16w16_persistent_kernels_list_4g_safe,
    **a16w16_persistent_kernels_list_4g_safe_nooob,
    **a16w16_mono_tile_kernels_list_4g_safe,
    **gfx942_kernels_list,
    **gfx1250_kernels_list,
    **gfx1250_clusterlaunch_kernels_list,
    **gfx1250_splitk_fuse_kernels_list,
}

# fmt: on


# Subset-compile kid taxonomy consumed by gen_instances.py.

# Splitk kids: a16w16_flatmm_splitk pipeline (kid 200..223 + nooob mirror).
SPLITK_KIDS = (
    frozenset(a16w16_flatmm_splitk_kernels_list.keys())
    | frozenset(a16w16_flatmm_splitk_kernels_list_nooob.keys())
    | frozenset(gfx942_splitk_kernels_list.keys())
    | frozenset(gfx1250_kernels_list.keys())
    | frozenset(gfx1250_clusterlaunch_kernels_list.keys())
    | frozenset(gfx1250_splitk_fuse_kernels_list.keys())
)

BMM_MXSCALE_WORKSPACE_TAGS = frozenset(
    {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
    }
)
BMM_MXSCALE_WORKSPACE_KIDS = frozenset(
    kid
    for kid, instance in a8w8_mxscale_bmm_kernels_list.items()
    if instance.kernel_tag in BMM_MXSCALE_WORKSPACE_TAGS and not instance.direct_only
)

_SUPPORTED_SPLITK_WORKSPACE_DTYPES = frozenset({"bf16_t", "fp32_t"})
for _workspace_kid in SPLITK_KIDS:
    _workspace_dtype = kernels_list[_workspace_kid].splitk_workspace_dtype
    if _workspace_dtype not in _SUPPORTED_SPLITK_WORKSPACE_DTYPES:
        raise ValueError(
            f"workspace kid {_workspace_kid} must explicitly declare "
            f"splitk_workspace_dtype, got {_workspace_dtype!r}"
        )

# Non-splitk a16w16-family kids: split-barrier 4..9 + cpol/nooob mirrors, persistent 300..315 +
# cpol/nooob mirrors.
NON_SPLITK_KIDS = (
    frozenset(a16w16_kernels_list.keys())
    | frozenset(a16w16_kernels_list_nooob.keys())
    | frozenset(a16w16_kernels_list_cpol.keys())
    | frozenset(a16w16_kernels_list_cpol_nooob.keys())
    | frozenset(a16w16_persistent_kernels_list.keys())
    | frozenset(a16w16_persistent_kernels_list_cpol.keys())
    | frozenset(a16w16_persistent_kernels_list_nooob.keys())
    | frozenset(a16w16_persistent_kernels_list_cpol_nooob.keys())
    | frozenset(a16w16_mono_tile_kernels_list.keys())
    | frozenset(gfx942_nosplit_kernels_list.keys())
)

# 4g_safe kid families. Per-WG-tight BR sizing -- selectable for any shape
# (M/N/K tail safe by BR num_records). All current 4g_safe kids are non-splitk
# (split-barrier / persistent / mono_tile variants). flatmm_splitk_4g_safe
# can be added later if needed.
SPLITK_4G_SAFE_KIDS = frozenset()
NON_SPLITK_4G_SAFE_KIDS = (
    frozenset(a16w16_kernels_list_4g_safe.keys())
    | frozenset(a16w16_kernels_list_4g_safe_nooob.keys())
    | frozenset(a16w16_persistent_kernels_list_4g_safe.keys())
    | frozenset(a16w16_persistent_kernels_list_4g_safe_nooob.keys())
    | frozenset(a16w16_mono_tile_kernels_list_4g_safe.keys())
)
# Per the opus kid pruning policy (project memory), 4g_safe kids are added
# additively -- they do NOT shadow or replace any existing kid.
NON_SPLITK_KIDS = NON_SPLITK_KIDS | NON_SPLITK_4G_SAFE_KIDS

# All-4g_safe-kids superset, consumed by the per-kid 4 GiB filter in
# opus_gemm_tune.py (legacy kids are dropped from the candidate pool when
# A/B/C bytes exceed UINT32_MAX; 4g_safe kids stay).
FOUR_G_SAFE_KIDS = SPLITK_4G_SAFE_KIDS | NON_SPLITK_4G_SAFE_KIDS

# Bias-aware kids: gfx950 split-barrier (4..9 + cpol/nooob mirrors), 4g_safe
# mirrors, and the entire splitk family (gfx950 a16w16_flatmm_splitk + gfx942
# splitk). Persistent excluded (launcher rejects bias).
BIAS_AWARE_KIDS = (
    frozenset(a16w16_kernels_list.keys())
    | frozenset(a16w16_kernels_list_nooob.keys())
    | frozenset(a16w16_kernels_list_cpol.keys())
    | frozenset(a16w16_kernels_list_cpol_nooob.keys())
    | frozenset(a16w16_kernels_list_4g_safe.keys())
    | frozenset(a16w16_kernels_list_4g_safe_nooob.keys())
    | SPLITK_KIDS
)

# Exact-id kernels kept in every default build.  The high-level A16 caller-side
# heuristics below the tuned lookup are constrained to these ids; the unified
# public/C++ launch path still receives one already-resolved exact kid.
DEFAULT_COMPILED_KIDS_GFX950 = frozenset(
    {
        # splitk fallback (small M / non-aligned big M)
        200,
        1200,  # cc tile 0: (64, 64, 64) WG=2
        206,
        1206,  # cc tile 6: (64, 32, 128) WG=2
        208,
        1208,  # cc tile 8: (64, 64, 128) WG=1
        # persistent fallback (large M, tile-aligned)
        300,
        1300,  # persistent (256, 256, 64)
    }
)

DEFAULT_COMPILED_KIDS_GFX942 = frozenset(
    {
        # Representative exact-id launchers kept in default gfx942 builds.
        10000,  # gfx942 split-barrier    512x128x128x64 16x16x16 (large problem)
        10001,  # gfx942 p1               256x64x64x64
        10003,  # gfx942 p1_bk128         256x64x64x128
        10200,  # gfx942 splitk          512x128x128x64 16x16x16 (N > 128)
        10201,  # gfx942 splitk_p1        256x64x64x64  (depth=2 + workspace + reduce)
        10203,  # gfx942 splitk_p1_bk128  256x64x64x128 (B_K=128 Option B; dev/bench)
        10204,  # gfx942 splitk_em3en4_lds1_pgr2 256x128x96x128 hipb-orientation
        10205,  # gfx942 splitk_legacy    512x64x128x64 16x16x16
        10210,  # gfx942 splitk_legacy_bf16ws 512x128x128x64
        10213,  # gfx942 splitk_p1_bk128_bf16ws 256x64x64x128
        10300,  # gfx942 wave_k_coop     512x16x16x64 T_K=8
        10301,  # gfx942 wave_k_coop     512x16x32x32 T_K=8
        10302,  # gfx942 wave_k_coop     512x32x16x64 T_K=8
        10303,  # gfx942 wave_k_coop     256x32x32x64 T_K=4
        10305,  # gfx942 wave_k_coop     512x16x32x64 T_K=8
    }
)

# Keep six representative gfx1250 exact-id launchers in default builds. Every
# other plain kid and all clusterlaunch kids are compiled on demand by the
# tuner (candidate selection + sidecar expansion), so default builds stay
# small.
DEFAULT_COMPILED_KIDS_GFX1250 = frozenset(
    GFX1250_PLAIN_KID_OF[_t]
    for _t in (
        (16, 32, 128),
        (16, 64, 128),
        (16, 128, 128),
        (32, 32, 128),
        (32, 64, 128),
        (32, 128, 128),
    )
)

DEFAULT_COMPILED_KIDS = (
    DEFAULT_COMPILED_KIDS_GFX950
    | DEFAULT_COMPILED_KIDS_GFX942
    | DEFAULT_COMPILED_KIDS_GFX1250
)

DEFAULT_COMPILED_KIDS_BY_ARCH = {
    "gfx950": DEFAULT_COMPILED_KIDS_GFX950,
    "gfx942": DEFAULT_COMPILED_KIDS_GFX942,
    "gfx1250": DEFAULT_COMPILED_KIDS_GFX1250,
}


def default_compiled_kids_for_arch(arches):
    """Return the default exact-id compile floor for requested arches."""
    if arches is None:
        return DEFAULT_COMPILED_KIDS
    arches = {a.lower() for a in arches}
    out = frozenset()
    for arch in arches:
        out = out | DEFAULT_COMPILED_KIDS_BY_ARCH.get(arch, frozenset())
    return out


# Map each architecture and interface to its registered kernel tags.
# An empty set means that the interface exists but has no kernel on that arch.
OPUS_KERNEL_TAGS_BY_ARCH_FAMILY = {
    "gfx950": {
        "a16w16": frozenset(
            {
                "a16w16",
                "a16w16_flatmm",
                "a16w16_flatmm_splitk",
                "a16w16_mono_tile",
                "a16w16_persistent",
            }
        ),
        "a8w8": frozenset({"a8w8"}),
        "a8w8_blockscale": frozenset({"a8w8_scale"}),
        "a8w8_mxscale_bmm": frozenset(
            {
                "a8w8_mxscale_bmm_flatmm_splitk",
                "a8w8_mxscale_bmm_fused",
                "a8w8_mxscale_bmm_minterleave",
                "a8w8_mxscale_bmm_mouter",
                "a8w8_mxscale_bmm_mouter_tunable",
                "a8w8_mxscale_bmm_pipeline",
                "a8w8_mxscale_bmm_wave8n2",
                "a8w8_mxscale_bmm_wave4m2_selfload",
            }
        ),
        "a8w8_blockscale_bpreshuffle": frozenset(),
    },
    "gfx942": {
        "a16w16": frozenset(
            {
                "a16w16_em3en4_lds1_pgr2_sk",
                "a16w16_kbuf1_large_tile",
                "a16w16_kbuf1_sk",
                "a16w16_kbuf2v",
                "a16w16_kbuf2v_bk128",
                "a16w16_kbuf2v_bk128_sk",
                "a16w16_kbuf2v_sk",
                "a16w16_quad_mfma32_kbuf1",
                "a16w16_quad_mfma32_kbuf1_sk",
                "a16w16_wave_k_coop",
                "a16w16_wave_k_coop_accum",
            }
        ),
        "a8w8": frozenset(),
        "a8w8_blockscale": frozenset(),
        "a8w8_mxscale_bmm": frozenset(),
        "a8w8_blockscale_bpreshuffle": frozenset(
            {"a8w8_blockscale_bpreshuffle_singlebuf"}
        ),
    },
    "gfx1250": {
        "a16w16": frozenset(
            {
                "a16w16_cluster_tdm_splitk_ws",
                "a16w16_clusterlaunch_tdm_splitk_fuse",
                "a16w16_clusterlaunch_tdm_splitk_ws",
            }
        ),
        "a8w8": frozenset(),
        "a8w8_blockscale": frozenset(),
        "a8w8_mxscale_bmm": frozenset(),
        "a8w8_blockscale_bpreshuffle": frozenset(),
    },
}

# Always include these A8W8 kernels in matching-architecture subset builds.
OPUS_MANDATORY_A8_KIDS = {
    "gfx950": frozenset({1, 2}),
    "gfx942": frozenset({11000}),
    "gfx1250": frozenset(),
}


def canonical_output_dtype(output_dtype) -> str | None:
    """Normalize supported output dtype names for registry lookup."""
    if output_dtype is None:
        return None
    value = str(output_dtype).strip().lower()
    aliases = {
        "bf16": "bf16_t",
        "bfloat16": "bf16_t",
        "bf16_t": "bf16_t",
        "torch.bfloat16": "bf16_t",
        "fp32": "fp32_t",
        "float": "fp32_t",
        "float32": "fp32_t",
        "fp32_t": "fp32_t",
        "torch.float32": "fp32_t",
    }
    return aliases.get(value, value)


def get_kernel_instance(
    arch: str,
    family: str,
    kid: int,
    output_dtype=None,
) -> OpusGemmInstance | None:
    """Return a kernel registered for ``(arch, interface, kid, Y.dtype)``."""
    arch = str(arch).lower()
    family = str(family).lower()
    family_tags = OPUS_KERNEL_TAGS_BY_ARCH_FAMILY.get(arch, {}).get(family)
    if family_tags is None:
        return None

    try:
        kid = int(kid)
    except (TypeError, ValueError):
        return None

    instance = kernels_list.get(kid)
    if instance is None or instance.kernel_tag not in family_tags:
        return None
    instance_arch = (instance.arch_prefix or "gfx950").lower()
    if instance_arch != arch:
        return None

    dtype = canonical_output_dtype(output_dtype)
    if dtype is not None:
        # Workspace reducers own the final cast, so their host dispatch
        # specialization in ``output_dtypes`` is not the Y dtype contract.
        if family == "a16w16" and kid in SPLITK_KIDS:
            # The current gfx942 BF16-workspace reducer is exact-N and writes
            # BF16 only.  Other A16 workspace reducers support BF16/FP32 Y.
            allowed = (
                {"bf16_t"}
                if arch == "gfx942"
                and instance.splitk_workspace_dtype == "bf16_t"
                else {"bf16_t", "fp32_t"}
            )
            output_compatible = dtype in allowed
        elif family == "a8w8_mxscale_bmm":
            output_compatible = dtype in {"bf16_t", "fp32_t"}
        else:
            output_compatible = dtype in instance.output_dtypes
        if not output_compatible:
            return None
    return instance


def kernel_needs_external_workspace(arch: str, family: str, kid: int) -> bool:
    """Return whether a registered kernel requires caller-owned workspace.

    Unknown logical keys are errors rather than ``False``: treating an unknown
    kid as a non-workspace kernel would let a caller launch it without the
    allocation required for memory safety.  Capability comes from the existing
    ``SPLITK_KIDS`` registry, never from a numeric kid range or tag substring.
    The registry includes all enabled two-stage reducers. If the experimental
    gfx1250 fused family is re-enabled, its first SplitK-1 WGs also publish
    external partial tiles and therefore enter this same capability set.
    """
    instance = get_kernel_instance(arch, family, kid)
    if instance is None:
        raise KeyError(
            f"unknown OPUS kernel (arch={arch!r}, family={family!r}, kid={kid!r})"
        )
    return int(kid) in SPLITK_KIDS


def _opus_sidecar_path():
    """Return the on-disk path of the subset-compile sidecar.

    Lives in ``{bd_dir}/`` (one level above the per-module build dir) so
    it survives ``aiter.jit.core.clear_build("module_deepgemm_opus")`` --
    which ``build_module()`` calls when ``AITER_REBUILD == 1`` -- and is
    therefore the canonical "what kids should be in the next .so" source
    that ``gen_instances.py`` consumes. The tuner expands this sidecar
    BEFORE triggering the rebuild; if it lived inside the build dir,
    clear_build would wipe it out before gen_instances could read it.
    """
    # Import lazily to avoid circular import at module load (aiter imports
    # opus_gemm_common, opus_gemm_common imports aiter.jit.core).
    from aiter.jit.core import bd_dir

    return os.path.join(bd_dir, "compiled_kids_opus.json")
