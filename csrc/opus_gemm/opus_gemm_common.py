# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS kernel registrations shared by selection and code generation."""

import os
import sys
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation

_A16W16_CO_TAGS = frozenset(
    {"a16w16_4wave_co", "a16w16_4wave_wl_co", "a16w16_4wave_wlr_co"}
)

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


_SF_SHUF_SUB_HEADER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "include",
    "gfx950",
    "opus_gemm_traits_a8w8_scale_gfx950.cuh",
)
_SF_SHUF_SUB_CACHE: list[int] = []


def _opus_sf_shuf_sub() -> int:
    """``OPUS_SF_SHUF_SUB_VALUE``, parsed from the traits header.

    One source of truth for the A-scale layout's ``sub``. The host picks which
    ``shuffle_scale_a(x, K, sub)`` to feed a kid and the kernel decides how to
    read it; if those two numbers disagree the kid does not fail, it returns
    wrong numbers -- so this is parsed rather than restated. Raises if the
    #define is absent, because a default here would resurrect exactly the
    silent divergence it exists to prevent.
    """
    if not _SF_SHUF_SUB_CACHE:
        import re

        with open(_SF_SHUF_SUB_HEADER) as f:
            m = re.search(
                r"^#define\s+OPUS_SF_SHUF_SUB_VALUE\s+(\d+)\s*$",
                f.read(),
                re.MULTILINE,
            )
        if not m:
            raise RuntimeError(
                f"OPUS_SF_SHUF_SUB_VALUE not found in {_SF_SHUF_SUB_HEADER}; "
                "the A-scale layout is undefined and any shuffled kid handed a "
                "guessed sub would return wrong numbers silently"
            )
        _SF_SHUF_SUB_CACHE.append(int(m.group(1)))
    return _SF_SHUF_SUB_CACHE[0]


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
    # canonical local kid registry, but their generated symbols share the
    # ``opus_bmm`` root and a uniform exact-kid launcher signature.
    direct_only: bool = False
    prefetch_scale: bool = False
    fused_reduce: bool = False
    preload_sf: bool = False
    # a8w8_mxscale BMM wave8 families only: the A scale panel arrives M-packed
    # from the host (shuffle_scale_mxsk_mpack) and is read straight from global,
    # so there is no panel to stage. Maps to the wave8 kernel's trailing
    # `bool SFA_MPACK_GLOBAL`; see needs_mpacked_sfa for the caller's side of it.
    mpack_sfa: bool = False
    # a8w8_mxscale BMM wave8 families and flatmm-splitK: both scale panels arrive
    # in the reference kernel's layout (shuffle_scale_a / _b) and are read from
    # global one dword per (M subtile pair, K tile pair). Maps to the trailing
    # `bool SHUFFLE_SCALE` of either kernel; see needs_shuffle_scale for the
    # caller's side. Mutually exclusive with preload_sf, which it replaces.
    shuffle_scale: bool = False
    # a8w8_mxscale BMM wave8 families and flatmm-splitK: stage the shuffled scale
    # words through the LDS panel instead of reading them from global on every K
    # tile. Requires shuffle_scale. Maps to the kernel's trailing
    # `bool SF_SHUF_IN_LDS`; see the _sfshuf_lds name suffix.
    # The panel does not fit every tile and the kernel static_asserts rather than
    # degrading, so an ill-fitting kid is a build error rather than a mislabelled
    # reg build.
    sf_shuf_in_lds: bool = False
    # a8w8_mxscale BMM wave8 families only: band height in M tiles for the L2
    # rasterization of the workgroup -> tile map, 0 for the plain linear map. Maps
    # to the wave8 kernel's trailing `int XCD_WGM`.
    xcd_wgm: int = 0
    # a8w8_mxscale BMM wave1 only: register ring depth in K tiles, 0 for the
    # traits' VGPR-budget default. Maps to the wave1 traits' trailing `int RING_`.
    wave1_ring: int = 0
    # Compact BMM: issue future B before current LDS reads, then issue A.
    early_b: bool = False
    # a8w8_mxscale BMM specialized-pipeline axis (minterleave / mouter /
    # mouter_tunable / wave4m2_selfload families). Maps to the kernel's trailing
    # `bool SKIP_SCALE_WAIT` template param: skip the s_waitcnt on the per-K-tile
    # scale load (the scale is issued a tile ahead), trading a correctness margin
    # for pipeline overlap. Drives both the launcher body and the device
    # instantiation set for the kid.
    # Optional override for the D_OUT=void split-K specialization.  None keeps
    # the direct-output specialization's preload_sf setting; False lets an
    # exact kid retain its tuned splitK=1 preload while using the equivalent
    # lower-register-pressure path for FP32 workspace partials.
    workspace_preload_sf: bool | None = None
    skip_scale_wait: bool = False
    pack_scale_on_demand: bool = False
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

    # Pre-compiled (.co) family metadata. These fields participate in the
    # stable symbol/image name and are otherwise consumed only by gfx1250
    # CO code generation.
    co_num_vgpr: int = 0
    co_min_waves_per_eu: int = 1
    co_device_flags: tuple = ()
    co_variant: str = ""
    co_dtypes: tuple = ("bf16_t", "bf16_t", "bf16_t", "fp32_t")
    co_wave_layout: tuple = (4, 1)

    # Optional logical-M limit imposed by this exact kernel's launch geometry.
    max_m: int | None = None

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
            # opus_bmm_a8w8_mxscale_flatmm_splitk_<geom>_wgpcu{N}[_selfload]
            #     [_scaleprefetch][_sfpreload][_sfshuf]
            parts.insert(tag_at, "a8w8_mxscale_flatmm_splitk")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.direct_only:
                parts.append("selfload")
            if self.prefetch_scale:
                parts.append("scaleprefetch")
            if self.preload_sf:
                parts.append("sfpreload")
            # Same trap as the bdirect branch below, and this one was live: the
            # flatmm split-K kernel has implemented SHUFFLE_SCALE all along and the
            # codegen spells it on every kid of that kernel, but no suffix here
            # meant a shuffle_scale kid would have deduplicated onto its plain
            # sibling and been emitted as the plain one -- measuring as "the layout
            # makes no difference" with nothing raised.
            if self.shuffle_scale:
                parts.append("sfshuf")
        elif self.kernel_tag in (
            "a8w8_mxscale_bmm_bpreshuffle_bdirect",
            "a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen",
        ):
            # opus_bmm_a8w8_mxscale_bpreshuffle_bdirect[_tilen]_<geom>_wgpcu{N}
            #     [_scaleprefetch][_sfpreload]
            # The two flag suffixes matter: instances are deduplicated by name, so
            # a tile that differs only by a bool would otherwise silently collapse
            # onto its plain sibling and never be emitted.
            #
            # tilen shows in the T_MxT_N geom field as well (1x2 vs 2x1), so the
            # tag token is belt and braces -- but the geom field is name-only and
            # supplied by the factory, and this is the one that tracks the traits
            # struct actually instantiated.
            parts.insert(
                tag_at,
                (
                    "a8w8_mxscale_bpreshuffle_bdirect_tilen"
                    if self.kernel_tag.endswith("_tilen")
                    else "a8w8_mxscale_bpreshuffle_bdirect"
                ),
            )
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.prefetch_scale:
                parts.append("scaleprefetch")
            if self.preload_sf:
                parts.append("sfpreload")
            if self.shuffle_scale:
                # "sfshuf_lds" rather than a separate token, so the reg/lds pair of
                # one kid differs in exactly this suffix and nothing else. Same
                # rule as the wave8 families below -- and the same reason as the
                # dedup note above: without it the panel kid hashes to the reg
                # kid's filename, is never emitted, and the tuner ranks one kernel
                # against itself while reporting that the panel changed nothing.
                parts.append("sfshuf_lds" if self.sf_shuf_in_lds else "sfshuf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_allwave":
            # opus_bmm_a8w8_mxscale_bpreshuffle_allwave_<geom>_wgpcu{N}_sfpreload
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_allwave")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.preload_sf:
                parts.append("sfpreload")
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_blds":
            # opus_bmm_a8w8_mxscale_bpreshuffle_blds_<geom>_wgpcu{N}
            #     [_scaleprefetch][_sfpreload]
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_blds")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.prefetch_scale:
                parts.append("scaleprefetch")
            if self.preload_sf:
                parts.append("sfpreload")
            if self.shuffle_scale:
                parts.append("sfshuf_lds" if self.sf_shuf_in_lds else "sfshuf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_wave8n4":
            # opus_bmm_a8w8_mxscale_bpreshuffle_wave8n4_<geom>_wgpcu{N}_sfpreload
            #     [_xcd{N}][_sfgmpack][_sfshuf]
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_wave8n4")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.preload_sf:
                parts.append("sfpreload")
            if self.xcd_wgm:
                parts.append(f"xcd{self.xcd_wgm}")
            if self.mpack_sfa:
                parts.append("sfgmpack")
            if self.shuffle_scale:
                # "sfshuf_lds" rather than a separate token, so the reg/lds pair of
                # one kid differs in exactly this suffix and nothing else.
                parts.append("sfshuf_lds" if self.sf_shuf_in_lds else "sfshuf")
        elif self.kernel_tag in (
            "a8w8_mxscale_bmm_bpreshuffle_wavetm1",
            "a8w8_mxscale_bmm_bpreshuffle_wavetm1_blds",
        ):
            # opus_bmm_a8w8_mxscale_bpreshuffle_wavetm1[_blds]_<geom>_wgpcu{N}_sfpreload
            #     [_xcd{N}][_sfgmpack]
            parts.insert(
                tag_at, self.kernel_tag.replace("a8w8_mxscale_bmm_", "a8w8_mxscale_")
            )
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.preload_sf:
                parts.append("sfpreload")
            if self.xcd_wgm:
                parts.append(f"xcd{self.xcd_wgm}")
            if self.mpack_sfa:
                parts.append("sfgmpack")
            if self.shuffle_scale:
                parts.append("sfshuf_lds" if self.sf_shuf_in_lds else "sfshuf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_compact":
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_compact")
            parts.append(f"buf{self.num_slots}")
            if self.early_b:
                parts.append("earlyb")
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_wave1":
            # opus_bmm_a8w8_mxscale_bpreshuffle_wave1_<geom>_wgpcu{N}
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_wave1")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.wave1_ring:
                parts.append(f"ring{self.wave1_ring}")
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
            if self.preload_sf_lds:
                parts.append("preload_sf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_pipeline_bpreshuffle":
            # opus_bmm_a8w8_mxscale_pipeline_bpreshuffle_<geom>[_preload_sf].
            # Without this branch the tag fell through to the bare default, so
            # kid31 was named opus_bmm_<geom> alone: it advertised neither the
            # preshuffled B its caller must pass nor the scale preload, on a
            # codegen that deduplicates instances by name.
            parts.insert(tag_at, "a8w8_mxscale_pipeline_bpreshuffle")
            if self.preload_sf_lds:
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
                "wsf32" if self.splitk_workspace_dtype == "fp32_t" else "wsbf16"
            )
            parts.append(f"p{self.num_slots}w{self.wg_per_cu}")
        elif self.kernel_tag in _A16W16_CO_TAGS:
            # The host launcher symbol, ELF entry point and .co filename must
            # remain identical. Encode every device configuration axis that
            # can distinguish two pre-built images in this stable name.
            parts.insert(
                tag_at,
                {
                    "a16w16_4wave_wl_co": "4wave_wl_co",
                    "a16w16_4wave_wlr_co": "4wave_wlr_co",
                }.get(self.kernel_tag, "4wave_co"),
            )
            # Wave layout only shows up for the families that can vary it, so the
            # 4wave_co names already on disk are untouched.
            if self.kernel_tag in ("a16w16_4wave_wl_co", "a16w16_4wave_wlr_co"):
                parts.append(f"w{self.co_wave_layout[0]}x{self.co_wave_layout[1]}")
            parts.append(f"c{self.cluster_wg_m}x{self.cluster_wg_n}")
            parts.append(f"p{self.num_slots}")
            parts.append(f"v{self.co_num_vgpr}w{self.co_min_waves_per_eu}")
            if self.co_variant:
                parts.append(self.co_variant)
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

    @property
    def needs_preshuffled_b(self) -> bool:
        """Whether this kid reads B from shuffle_weight(w, layout=(16, 16)).

        Read it from here for the same reason as m_align: a caller that hands a
        row-major B to one of these kids gets no error, just a wrong answer, so
        a hand-maintained second list of "the preshuffled ones" fails silently.
        The m_align guard did exactly that -- it fed every kid the plain weight
        and reported ~1.47 relative error for all 15 of them.
        """
        return "bpreshuffle" in self.kernel_tag

    @property
    def needs_mpacked_sfa(self) -> tuple[int, int] | None:
        """(B_M, SFA_MB) for shuffle_scale_mxsk_mpack, or None for a plain A scale.

        Same silent-wrong-answer hazard as needs_preshuffled_b: the layout is a
        permutation of the same byte count, so a kid handed the plain (M, K/128)
        scale runs and returns wrong numbers. SFA_MB = T_M*W_M is the row block one
        M subtile steps over, which is what the packing folds into the low axis.
        """
        if not self.mpack_sfa:
            return None
        return self.B_M, self.T_M * self.W_M

    @property
    def needs_shuffle_scale(self) -> int | None:
        """shuffle_scale_a's ``sub``, or None for a plain A scale.

        sub is the row distance between the two M subtiles a dword pairs. It is a
        property of the *layout*, not of this kid's wave grid: the quantize kernel
        emits the A scale once per launch, so a deployment holds exactly one sub
        and every kid must read that one. The value is READ FROM THE HEADER
        (``_opus_sf_shuf_sub()``) rather than restated here -- if the host's sub
        and the kernel's disagree the kid does not fail, it returns plausible
        wrong numbers.

        Same silent-wrong-answer hazard as needs_mpacked_sfa: a kid handed the
        plain panels runs and returns wrong numbers. The A panel goes in with
        stride(1) as the per-batch slab and stride(0) zeroed; the kernel derives
        every other term, including the K block pair count, from the problem shape.

        The two pipelines assert their own tile against this sub, so a kid whose
        grid cannot read it fails to compile.
        """
        if not self.shuffle_scale:
            return None
        return _opus_sf_shuf_sub()


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


def a8w8_mxscale_flatmm_prefetch_k_iter(instance: OpusGemmInstance) -> int:
    """Mirror gfx950 MXFP8 flatmm ``Traits::prefetch_k_iter``."""
    if instance.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_wave1":
        return 1  # the wave1 traits' per-split floor: its ring pads past the end
    sizeof_da = 1  # FP8
    is_tile_n = instance.B_M == 16
    load_group_m = 16 if is_tile_n else 32
    load_group_n = 16 if is_tile_n else 32
    load_group_k = instance.W_K
    num_m = instance.B_M // load_group_m
    num_n = instance.B_N // load_group_n
    num_k = instance.B_K // load_group_k
    smem_linear = 64 * 16 // sizeof_da  # WARP_SIZE=64
    smem_sub = smem_linear // load_group_k
    slots = load_group_m // smem_sub
    padding = 2 * 16 // sizeof_da
    per_group_load = slots * (smem_linear + padding) * sizeof_da
    per_iter = (num_m + num_n) * num_k * per_group_load
    lds_total = 163840
    return max(
        1,
        (lds_total // max(instance.WG_PER_CU, 1)) // max(per_iter, 1),
    )


_BMM_M_ALIGN_TILES = {
    "a8w8_mxscale_bmm_flatmm_splitk": 0,
    "a8w8_mxscale_bmm_bpreshuffle_bdirect": 0,
    "a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen": 0,
    "a8w8_mxscale_bmm_bpreshuffle_blds": 0,
    "a8w8_mxscale_bmm_bpreshuffle_allwave": 0,
    "a8w8_mxscale_bmm_bpreshuffle_wave8n4": 0,
    "a8w8_mxscale_bmm_bpreshuffle_wavetm1": 0,
    "a8w8_mxscale_bmm_bpreshuffle_wavetm1_blds": 0,
    "a8w8_mxscale_bmm_bpreshuffle_wave1": 0,
    "a8w8_mxscale_bmm_bpreshuffle_compact": 0,
    "a8w8_mxscale_bmm_pipeline": 0,
    "a8w8_mxscale_bmm_pipeline_bpreshuffle": 0,
    "a8w8_mxscale_bmm_fused": 0,
    "a8w8_mxscale_bmm_minterleave": 2,
    "a8w8_mxscale_bmm_wave4m2_selfload": 2,
    "a8w8_mxscale_bmm_wave8n2": 1,
    "a8w8_mxscale_bmm_mouter": 1,
    "a8w8_mxscale_bmm_mouter_tunable": 1,
}

# GROUP_N=GROUP_K=32 counterparts use the base local kid plus this stride.
MX32_KID_STRIDE = 1000


def _a16w16(bs, bm, bn, bk, tn, wm, wn, wk, has_oob=True, cachectl_a=0, cachectl_b=17):
    """Factory for a16w16 split-barrier kid instances.

    cachectl_a / cachectl_b default to (0, 17) = (LRU, BYPASS_L2), which
    matches the traits-default cache policy for the split-barrier pipeline
    (see opus_gemm_a16w16_traits_gfx950 in
    csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh).
    This is the "legacy" policy used by KID 0..5 and 46..51 -- the
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
    0: OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
}


def _a8w8_mxscale_bmm_flatmm_splitk(
    bm, bn, bk, wg_per_cu, direct_only=False, prefetch_scale=False, preload_sf=False,
    shuffle_scale=False,
    quant_block=128,
):
    # quant_block is GROUP_N and GROUP_K together: A and B quantise on the same
    # block, either DSv4's 128 or MX's 32. GROUP_M stays 1 (per token) for both.
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256, bm, bn, bk, t_m, t_n, 16, 16, 128, 16, 16, 4,
        1, quant_block, quant_block, "a8w8_mxscale_bmm_flatmm_splitk", ["fp32_t"],
        wg_per_cu, splitk_workspace_dtype="fp32_t",
    )
    inst.name_root = "opus_bmm"
    inst.direct_only = direct_only
    inst.prefetch_scale = prefetch_scale
    inst.preload_sf = preload_sf
    # The kernel's static_asserts: the shuffle_scale dword pairs adjacent M
    # subtiles and spans at most two K blocks, and it replaces the LDS panels
    # rather than filling them. SFA_MB = T_M*W_M = 32 for this family.
    if shuffle_scale:
        assert not preload_sf, "shuffle_scale replaces the LDS scale panels"
        assert bm % 64 == 0, f"B_M={bm}: shuffle_scale needs COM_REP_M even"
        assert bk // 128 <= 2, f"B_K={bk}: shuffle_scale spans at most two K blocks"
    inst.shuffle_scale = shuffle_scale
    return inst


_BMM_MXSCALE_SPLITK_TILES = {
    67: (16,  32, 256, 2, False, False),
    68: (16,  32, 256, 2, False, True),
    69: (16,  32, 128, 2, False, False),
    70: (16,  32, 256, 4, False, False),
    66: (16,  32, 512, 2, False, False),
    65: (16,  64, 256, 2, False, False),
    64: (16, 128, 256, 1, False, False),
    63: (16,  32, 512, 2, False, True),
    72: (32,  32, 256, 2, False, True),
    74: (32,  32, 128, 2, False, True),
    71: (64,  32, 256, 2, False, False),
    73: (64,  32, 256, 1, False, False),
    140: (32,  64, 256, 2, False, False),
    141: (32,  64, 256, 1, False, False),
    142: (32,  64, 256, 2, True,  False),
    143: (64,  64, 128, 2, False, False),
    144: (64,  64, 128, 2, False, True),
    4: (128, 128, 128, 1, False, False),
    8: (128, 128, 128, 1, False, True),
    9: (64,  128, 256, 1, False, False),
    10: (128, 64,  256, 1, False, False),
    62: (32, 256, 128, 1, False, False),
    2:  (64, 128, 128, 2, False, False),
    0:   (32, 128, 128, 2, False, False),
    1:  (32, 128, 128, 2, False, False),
}
_bmm_flatmm_local = {
    kid: _a8w8_mxscale_bmm_flatmm_splitk(bm, bn, bk, wg, direct, prefetch)
    for kid, (bm, bn, bk, wg, direct, prefetch) in _BMM_MXSCALE_SPLITK_TILES.items()
}
_BMM_MXSCALE_SPLITK_PRELOAD_TILES = {
    75: (64, 32, 256, 2),
    76: (128, 128, 128, 1),
    77: (128, 64, 256, 1),
    78: (64, 128, 256, 1),
}
_bmm_flatmm_local.update({
    kid: _a8w8_mxscale_bmm_flatmm_splitk(bm, bn, bk, wg, preload_sf=True)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_SPLITK_PRELOAD_TILES.items()
})



# Preshuffled-weight MXFP8 BMM kernel registrations.


def _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg_per_cu, prefetch_scale=False,
                                          preload_sf=False, shuffle_scale=False,
                                          tilen=False, sf_shuf_in_lds=False,
                                          quant_block=128):
    """Preshuffled B bypassing LDS, on the flatmm producer/consumer split.

    Same kernel, launcher and tile geometry as the plain flatmm split-K family
    (BLOCK_SIZE=256, MFMA 16x16x128, VEC=(16,16,4), GROUP=(1,128,128)); the
    traits alias flips the B layout, the MFMA scale_op_sel byte select and B's
    path. The 16x16 preshuffle order already IS the mfma_16x16x128 B fragment
    order, so the consumer waves buffer_load B straight into their MFMA registers
    and the producer waves stage A only. Callers pass shuffle_weight(w, (16, 16)).
    """
    # Same tileN/tileM naming rule as the plain flatmm split-K family: the real
    # T_M/T_N comes from B_M in the traits, these only drive the symbol name.
    # tilen forces the B_M == 16 grid onto a wider tile, so the name has to follow
    # the traits rather than B_M.
    t_m, t_n = (1, 2) if (bm == 16 or tilen) else (2, 1)
    inst = OpusGemmInstance(
        256,            # BLOCK_SIZE
        bm, bn, bk,     # BLOCK tile
        t_m, t_n,       # T_M, T_N (4-wave warp-spec; tileN=1,2 / tileM=2,1)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        # GROUP_M=1 (per-token); GROUP_N=GROUP_K=quant_block, DSv4's 128 or MX's
        # 32. The traits this family derives from carry the 32 path, so the only
        # thing that made these 128-only was the literal.
        1, quant_block, quant_block,
        ("a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen" if tilen
         else "a8w8_mxscale_bmm_bpreshuffle_bdirect"),
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.prefetch_scale = prefetch_scale
    inst.preload_sf = preload_sf
    inst.shuffle_scale = shuffle_scale
    # The shuffled layout's own LDS scale panel. Distinct from preload_sf, which
    # stages the *plain* panel and is mutually exclusive with shuffle_scale in
    # this pipeline (both fill the same LDS, with different addressing) -- so a
    # panel kid here reads preload_sf False and sf_shuf_in_lds True, where a wave8
    # panel kid carries both.
    assert not sf_shuf_in_lds or shuffle_scale, (
        "sf_shuf_in_lds stages the shuffled scale words; it means nothing without "
        "shuffle_scale"
    )
    assert not (sf_shuf_in_lds and preload_sf), (
        "the plain and shuffled scale panels are alternative fills of the same "
        "LDS; the pipeline static_asserts they never coexist"
    )
    inst.sf_shuf_in_lds = sf_shuf_in_lds
    return inst


# bdirect tiles use producer waves for A and consumer waves for direct B loads.
# Scale-panel variants preload E8M0 scales; the small-M tiles retain the
# producer/consumer split with a 16-row output tile.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)
    28: (16, 32, 512, 2),
    26: (16, 32, 256, 2),
    # kid24 omits the scale panels used by kid25.
    24: (64, 32, 256, 2),
}
_bmm_bpre_bdirect_local = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_TILES.items()
}
# Plain-scale panel variant of the BM16 tile. COM_REP_M=1 lets the panel
# use the caller's row-major scale layout directly.
_bmm_bpre_bdirect_local[89] = (
    _a8w8_mxscale_bmm_bpreshuffle_bdirect(
        16, 32, 512, 2, prefetch_scale=False, preload_sf=True
    )
)

_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_PRELOAD_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)
    25: (64, 32, 256, 2),
    # 128x128x128 bdirect tile with preloaded scale panels.
    29: (128, 128, 128, 1),
}
# Larger B_K=256 tiles require additional staging and fragment registers.
# This family uses B_K=128 for its wider tiles to stay within those budgets.
_bmm_bpre_bdirect_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg, preload_sf=True)
    for kid, (bm, bn, bk, wg)
    in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_PRELOAD_TILES.items()
})


def _a8w8_mxscale_bmm_bpreshuffle_blds(bm, bn, bk, wg_per_cu, prefetch_scale=False,
                                       preload_sf=False, shuffle_scale=False,
                                       sf_shuf_in_lds=False, quant_block=128):
    """Preshuffled B staged through LDS by producer waves.

    Consumers read their MFMA fragments from LDS. The tile geometry matches
    the plain flatmm family; the traits select B's byte order and scale_op_sel.
    Callers pass shuffle_weight(w, (16, 16))."""
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256,            # BLOCK_SIZE
        bm, bn, bk,     # BLOCK tile
        t_m, t_n,       # T_M, T_N
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        # GROUP_M=1 (per-token); GROUP_N=GROUP_K=quant_block, 128 or MX's 32.
        1, quant_block, quant_block,
        "a8w8_mxscale_bmm_bpreshuffle_blds",
        ["fp32_t"],
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.prefetch_scale = prefetch_scale
    inst.preload_sf = preload_sf
    inst.shuffle_scale = shuffle_scale
    assert not (shuffle_scale and preload_sf), (
        "preload_sf stages the *plain* scale panel and reads it with the plain "
        "layout's addressing; the two flags are mutually exclusive (static_assert "
        "in opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh). A shuffled "
        "kid that wants an LDS panel asks for sf_shuf_in_lds instead."
    )
    assert not sf_shuf_in_lds or shuffle_scale, (
        "sf_shuf_in_lds stages the shuffled scale words; it means nothing without "
        "shuffle_scale"
    )
    inst.sf_shuf_in_lds = sf_shuf_in_lds
    return inst


# Map each plain flatmm tile to its preshuffled-B/LDS counterpart.
# Keep explicit IDs so existing tuned configurations remain valid.
_BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF = {
    72: 35,
    144: 36,
    75: 37,
    76: 38,
    77: 39,
    1: 60,
    2: 61,
    4: 40,
    8: 41,
    9: 42,
    10: 43,
    62: 44,
    63: 45,
    64: 46,
    65: 47,
    66: 48,
    67: 49,
    68: 50,
    69: 51,
    70: 52,
    71: 53,
    73: 54,
    74: 55,
    78: 56,
    140: 57,
    141: 58,
    143: 59,
}
_bmm_bpre_blds_local = {
    twin: _a8w8_mxscale_bmm_bpreshuffle_blds(
        plain.B_M, plain.B_N, plain.B_K, plain.WG_PER_CU,
        prefetch_scale=plain.prefetch_scale,
        preload_sf=plain.preload_sf,
    )
    for plain, twin in (
        # The twin map uses this family's local kid numbers.
        (_bmm_flatmm_local[plain_kid], twin_kid)
        for plain_kid, twin_kid in _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF.items()
    )
}

# Plain-scale panel variant of the BM16 tile with prefetched A scales.
# COM_REP_M=1 allows direct panel loads from the caller's scale layout.
_bmm_bpre_blds_local[88] = (
    _a8w8_mxscale_bmm_bpreshuffle_blds(
        32, 32, 256, 2, prefetch_scale=False, preload_sf=True
    )
)

# kid142 is the DIRECT_ONLY persistent schedule, which carries its own B staging
# and rejects the flags this family sets. kid0 is the heuristic default and an
# alias of kid1's geometry, whose preshuffled twin covers its tile.
_BLDS_NO_TWIN = {0, 142}
_blds_untwinned = sorted(
    kid
    for kid in _bmm_flatmm_local
    if kid not in _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF and kid not in _BLDS_NO_TWIN
)
assert not _blds_untwinned, (
    f"plain flatmm kids {_blds_untwinned} have no preshuffled-B twin. A preshuffled "
    "deployment cannot dispatch a row-major kid, so any cell one of these wins is a "
    "cell preshuffling B costs performance on. Add an id to "
    "_BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF, or to _BLDS_NO_TWIN with the reason."
)

# Small tiles compatible with the shuffled-scale layout.
# Each shuffled-scale variant retains its plain counterpart's geometry.
_BMM_MXSCALE_BPRESHUFFLE_BLDS_SHUFFLE_SCALE_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU, prefetch_scale)   plain winner it answers
    270: (16, 32, 256, 4, False),   # kid52
    271: (32, 32, 256, 2, True),    # kid35
    # kid272 wins nothing, and neither does its plain twin kid47. Kept only so
    # kid47's tile stays expressible under the wholesale switch; retire together.
    272: (16, 64, 256, 2, False),   # kid47
    # kid48 is the same tile without prefetch and deliberately gets no twin: it
    # wins nothing, so a twin would measure a kid the tuner never picks.
    274: (16, 32, 512, 2, True),    # kid45
}
_bmm_bpre_blds_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_blds(bm, bn, bk, wg, prefetch_scale=pf,
                                            shuffle_scale=True)
    for kid, (bm, bn, bk, wg, pf)
    in _BMM_MXSCALE_BPRESHUFFLE_BLDS_SHUFFLE_SCALE_TILES.items()
})

# blds variant with an LDS shuffled-scale panel.
# The registered BM32 tile fully consumes each packed scale word.
_BMM_MXSCALE_BPRESHUFFLE_BLDS_SHUFFLE_PANEL_TILES = {
    276: (32, 32, 256, 2, True),    # kid271 + panel
}
_bmm_bpre_blds_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_blds(bm, bn, bk, wg, prefetch_scale=pf,
                                            shuffle_scale=True, sf_shuf_in_lds=True)
    for kid, (bm, bn, bk, wg, pf)
    in _BMM_MXSCALE_BPRESHUFFLE_BLDS_SHUFFLE_PANEL_TILES.items()
})

# The shuffled-scale bdirect twins. A shuffled kid reads its A scale as packed
# dwords instead of per-subtile bytes, so it replaces the plain LDS scale panel
# rather than filling it -- one twin therefore answers both the panelled and the
# un-panelled member of a plain pair (kid248 stands against kid24 and kid25).
#
# Flags track the plain twin; the per-kid comment names it.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_SHUFFLE_SCALE_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU, prefetch_scale)     twin of
    248: (64, 32, 256, 2, False),    # kid24 / kid25
    249: (128, 128, 128, 1, False),  # kid29
    # ... and the same two tiles with the scale load hoisted ahead of the LDS wait.
    # PREFETCH_SCALE is only testable here: the wave8n4/wavetm1 pipeline
    # static_asserts !PREFETCH_SCALE.
    250: (64, 32, 256, 2, True),     # kid248 + prefetch
    251: (128, 128, 128, 1, True),   # kid249 + prefetch
    269: (16, 32, 256, 2, False),    # kid26
    # B_K=512, i.e. COM_REP_K=4, which spends two A-scale dwords along K
    # (SF_GEOM::KD). prefetch_scale is off because kid28 does not have it and
    # SF_PREFETCH is gated on COM_REP_K==1 anyway, so it would be inert.
    273: (16, 32, 512, 2, False),    # kid28
}
_bmm_bpre_bdirect_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg, prefetch_scale=pf,
                                               shuffle_scale=True)
    for kid, (bm, bn, bk, wg, pf)
    in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_SHUFFLE_SCALE_TILES.items()
})

# Shuffled-scale variants with an LDS panel. The prologue stages scales
# once so the K loop reads the panel from LDS. SF_SHUF_IN_LDS and
# SUBTILE_TILE remain separate registered traits.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_SHUFFLE_PANEL_TILES = {
    275: (16, 32, 512, 2, False),    # kid273 + panel
}
_bmm_bpre_bdirect_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg, prefetch_scale=pf,
                                               shuffle_scale=True, sf_shuf_in_lds=True)
    for kid, (bm, bn, bk, wg, pf)
    in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_SHUFFLE_PANEL_TILES.items()
})

# Paired plain-panel and shuffled-scale bdirect configurations.
# B_M=128 uses B_K=128 to satisfy the minimum prefetch depth.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_ISOLATE = {
    #    (B_M, B_N, B_K, WG_PER_CU)   term moved, and what it moves
    (79, 252): (64,  32,  128, 2),   # B_K  256->128: COM_REP_K 2->1
    (80, 253): (128, 32,  128, 2),   # B_M   64->128: COM_REP_M 2->4, vs (336,337)
    (81, 254): (64,  32,  256, 1),   # WG_PER_CU 2->1: occupancy only, geometry fixed
    # (344, 345): (64, 128, 256, 1) -- B_N 32->128, vs (342,343). Unwired, see below.
}
# The 64x128x256 bdirect variants are not instantiated because of register
# pressure. LDS feasibility alone does not establish register feasibility.
for (_plain_kid, _shuf_kid), (_bm, _bn, _bk, _wg) in (
    _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_ISOLATE.items()
):
    _bmm_bpre_bdirect_local[_plain_kid] = (
        _a8w8_mxscale_bmm_bpreshuffle_bdirect(_bm, _bn, _bk, _wg, preload_sf=True)
    )
    _bmm_bpre_bdirect_local[_shuf_kid] = (
        _a8w8_mxscale_bmm_bpreshuffle_bdirect(_bm, _bn, _bk, _wg, prefetch_scale=True,
                                              shuffle_scale=True)
    )


# bdirect variants with the T_M=1, T_N=2 consumer grid at B_M > 16.
# These remain registered but are excluded from the default tuning policy.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_TILEN_TILES = {
    #    (B_M, B_N, B_K, WG_PER_CU, preload_sf)   corresponding shuffled-scale ID
    85: (64, 32, 256, 1, True),    # kid81 (wg1 + panels)
    86: (64, 32, 256, 2, True),    # kid25 (wg2 + panels), the family's m<=256 winner
    87: (64, 32, 256, 2, False),   # kid24 (wg2, no panels), owns the K=1024 end
}
_bmm_bpre_bdirect_tilen_local = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg, preload_sf=pre,
                                               tilen=True)
    for kid, (bm, bn, bk, wg, pre)
    in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_TILEN_TILES.items()
}


def _a8w8_mxscale_bmm_bpreshuffle_allwave(bm, bn, bk, wg_per_cu, quant_block=128):
    """Four compute waves on a 2x2 grid, with both A and preshuffled B in LDS.

    All waves stage operands into the ring and perform MFMA."""
    inst = OpusGemmInstance(
        256,            # BLOCK_SIZE (4 waves)
        bm, bn, bk,     # BLOCK tile
        2, 2,           # T_M, T_N (name only; ALL_WAVE derives the 2x2 grid)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, quant_block, quant_block,  # GROUP_M=1 (per-token), GROUP_N=GROUP_K
        "a8w8_mxscale_bmm_bpreshuffle_allwave",
        ["fp32_t"],
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.preload_sf = True
    return inst


# A and B both staged means twice the LDS per ring slot: 64x64x256 at two
# workgroups a CU and 128x128x256 at one leave fewer than the three slots the
# pipeline needs, so neither is here.
_BMM_MXSCALE_BPRESHUFFLE_ALLWAVE_TILES = {
    #    (B_M, B_N, B_K, WG_PER_CU)
    100: (64, 64, 256, 1),
    101: (64, 64, 128, 2),
    102: (64, 128, 256, 1),
    103: (64, 128, 128, 2),
    104: (128, 64, 256, 1),
    105: (128, 128, 128, 1),
    106: (128, 64, 128, 2),
}
_bmm_bpre_allwave_local = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_allwave(bm, bn, bk, wg)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_BPRESHUFFLE_ALLWAVE_TILES.items()
}
# No GROUP_K=32 twins yet: the all-wave traits cap B_N at GROUP_N, which at 32
# rejects every tile here.


# Derive GROUP_N=GROUP_K=32 variants from existing plain-scale instances.
# Geometry and scheduling stay aligned with their group128 counterparts.
# Shuffled-scale variants require a different producer layout and are excluded.
def _mx32_twins(local, ctor, skip=frozenset(), **ctor_kwargs):
    """{mirror + MX32_KID_STRIDE: the same tile at quant_block=32}.

    `skip` holds mirrors whose twin clang 22 cannot compile, the same defect
    _MX32_CLANG_REGCLASS_SKIP covers for the split-K family, plus any whose twin
    is held out for returning wrong answers.
    """
    twins = {}
    for kid, inst in local.items():
        if kid in skip:
            continue
        if inst.shuffle_scale or getattr(inst, "sf_shuf_in_lds", False):
            continue
        twins[kid + MX32_KID_STRIDE] = ctor(
            inst.B_M, inst.B_N, inst.B_K, inst.WG_PER_CU,
            prefetch_scale=inst.prefetch_scale,
            preload_sf=inst.preload_sf,
            quant_block=32,
            **ctor_kwargs,
        )
    return twins


_bmm_bpre_bdirect_local.update(
    _mx32_twins(_bmm_bpre_bdirect_local, _a8w8_mxscale_bmm_bpreshuffle_bdirect)
)
_bmm_bpre_blds_local.update(
    # Exclude group32 variants affected by ROCm 7.2.4 clang-22 register
    # allocation defects: kid44 fails register-class validation, while
    # kids40/41 can reload an incomplete spilled MFMA accumulator.
    _mx32_twins(_bmm_bpre_blds_local, _a8w8_mxscale_bmm_bpreshuffle_blds,
                skip=frozenset({44, 40, 41}))
)
_bmm_bpre_bdirect_tilen_local.update(
    _mx32_twins(_bmm_bpre_bdirect_tilen_local,
                _a8w8_mxscale_bmm_bpreshuffle_bdirect, tilen=True)
)


def _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg_per_cu, xcd_wgm=0,
                                          mpack_sfa=False, shuffle_scale=False,
                                          sf_shuf_in_lds=False, quant_block=128):
    """Eight compute waves on a 2x4 grid with preshuffled B loaded directly.

    SCALE_OPSEL requires COM_REP_M = B_M/(W_M*T_M) to be a multiple of four.
    Eight waves distribute the accumulator and fragment register footprint.
    Only A is staged in LDS; B loads are consumed one N repeat at a time."""
    inst = OpusGemmInstance(
        512,            # BLOCK_SIZE (8 waves)
        bm, bn, bk,     # BLOCK tile
        2, 4,           # T_M, T_N (name only; traits derive the real 2x4 grid)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, quant_block, quant_block,  # GROUP_M=1 (per-token), GROUP_N=GROUP_K
        "a8w8_mxscale_bmm_bpreshuffle_wave8n4",
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.preload_sf = True
    inst.xcd_wgm = xcd_wgm
    inst.mpack_sfa = mpack_sfa
    inst.shuffle_scale = shuffle_scale
    assert not sf_shuf_in_lds or shuffle_scale, (
        "sf_shuf_in_lds stages the shuffled scale words; it means nothing without "
        "shuffle_scale"
    )
    inst.sf_shuf_in_lds = sf_shuf_in_lds
    return inst


_BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)
    30: (256, 256, 128, 1),
    # Half-M tile satisfying the SCALE_OPSEL COM_REP_M alignment.
    23: (128, 256, 128, 1),
    # 128x128 tile with the wave8n4 schedule.
    27: (128, 64,  256, 1),
    # 128x128 wave8n4 variants at two K tile depths.
    83: (128, 128, 256, 1),
    84: (128, 128, 128, 1),
}
_bmm_bpre_wave8n4_local = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_TILES.items()
}

# wave8n4 variants with XCD-aware banded workgroup mapping.
# XCD_WGM selects the M-band height; zero keeps the linear tile order.
_BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_XCD_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU, XCD_WGM)     twin of
    82: (256, 256, 128, 1, 4),               # kid30 + FlyDSL's band height
    # Apply band height 4 to the additional 2x4 tiles.
    90: (128, 64,  256, 1, 4),               # kid27 + band 4
    91: (128, 128, 256, 1, 4),               # kid83 + band 4
    # 347: (256, 256, 128, 1, 2),             # kid30 + half of it; noise, see above
}
_bmm_bpre_wave8n4_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg, xcd_wgm=wgm)
    for kid, (bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_XCD_TILES.items()
})

# Shuffled-scale counterparts of the wave8n4 tiles.
# The scale layout is fixed by its producer; every selected kernel must
# consume that same layout. This wave grid uses SHUF_SUB=32 to pair
# M subtiles in each packed scale word.
_BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_SHUFFLE_SCALE_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)     twin of
    245: (256, 256, 128, 1),         # kid30
    246: (128, 256, 128, 1),         # kid23
    247: (128, 64,  256, 1),         # kid27 -- also the only COM_REP_K=2 shuffled kid
}
_bmm_bpre_wave8n4_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg, shuffle_scale=True)
    for kid, (bm, bn, bk, wg)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_SHUFFLE_SCALE_TILES.items()
})

# Additional wave8n4 shuffled-scale counterparts with matching geometry.
_BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_SHUF_TWIN_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU, XCD_WGM, SF_SHUF_IN_LDS)   twin of
    # -- reg form, the tiles that had no shuffled twin at all --
    255: (256, 256, 128, 1, 4, False),                    # kid82  (15 rows)
    256: (128, 128, 256, 1, 0, False),                    # kid83
    257: (128, 128, 128, 1, 0, False),                    # kid84
    # -- lds form, one per tile above plus one per kid245/246/247 --
    258: (256, 256, 128, 1, 0, True),                     # kid30 / kid245
    259: (128, 256, 128, 1, 0, True),                     # kid23 / kid246
    260: (128, 64,  256, 1, 0, True),                     # kid27 / kid247
    261: (256, 256, 128, 1, 4, True),                     # kid82 / kid255
    262: (128, 128, 256, 1, 0, True),                     # kid83 / kid256
    263: (128, 128, 128, 1, 0, True),                     # kid84 / kid257
}
_bmm_bpre_wave8n4_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg, xcd_wgm=wgm,
                                               shuffle_scale=True,
                                               sf_shuf_in_lds=lds)
    for kid, (bm, bn, bk, wg, wgm, lds)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_SHUF_TWIN_TILES.items()
})

# B_K=256 variants with matched plain and shuffled-scale configurations.





def _a8w8_mxscale_bmm_bpreshuffle_wavetm1(block_size, bm, bn, bk, wg_per_cu,
                                          xcd_wgm=0, mpack_sfa=False,
                                          shuffle_scale=False, sf_shuf_in_lds=False,
                                          quant_block=128, blds=False):
    """T_M=1 tiles with A fragments retained in registers at B_M=128.

    BLOCK_SIZE selects a 1x8 or 1x4 grid. The blds variant stages B through
    the LDS ring; the direct-B variant loads it into registers."""
    inst = OpusGemmInstance(
        block_size,     # BLOCK_SIZE (512 -> 8 waves / 1x8, 256 -> 4 waves / 1x4)
        bm, bn, bk,     # BLOCK tile
        1, block_size // 64,  # T_M, T_N (name only; traits derive the real grid)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, quant_block, quant_block,  # GROUP_M=1 (per-token), GROUP_N=GROUP_K
        "a8w8_mxscale_bmm_bpreshuffle_wavetm1" + ("_blds" if blds else ""),
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.preload_sf = True
    inst.xcd_wgm = xcd_wgm
    inst.mpack_sfa = mpack_sfa
    inst.shuffle_scale = shuffle_scale
    assert not sf_shuf_in_lds or shuffle_scale, (
        "sf_shuf_in_lds stages the shuffled scale words; it means nothing without "
        "shuffle_scale"
    )
    inst.sf_shuf_in_lds = sf_shuf_in_lds
    return inst


# T_M=1 tiles. BLOCK_SIZE selects the number of N waves.
# At B_M=128 the A fragments stay in registers; larger tiles use the
# family's LDS streaming schedule.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU)
    32: (512, 128, 256, 128, 1),
    33: (256, 128, 256, 128, 1),
}
_bmm_bpre_wavetm1_local = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg)
    for kid, (bs, bm, bn, bk, wg) in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_TILES.items()
}

# wavetm1 variants with XCD-aware workgroup rasterization.
# The mapping groups nearby M tiles into bands to share cache-resident
# operands. XCD_WGM selects the band height.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_XCD_TILES = {
    # (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    34: (256, 128, 256, 128, 1, 4),   # kid33 with band height 4
    92: (512, 128, 256, 128, 1, 4),   # = kid32 (1x8) + band 4
}
_bmm_bpre_wavetm1_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_XCD_TILES.items()
})

# kid243 uses SFA_MPACK_GLOBAL: the caller supplies the A scale panel
# packed along M instead of having the prologue pack ordinary scales.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_MPACK_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    243: (256, 128, 256, 128, 1, 4),
}
_bmm_bpre_wavetm1_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm,
                                               mpack_sfa=True)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_MPACK_TILES.items()
})

# Narrower wavetm1 tiles on the 1x4 compute-wave grid.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_1X4_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    93: (256, 64, 128, 256, 1, 4),
    94: (256, 128, 128, 128, 1, 4),
    95: (256, 64, 256, 128, 1, 4),
    96: (256, 64, 64, 256, 1, 4),
    97: (256, 128, 64, 256, 1, 4),
    98: (256, 128, 128, 256, 1, 4),
    99: (256, 64, 128, 128, 1, 4),
}
_bmm_bpre_wavetm1_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_1X4_TILES.items()
})

# The 1x4 tiles at two workgroups per CU.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_1X4_WG2_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    117: (256, 128, 128, 128, 2, 4),
    118: (256, 64, 128, 256, 2, 4),
    119: (256, 64, 256, 128, 2, 4),
}
_bmm_bpre_wavetm1_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_1X4_WG2_TILES.items()
})

# The 1x4 tiles with B staged through the LDS ring (wavetm1_blds). kid120 is
# flydsl's b2/m2048 tile, 128x128x128 at four slots; the others are the 1x4
# tiles whose A+B slot still leaves three.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_BLDS_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    120: (256, 128, 128, 128, 1, 4),
    121: (256, 64, 128, 128, 1, 4),
    122: (256, 64, 128, 256, 1, 4),
    123: (256, 128, 256, 128, 1, 4),
    # Two-wave decode tiles using the shared A/B LDS staging schedule.
    # The B-LDS variant also participates in the scale-panel residency budget.
    124: (128, 16, 32, 256, 2, 0),
}
_bmm_bpre_wavetm1_blds_local = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm, blds=True)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_BLDS_TILES.items()
}

# kid244 reads both scales in the shuffle_scale_a / shuffle_scale_b layout.
# Each scale word packs two M subtiles crossed with two K blocks; B bytes
# are duplicated for the shared scale_op_sel selection. At B_K=128 the
# K loop unrolls by two to select the two K blocks in each word.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_SHUFFLE_SCALE_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    244: (256, 128, 256, 128, 1, 4),
}
_bmm_bpre_wavetm1_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm,
                                               shuffle_scale=True)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_SHUFFLE_SCALE_TILES.items()
})

# Matched plain and shuffled-scale wavetm1 configurations.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_SHUF_TWIN_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM, SF_SHUF_IN_LDS)  twin of
    # -- reg form; kid34's is kid244 above --
    264: (512, 128, 256, 128, 1, 0, False),                          # kid32
    265: (256, 128, 256, 128, 1, 0, False),                          # kid33
    # -- lds form --
    266: (256, 128, 256, 128, 1, 4, True),                           # kid34 / kid244
    267: (512, 128, 256, 128, 1, 0, True),                           # kid32 / kid264
    268: (256, 128, 256, 128, 1, 0, True),                           # kid33 / kid265
}
_bmm_bpre_wavetm1_local.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm,
                                               shuffle_scale=True,
                                               sf_shuf_in_lds=lds)
    for kid, (bs, bm, bn, bk, wg, wgm, lds)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_SHUF_TWIN_TILES.items()
})

# GROUP_K=32 counterparts of the all-compute direct-B families.
# Construct them from the original instance's traits.
def _mx32_wave8_twins(local, ctor, *lead, skip=frozenset(), **ctor_kwargs):
    # ctor_kwargs is not decoration. This builds the twin from the mirror's
    # *tile* attributes only, so any ctor flag that is not one of those -- blds
    # being the one that exists today -- is silently lost and the twin comes out
    # as a different kernel wearing the twin's id. It has to be passed back in
    # here, by the caller, the same way the mirror's own dict passes it.
    return {
        kid + MX32_KID_STRIDE: ctor(
            *[getattr(inst, a) for a in lead],
            inst.B_M, inst.B_N, inst.B_K, inst.WG_PER_CU,
            xcd_wgm=inst.xcd_wgm, quant_block=32, **ctor_kwargs,
        )
        for kid, inst in local.items()
        if kid not in skip
        and not (inst.shuffle_scale or inst.mpack_sfa or inst.sf_shuf_in_lds)
    }


_bmm_bpre_wave8n4_local.update(
    _mx32_wave8_twins(_bmm_bpre_wave8n4_local, _a8w8_mxscale_bmm_bpreshuffle_wave8n4)
)
_bmm_bpre_wavetm1_local.update(
    # The smaller B_K=128 tiles use the ring path when the whole-split scale
    # panel would exceed their residency budget.
    _mx32_wave8_twins(_bmm_bpre_wavetm1_local, _a8w8_mxscale_bmm_bpreshuffle_wavetm1,
                      "BLOCK_SIZE", skip=frozenset({95, 99, 119}))
)
_bmm_bpre_wavetm1_blds_local.update(
    # GROUP_K=32 counterparts of the wave8n4 tiles.
    _mx32_wave8_twins(_bmm_bpre_wavetm1_blds_local,
                      _a8w8_mxscale_bmm_bpreshuffle_wavetm1,
                      "BLOCK_SIZE", skip=frozenset({123}), blds=True)
)


def _a8w8_mxscale_bmm_bpreshuffle_wave1(bm, bn, bk, wg_per_cu, ring=0, quant_block=128, group_n=None):
    """Single-wave decode kernel with operands loaded directly into registers.

    The pipeline uses no LDS staging or inter-wave barriers."""
    inst = OpusGemmInstance(
        64,             # BLOCK_SIZE: one wave64
        bm, bn, bk,     # BLOCK tile
        1, 1,           # T_M, T_N
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, quant_block if group_n is None else group_n, quant_block,
        "a8w8_mxscale_bmm_bpreshuffle_wave1",
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.wave1_ring = ring
    return inst


# Single-wave decode tiles with several N/K depths and load schedules.
_BMM_MXSCALE_BPRESHUFFLE_WAVE1_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)
    107: (16, 32, 256, 2),
    108: (16, 64, 256, 2),
    109: (16, 32, 512, 2),
    110: (32, 32, 256, 2),
    111: (16, 32, 128, 2),
    112: (16, 32, 256, 1),
    # kid112 with the ring pinned shallower (default 4 stages, 192 VGPRs).
    113: (16, 32, 256, 1, 2),
    114: (16, 32, 256, 1, 3),
    # B_N=16 variants increase the number of N tiles per workgroup row.
    115: (16, 16, 256, 2),
    116: (16, 16, 256, 1),
}
_bmm_bpre_wave1_local = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave1(*geom)
    for kid, geom in _BMM_MXSCALE_BPRESHUFFLE_WAVE1_TILES.items()
}

# Independent per-N-row K32 scales.
_bmm_bpre_wave1_local.update({
    kid + 124: _a8w8_mxscale_bmm_bpreshuffle_wave1(
        *geom, quant_block=32, group_n=1
    )
    for kid, geom in _BMM_MXSCALE_BPRESHUFFLE_WAVE1_TILES.items()
})
_bmm_bpre_wave1_local[241] = _a8w8_mxscale_bmm_bpreshuffle_wave1(
    16, 16, 128, 2, quant_block=32, group_n=1
)


# Larger per-row K32 tile for shapes with enough output tiles to fill the GPU.
# A two-stage register ring bounds operand storage while each load feeds more
# MFMAs than the decode tiles above.
_bmm_bpre_wave1_local[242] = _a8w8_mxscale_bmm_bpreshuffle_wave1(
    64, 64, 128, 2, 2, quant_block=32, group_n=1
)


# Derive GROUP_N=GROUP_K=32 flatmm variants with kid + MX32_KID_STRIDE.
# Exclude tiles affected by the ROCm 7.2.4 clang-22
# register-class defect at the larger register footprints.
_MX32_CLANG_REGCLASS_SKIP = frozenset({4, 10, 62})
_bmm_flatmm_local.update({
    kid + MX32_KID_STRIDE: _a8w8_mxscale_bmm_flatmm_splitk(
        bm, bn, bk, wg, direct, prefetch, quant_block=32
    )
    for kid, (bm, bn, bk, wg, direct, prefetch) in _BMM_MXSCALE_SPLITK_TILES.items()
    if kid not in _MX32_CLANG_REGCLASS_SKIP
})
# GROUP_K=32 PRELOAD_SF_LDS variants use the bounded scale ring selected
# by SF_USE_RING, instead of allocating a whole-split scale panel.
_bmm_flatmm_local.update({
    kid + MX32_KID_STRIDE: _a8w8_mxscale_bmm_flatmm_splitk(
        bm, bn, bk, wg, preload_sf=True, quant_block=32
    )
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_SPLITK_PRELOAD_TILES.items()
})


# ROCm 7.2.4 clang-22 assigns an illegal register class while compiling this
# exact high-pressure PRELOAD_SF_LDS + D_OUT=void specialization after the
# workspace kargs moved to a direct pointer.  Its splitK=1 BF16/FP32 kernels
# keep PRELOAD_SF_LDS; only the split-K workspace specialization uses the
# semantically equivalent non-preload implementation (the same geometry as
# local kid 10).
_bmm_flatmm_local[77].workspace_preload_sf = False


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
    20: _a8w8_mxscale_bmm_minterleave(128, 128, 128, 1, False),
    21: _a8w8_mxscale_bmm_minterleave(128, 128, 128, 1, True),
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
    3: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_fused", 32, 128, 128, 2),
}

# pipeline (kids 14/15/151/152/16/17/22): BLOCK_SIZE 512, m{128,256}n{128,256}k128, dual
# bf16/fp32 traits (output dtype baked into the traits tuple), non-splitk scale
# kargs. One of the gemm_a8w8_scale_* kernels selected by flags. The wave
# layout (T_M/T_N/W_*) is derived inside opus_gemm_a8w8_scale_traits_gfx950 from
# BLOCK + <B_M,B_N,B_K>, so only B_M/B_N/B_K matter here (the T_M/T_N passed to
# OpusGemmInstance are cosmetic for this tag).
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
    14: _a8w8_mxscale_bmm_pipeline(B_M=128),
    15: _a8w8_mxscale_bmm_pipeline(),
    16: _a8w8_mxscale_bmm_pipeline(preload_sf_lds=True),
    # kid17 uses preloaded scales at the half-M tile.
    17: _a8w8_mxscale_bmm_pipeline(B_M=128, preload_sf_lds=True),
    # kid22 uses preloaded scales with the N tile halved.
    22: _a8w8_mxscale_bmm_pipeline(B_N=128, preload_sf_lds=True),
}


def _a8w8_mxscale_bmm_pipeline_bpreshuffle(**flags):
    """Preshuffled-weight variant of the quadrant-scheduled pipeline.

    The tile, 4x2 wave grid and LDS staging match kid158. The producer's
    B addressing consumes shuffle_weight(w, (16, 16))."""
    inst = OpusGemmInstance(
        512, 256, 256, 128, 2, 1, 16, 16, 128, 16, 16, 4, 1, 128, 128,
        "a8w8_mxscale_bmm_pipeline_bpreshuffle", ["fp32_t"], 1,
    )
    inst.name_root = "opus_bmm"
    for key, val in flags.items():
        setattr(inst, key, val)
    return inst


_bmm_pipeline_bpre_local = {
    # kid31: kid16 (preload_sf_lds) with a preshuffled B.
    31: _a8w8_mxscale_bmm_pipeline_bpreshuffle(preload_sf_lds=True),
}
_bmm_mouter_local = {
    5: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter", 128, 128, 128, 1),
    12: _a8w8_mxscale_bmm_spec(
        "a8w8_mxscale_bmm_mouter", 128, 128, 128, 1, skip_scale_wait=True
    ),
}
_bmm_mouter_tunable_local = {
    18: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter_tunable", 128, 128, 128, 1),
    19: _a8w8_mxscale_bmm_spec(
        "a8w8_mxscale_bmm_mouter_tunable", 128, 128, 128, 1,
        skip_scale_wait=True,
    ),
}
_bmm_wave8n2_local = {
    6: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_wave8n2", 128, 128, 128, 1),
}
_BMM_WAVE4M2_TILES = {
    7: (False, False),
    11: (True, False),
    13: (True, True),
}
_bmm_wave4m2_local = {
    kid: _a8w8_mxscale_bmm_spec(
        "a8w8_mxscale_bmm_wave4m2_selfload", 128, 128, 128, 1,
        skip_scale_wait=skip, pack_scale_on_demand=pack,
    )
    for kid, (skip, pack) in _BMM_WAVE4M2_TILES.items()
}


# All name-keyed a8w8_mxscale BMM kernel families (gfx950-only). Kept as a tuple
# of the per-family kid-keyed dicts -- NOT merged into one dict, because int kids
# repeat across families and are deduped downstream by launcher NAME (see
# gen_instances.py). Single source of truth for both consumers there: the codegen
# kdict merge and the BMM int-kid tune-lookup emitter.
# Kids that read their scales in a layout the quantiser has to produce -- the
# shuffle_scale words (with or without the LDS panel) and the host M-packed A
# panel. Nothing upstream emits those layouts yet, and no tuned row uses these
# kids, so they stay defined but are not built; set this to True to put all
# of them back into the catalogue, the codegen and the tuner.
BMM_BUILD_RELAID_SCALE_KIDS = False


def _relaid_scale(inst):
    return bool(inst.shuffle_scale or getattr(inst, "sf_shuf_in_lds", False)
                or inst.mpack_sfa)


def _bmm_kernel_table(kernels):
    return {
        kid: inst
        for kid, inst in kernels.items()
        if BMM_BUILD_RELAID_SCALE_KIDS or not _relaid_scale(inst)
    }


a8w8_mxscale_bmm_flatmm_splitk_kernels_list = _bmm_kernel_table(_bmm_flatmm_local)
a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list = _bmm_kernel_table(_bmm_bpre_bdirect_local)
a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen_kernels_list = _bmm_kernel_table(_bmm_bpre_bdirect_tilen_local)
a8w8_mxscale_bmm_bpreshuffle_blds_kernels_list = _bmm_kernel_table(_bmm_bpre_blds_local)
a8w8_mxscale_bmm_bpreshuffle_allwave_kernels_list = _bmm_kernel_table(_bmm_bpre_allwave_local)
a8w8_mxscale_bmm_bpreshuffle_wave8n4_kernels_list = _bmm_kernel_table(_bmm_bpre_wave8n4_local)
a8w8_mxscale_bmm_bpreshuffle_wavetm1_kernels_list = _bmm_kernel_table(_bmm_bpre_wavetm1_local)
a8w8_mxscale_bmm_bpreshuffle_wavetm1_blds_kernels_list = _bmm_kernel_table(
    _bmm_bpre_wavetm1_blds_local
)
a8w8_mxscale_bmm_bpreshuffle_wave1_kernels_list = _bmm_kernel_table(_bmm_bpre_wave1_local)
a8w8_mxscale_bmm_pipeline_bpreshuffle_kernels_list = _bmm_kernel_table(_bmm_pipeline_bpre_local)
a8w8_mxscale_bmm_fused_kernels_list = _bmm_kernel_table(_bmm_fused_local)
a8w8_mxscale_bmm_minterleave_kernels_list = _bmm_kernel_table(_bmm_minterleave_local)
a8w8_mxscale_bmm_mouter_kernels_list = _bmm_kernel_table(_bmm_mouter_local)
a8w8_mxscale_bmm_mouter_tunable_kernels_list = _bmm_kernel_table(
    _bmm_mouter_tunable_local
)
a8w8_mxscale_bmm_pipeline_kernels_list = _bmm_kernel_table(_bmm_pipeline_local)
a8w8_mxscale_bmm_wave8n2_kernels_list = _bmm_kernel_table(_bmm_wave8n2_local)
a8w8_mxscale_bmm_wave4m2_selfload_kernels_list = _bmm_kernel_table(
    _bmm_wave4m2_local
)


def _a8w8_mxscale_bmm_bpreshuffle_compact(bm, bn, bk, wm, wn, nb, nt, early_b):
    return OpusGemmInstance(
        wm * wn * 64,
        bm,
        bn,
        bk,
        wm,
        wn,
        16,
        16,
        128,
        16,
        16,
        4,
        1,
        128,
        128,
        "a8w8_mxscale_bmm_bpreshuffle_compact",
        ["fp32_t"],
        WG_PER_CU=1,
        cachectl_a=0,
        cachectl_b=2 if nt else 0,
        num_slots=nb,
        early_b=early_b,
        name_root="opus_bmm",
        max_m=2047,
    )


# Explicit local ids keep existing tuned CSVs stable when adding configurations.
# Arguments: BM, BN, BK, WM, WN, LDS slots, non-temporal B, early B prefetch.
a8w8_mxscale_bmm_bpreshuffle_compact_kernels_list = _bmm_kernel_table(
    {
        125: _a8w8_mxscale_bmm_bpreshuffle_compact(64, 64, 256, 2, 2, 4, True, False),
        126: _a8w8_mxscale_bmm_bpreshuffle_compact(96, 64, 256, 2, 2, 3, False, False),
        127: _a8w8_mxscale_bmm_bpreshuffle_compact(64, 64, 256, 4, 2, 4, True, False),
        128: _a8w8_mxscale_bmm_bpreshuffle_compact(
            192, 128, 128, 2, 4, 3, False, False
        ),
        129: _a8w8_mxscale_bmm_bpreshuffle_compact(256, 128, 128, 4, 2, 3, True, False),
        130: _a8w8_mxscale_bmm_bpreshuffle_compact(192, 128, 128, 2, 4, 3, False, True),
        131: _a8w8_mxscale_bmm_bpreshuffle_compact(256, 128, 128, 4, 2, 3, True, True),
        132: _a8w8_mxscale_bmm_bpreshuffle_compact(16, 32, 512, 1, 2, 3, False, False),
        133: _a8w8_mxscale_bmm_bpreshuffle_compact(32, 32, 512, 2, 2, 2, False, False),
        134: _a8w8_mxscale_bmm_bpreshuffle_compact(16, 64, 512, 1, 4, 2, False, False),
        135: _a8w8_mxscale_bmm_bpreshuffle_compact(16, 64, 256, 1, 4, 3, False, False),
        136: _a8w8_mxscale_bmm_bpreshuffle_compact(32, 64, 256, 2, 2, 3, False, False),
        137: _a8w8_mxscale_bmm_bpreshuffle_compact(64, 32, 256, 2, 2, 3, False, False),
        138: _a8w8_mxscale_bmm_bpreshuffle_compact(64, 64, 128, 2, 2, 3, False, False),
        139: _a8w8_mxscale_bmm_bpreshuffle_compact(
            128, 128, 128, 2, 2, 2, False, False
        ),
    }
)


a8w8_mxscale_bmm_kernel_lists = (
    a8w8_mxscale_bmm_bpreshuffle_compact_kernels_list,
    a8w8_mxscale_bmm_flatmm_splitk_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_blds_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_allwave_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_wave8n4_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_wavetm1_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_wavetm1_blds_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_wave1_kernels_list,
    a8w8_mxscale_bmm_fused_kernels_list,
    a8w8_mxscale_bmm_minterleave_kernels_list,
    a8w8_mxscale_bmm_mouter_kernels_list,
    a8w8_mxscale_bmm_mouter_tunable_kernels_list,
    a8w8_mxscale_bmm_pipeline_kernels_list,
    a8w8_mxscale_bmm_pipeline_bpreshuffle_kernels_list,
    a8w8_mxscale_bmm_wave8n2_kernels_list,
    a8w8_mxscale_bmm_wave4m2_selfload_kernels_list,
)


a8w8_kernels_list = {
    0: OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0, "a8w8", ["fp32_t"]),
}

a16w16_kernels_list = {
    # -- MFMA 16x16x32, T_N=2, BS=256 (2-block/CU capable) --
    # disabled: _a16w16(256, 128, 128, 32,  2, 16, 16, 32),  # disabled: intermittent accuracy (suspected compiler issue with VGPR=104/AGPR=64)
    0:  _a16w16(256, 128, 256, 32,  2, 16, 16, 32),
    1:  _a16w16(256, 256, 128, 32,  2, 16, 16, 32),
    # -- MFMA 16x16x32, T_N=4, BS=512 (1-block/CU) --
    2:  _a16w16(512, 128, 128, 64,  4, 16, 16, 32),
    3:  _a16w16(512, 256, 128, 64,  4, 16, 16, 32),
    4:  _a16w16(512, 128, 256, 64,  4, 16, 16, 32),
    5:  _a16w16(512, 256, 256, 64,  4, 16, 16, 32),  # existing / current default
}

# Removed a16w16_flatmm non-splitk: Rationale: the non-splitk a16w16_flatmm
# pipeline has two latent correctness b...
a16w16_flatmm_kernels_list = {}

# 11 splitk tiles mirroring gcnasm/opus_fmm/flatmm_a16w16_4wave_wasp_splitk.cc -t 0..10 dispatch
# exactly: * 8 WG_PER_CU=2 tiles (...
a16w16_flatmm_splitk_kernels_list = {
    # WG_PER_CU=2, cc tile 0..7
    6: _a16w16_flatmm_splitk( 64,  64,  64, 2),   # cc tile 0: M>=128 sweet spot (default)
    7: _a16w16_flatmm_splitk( 32,  32,  64, 2),   # cc tile 1
    8: _a16w16_flatmm_splitk( 32,  32, 128, 2),   # cc tile 2
    9: _a16w16_flatmm_splitk( 32,  64,  64, 2),   # cc tile 3
    10: _a16w16_flatmm_splitk( 32, 128,  64, 2),   # cc tile 4
    11: _a16w16_flatmm_splitk( 64,  32,  64, 2),   # cc tile 5
    12: _a16w16_flatmm_splitk( 64,  32, 128, 2),   # cc tile 6: recommended for medium M
    13: _a16w16_flatmm_splitk(128,  32,  64, 2),   # cc tile 7
    # WG_PER_CU=1, cc tile 8..10 (160 KB/wg LDS; zero VGPR spill only)
    14: _a16w16_flatmm_splitk( 64,  64, 128, 1),   # cc tile 8: deep K, high compute/load ratio
    15: _a16w16_flatmm_splitk(256,  32,  64, 1),   # cc tile 9: very tall, narrow N
    16: _a16w16_flatmm_splitk( 32, 256,  64, 1),   # cc tile 10: very wide, narrow M
    # Tile coverage extension (kids 17..29): B_M=96 OR B_N=96 lanes for shapes whose M or N is a
    # multiple of 96.
    17: _a16w16_flatmm_splitk( 32,  96,  64, 1),   # pfk=9, VGPR=176/512, AGPR=24
    18: _a16w16_flatmm_splitk( 32,  96,  64, 2),   # pfk=4, VGPR=176/256, AGPR=24
    19: _a16w16_flatmm_splitk( 32,  96, 128, 1),   # pfk=4, VGPR=288/512, AGPR=24
    20: _a16w16_flatmm_splitk( 64,  96,  64, 1),   # pfk=7, VGPR=192/512, AGPR=48
    21: _a16w16_flatmm_splitk( 64,  96,  64, 2),   # pfk=3, VGPR=192/256, AGPR=48
    22: _a16w16_flatmm_splitk( 64,  96, 128, 1),   # pfk=3, VGPR=320/512, AGPR=48
    23: _a16w16_flatmm_splitk( 96,  32,  64, 1),   # pfk=9, VGPR=144/512, AGPR=24
    24: _a16w16_flatmm_splitk( 96,  32,  64, 2),   # pfk=4, VGPR=144/256, AGPR=24
    25: _a16w16_flatmm_splitk( 96,  32, 128, 1),   # pfk=4, VGPR=224/512, AGPR=24
    26: _a16w16_flatmm_splitk( 96,  64,  64, 1),   # pfk=7, VGPR=176/512, AGPR=48
    27: _a16w16_flatmm_splitk( 96,  64,  64, 2),   # pfk=3, VGPR=176/256, AGPR=48
    28: _a16w16_flatmm_splitk( 96,  64, 128, 1),   # pfk=3, VGPR=288/512, AGPR=48
    29: _a16w16_flatmm_splitk( 96,  96,  64, 2),   # pfk=3, VGPR=208/256, AGPR=72  (81% VGPR -- watch)
}

# non-OOB variants: kid + 46, same tile but HAS_OOB=false.
a16w16_kernels_list_nooob = {
    kid + 46: _a16w16(
        inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
        inst.T_N, inst.W_M, inst.W_N, inst.W_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_kernels_list.items()
}

# CPOL variants for a16w16: 3 policies per kid, tuner picks best per shape.
# Explicit local OOB/nooob ranges keep the cache policies disjoint.
_CACHECTL_CONFIGS = [
    # (oob_offset, nooob_offset, cachectl_a, cachectl_b, tag)
    (97, 103, 1, 17, "Mheavy"),
    (140, 109, 17, 1, "Nheavy"),
    (146, 152, 0, 0, "balanced"),
]
a16w16_kernels_list_cpol = {}
for offset, _nooob_offset, ca, cb, _tag in _CACHECTL_CONFIGS:
    for kid, inst in a16w16_kernels_list.items():
        new_inst = _a16w16(
            inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
            inst.T_N, inst.W_M, inst.W_N, inst.W_K,
        )
        new_inst.cachectl_a = ca
        new_inst.cachectl_b = cb
        a16w16_kernels_list_cpol[kid + offset] = new_inst

a16w16_kernels_list_cpol_nooob = {}
for _oob_offset, offset, ca, cb, _tag in _CACHECTL_CONFIGS:
    for kid, inst in a16w16_kernels_list.items():
        new_inst = _a16w16(
            inst.BLOCK_SIZE, inst.B_M, inst.B_N, inst.B_K,
            inst.T_N, inst.W_M, inst.W_N, inst.W_K, has_oob=False,
        )
        new_inst.cachectl_a = ca
        new_inst.cachectl_b = cb
        a16w16_kernels_list_cpol_nooob[kid + offset] = new_inst

a16w16_flatmm_splitk_kernels_list_nooob = {
    kid + 46: _a16w16_flatmm_splitk(
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
    30 + i: _a16w16_persistent(bm, bn, bk)
    for i, (bm, bn, bk) in enumerate(_PERSISTENT_TILES)
}

# Cpol variants (304..315): 3 groups x 4 tiles, mirroring _CACHECTL_CONFIGS but with a single
# compact base offset per cpol group.
_PERSISTENT_CPOL_GROUPS = [
    # (base_kid, cachectl_a, cachectl_b)
    (34, 1, 17),   # Mheavy
    (38, 17, 1),   # Nheavy
    (42, 0, 0),    # balanced
]
a16w16_persistent_kernels_list_cpol = {}
for _base, _ca, _cb in _PERSISTENT_CPOL_GROUPS:
    for i, (bm, bn, bk) in enumerate(_PERSISTENT_TILES):
        a16w16_persistent_kernels_list_cpol[_base + i] = _a16w16_persistent(
            bm, bn, bk, cachectl_a=_ca, cachectl_b=_cb
        )

# Nooob mirrors at +1000 for both legacy (1300..1303) and cpol (1304..1315).
# Explicit cachectl inheritance keeps name() consistent with parents.
a16w16_persistent_kernels_list_nooob = {
    kid + 46: _a16w16_persistent(
        inst.B_M, inst.B_N, inst.B_K, has_oob=False,
        cachectl_a=inst.cachectl_a, cachectl_b=inst.cachectl_b,
    )
    for kid, inst in a16w16_persistent_kernels_list.items()
}
a16w16_persistent_kernels_list_cpol_nooob = {
    kid + 46: _a16w16_persistent(
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


# 1 mono-tile tiles, local kids 92..96.
#
# B_K=128 tiles (e.g. (64,256,128), (128,128,128)) are intentionally
# excluded: the pipeline uses 2x smem_a + 3x smem_b (A double-buffered,
# B triple-buffered as r0/r1/w), which pushes those tiles to 165-231 KiB
# of LDS -- over gfx950's 160 KiB budget. Re-enable only after the
# pipeline drops B to two slots.
_MONO_TILE_TILES = [
    # (B_M, B_N, B_K)
    (192, 256, 64),   # 92
    (128, 256, 64),   # 93
    (192, 128, 64),   # 94
    (128, 128, 64),   # 95
    ( 64, 128, 64),   # 96
]
a16w16_mono_tile_kernels_list = {
    92 + i: _a16w16_mono_tile(bm, bn, bk)
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
# Offset choice: +5000 is reserved for legacy 4g_safe variants; cpol variants
# use explicit disjoint offsets rather than an unconditional +1000 mirror.
# 4g_safe kids carry HAS_OOB
# from their parent (M/N tail is absorbed by the per-WG BR num_records, so
# the per-thread predicate is structurally a no-op for valid in-tile threads;
# we still emit both has_oob variants for consistency with the legacy axis).


def _make_4g_safe(inst: "OpusGemmInstance") -> "OpusGemmInstance":
    """Clone an OpusGemmInstance with is_4g_safe=True; everything else
    (kernel_tag, traits, kargs, BLOCK/B_*/T_*/W_*/VEC_*, cachectl, has_oob)
    is inherited verbatim. The codegen dispatch in gen_instances.py reads
    is_4g_safe to pick the 4g_safe pipeline header + kernel symbol."""
    from dataclasses import replace
    return replace(inst, is_4g_safe=True)


a16w16_kernels_list_4g_safe = {
    115 + i: _make_4g_safe(inst)
    for i, inst in enumerate(a16w16_kernels_list.values())
}
a16w16_kernels_list_4g_safe_nooob = {
    125 + i: _make_4g_safe(inst)
    for i, inst in enumerate(a16w16_kernels_list_nooob.values())
}
a16w16_persistent_kernels_list_4g_safe = {
    121 + i: _make_4g_safe(inst)
    for i, inst in enumerate(a16w16_persistent_kernels_list.values())
}
a16w16_persistent_kernels_list_4g_safe_nooob = {
    131 + i: _make_4g_safe(inst)
    for i, inst in enumerate(a16w16_persistent_kernels_list_nooob.values())
}
a16w16_mono_tile_kernels_list_4g_safe = {
    135 + i: _make_4g_safe(inst)
    for i, inst in enumerate(a16w16_mono_tile_kernels_list.values())
}


# -- gfx942 kernel lists ------------------------------------------------

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
    """Factory for gfx942 a16w16 kbuf1-large-tile kid instances (local kid 0,
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
    0: _a16w16_gfx942        (512, 128, 128,  64,    4, 16, 16, 16),   # kbuf1_large_tile (4-phase, big tile)
    1: _a16w16_p1_gfx942     (256,  64,  64,  64,    2, 16, 16, 16),   # P1 depth=2 sibling of 10201
    2: _a16w16_kbuf2v_bk128_gfx942(256, 64,  64, 128,    2, 16, 16, 16),   # P1 B_K=128 sibling of 10203
    3: _a16w16_quad_mfma32_gfx942(256, 256, 256, 32, 2, 2, 32, 32, 8), # quad MFMA32 pipeline
    12: _a16w16_wave_k_coop_gfx942(512, 16, 16, 64,    1, 16, 16, 16),  # wave-K-coop 16x16, T_K=8
    13: _a16w16_wave_k_coop_gfx942(512, 16, 32, 32,    1, 16, 16, 16),  # WKC 16x32, B_K=32
    14: _a16w16_wave_k_coop_gfx942(512, 32, 16, 64,    1, 16, 16, 16),  # WKC 32x16, aliased partial
    15: _a16w16_wave_k_coop_gfx942(256, 32, 32, 64,    1, 16, 16, 16),  # WKC 32x32, T_K=4
    16: _a16w16_wave_k_coop_gfx942(512, 16, 32, 64,    1, 16, 16, 16),  # WKC 16x32, B_K=64
    17: _a16w16_wave_k_coop_accum_gfx942(256, 16, 16, 64, 1, 16, 16, 16),  # WKC 16x16 split8 atomic accumulate
    18: _a16w16_wave_k_coop_accum_gfx942(512, 16, 32, 32, 1, 16, 16, 16),  # WKC 16x32 split8 atomic accumulate
    19: _a16w16_wave_k_coop_accum_gfx942(512, 32, 16, 64, 1, 16, 16, 16),  # WKC 32x16 split8 atomic accumulate
    20: _a16w16_wave_k_coop_accum_gfx942(256, 32, 32, 64, 1, 16, 16, 16),  # WKC 32x32 split8 atomic accumulate
    21: _a16w16_wave_k_coop_accum_gfx942(256, 64, 16, 64, 1, 16, 16, 16),  # WKC 64x16 split8 atomic accumulate
}

gfx942_splitk_kernels_list = {
    4: _a16w16_kbuf1_sk_gfx942      (512, 128, 128,  64,    4, 16, 16, 16),                # legacy 4-phase large tile
    5: _a16w16_kbuf2v_sk_gfx942     (256,  64,  64,  64,    2, 16, 16, 16),                # P1 depth=2 + V-dbuf
    6: _a16w16_kbuf2v_bk128_sk_gfx942(256, 64,  64, 128,    2, 16, 16, 16),                # P1 B_K=128 sub-K decomp
    7: _a16w16_em3en4_lds1_pgr2_sk_gfx942 (256, 128,  96, 128,    2, 16, 16, 16),                # EM3EN4 LDS1/PGR2 hipb-orientation (host 128M x 96N)
    8: _a16w16_kbuf1_sk_gfx942      (512,  64, 128,  64,    4, 16, 16, 16),                # legacy 4-phase M64 x N128
    9: _a16w16_kbuf1_sk_bf16ws_gfx942(512, 128, 128,  64,    4, 16, 16, 16),                # legacy 4-phase large tile + bf16 workspace
    10: _a16w16_kbuf2v_bk128_sk_bf16ws_gfx942(256, 64,  64, 128,    2, 16, 16, 16),           # P1 B_K=128 + bf16 workspace
    11: _a16w16_quad_mfma32_sk_bf16ws_gfx942(256, 256, 256, 32, 2, 2, 32, 32, 8, group_m=6),  # 3 bf16 splitK sibling, group-M=6
}

gfx942_a8w8_kernels_list = {
    0: _a8w8_blockscale_bpreshuffle_singlebuf_gfx942(
        512, 128, 128, 128, 4, 2, 16, 16, 32, vec=16),
}
# NOTE: 10402 (a16w16_naive_64x64) was removed -- 32.85us never matched WKC's
# 11.88us on tuned shapes (bf16_tuned_ge...

gfx942_kernels_list = {
    **gfx942_nosplit_kernels_list,
    **gfx942_splitk_kernels_list,
}

# -- gfx1250 kernel lists ----------------------------------------------------
# Both #4246 families are represented: two-stage
# cluster/TDM kids and fused in-cluster-reduce kids.


def _a16w16_cluster_tdm_splitk_ws_gfx1250(bm, bn, bk, layout, num_slots=3, wg_per_cu=2):
    """Factory for the gfx1250 a16w16 cluster/TDM split-K (workspace + reduce) kid.

    Locked geometry from the kernel base
    (demon_gcn/wmma_opus_rdna4/gemm_a16w16_cluster_tdm_splitk_reduce_4wave.cc):
    BLOCK_SIZE=128 (4 waves x 32 = 2 producer + 2 consumer), MFMA 16x16x32,
    NO-CLUSTER (one WG per B_M x B_N tile). The main kernel WMMA-accumulates in
    fp32 and casts each split's partial into the exact kid's typed workspace; a
    separate reduce kernel sums the split slices in fp32, folds bias, and casts
    to the Y dtype. The two-stage families keep partials in fp32 workspace. The
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
        splitk_workspace_dtype="fp32_t",
        # The separate reducer places one logical row in each grid.y block.
        max_m=65535,
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
    #    variant).
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
#   plain (no-cluster) local kids start at 0, ONE P=3 kid per tile (P=2 is
#   dropped -- unvalidated). Tiles the picker rejects (>=256-request TDM
#   direct-copy, now FIXED) fall back to P=3, 1 WG/CU so every no-spill tile still
#   emits a plain kid (LDS(P=3) <= 320 KB for this set).
#
# The consumer kExpN stability guard (previously _GFX1250_MAX_KEXPN=8) is removed.
gfx1250_kernels_list = {}
GFX1250_PLAIN_KID_OF = {}  # (B_M,B_N,B_K) -> kid (P=3; tuner + Python heuristic)
_GFX1250_KID_BASE = 0
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

GFX1250_BASE_KIDS = frozenset(gfx1250_kernels_list.keys())


# -- gfx1250 CLUSTER-LAUNCH (multicast) variant ------------------------------
# Same 4-wave TDM split-K + typed workspace + reduce kernel, but launched as a
# (cluster_wg_m x cluster_wg_n x 1) workgroup CLUSTER: peers co-reside and share
# A/B TDM loads via CLUSTER_LOAD_ASYNC multicast (named-barrier producer/consumer
# handshake, same as the plain base). The host launcher rounds the grid up to the
# cluster dims; surplus workgroups take the pipeline's uniform tile_oob exit.
# Logical workspace strides use the unrounded tile counts.
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
# Clusterlaunch local kids follow the plain table.
_GFX1250_CLUSTERLAUNCH_KID_BASE = _p_kid
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


# Deterministic local numbering over (tile outer, then
# cluster dim (cwn outer, cwm inner)). The kExpN stability guard has been removed, so ALL 26
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
# is visible to exact dispatch/capability queries.
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
        # This family reduces in-kernel, without the separate grid.y launch.
        max_m=None,
        fuse_split_k=split_k,
        # Historical #4246 field name; physically this is an N-peer count.
        fuse_m_cluster=n_cluster,
    )


GFX1250_SPLITK_FUSE_ENABLED = False

gfx1250_splitk_fuse_kernels_list = {}
GFX1250_SPLITK_FUSE_KID_OF = {}

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


# -- gfx1250 symmetric 4-wave compute, pre-compiled .co ---------------------
# JSON is the single source of truth shared by the offline image builder and
# this host-launcher registry. The device kernels need compiler facilities not
# available in the release JIT toolchain, so gen_instances emits host launchers
# only and loads the matching image at runtime.
CO_KERNELS_JSON = os.path.join(os.path.dirname(__file__), "gen_co", "co_kernels.json")

_CO_DTYPE_BYTES = {
    "bf16_t": 2,
    "fp16_t": 2,
    "fp32_t": 4,
    "fp8_t": 1,
    "bf8_t": 1,
}


def _co_instance_from_json(arch, entry):
    """Convert one JSON record to the exact host/device CO configuration."""
    tag = entry["tag"]
    bm, bn, bk = entry["tile"]
    cwm, cwn = entry["cluster"]
    lb_threads, lb_waves = entry["launch_bounds"]
    dtypes = entry["dtypes"]
    assert lb_threads == entry["block_size"], (
        f"co kid {entry['kid']}: launch_bounds[0]={lb_threads} must equal "
        f"block_size={entry['block_size']}"
    )
    vec_a = 16 // _CO_DTYPE_BYTES[dtypes["a"]]
    vec_b = 16 // _CO_DTYPE_BYTES[dtypes["b"]]
    return OpusGemmInstance(
        entry["block_size"],
        bm,
        bn,
        bk,
        4,
        1,
        16,
        16,
        32,
        vec_a,
        vec_b,
        8,
        0,
        0,
        0,
        tag,
        [dtypes["c"]],
        arch_prefix=arch,
        num_slots=entry["num_slots"],
        cluster_wg_m=cwm,
        cluster_wg_n=cwn,
        co_num_vgpr=entry.get("num_vgpr", 0),
        co_min_waves_per_eu=lb_waves,
        co_device_flags=tuple(entry.get("device_flags", ())),
        co_variant=entry.get("variant", ""),
        co_dtypes=(dtypes["a"], dtypes["b"], dtypes["c"], dtypes["acc"]),
        co_wave_layout=tuple(entry.get("wave_layout", (4, 1))),
    )


def _validate_co_instance(kid, instance):
    """Validate the runtime contract baked into one pre-built image."""
    if instance.arch_prefix != "gfx1250":
        raise ValueError(
            f"CO kid {kid} must target gfx1250, got {instance.arch_prefix!r}"
        )
    if instance.output_dtypes != ["bf16_t"]:
        raise ValueError(f"CO kid {kid} must expose BF16 output only")
    if instance.co_dtypes != (
        "bf16_t",
        "bf16_t",
        "bf16_t",
        "fp32_t",
    ):
        raise ValueError(f"CO kid {kid} has unsupported dtype contract")
    if instance.splitk_workspace_dtype is not None:
        raise ValueError(f"CO kid {kid} must not declare workspace storage")


def co_image_path(path, instance):
    """Return the pre-built image corresponding to an instance JSON file."""
    return os.path.join(
        os.path.dirname(path), instance.arch_prefix, f"{instance.name}.co"
    )


def _load_co_kernels(path, require_image=True):
    """Load pre-built CO instances, safely dropping records with no image."""
    if not os.path.exists(path):
        return {}

    import json

    with open(path) as stream:
        document = json.load(stream)

    instances = {}
    seen_names = {}
    missing = []
    for arch, entries in document.items():
        if arch.startswith("_"):
            continue
        for entry in entries:
            kid = entry["kid"]
            tag = entry["tag"]
            if tag not in _A16W16_CO_TAGS:
                raise ValueError(f"unsupported CO kernel tag {tag!r} for kid {kid}")
            if type(kid) is not int or kid < 0:
                raise ValueError(f"invalid local CO kid {kid}")
            assert kid not in instances, f"duplicate co kid {kid} in {path}"
            instance = _co_instance_from_json(arch, entry)
            _validate_co_instance(kid, instance)
            assert instance.name not in seen_names, (
                f"co kids {seen_names[instance.name]} and {kid} generate the same "
                f"symbol {instance.name!r}"
            )
            seen_names[instance.name] = kid
            if require_image and not os.path.exists(co_image_path(path, instance)):
                missing.append((kid, instance.name))
                continue
            instances[kid] = instance

    if missing:
        preview = ", ".join(f"{kid} ({name}.co)" for kid, name in missing[:5])
        suffix = f", ... and {len(missing) - 5} more" if len(missing) > 5 else ""
        print(
            f"[opus] {len(missing)} pre-compiled (.co) kid(s) dropped -- image "
            f"not found under {os.path.dirname(path)}: {preview}{suffix}. Build "
            "them with csrc/opus_gemm/gen_co/build_co.py.",
            file=sys.stderr,
        )
    return instances


gfx1250_4wave_co_kernels_list = {
    kid: instance
    for kid, instance in _load_co_kernels(CO_KERNELS_JSON).items()
    if instance.kernel_tag in _A16W16_CO_TAGS
}
# The offline builder must see declared records even before their images exist.
gfx1250_4wave_co_kernels_declared = {
    kid: instance
    for kid, instance in _load_co_kernels(
        CO_KERNELS_JSON, require_image=False
    ).items()
    if instance.kernel_tag in _A16W16_CO_TAGS
}

GFX1250_SPLITK_FUSE_KID_BASE = max(
    (*gfx1250_kernels_list, *gfx1250_clusterlaunch_kernels_list,
     *gfx1250_4wave_co_kernels_declared), default=-1,
) + 1000
_sf_kid = GFX1250_SPLITK_FUSE_KID_BASE
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

GFX1250_SPLITK_FUSE_KIDS = frozenset(gfx1250_splitk_fuse_kernels_list.keys())
assert bool(GFX1250_SPLITK_FUSE_KIDS) == GFX1250_SPLITK_FUSE_ENABLED

GFX1250_4WAVE_CO_KID_OF = {
    instance.name: kid for kid, instance in gfx1250_4wave_co_kernels_list.items()
}
GFX1250_4WAVE_CO_KIDS = frozenset(gfx1250_4wave_co_kernels_list)

_GFX1250_PRE_CO_KIDS = (
    frozenset(gfx1250_kernels_list)
    | frozenset(gfx1250_clusterlaunch_kernels_list)
    | GFX1250_SPLITK_FUSE_KIDS
)
assert not (GFX1250_4WAVE_CO_KIDS & _GFX1250_PRE_CO_KIDS), (
    "gfx1250 CO kids overlap an existing gfx1250 family"
)

# Flatten the eight BMM launcher tags into the same canonical exact-kid
# registry used by every other OPUS family.
a8w8_mxscale_bmm_kernels_list = {
    kid: instance
    for family in a8w8_mxscale_bmm_kernel_lists
    for kid, instance in family.items()
}

# combined list (used by production gen_instances / dispatch)
gfx950_a16w16_kernels_list = {
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
}

gfx1250_a16w16_kernels_list = {
    **gfx1250_kernels_list,
    **gfx1250_clusterlaunch_kernels_list,
    **gfx1250_splitk_fuse_kernels_list,
    **gfx1250_4wave_co_kernels_declared,
}

KERNEL_REGISTRY = {
    "a16w16": {
        "gfx950": {cu: dict(sorted(gfx950_a16w16_kernels_list.items())) for cu in (128, 256)},
        "gfx942": {cu: dict(sorted(gfx942_kernels_list.items())) for cu in (80, 228, 304)},
        "gfx1250": {256: dict(sorted(gfx1250_a16w16_kernels_list.items()))},
    },
    "a8w8": {"gfx950": {cu: dict(a8w8_kernels_list) for cu in (128, 256)}},
    "a8w8_blockscale": {"gfx950": {cu: dict(a8w8_scale_kernels_list) for cu in (128, 256)}},
    "a8w8_mxscale_bmm": {"gfx950": {cu: dict(sorted(a8w8_mxscale_bmm_kernels_list.items())) for cu in (128, 256)}},
    "a8w8_blockscale_bpreshuffle": {"gfx942": {cu: dict(gfx942_a8w8_kernels_list) for cu in (80, 228, 304)}},
}

# fmt: on


# Generated names must identify one configuration. Multiple IDs may alias
# the same configuration, but distinct traits must not share a name.
_name_owner = {}
_name_clashes = []
for _cat in (
    *a8w8_mxscale_bmm_kernel_lists,
    gfx950_a16w16_kernels_list,
    gfx942_kernels_list,
    gfx1250_a16w16_kernels_list,
):
    for _kid, _inst in _cat.items():
        _sig = repr(asdict(_inst))
        _owner = _name_owner.setdefault(_inst.name, (_kid, _sig))
        if _owner[1] != _sig:
            _name_clashes.append((_owner[0], _kid, _inst.name))
assert not _name_clashes, (
    "these kid pairs differ in configuration yet share one instance name, so only "
    f"one of each pair can be emitted: {_name_clashes}. Whatever distinguishes them "
    "has to appear in OpusGemmInstance.name for their kernel_tag."
)


@dataclass(frozen=True, order=True)
class KernelKey:
    family: str
    gfx: str
    cu_num: int
    kid: int

    def __post_init__(self):
        object.__setattr__(self, "family", self.family.strip().lower())
        object.__setattr__(self, "gfx", self.gfx.split(":", 1)[0].strip().lower())
        if type(self.cu_num) is not int or self.cu_num <= 0:
            raise ValueError("kernel identity requires a positive integer cu_num")
        if type(self.kid) is not int or self.kid < 0:
            raise ValueError("kernel identity requires a non-negative integer kid")

    def encode(self) -> str:
        return f"{self.family}/{self.gfx}/{self.cu_num}/{self.kid}"

    @classmethod
    def decode(cls, value: str) -> "KernelKey":
        family, gfx, cu_num, kid = value.split("/")
        return cls(family, gfx, int(cu_num), int(kid))


def get_registry(family: str, gfx: str, cu_num: int) -> dict[int, object]:
    """Select available kernels from the family/gfx/CU table."""
    family = family.strip().lower()
    gfx = gfx.split(":", 1)[0].strip().lower()
    tables = KERNEL_REGISTRY.get(family, {}).get(gfx, {})
    if not tables:
        return {}
    if type(cu_num) is not int or cu_num not in tables:
        raise ValueError(f"unregistered OPUS namespace {family}/{gfx}/{cu_num}")
    records = tables[cu_num]
    if family == "a16w16" and gfx == "gfx1250":
        return {
            kid: instance
            for kid, instance in records.items()
            if instance.kernel_tag not in _A16W16_CO_TAGS
            or os.path.isfile(co_image_path(CO_KERNELS_JSON, instance))
        }
    return records


def parse_integer(value, *, name, minimum=0):
    try:
        number = Decimal(str(value))
        integer = int(number)
        if number != integer or integer < minimum:
            raise ValueError
        return integer
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            f"invalid {name} {value!r}; expected an integer >= {minimum}"
        ) from exc


def kid_for_name(family: str, gfx: str, cu_num: int, name: str) -> int:
    kid = min(
        (
            kid
            for kid, instance in get_registry(family, gfx, cu_num).items()
            if instance.name == name
        ),
        default=None,
    )
    if kid is None:
        raise ValueError(
            f"kernel symbol {name!r} is absent from {family}/{gfx}/{cu_num}"
        )
    return kid


def default_compiled_keys(gfx: str, cu_num: int) -> set[KernelKey]:
    """Select the default compile set from the current CU table."""
    defaults = DEFAULT_COMPILED_KIDS_BY_NAMESPACE.get((gfx, cu_num), ())
    return {
        KernelKey("a16w16", gfx, cu_num, kid)
        for kid in get_registry("a16w16", gfx, cu_num)
        if kid in defaults
    }


def _in_group(name, family, instance):
    if family == "a16w16":
        workspace = instance.splitk_workspace_dtype is not None
        return {
            "SPLITK_KIDS": workspace,
            "NON_SPLITK_KIDS": not workspace,
            "BIAS_AWARE_KIDS": workspace or instance.kernel_tag == "a16w16",
            "GFX1250_4WAVE_CO_KIDS": instance.kernel_tag in _A16W16_CO_TAGS,
            "SPLITK_4G_SAFE_KIDS": workspace and instance.is_4g_safe,
            "NON_SPLITK_4G_SAFE_KIDS": not workspace and instance.is_4g_safe,
            "FOUR_G_SAFE_KIDS": instance.is_4g_safe,
        }.get(name, False)
    return family == "a8w8_mxscale_bmm" and name == "BMM_MXSCALE_KIDS"


def group_kids(name, gfx, family="a16w16", *, cu_num):
    """Derive capability groups from the selected namespace's instances."""
    return frozenset(
        kid
        for kid, instance in get_registry(family, gfx, cu_num).items()
        if _in_group(name, family, instance)
    )


BMM_MXSCALE_WORKSPACE_TAGS = frozenset(
    {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
    }
)
BMM_MXSCALE_WORKSPACE_KIDS = frozenset(
    ("a8w8_mxscale_bmm", "gfx950", kid)
    for kid, instance in a8w8_mxscale_bmm_kernels_list.items()
    if instance.kernel_tag in BMM_MXSCALE_WORKSPACE_TAGS and not instance.direct_only
)

_SUPPORTED_SPLITK_WORKSPACE_DTYPES = frozenset({"bf16_t", "fp32_t"})
for _tables in KERNEL_REGISTRY["a16w16"].values():
    for _workspace_kid, _workspace_instance in next(iter(_tables.values())).items():
        _workspace_dtype = _workspace_instance.splitk_workspace_dtype
        if (
            _workspace_dtype is not None
            and _workspace_dtype not in _SUPPORTED_SPLITK_WORKSPACE_DTYPES
        ):
            raise ValueError(
                f"workspace kid {_workspace_kid} must explicitly declare "
                f"splitk_workspace_dtype, got {_workspace_dtype!r}"
            )

# Exact-id kernels kept in every default build.  The high-level A16 caller-side
# heuristics below the tuned lookup are constrained to these ids; the unified
# public/C++ launch path still receives one already-resolved exact kid.
DEFAULT_COMPILED_KIDS_GFX950 = frozenset(
    {
        # splitk fallback (small M / non-aligned big M)
        6,
        52,  # cc tile 0: (64, 64, 64) WG=2
        12,
        58,  # cc tile 6: (64, 32, 128) WG=2
        14,
        60,  # cc tile 8: (64, 64, 128) WG=1
        # persistent fallback (large M, tile-aligned)
        30,
        76,  # persistent (256, 256, 64)
    }
)

DEFAULT_COMPILED_KIDS_GFX942 = frozenset(
    {
        # Representative exact-id launchers kept in default gfx942 builds.
        0,  # gfx942 split-barrier    512x128x128x64 16x16x16 (large problem)
        1,  # gfx942 p1               256x64x64x64
        2,  # gfx942 p1_bk128         256x64x64x128
        4,  # gfx942 splitk          512x128x128x64 16x16x16 (N > 128)
        5,  # gfx942 splitk_p1        256x64x64x64  (depth=2 + workspace + reduce)
        6,  # gfx942 splitk_p1_bk128  256x64x64x128 (B_K=128 Option B; dev/bench)
        7,  # gfx942 splitk_em3en4_lds1_pgr2 256x128x96x128 hipb-orientation
        8,  # gfx942 splitk_legacy    512x64x128x64 16x16x16
        9,  # gfx942 splitk_legacy_bf16ws 512x128x128x64
        10,  # gfx942 splitk_p1_bk128_bf16ws 256x64x64x128
        12,  # gfx942 wave_k_coop     512x16x16x64 T_K=8
        13,  # gfx942 wave_k_coop     512x16x32x32 T_K=8
        14,  # gfx942 wave_k_coop     512x32x16x64 T_K=8
        15,  # gfx942 wave_k_coop     256x32x32x64 T_K=4
        16,  # gfx942 wave_k_coop     512x16x32x64 T_K=8
    }
)

# Keep representative plain/clusterlaunch workspace kids and all CO host kids
# in default gfx1250 builds; the tuner compiles other device kids on demand.
DEFAULT_COMPILED_KIDS_GFX1250 = (
    frozenset(
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
    | frozenset({GFX1250_CLUSTERLAUNCH_KID_OF[(16, 32, 128, 2, 1)]})
    | frozenset(gfx1250_4wave_co_kernels_declared)
)


DEFAULT_COMPILED_KIDS_BY_ARCH = {
    "gfx950": DEFAULT_COMPILED_KIDS_GFX950,
    "gfx942": DEFAULT_COMPILED_KIDS_GFX942,
    "gfx1250": DEFAULT_COMPILED_KIDS_GFX1250,
}


DEFAULT_COMPILED_KIDS = frozenset(
    ("a16w16", gfx, kid)
    for gfx, kids in DEFAULT_COMPILED_KIDS_BY_ARCH.items()
    for kid in kids
)

DEFAULT_COMPILED_KIDS_BY_NAMESPACE = {
    (gfx, cu): DEFAULT_COMPILED_KIDS_BY_ARCH[gfx] & table.keys()
    for gfx, tables in KERNEL_REGISTRY["a16w16"].items()
    for cu, table in tables.items()
}


def default_compiled_kids_for_arch(arches, *, cu_num=None):
    arches = set(arches or OPUS_KERNEL_TAGS_BY_ARCH_FAMILY)
    if cu_num is not None:
        return frozenset(
            (key.family, key.gfx, key.kid)
            for gfx in arches
            for key in default_compiled_keys(gfx, cu_num)
        )
    return frozenset(
        ("a16w16", gfx, kid)
        for gfx in arches
        for kid in DEFAULT_COMPILED_KIDS_BY_ARCH.get(gfx, ())
    )


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
        # Keep in step with _A8W8_MXSCALE_BMM_TAGS in aiter/ops/opus/launch_plan.py.
        # They are two hand-maintained copies of the same list and both are
        # consulted, this one first: get_kernel_instance rejects a kid whose tag
        # is missing here before the plan builder ever checks its own copy. A tag
        # added to only one of them therefore looks like "no registered OPUS
        # kernel", which reads as a missing kernel rather than a missing list
        # entry -- that is exactly how all 133 preshuffled rows were dropped
        # while the kids were present in kernels_list the whole time.
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
                # The preshuffled-B families: unreachable upstream, where gfx950
                # preshuffle is off, but every row of the preshuffled tuned table
                # names one.
                "a8w8_mxscale_bmm_bpreshuffle_bdirect",
                "a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen",
                "a8w8_mxscale_bmm_bpreshuffle_blds",
                "a8w8_mxscale_bmm_bpreshuffle_allwave",
                "a8w8_mxscale_bmm_bpreshuffle_wave8n4",
                "a8w8_mxscale_bmm_bpreshuffle_wavetm1",
                "a8w8_mxscale_bmm_bpreshuffle_wavetm1_blds",
                "a8w8_mxscale_bmm_bpreshuffle_wave1",
                "a8w8_mxscale_bmm_bpreshuffle_compact",
                "a8w8_mxscale_bmm_pipeline_bpreshuffle",
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
                "a16w16_4wave_co",
                "a16w16_4wave_wl_co",
                "a16w16_4wave_wlr_co",
            }
        ),
        "a8w8": frozenset(),
        "a8w8_blockscale": frozenset(),
        "a8w8_mxscale_bmm": frozenset(),
        "a8w8_blockscale_bpreshuffle": frozenset(),
    },
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
    *,
    cu_num: int,
) -> OpusGemmInstance | None:
    """Return a kernel registered for ``(arch, interface, kid, Y.dtype)``."""
    arch = str(arch).lower().split(":", 1)[0]
    family = str(family).lower()
    family_tags = OPUS_KERNEL_TAGS_BY_ARCH_FAMILY.get(arch, {}).get(family)
    if family_tags is None:
        return None

    try:
        kid = int(kid)
    except (TypeError, ValueError):
        return None

    instance = get_registry(family, arch, cu_num).get(kid)
    if instance is None or instance.kernel_tag not in family_tags:
        return None
    instance_arch = (instance.arch_prefix or "gfx950").lower()
    if instance_arch != arch:
        return None

    dtype = canonical_output_dtype(output_dtype)
    if dtype is not None:
        # Workspace reducers own the final cast, so their host dispatch
        # specialization in ``output_dtypes`` is not the Y dtype contract.
        if family == "a16w16" and instance.splitk_workspace_dtype is not None:
            # The current gfx942 BF16-workspace reducer is exact-N and writes
            # BF16 only.  Other A16 workspace reducers support BF16/FP32 Y.
            allowed = (
                {"bf16_t"}
                if arch == "gfx942" and instance.splitk_workspace_dtype == "bf16_t"
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


def kernel_needs_external_workspace(
    arch: str, family: str, kid: int, *, cu_num: int
) -> bool:
    """Return whether a registered kernel requires caller-owned workspace.

    Unknown logical keys are errors rather than ``False``: treating an unknown
    kid as a non-workspace kernel would let a caller launch it without the
    allocation required for memory safety.  Capability comes from the existing
    selected kernel instance, never from a numeric kid range or tag substring.
    The registry includes all enabled two-stage reducers. If the experimental
    gfx1250 fused family is re-enabled, its first SplitK-1 WGs also publish
    external partial tiles and therefore enter this same capability set.
    """
    instance = get_kernel_instance(arch, family, kid, cu_num=cu_num)
    if instance is None:
        raise KeyError(
            f"unknown OPUS kernel (arch={arch!r}, family={family!r}, kid={kid!r})"
        )
    return instance.splitk_workspace_dtype is not None and not instance.direct_only


def _opus_sidecar_path():
    """Return the on-disk path of the subset-compile sidecar.

    Lives in ``{bd_dir}/`` (one level above the per-module build dir) so
    it survives ``aiter.jit.core.clear_build("module_deepgemm_opus")`` --
    which ``build_module()`` calls when ``AITER_REBUILD == 1`` -- and
    seeds the last successfully compiled set into the next codegen.
    The tuner passes new candidates through ``--extra_kernel_keys``; it does not
    advance this file before compiling. JIT atomically copies the generated
    sidecar back here after installing the .so, independently of source-cache
    publication. Its adjacent ``.receipt`` binds the contents to that binary;
    the tuner requires both to match before skipping a rebuild. Runtime exact
    dispatch resolves the caller-selected kid without reading this file.
    """
    # Import lazily to avoid circular import at module load (aiter imports
    # opus_gemm_common, opus_gemm_common imports aiter.jit.core).
    from aiter.jit.core import bd_dir

    return os.path.join(bd_dir, "compiled_kids_opus.json")
