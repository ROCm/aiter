# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
from dataclasses import asdict, dataclass, field

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
    # SplitK workspace storage dtype; splitK launchers still use fp32 tune dispatch.
    splitk_workspace_dtype: str = "fp32_t"

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

    # gfx1250 FUSED single-kernel split-K (a16w16_clusterlaunch_tdm_splitk_fuse):
    # SplitK and MClusterWg are COMPILE-TIME (cluster dims (SplitK, MClusterWg, 1)),
    # so each kid bakes one combo. fuse_ws_dtype = DataWs partial storage
    # ("bf16_t" default; "fp32_t" for higher reduce precision). Ignored by every
    # other pipeline. fuse_split_k == 0 marks "not a fuse kid".
    fuse_split_k: int = 0
    fuse_m_cluster: int = 1
    fuse_ws_dtype: str = "bf16_t"

    # --- a8w8_mxscale BMM flatmm-splitK axes (kernel_tag ==
    # "a8w8_mxscale_bmm_flatmm_splitk"). The BMM main kernel template is
    #   gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, D_OUT, DIRECT_ONLY,
    #                                          PREFETCH_SCALE>
    # so unlike a16w16 each kid carries two compile-time booleans in addition to
    # the tile. direct_only == consumer-self-load direct-store (splitK==1 only);
    # prefetch_scale == scale-prefetch variant; fused_reduce == splitK==2 fused
    # tail-reduce launch path. These drive both the launcher body and the set of
    # device instantiations gen_instances emits for the kid.
    direct_only: bool = False
    prefetch_scale: bool = False
    fused_reduce: bool = False
    # a8w8_mxscale BMM flatmm-splitK only: preload this split's SFA (per-token) +
    # SFB (block) scale panels into LDS once, then read scales from LDS in the
    # consumer instead of a per-K-tile global buffer_load. Maps to the kernel's
    # 5th template bool PRELOAD_SF_LDS.
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
    # a8w8_mxscale BMM wave8 families only: band height in M tiles for the L2
    # rasterization of the workgroup -> tile map, 0 for the plain linear map. Maps
    # to the wave8 kernel's trailing `int XCD_WGM`.
    xcd_wgm: int = 0
    # a8w8_mxscale BMM specialized-pipeline axis (minterleave / mouter /
    # mouter_tunable / wave4m2_selfload families). Maps to the kernel's trailing
    # `bool SKIP_SCALE_WAIT` template param: skip the s_waitcnt on the per-K-tile
    # scale load (the scale is issued a tile ahead), trading a correctness margin
    # for pipeline overlap. Drives both the launcher body and the device
    # instantiation set for the kid.
    skip_scale_wait: bool = False
    # a8w8_mxscale BMM wave4m2_selfload family extra bool axis (kernel template
    # order: <Traits, D_OUT, SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND>).
    pack_scale_on_demand: bool = False
    # a8w8_mxscale BMM pipeline family (kids 150/151/152): dual bf16/fp32
    # traits + one of the gemm_a8w8_scale_* kernels selected by these flags
    # (all-false = plain scale kernel).
    k1024_only: bool = False
    k1024_lb1: bool = False
    # a8w8_mxscale BMM pipeline family (kid158): preload BOTH SFA (per-token) and
    # SFB (block) scale panels into LDS. Maps to the pipeline kernel
    # gemm_a8w8_scale_preload_sf_kernel.
    preload_sf_lds: bool = False
    # Symbol root ("opus_gemm" for GEMM, "opus_bmm" for the batched frontends).
    name_root: str = "opus_gemm"

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
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_bdirect":
            # opus_bmm_a8w8_mxscale_bpreshuffle_bdirect_<geom>_wgpcu{N}
            #     [_scaleprefetch][_sfpreload]
            # The two flag suffixes matter: instances are deduplicated by name, so
            # a tile that differs only by a bool would otherwise silently collapse
            # onto its plain sibling and never be emitted.
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_bdirect")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.prefetch_scale:
                parts.append("scaleprefetch")
            if self.preload_sf:
                parts.append("sfpreload")
            if self.shuffle_scale:
                parts.append("sfshuf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_blds":
            # opus_bmm_a8w8_mxscale_bpreshuffle_blds_<geom>_wgpcu{N}
            #     [_scaleprefetch][_sfpreload]
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_blds")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.prefetch_scale:
                parts.append("scaleprefetch")
            if self.preload_sf:
                parts.append("sfpreload")
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
                parts.append("sfshuf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_bpreshuffle_wavetm1":
            # opus_bmm_a8w8_mxscale_bpreshuffle_wavetm1_<geom>_wgpcu{N}_sfpreload
            #     [_xcd{N}][_sfgmpack]
            parts.insert(tag_at, "a8w8_mxscale_bpreshuffle_wavetm1")
            parts.append(f"wgpcu{self.WG_PER_CU}")
            if self.preload_sf:
                parts.append("sfpreload")
            if self.xcd_wgm:
                parts.append(f"xcd{self.xcd_wgm}")
            if self.mpack_sfa:
                parts.append("sfgmpack")
            if self.shuffle_scale:
                parts.append("sfshuf")
        elif self.kernel_tag == "a8w8_mxscale_bmm_minterleave":
            # opus_bmm_a8w8_mxscale_flatmm_minterleave_<geom>_wgpcu{N}[_skip_scale_wait]
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
        elif self.kernel_tag == "a8w8_mxscale_bmm_pipeline_bpreshuffle":
            # opus_bmm_a8w8_mxscale_pipeline_bpreshuffle_<geom>[_preload_sf].
            # Without this branch the tag fell through to the bare default, so
            # kid196 was named opus_bmm_<geom> alone: it advertised neither the
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
            # gfx1250 fp32-workspace split-K with a separate reduce kernel.
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
            # FUSED single-kernel split-K. The visible segment is "skfuse" (NOT
            # "splitk_...") so the reduce-TU detection (keys on "_splitk_" in the
            # kernel name, gen_instances.py) never emits a reduce kernel for it.
            # It IS still in SPLITK_TAGS (fp32 lookup ABI). m{m}s{split_k}ws{dt}
            # + cluster geometry keep each (tile, split_k, m_cluster, ws_dtype)
            # symbol unique.
            parts.insert(tag_at, "skfuse")
            # fuse_m_cluster now groups N-tile peers (A-multicast); tag as n{}.
            parts.append(f"n{self.fuse_m_cluster}s{self.fuse_split_k}")
            parts.append("wsf32" if self.fuse_ws_dtype == "fp32_t" else "wsbf16")
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
        """M multiple this kid's generated host guard enforces (1 == any M).

        The launcher family decides it, not the kid: see _BMM_M_ALIGN_TILES and
        the AITER_CHECK blocks the matching launcher body in
        codegen/gen_instances_gfx950.py emits. Consumers that pick a kid for a
        shape (the tuner's candidate filter, the runtime's padded-M lookup) must
        read it from here rather than keep their own list -- two hand-maintained
        copies is exactly how kid326 ended up excluded from tuning while the
        runtime dispatched it anyway.
        """
        mult = _BMM_M_ALIGN_TILES.get(self.kernel_tag)
        if mult is not None:
            return self.B_M * mult if mult else 1
        # Non-BMM families: has_oob is the codegen flag that says whether the
        # tail is masked, and opus_gemm_tune.py already gates on it this way.
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

        Same silent-wrong-answer hazard as needs_mpacked_sfa: shuffle_scale_a
        and shuffle_scale_b permute the same bytes, so a kid handed the plain
        panels runs and returns wrong numbers. The A panel goes in with stride(1)
        as the per-batch slab and stride(0) zeroed -- the kernel derives every
        other term, including the K block pair count, from the problem shape.

        sub = T_M*W_M is the row distance between adjacent M subtiles, which is
        the axis the dword pairs. 16 is FlyDSL's own layout (T_M=1).
        """
        if not self.shuffle_scale:
            return None
        return self.T_M * self.W_M


# a8w8_mxscale BMM launcher family -> the B_M multiple its host guard requires,
# or 0 when the launcher masks a partial M tile and emits no M check at all.
# Mirrors the AITER_CHECK blocks in the launcher bodies of
# codegen/gen_instances_gfx950.py (_BMM_*_LAUNCHER_BODY); gen_instances asserts
# the two agree, so a guard edit that forgets this table fails the build.
_BMM_M_ALIGN_TILES = {
    "a8w8_mxscale_bmm_flatmm_splitk": 0,
    "a8w8_mxscale_bmm_bpreshuffle_bdirect": 0,
    "a8w8_mxscale_bmm_bpreshuffle_blds": 0,
    "a8w8_mxscale_bmm_bpreshuffle_wave8n4": 0,
    "a8w8_mxscale_bmm_bpreshuffle_wavetm1": 0,
    "a8w8_mxscale_bmm_pipeline": 0,
    "a8w8_mxscale_bmm_pipeline_bpreshuffle": 0,
    "a8w8_mxscale_bmm_fused": 0,
    "a8w8_mxscale_bmm_minterleave": 2,  # MI=2 M tiles per WG, baked in
    "a8w8_mxscale_bmm_wave4m2_selfload": 2,  # LOGICAL_B_M = B_M * 2
    "a8w8_mxscale_bmm_wave8n2": 1,
    "a8w8_mxscale_bmm_mouter": 1,
    "a8w8_mxscale_bmm_mouter_tunable": 1,
}


def _a16w16(bs, bm, bn, bk, tn, wm, wn, wk, has_oob=True, cachectl_a=0, cachectl_b=17):
    """Factory for a16w16 split-barrier kid instances.

    cachectl_a / cachectl_b default to (0, 17) = (LRU, BYPASS_L2), which
    matches the traits-default cache policy for the split-barrier pipeline
    (see opus_gemm_a16w16_traits_gfx950 in
    csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh).
    This is the "legacy" policy used by KID 4..9 and 1004..1009 -- the
    `_LEGACY_CACHECTL` special-case in OpusGemmInstance.name keeps these
    kids emitting the bare `..._0x0x0` symbol (no `_cA0cB17` suffix) so
    the production heuristic dispatcher and the opus tuned CSV stay
    bit-compatible.
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
    # kid 1 (256x256) is the launcher hardcoded by opus_gemm.cu's
    # opus_dispatch_scale (the only a8w8_scale GEMM path). The 128x256 sibling
    # kid 720 was removed below.
    1: OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
}

# Dead 128x256 scale GEMM tiles removed (no CSV/dispatch caller):
#   - kid 720 (a8w8_scale, fp32 block-scale): only consumer was the removed
#     opus_bmm_a8w8_scale mmajor path.
#   - kid 710 (a8w8_mxscale, e8m0 block-scale): only consumer was the opus_bmm
#     kid 149 hand-written adapter (via the _mmajor sibling), now replaced by
#     the BMM-native a8w8_mxscale_bmm_pipeline 128x256 instance.
# Both were the same gemm_a8w8_scale_kernel specialization, differing only in
# scale dtype; opus_dispatch_scale still uses the 256x256 kid 1 above.


def _a8w8_mxscale_bmm_flatmm_splitk(
    bm, bn, bk, wg_per_cu, direct_only=False, prefetch_scale=False, preload_sf=False,
    shuffle_scale=False,
):
    """fp8 e8m0 mxscale BATCHED matmul flatmm split-K tile.

    Backs opus_bmm_a8w8_mxscale(); the main kernel
    (gemm_a8w8_mxscale_flatmm_splitk_kernel) writes an fp32 workspace and a
    shared reduce kernel casts to the Y dtype (bf16/fp32), so output_dtypes is
    fp32 workspace here. Locked geometry (matches the hand-written traits in
    opus_bmm.cu): BLOCK_SIZE=256 (4 waves), T_M=2/T_N=1, MFMA 16x16x128 (fp8),
    VEC=(16,16,4), GROUP=(1,128,128) (per-token M, 128x128 block scale).
    direct_only / prefetch_scale are the two kernel compile-time booleans.
    """
    # tileN (bm==16): consumers split N (T_M=1, T_N=2). tileM (bm>=32): split M
    # (T_M=2, T_N=1). The real T_M/T_N is derived in the C++ traits from B_M;
    # these values only drive the generated symbol name, so keep them honest.
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256,            # BLOCK_SIZE
        bm, bn, bk,     # BLOCK tile
        t_m, t_n,       # T_M, T_N (4-wave warp-spec; tileN=1,2 / tileM=2,1)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, 128, 128,    # GROUP_M=1 (per-token), GROUP_N=GROUP_K=128
        "a8w8_mxscale_bmm_flatmm_splitk",
        # Single <fp32_t> host instantiation: the launcher is templated on D_C
        # only to satisfy the codegen host-decl machinery; its body ignores D_C
        # and branches on Y.dtype() at runtime (native __bf16/float), exactly
        # like the hand-written _impl. The fp32 split-K workspace dtype is fixed
        # inside the traits, and the reduce kernel casts to the runtime Y dtype.
        ["fp32_t"],
        wg_per_cu,
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


# fp8 e8m0 mxscale BMM flatmm split-K tiles. kid numbers preserved from the old
# opus_bmm.cu switch so existing tuned CSVs / heuristics keep working. Each kid =
# (B_M, B_N, B_K, WG_PER_CU, direct_only, prefetch_scale). Big-tile pipelines
# (mouter / minterleave / wave*n* / pipeline, kids 131/132/134/140-163/149-152)
# stay monolithic in opus_bmm.cu and are NOT migrated here.
_BMM_MXSCALE_SPLITK_TILES = {
    # tileN (B_M=16): single 16-row MFMA M-wave so small-M/decode shapes (M<=32)
    # don't over-compute a fat B_M tile. Targets the G=2 K=4096 M<=32 gap vs bf16.
    316: (16,  32,  256, 2, False, False),
    317: (16,  32,  256, 2, False, True),    # scale prefetch
    318: (16,  32,  128, 2, False, False),
    # prefetch-depth sweep: higher WG_PER_CU shrinks per-WG LDS -> shallower
    # prefetch_k_iter + more occupancy (small-M/few-tile shapes want this).
    319: (16,  32,  256, 4, False, False),
    314: (16,  32,  512, 2, False, False),   # fewer K-iters (8) per WG
    # wider-N tileN: larger B_N raises COM_REP_N (more MFMA/iter) to hide
    # ds_read+scale latency; WG_PER_CU keeps prefetch_k_iter >= 3.
    313: (16,  64,  256, 2, False, False),   # COM_REP_N=2
    312: (16, 128,  256, 1, False, False),   # COM_REP_N=4
    # M=16/32 last-mile (G=2 N=1024 K=4096): 311 = wide-K tileN + scale prefetch;
    # 321/323 = 32x32 tileM (exact M=32 fit, no OOB waste, COM_REP_N=2).
    311: (16,  32,  512, 2, False, True),
    321: (32,  32,  256, 2, False, True),
    323: (32,  32,  128, 2, False, True),
    # fine tiles (small / mid M)
    320: (64,  32,  256, 2, False, False),
    322: (64,  32,  256, 1, False, False),
    # kid324 = kid320 tile + SFA+SFB scale panels preloaded into LDS
    # (PRELOAD_SF_LDS; wired via the preload-tiles dict below, not the 6-tuple).
    # ATT on kid320 showed ~20% of consumer cycles stalled on vmcnt for the
    # per-K-tile global scale load; staging both panels into LDS once (ds_read /
    # lgkmcnt) breaks the mid-M valley: G4 K4096 M256 0.93->1.00x, M512
    # 0.94->1.01x, M192 0.91->0.98x vs bf16 (+8-26% TFLOPS over kid320, M128-1024).
    # Other attempts (scaleprefetch, B_K=128/512, wg4, 64x64 splitK) all <= kid320.
    640: (32,  64,  256, 2, False, False),
    642: (32,  64,  256, 1, False, False),
    646: (32,  64,  256, 2, True,  False),   # consumer self-load (splitK==1)
    650: (64,  64,  128, 2, False, False),
    653: (64,  64,  128, 2, False, True),    # scale prefetch
    # No 64x64x256 kid: mirroring bf16's MT64x64x256 forces wg_per_cu=1 (LDS
    # ~198KB), so at M=256 it runs half the WGs and lands 0.77x vs bf16. bf16 only
    # wins it via stream-K (refills low tile count), which the flatmm pipeline lacks.
    128: (128, 128, 128, 1, False, False),
    137: (128, 128, 128, 1, False, True),    # scale prefetch
    138: (64,  128, 256, 1, False, False),
    139: (128, 64,  256, 1, False, False),
    # baseline tiles (guaranteed-runnable fallbacks; kid 0 is the heuristic default)
    256: (32,  256, 128, 1, False, False),
    64:  (64,  128, 128, 2, False, False),
    0:   (32,  128, 128, 2, False, False),
    32:  (32,  128, 128, 2, False, False),
}
a8w8_mxscale_bmm_flatmm_splitk_kernels_list = {
    kid: _a8w8_mxscale_bmm_flatmm_splitk(bm, bn, bk, wg, direct, prefetch)
    for kid, (bm, bn, bk, wg, direct, prefetch) in _BMM_MXSCALE_SPLITK_TILES.items()
}

# SFA/SFB-into-LDS preload variants (PRELOAD_SF_LDS). Kept in a separate dict so
# the base 6-tuple stays untouched; each entry is (B_M, B_N, B_K, WG_PER_CU) and
# always sets preload_sf=True (non-direct, non-prefetch).
_BMM_MXSCALE_SPLITK_PRELOAD_TILES = {
    324: (64, 32, 256, 2),  # = kid320 + SFA/SFB scale panels preloaded to LDS
    # mid-M wg1 tiles + SFA/SFB preload (same mechanism as kid324/kid158): staging
    # both scale panels into LDS removes the per-K-tile global scale vmcnt load that
    # gated the plain/scaleprefetch tiles. On K=4096 M256-2048 this wins +13-17%
    # over the old kid137/653/139 picks (kid325 ships G2/M2048, G4/M1024, G8/M512,
    # G16/M256; kid326 ships G8/M256). K=1024 gains are ~noise (few K-tiles). kid327
    # kept as a candidate but wins nothing robustly (clock-fragile at cold sclk).
    325: (128, 128, 128, 1),  # = kid128/137 tile + preload
    326: (128, 64,  256, 1),  # = kid139 tile + preload
    327: (64,  128, 256, 1),  # = kid138 tile + preload
}
a8w8_mxscale_bmm_flatmm_splitk_kernels_list.update({
    kid: _a8w8_mxscale_bmm_flatmm_splitk(bm, bn, bk, wg, preload_sf=True)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_SPLITK_PRELOAD_TILES.items()
})

# --- shuffle_scale on this family: a retired measurement, kept for its verdict ----
#
# The flatmm split-K kernel has implemented SHUFFLE_SCALE all along and the codegen
# spells it on every kid of this kernel, but no instance had ever set the flag, and
# _name had no suffix for it -- so one would have deduplicated onto its plain
# sibling and been emitted as the plain kid, measuring as "the layout changes
# nothing" with nothing raised. That is fixed (see the sfshuf note in _name).
#
# Six instances then priced the layout here and are retired now that it has come out
# no; the numbers are below because the axis reversed twice under better measurement, and
# the two mis-framed comparisons that caused it are the reusable part. 328/329/330 are
# the three tiles as first measured, bit-exact against their plain twins; twin vs twin
# over the 77 cells of the plain tuned table, 3 draws:
#
#   kid328 vs kid320 (same tile, no panel)   1.099x  76/77 won
#   kid328 vs kid324 (same tile + preload)   0.893x   9/77 won
#   kid329 vs kid325 (same tile + preload)   0.705x   0/77 won
#   kid330 vs kid326 (same tile + preload)   0.748x   0/77 won
#
# -- but that first pass forgot PREFETCH_SCALE, which is the axis the two mechanisms
# are actually competing on: it moves load_scale_regs ahead of the lgkmcnt wait, and it
# carries no static_assert against SHUFFLE_SCALE, so the combination was instantiable
# all along. kid331/332/333 are the same three tiles with it on, and it is worth 1.139x
# to the layout, which changes the verdict at these tiles from a loss to a tie:
#
#   kid331 vs kid328 (the prefetch axis alone)  1.139x  69/77 won
#   kid331 vs kid324 (prefetched vs the panel)  1.013x  66/77 won
#   kid331 vs kid320 (vs the naive load)        1.268x  74/77 won
#   kid332 vs kid325                            0.714x   0/77 won
#   kid333 vs kid326                            0.913x   0/77 won
#
# kid332 is the informative failure: prefetch moved it 0.705 -> 0.714, i.e. not a
# latency problem. It is the only one with COM_REP_K==1, where load_scale_regs issues
# load<1>(g_sfa_shuf_h, 2*w + kp) every K tile against the two halves of one dword --
# two loads where one would do. Loading the dword once per two tiles needs the K parity
# in op_sel, which is an instruction immediate, so it has to be compile-time; today it
# rides in the address instead. The main K loop's parity is static (it steps by 2 from
# a compile-time start), the tail and epilogue are where it is not. Not implemented.
#
# So at the tables' K the layout is a wash on its best tile rather than a loss, and the
# pool-level answer is that it buys nothing: best shuffle_scale kid against best
# plain-scale kid, both sides drawing the whole pool, is -0.80% at K=1024, -4.25% at
# K=4096 and -5.74% at K=8192, 4 of 36 cells better by >1% and none at K=8192. (Read
# with kid334/335 in the pool; without the prefetch axis the same sweep reads -0.43% /
# -5.97% / -10.82%, so half the apparent decay with K was the missing prefetch on the
# bdirect twins rather than the layout.) Three of the 36 cells -- g2/m16, at -18% to
# -56% -- are not measuring the layout at all: no shuffle_scale kid has a 16-row tile,
# so they go to kid179/kid243/kid236 and record a coverage gap. The medians are robust
# to them either way.
#
# What decay is left is the panel amortizing: PRELOAD_SF_LDS pays a fixed one-shot fill
# and then reads from LDS, so more K tiles amortize it better, while the shuffled
# layout's cost is per tile and flat.
#
# Past K=8192 the panel does not fit, 21 of 99 kids stop dispatching, and at splitK=1
# the fastest kid at K=16384/32768 is a shuffle_scale kid in all 10 cells -- which is
# an artefact of pinning splitK=1. The bound is per split (the launcher checks
# ceil(total_iters/split_k) <= SF_PRELOAD_K_MAX/B_K), so a plain-scale preload kid
# reaches K=32768 at split_k>=4 today. Swept over split_k in {1,2,4,8}: median -3.1%,
# 0 of 10 cells better by >1%, kid194 taking 6 of 10. g2/m256/K16384 alone moves from
# "kid331 at 36.0us is fastest" to kid326 at sk4 in 26.6us.
#
# That verdict does not extend past these shapes, and the paragraph that used to stand
# here wrongly closed the K-bound question with it. Every shape above leaves the machine
# half empty -- g2/m4096, the largest, is 128 workgroups at B_M=256 against 256 CUs -- so
# split-K was partly buying parallelism the shape lacked. On shapes that fill it
# (g16/m4096 = 2048 WGs) at K=16384/32768, split_k=1 wins 6 of 6 cells and no panel kid
# dispatches at split_k=1, so kid213 takes every cell with the best panel kid 1.068-1.144x
# back. The kid x split_k grid attributes that to the column, not the layout: wherever both
# families reach the same split_k, the panel kid is 0.75-0.83x the shuffle_scale kid.
#
# And the bound is one constant set by the worst tile rather than each kid's own. The panel
# is (SFA rows + SFB rows) * K/GROUP_K with SFA rows == B_M, so it varies 8x across kids at
# equal K; .group_segment_fixed_size from the built code objects puts 19
# of 25 panel kids at 3-884x spare -- kid205 could hold 111,360 of per-split K, kid194
# 30,976, kid208 (mpack_sfa, so only 2 SFB rows in LDS) 7.2M. The 151,680 figure the traits
# comment cites is kid158/196's, whose staging is the 2*(B_M+B_N)*B_K double buffer, not the
# flatmm/wave8 families the constant also governs; only 158/196/228/230/324/326 are really
# full. So chunked refills are forced for 6 kids, and the wave8 traits now derives its bound
# from the LDS its staging leaves over instead of copying the flat 8192 -- 32768 for the
# B_M=128 kids, 30,848 for kid194. That is worth 3.0-4.0% at K=16384 and split_k=1 on
# machine-filling shapes (8/8 paired draws, bit-exact), and changes no shipped row, since
# this table is K in {1024, 4096} at splitK=1. K=32768 still wants a chunked refill: a
# 256x256 tile needs 66,048 panel bytes against 62,208 of headroom.
#
# kid331 is also not the answer at large K: it ranks 14th-21st of 76 there, 64x32x256
# being too small a tile at large M, and the four-kid sweep that first suggested it
# (1.3-1.7x over kid320) only looked convincing because all four were small or mediocre
# tiles.
#
# The one lever left unpulled is the COM_REP_K==1 amortization above, and its value is
# unmeasured rather than small: the ~2% estimate here counted u16 loads against the tile's
# ~86 buffer_loads, and the bdirect twins show a 25% spread between tiles, but kid215
# (B_K=256, COM_REP_K=2, still losing 7-13%) shows COM_REP_K is not what that spread
# tracks. See the kid334/335 entry for the pairs that would separate the terms.
#
# The flatmm half of the axis is therefore closed. To reopen it, restore the dict below
# and _BLDS_NO_TWIN's reference to it; the factory keeps its shuffle_scale kwarg for
# that and asserts the kernel's static_asserts on the Python side.
#
# _BMM_MXSCALE_SPLITK_SHUFSCALE_TILES = {   # (B_M, B_N, B_K, WG_PER_CU, prefetch_scale)
#     328: (64,  32,  256, 2, False),   # = kid320/324 tile, kid216's geometry
#     329: (128, 128, 128, 1, False),   # = kid325 tile, the plain table's workhorse
#     330: (128, 64,  256, 1, False),   # = kid326 tile
#     331: (64,  32,  256, 2, True),    # the same three, scale load hoisted ahead
#     332: (128, 128, 128, 1, True),    # of the LDS wait
#     333: (128, 64,  256, 1, True),
# }


# --- Removed bpreshuffle experiments -------------------------------------------
#
# The families below were built to isolate one axis each, all reached a verdict,
# and all are strictly dominated by a surviving kid at every shape measured (g in
# {2,4,8,16} x m in {16..32768} x n1024 x k{1024,4096}). They are gone from the
# catalog; the verdicts are kept here because each one closes off a direction.
#
#   kid170/176  bpreshuffle, B via LDS + op_sel scales. The original preshuffle
#               points on kid320/kid325's tiles. Both land within noise of their
#               row-major twins, which is the useful result -- the 16x16 shuffle
#               and the op_sel byte select are free -- but the tile itself is 2x
#               off the wave-grid families at large M.
#   kid177      bcast: B_PRESHUFFLE without SCALE_OPSEL, the third point that
#               split kid176's two axes. Measured within 0.5% of kid176, so the
#               op_sel select costs nothing and neither axis explains any gap.
#   kid178/180/ bdirect on 128x64, and 128x128 without the scale panels. Two
#   185         consumer waves is too few for tiles this wide: at m<=256 they sit
#               at 17-38 us where the 64-row kid172 needs 10-20 and the 16-row
#               kid179 7-10, and at m>=4096 the wave-grid families are 1.4-2.3x
#               ahead. Unlike kid184 they do not win the m=512..2048 band either
#               -- kid178 comes closest at g2/m512/K=1024 and is still 1.13x of
#               the row-major kid653. (kid171 and kid184, once grouped here, are
#               back; see their notes below.)
#   kid190/191  sfmpack: the A scale panel packed along M *inside LDS*, turning
#               4 ds_read_u8 into 1 ds_read_b32. It shortens the steady-state
#               loop (173 -> 161 instructions) and buys nothing, because the loop
#               is paced by MFMA between two barriers, not by instruction count.
#               Not to be confused with kid198's host-side packing below.
#   kid187      allwave: four all-consumer waves on a 2x2 grid sharing B through
#               LDS. 9% behind the two-consumer form -- splitting a tile further
#               cuts MFMA per wave without removing any wait.
#   kid189/197/ allwave_bdirect, 2x2 with direct B. kid189 and kid197 differed
#   kid201      only in WG_PER_CU and measured identically (948.8 us both), which
#               is itself the result: WG_PER_CU does not gate this pipeline.
#               kid201 doubled B_N and moved it 0.6%. The family's ceiling is its
#               2x2 grid, which its traits fix: T_M=2 makes both M-waves fetch
#               the same B from global with no LDS to share it through, so B
#               crosses L1 twice per K tile. kid203 runs the same 128x256 tile
#               and the same four waves on a 1x4 grid and that alone takes
#               MfmaUtil from 22.8% to 48.5% (1.85x). L1 tag-conflict stalls:
#               17.8M cycles here, 0 there.
#   kid188      wave4m2_bdirect, two stacked M phases. Same verdict as kid189 --
#               it already matched FlyDSL on per-wave wait cycles and non-MFMA
#               instructions per MFMA, so its 1.7x gap was never the wave/register
#               structure; it was the workgroup -> tile map, which is what the
#               XCD swizzle on kid205 addresses.
#   kid192/193  wave8, the T_M=4 end of the T_M sweep, at 256x256 and 256x128.
#               The 4x2 grid is superseded by 2x4/1x4 on every tile, and the
#               sweep's conclusion outlived the kid that anchored it: kid192 won
#               none of the 116 shapes and ran 1.14-1.39x the best kid on all six
#               of the large-M shapes its 256x256 tile was built for. The one
#               thing it showed that its LDS-B siblings do not is that direct-B
#               pays at short K (g2/m8192/k1024: 31.5us against kid196's 39.1 and
#               kid158's 38.5) -- but 2x4 direct-B takes the same cell in 27.6, so
#               what T_M=4 costs is the grid and not the B path. That argument now
#               lives on wave8n4, which is the sweep's shallow end.
#   kid193      see kid192; same grid at 256x128.
#   kid165/174  wave8n4 at 256x128 and 128x128, both dominated by kid168/175.
#   kid195      wave8n8: 1x8 with A streamed from LDS at B_M=256. Halving B_M
#               instead of B_N is what makes T_M=1 pay, because that is what gets
#               A back into registers -- which is the wavetm1 family, and it is
#               the fastest thing here. The wave8n8 traits and the whole streamed
#               MMA path went with it; what that path had to teach is that cover
#               over an LDS read is counted in MFMAs and a buffer costs a group's
#               registers, so many small groups beat few large ones. Four subtiles
#               a group put three buffers at 96 registers, which left the
#               allocator nothing: it folded every group onto one buffer and
#               emitted a full lgkmcnt(0) per pair of reads, and a thread trace
#               showed 29 cycles of stall per LDS read against kid158's 7 at a
#               comparable read count -- a serialized dependence chain, not a
#               bandwidth wall. One subtile a group is 8 registers and two MFMAs,
#               so four buffers cost 32 and still ran six MFMAs ahead of the read
#               they covered; six buffers were worse again.
#   kid198      sfgmpack: the A scale panel packed along M by the *host* and read
#               straight from global. Bank conflicts go to zero exactly as
#               intended (LdsBankConflict 0.884 -> 0.000) and it still loses 8.8%,
#               because it costs one more VMEM instruction per K tile and ~89% of
#               kid194's stalls are already on that pipe.
#   kid204/206/ wavetm1 XCD band heights 2 and 8, and the 1x8 form at band 4. The
#   kid207      band height barely matters (within 1% across 2/4/8); kid205 keeps
#               FlyDSL's 4. The 1x8 forms lose to their 1x4 siblings at large M.
#   kid224/225  bdirect 32x32x256 and 64x64x128, tried on the theory that the
#               mid-M losses to the row-major family were a missing geometry. The
#               theory is untested: both kids measured ~7.8us at every shape and so
#               did all five blds kids on their first pass, five different tiles
#               agreeing to 0.1%, because none of them had been compiled. A kid
#               with no emitted instance silently runs a different kernel and
#               passes the correctness gate, and the JIT does not rebuild on a
#               codegen edit unless AITER_REBUILD=1 (see the blds note below --
#               adding a tag means three hand-maintained lists in gen_instances.py
#               on top of the five gfx950 maps, and missing them is silent). What
#               actually closed those cells was blds, which says the losses were
#               B's path and not its tile, so these two were not brought back.
#               Both ids stay retired rather than being handed to something else.
#
# What survives, and the region each one owns:
#
#   kid172   bdirect 64x32x256 + panels     -- fastest bpreshuffle kid at m<=128
#   kid175   wave8n4 128x64x256             -- best in the 8-wave family at m<=256
#   kid168   wave8n4 128x256                -- mid M
#   kid194   wave8n4 256x256                -- the 8-wave large-M tile
#   kid202   wavetm1 1x8 (BLOCK_SIZE=512)   -- the 8-wave T_M=1 control
#   kid203   wavetm1 1x4, linear map        -- the XCD swizzle's control
#   kid205   wavetm1 1x4, XCD band 4        -- fastest overall at large M
#   kid196   pipeline_bpreshuffle           -- prices the B shuffle against kid158
#
# -------------------------------------------------------------------------------


def _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg_per_cu, prefetch_scale=False,
                                          preload_sf=False, shuffle_scale=False):
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
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256,            # BLOCK_SIZE
        bm, bn, bk,     # BLOCK tile
        t_m, t_n,       # T_M, T_N (4-wave warp-spec; tileN=1,2 / tileM=2,1)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, 128, 128,    # GROUP_M=1 (per-token), GROUP_N=GROUP_K=128
        "a8w8_mxscale_bmm_bpreshuffle_bdirect",
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.prefetch_scale = prefetch_scale
    inst.preload_sf = preload_sf
    inst.shuffle_scale = shuffle_scale
    return inst


# kid172: kid170's 64x32x256 geometry with B read straight from global into
# registers, plus the SFA/SFB LDS scale panels. Dropping B from LDS frees ~25 KiB
# per WG; spending that on pipeline depth or occupancy did not pay (4 prefetch
# slots and WG_PER_CU=3 both measured ~25% slower -- occupancy is VGPR-bound at
# 2 waves/SIMD, not LDS-bound). The panels are worth a further 1-6% on this tile.
#
# The two mechanisms are independent and additive: do_scaled_mma waits on vmcnt
# for B and lgkmcnt for the panels separately, and against the plain tile
# (kid128, 176.75us at g2 N1024 K4096 M8192) direct B alone is -18% and the
# panels alone are -34%.
#
# This owns the m=64..256 band at K=4096 (kid171, the same tile without the
# panels, owns it at K=1024). Below that the 16-row kid179/kid173 take over, and
# above it the all-wave families do: with two of four waves staging, a
# compute-bound tile only gets two MFMA waves, so every larger tile in this family
# is 1.4-2.3x behind them. The tiles that chased large M (kid178/180/184/185) are
# gone -- see the removed-experiments note above.
# kid179 and kid173 are the 16-row tiles: every other preshuffled kid computes at
# least 64 rows, which on a decode shape is 4x the rows there are. That is
# measurable and it is what the tuner ships kid311 (the row-major 16x32x512) for
# -- against kid172 at g2..g8 / m<=16 / k4096, kid311 runs 9.5-10.8us where
# kid172 needs 11.8-13.1, a 22-26% gap no wider tile closes.
#
# Each consumer wave owns its own 16 columns, which the direct-B layout absorbs
# as a base offset. Spending the LDS this frees on occupancy loses badly (wg4
# measured 20us vs 8us at wg2), B_K=512 beats 256 wherever 512 is legal, and
# B_N=64 loses to 32. The
# scale panels are a loss here too (g2/m1: 8.17 vs 7.57; g2/m256: 18.34 vs
# 16.49): on a 16-row tile their setup is not amortised, and the direct-B vmcnt
# wait already retires the plain scale loads.
#
# kid173 is the same 16-row tile at B_K=256, and it exists because B_K=512 is not
# always available: K=1024 leaves the split-K tile too few K-tiles and kid179 --
# along with its row-major twin kid311 -- is rejected outright there. Without a
# 256-deep entry every K=1024 cell below m=256 fell to the 64-row kid172, 4x a
# decode shape's rows, which cost 1.20-1.33x against the row-major kid319/321 on
# 21 cells. kid173 closes that: at g2/m16/K=1024 it is the fastest kid measured,
# 3.84us against kid319's 3.88 and kid172's 5.28. WG_PER_CU=4 was measured
# alongside and lands within 0.03us at every shape, so only wg2 is kept -- the
# wg4-is-much-worse note above was taken at B_K=512, i.e. twice the LDS a stage.
# Where B_K=512 does run it still wins by a wide margin (g2/m16/K=4096: kid179
# 7.35 vs kid173 9.43), so the two split the K axis rather than one replacing the
# other.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)
    179: (16, 32, 512, 2),
    173: (16, 32, 256, 2),
    # kid171 is kid172's tile without the scale panels, and those two split the K
    # axis for the same reason: the panels earn their setup back at K=4096 (kid172
    # 10.68 vs 11.39us at g2/m16, 12.02 vs 12.96 at g2/m256) and lose at K=1024,
    # where there is 4x less K to amortise them over (kid171 6.04 vs 6.88us at
    # g8/m64, 5.94 vs 6.76 at g2/m256, beating the row-major kid321 as well).
    171: (64, 32, 256, 2),
}
a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_TILES.items()
}

_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_PRELOAD_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)
    172: (64, 32, 256, 2),
    # kid184 is kid325's tile -- the row-major mid-M winner -- with B read direct
    # from global, and it owns m=512..2048 the way kid172 owns m=64..256: 0.93-0.99x
    # of kid325 there (g8/m512/K=4096 33.12 vs 34.80us, g4/m1024/K=1024 14.26 vs
    # 15.15), which is 6 of the cells the family otherwise lost by 1.10-1.21x.
    # It is only good in that band. Below it the tile is far too wide for two
    # consumer waves (27-29us at m<=256 against kid172's 10-20), and above it the
    # wave-grid families are 1.4-2.3x ahead, so it is bracketed on both sides.
    184: (128, 128, 128, 1),
}
# 128x128x256 has no home in this file, and the two ways it fails are worth
# recording because the tile looks reasonable on paper (128 FLOP/byte, between
# kid184's tile and the wave-grid families'). Staged, it wants (128+128)*256 =
# 64KiB a prefetch stage and the pipelines assert three stages, i.e. 192KiB
# against the 160KiB a CU has. Direct-B drops B's stage, so the LDS fits -- but
# then a 128-column B fragment 256 bytes deep lives in VGPRs alongside a 128x128
# fp32 accumulator, and instead of failing an assert the TU simply does not
# compile: six such TUs were left >16 minutes without one finishing, in a module
# whose slowest other TU is 6 seconds. That is why the B_K=256 tiles stop at
# 128x64 / 64x128 and everything wider is B_K=128.
a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg, preload_sf=True)
    for kid, (bm, bn, bk, wg)
    in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_PRELOAD_TILES.items()
})


def _a8w8_mxscale_bmm_bpreshuffle_blds(bm, bn, bk, wg_per_cu, prefetch_scale=False,
                                       preload_sf=False):
    """Preshuffled B that still goes through LDS: bdirect with B's path put back.

    Same traits family as bdirect with B_DIRECT_REG left false, so the producer
    waves stage B into LDS and the consumers ds_read their fragments out of it,
    exactly as the row-major flatmm kids do. Only B's byte order and the MFMA
    scale_op_sel select differ from the plain kid of the same tile. Callers still
    pass shuffle_weight(w, (16, 16)).

    This is the control the family was missing, and it is what separates the two
    things the bdirect tag changes at once. Preshuffling B does not make B's global
    read faster -- the row-major read is already fully coalesced, since the kernel
    reads B in whatever chunks it likes and fixes the order on the way into LDS --
    it makes the LDS hop unnecessary. Measured on the one same-pipeline pair that
    already existed, kid158 against kid196, flipping only the layout is a wash:
    median -0.2% over all 116 shapes of the dsv4 envelope, 100 of them inside
    +-1%. So a kid here should track its plain twin, and every bit of bdirect's
    gain (and of its seven mid-M losses) belongs to skipping LDS instead. It came
    out slightly better than that: a twin runs 1-2% ahead of its plain kid, which
    is why the family is twinned in full rather than only where it was needed.

    That holds only where the staged load can be re-mapped to contiguous issues,
    which is b_preshuffle_contig_mxsk in the pipeline header. The B_M=16 tiles are
    all tileN (T_N=2) and the predicate used to want T_N==1, so all eight of their
    twins fell back to issuing the preshuffled bytes through the row-major
    thread -> (n, k) mapping -- 16 B out of each of 16 cache lines per issue. They
    were not 1-2% ahead of their plain kids but 8-93% behind, and the shape of that
    loss is worth recording because the size of it is not obvious from the mapping
    alone. Enumerating both layouts' addresses over a wave gives 16 lines per issue
    scattered against 8 contiguous, for the same 1024 B of payload: a 2x line-touch
    rate, not 2x the bytes. A rate penalty only shows up once those touches are what
    the kernel is waiting on, and B_M=16 gets there faster than any other tile,
    since M/16 workgroup rows each re-read the whole of B. So the loss tracks the M
    tile count and converges on the 2x, which is what kid238 against kid313 at
    g8/K=4096 does: 0.79x the plain time at m64 (4 M tiles), 1.08x at m128,
    1.66x at m256, 1.93x at m512 (32).
    Nothing in either layout was actually T_N-dependent (see the predicate's own
    note) and dropping the term puts them on the contiguous path: the same four
    cells become 0.78x, 1.05x, 1.02x and 0.98x, and across eight tiles x ten shapes
    the family now straddles zero instead of falling off it.
    That bought nothing on the dsv4 envelope, which is worth knowing before
    spending anything here again: median went +10.6% -> +10.0% and no cell's best
    preshuffled kid is a B_M=16 twin either way, because bdirect (kid173/kid179)
    reads the same tiles without LDS at all and is never worse. The value is that a
    tile the tuner can reach is no longer a trap -- it now costs a few percent to
    pick instead of 93% -- and picking it is what the tuner does on 11 of 133 rows.
    """
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256,            # BLOCK_SIZE
        bm, bn, bk,     # BLOCK tile
        t_m, t_n,       # T_M, T_N
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, 128, 128,    # GROUP_M=1 (per-token), GROUP_N=GROUP_K=128
        "a8w8_mxscale_bmm_bpreshuffle_blds",
        ["fp32_t"],
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.prefetch_scale = prefetch_scale
    inst.preload_sf = preload_sf
    return inst


# One twin per plain flatmm tile, keyed plain kid -> twin kid.
#
# Only the ids are written down here; the geometry and both flag axes are read off
# the plain kid below, so a pair cannot drift into differing by anything but B's
# layout. Ids have to be explicit rather than derived from position because a tuned
# CSV row names a kid, and a twin whose number moved when someone inserted a plain
# tile would silently point a shipped table at a different kernel.
#
# The set is complete over the family by assertion, not by judgement. The first cut
# held only the five tiles that beat the preshuffled families on the dsv4 envelope,
# which left the same gap one shape-set change away from reopening: a plain tile
# with no twin means a preshuffled deployment has nothing to answer that cell with,
# and nothing would have flagged it.
_BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF = {
    # the five measured against their plain twin first (see the note below)
    321: 226,
    653: 227,
    324: 228,
    325: 229,
    326: 230,
    # the rest of the family, so the pool is closed. 224 and 225 are skipped: they
    # named the two withdrawn bdirect experiments below, and an id that has meant
    # two different kernels makes any table row carrying it ambiguous.
    32: 251,
    64: 252,
    128: 231,
    137: 232,
    138: 233,
    139: 234,
    256: 235,
    311: 236,
    312: 237,
    313: 238,
    314: 239,
    316: 240,
    317: 241,
    318: 242,
    319: 243,
    320: 244,
    322: 245,
    323: 246,
    327: 247,
    640: 248,
    642: 249,
    650: 250,
}
a8w8_mxscale_bmm_bpreshuffle_blds_kernels_list = {
    twin: _a8w8_mxscale_bmm_bpreshuffle_blds(
        plain.B_M, plain.B_N, plain.B_K, plain.WG_PER_CU,
        prefetch_scale=plain.prefetch_scale,
        preload_sf=plain.preload_sf,
    )
    for plain, twin in (
        (a8w8_mxscale_bmm_flatmm_splitk_kernels_list[plain_kid], twin_kid)
        for plain_kid, twin_kid in _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF.items()
    )
}

# kid646 is the DIRECT_ONLY persistent schedule, which carries its own B staging
# and rejects the flags this family sets. kid0 is the heuristic default and an
# alias of kid32's geometry, so kid224 already covers its tile.
#
# A shuffle_scale flatmm kid would be exempt for a third reason: it wants a host-side
# scale relayout, so it is not dispatchable under the layout the tables ship and cannot
# win a shipped cell either way. None exist right now (the kid328-333 block above is
# retired), but restoring that dict means adding its ids here too, since promoting one
# into a table would need a preshuffled-B twin, i.e. a blds kid carrying shuffle_scale,
# which does not exist.
_BLDS_NO_TWIN = {0, 646}
_blds_untwinned = sorted(
    kid
    for kid in a8w8_mxscale_bmm_flatmm_splitk_kernels_list
    if kid not in _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF and kid not in _BLDS_NO_TWIN
)
assert not _blds_untwinned, (
    f"plain flatmm kids {_blds_untwinned} have no preshuffled-B twin. A preshuffled "
    "deployment cannot dispatch a row-major kid, so any cell one of these wins is a "
    "cell preshuffling B costs performance on. Add an id to "
    "_BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF, or to _BLDS_NO_TWIN with the reason."
)

# kid216: the 64x32x256 tile again with both scale panels in the shuffle_scale layout, i.e.
# one twin standing against both kid171 (no panel) and kid172 (panel), since the
# layout replaces the panel rather than filling it.
#
# This is the tile where the layout should cost nothing to adopt, and it is the
# reason to try this family at all. B_K=256 means COM_REP_K=2, so the dword's two
# 128-blocks of K land in the same K tile and op_sel's high bit is just the K
# repeat -- no unrolled loop parity, unlike the B_K=128 wave8 kids. COM_REP_M=2
# on top of that makes one dword the tile's entire A scale: the plain path issues
# COM_REP_M=2 buffer_load_b32, one per M subtile, whose 16 lanes sit on 16
# different 64B lines at stride_sfa apart; the shuffle_scale path issues one, coalesced.
# Half the loads and a sixteenth of the lines, for identical bytes.
#
# The B_M=16 tiles (kid179/kid173) cannot take the layout at all: a dword pairs
# two M subtiles and those tiles are one subtile tall, so there is no pair to
# make. If the producer ever emits the shuffle_scale layout globally, they need a different
# answer -- either a 32-row tile or a scalar unpack.
#
# What it does is collapse the kid171/kid172 K split into one kid. Those two are
# the same tile with and without the panels, and they divide the K axis because
# the panels earn their setup back at K=4096 and lose it at K=1024. kid216 is at
# the better of the pair in every cell of the band, because it gets the panel's
# access shape without the panel's setup (us, medians of four runs, each the min
# of both kid orders; every run agreed on the sign):
#
#     shape             | kid171  kid172  kid216 | vs best of the two
#     g8  m256   k4096  |   29.8    27.2    26.9 |  -1.1%
#     g8  m512   k4096  |   53.8    48.7    49.0 |  +0.6%
#     g8  m512   k1024  |   21.1    23.0    21.1 |   0.0%
#     g8  m1024  k4096  |  101.9    82.1    87.8 |  +6.9%
#
# At m<=256 with K=4096 and everything at g2, all three sit on the ~18.2-18.7us
# fixed-cost floor and the differences there are not readable.
#
# The two ends of that table are the whole mechanism. Against kid171 -- the
# sibling with no panel, whose scale reads are the uncoalesced ones -- shuffle_scale is
# 8.5-14% ahead at K=4096, which is the coalescing alone and the same effect the
# kid208/kid210 pair measured. Against kid172 it is level until m1024, where the
# panel pulls 7% back: by then there is enough M x K to amortise a one-shot
# cooperative fill, and a fill amortised over that much work beats re-reading the
# words per K tile however well those reads coalesce. That is the same crossover
# the wave8 twins show, just reached from the other side.
#
# That m1024 cell is not one to read as a loss, though, because neither kid in
# the pair belongs there: a 64-row tile needs 16 rounds at m1024 and both run
# about 2x off the best kid. Whom the cell actually goes to, and what the whole
# envelope does across the band above kid172's, at g8/n1024/k4096 (us, min of
# both kid orders):
#
#     kid                      | m512   m1024  m2048
#     172  bdirect 64x32x256   |  49.2   82.3  151.8
#     216  shuffle_scale     64x32x256   |  49.5   88.8  166.6
#     184  bdirect 128x128x128 |  30.7   57.8  107.7
#     175 / 215  128x64x256    |  32.5 / 35.0   60.7 / 66.6   111.3 / 113.3
#     205 / 210  128x256x128   |  39.1 / 38.5   41.4 / 40.9    59.2 / 59.2
#     194        256x256x128   |  53.4   55.5   58.2
#
#     best plain               |  30.7   41.4   58.2
#     best shuffle_scale                 |  35.0   40.9   59.2
#                              | +14.0%  -1.2%  +1.7%
#
# So the m1024 cell is fine -- it goes to kid210, which is itself a shuffle_scale kid and
# takes it by a hair -- and what the band leaves open is m512, which kid184 owns.
# kid217 is that twin and does not close it; see its entry for why, and why the
# answer there is a shuffle_scale-fed panel fill rather than another twin. The gap is also
# narrower than one cell of one shape suggests: at m512/K4096 it is g8 only
# (30.7 kid184 against 34.9 kid215, +13.7%), because g2 goes to kid172 and g16 to
# kid205, and both of those have twins that match them.
#
# The kid is kept on the same terms as kid213/214/215: measurement, not a tuner
# candidate, until a producer emits the layout.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_SHUFFLE_SCALE_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU, prefetch_scale)     twin of
    216: (64, 32, 256, 2, False),    # kid171 / kid172
    # kid217: kid184's tile, built to close the one cell where the shuffle_scale envelope
    # gave anything up -- g8/m512/K4096, which kid184 owns -- and it does not
    # close it. COM_REP_K=1 here, so a tile owns one of the dword's two K blocks
    # and the read takes the u16 half whose block it is (see the pipeline note);
    # the twin is a fair one and still loses (us, min of both kid orders):
    #
    #     shape             | kid184  kid217
    #     g8  m512   k4096  |   30.7    35.2   +14.7%
    #     g8  m1024  k4096  |   58.1    65.7   +13.1%
    #     g8  m2048  k4096  |  109.0   121.9   +11.8%
    #     g2  m512   k4096  |   28.9    32.2   +11.4%
    #     g16 m512   k4096  |   58.1    65.8   +13.3%
    #     g8  m512   k1024  |   19.1    18.8    -1.6%
    #
    # The K axis is the whole story and it is the kid171/kid172 curve again. At
    # B_K=128 and K=4096 a tile is one of 32, so kid184's panel is filled once and
    # read from LDS 32 times; kid217 issues 32 rounds of global reads instead, and
    # coalescing them does not make up for giving the panel away. At K=1024 there
    # are 8 tiles, the fill no longer amortises, and the twin edges ahead.
    #
    # So this cell is not a missing-twin problem: the shuffle_scale layout's gain is
    # removing the panel, and on this tile the panel is worth paying for. The way
    # to make the layout free here is not another twin but a panel fill that reads
    # the shuffle_scale bytes -- which would decouple what the producer emits from whether a
    # kid stages its scales, and is what to build if the producer switch is real.
    217: (128, 128, 128, 1, False),  # kid184
    # kid216/217 with the scale load hoisted ahead of the LDS wait. Both entries
    # above were measured with PREFETCH_SCALE off, which is not the fair setting for
    # this layout: the shuffled read is a per-K-tile global load sitting in the
    # critical path, where the panel it replaces is a ds_read already hoisted, and on
    # the flatmm family turning it on was worth 1.139x to the layout. It is only
    # testable here -- the wave8n4/wavetm1 pipeline static_asserts !PREFETCH_SCALE,
    # so kid210/213/214/215 cannot have it -- and bdirect + shuffle_scale + prefetch
    # is the one combination of the three that had never been instantiated.
    #
    # It is not what was wrong here. The axis alone is worth 1-2% on both tiles
    # (kid334/kid216 1.008x/1.012x/1.018x and kid335/kid217 1.017x/1.016x/1.012x at
    # K=1024/4096/8192, 6 shapes), nowhere near flatmm's 1.139x. What the pair does
    # show, once the layout is measured at its best, is that the verdict is per tile
    # rather than global -- against the LDS panel each replaces:
    #
    #     K=            1024    4096    8192
    #     kid334/kid172  1.108x  1.041x  1.038x   64x32x256,  wg2, cRepM=2, cRepK=2
    #     kid335/kid184  0.984x  0.805x  0.793x   128x128x128, wg1, cRepM=4, cRepK=1
    #
    # so the layout beats the panel outright on one tile and loses a fifth on the other.
    # Those two kids differ in four terms at once (B_M, B_N, B_K, WG_PER_CU), so the pair
    # attributes nothing on its own; kids 336-345 below walk the path one term at a time
    # and find B_K carrying the largest attributable share (0.907x at K=8192), B_M little
    # (0.979x), WG_PER_CU exactly nothing (1.000x), and a 0.86x residual that B_N cannot
    # be separated from. kid215 is not the counterexample it looks like: 128x64x256 in
    # another family differs in B_M, B_N, family and the prefetch it cannot have, so it
    # was never a B_K-only comparison.
    334: (64, 32, 256, 2, True),     # kid216 + prefetch
    335: (128, 128, 128, 1, True),   # kid217 + prefetch
}
a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_bdirect(bm, bn, bk, wg, prefetch_scale=pf,
                                               shuffle_scale=True)
    for kid, (bm, bn, bk, wg, pf)
    in _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_SHUFFLE_SCALE_TILES.items()
})

# --- what the 25% spread between kid334/kid172 and kid335/kid184 is caused by -------
#
# Those are the only two geometries this family has a shuffle_scale/panel pair at, and
# they differ in four terms at once: B_M 64->128, B_N 32->128, B_K 256->128 and
# WG_PER_CU 2->1. The instruction-count reading blamed B_K, and kid215 refutes it by
# being B_K=256 with COM_REP_K=2 and losing anyway. So walk the path one term at a time,
# holding the family fixed: each row below is kid334's geometry with exactly one term
# moved toward kid335's, and each needs BOTH sides of the pair -- plain + preload_sf for
# the panel, shuffle_scale + prefetch_scale for the layout at its best -- because a ratio
# is only readable against a twin that moved the same term.
#
# Measured (shuf/plain per pair, then each term against its own reference, K=1024/4096/8192):
#
#     baseline  334/172   64x32x256  wg2   1.105x 1.045x 1.037x
#     B_K->128  337/336   64x32x128  wg2   1.035x 0.982x 0.941x   term: 0.937 0.940 0.907
#     B_M->128  339/338  128x32x128  wg2   1.051x 0.936x 0.921x   term: 1.016 0.953 0.979
#     wg->1     343/342   64x32x256  wg1   1.102x 1.049x 1.037x   term: 0.997 1.005 1.000
#     endpoint  335/184  128x128x128 wg1   0.997x 0.807x 0.792x
#
#     product of the three   0.949 0.900 0.888
#     endpoint / baseline    0.902 0.773 0.763
#     residual (B_N + int.)  0.951 0.859 0.860
#
# So B_K carries the largest attributable term and the instruction-count reading holds
# for it: COM_REP_K=1 takes a u16 half of a dword per K tile while tiles are twice as
# dense, 4 loads per 256 of K against 1 for the same bytes, worth 6-9%. WG_PER_CU is
# exactly neutral, which is worth keeping as a negative. The residual is the largest
# piece at K>=4096 and stays unattributed because B_N cannot be stepped alone.
#     kid334/kid172   64x32x256  wg2   COM_REP (M,N,K) = (2,2,2), 8 MFMA
#     kid335/kid184  128x128x128 wg1   COM_REP (M,N,K) = (4,8,1), 32 MFMA
#
# so the path between them is exactly four steps, one per row here.
#
# 128x32x256 at WG_PER_CU=2 would have been the cleaner B_M step and is not legal:
# prefetch_k_iter divides the per-WG LDS budget by the A staging, which scales with B_M
# (B contributes nothing under B_DIRECT_REG, so B_N is free -- 64x128x256 has the same
# per-iteration LDS as 64x32x256), and 128 rows at B_K=256 halves the budget below the
# `prefetch_k_iter >= 3` floor. So B_M is stepped at B_K=128 instead, against the B_K
# row rather than against the baseline. Instantiating the traits alone against a
# candidate geometry checks it in a second, worth doing before a 21-minute build.
_BMM_MXSCALE_BPRESHUFFLE_BDIRECT_ISOLATE = {
    #    (B_M, B_N, B_K, WG_PER_CU)   term moved, and what it moves
    (336, 337): (64,  32,  128, 2),   # B_K  256->128: COM_REP_K 2->1
    (338, 339): (128, 32,  128, 2),   # B_M   64->128: COM_REP_M 2->4, vs (336,337)
    (342, 343): (64,  32,  256, 1),   # WG_PER_CU 2->1: occupancy only, geometry fixed
    (344, 345): (64,  128, 256, 1),   # B_N   32->128: COM_REP_N 2->8, vs (342,343)
}
# B_N has no row because it cannot be stepped alone, which took three attempts to
# establish. 64x128x256 passes the traits at both occupancies -- LDS is fine, since
# B_DIRECT_REG keeps B out of it and 64x128x256 has the same per-iteration LDS as
# 64x32x256 -- and runs at 332,000us at wg2 (kid340/341) and 105,000-133,000us at wg1
# (kid344/345), against 19us for the baseline geometry, both sides of each pair alike.
# That is the VGPR spill the family's own notes predict ("occupancy here is VGPR-bound at
# 2 waves/SIMD, not LDS-bound"): COM_REP_N=8 with COM_REP_K=2 is 32 MFMA and 16 B
# fragments per tile. B_N=128 is only viable at B_K=128, which is what every B_N=128 kid
# in this family is. So B_N cannot move without B_K moving, and the two measured corners
# differ in four terms because the geometry space forbids the orthogonal walk -- not
# because the pair was picked carelessly.
#
# Two traps worth keeping. Instantiating the traits alone against a candidate geometry
# takes a second and catches the compile-time gate, worth doing before a 21-minute
# build; it does NOT catch the spill, a register allocator outcome rather than a
# static_assert, so a new geometry wants a timing smoke check too. And a 300,000us kid
# inside a sweep is indistinguishable from a hang -- that is what it was first taken for.
# So kid340/341 are not wired at all. 344/345 are, and only as the recorded evidence
# that wg1 does not rescue B_N=128; nothing should sweep or ship them.
for (_plain_kid, _shuf_kid), (_bm, _bn, _bk, _wg) in (
    _BMM_MXSCALE_BPRESHUFFLE_BDIRECT_ISOLATE.items()
):
    a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list[_plain_kid] = (
        _a8w8_mxscale_bmm_bpreshuffle_bdirect(_bm, _bn, _bk, _wg, preload_sf=True)
    )
    a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list[_shuf_kid] = (
        _a8w8_mxscale_bmm_bpreshuffle_bdirect(_bm, _bn, _bk, _wg, prefetch_scale=True,
                                              shuffle_scale=True)
    )




def _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg_per_cu, xcd_wgm=0,
                                          mpack_sfa=False, shuffle_scale=False):
    """256x256 preshuffled-B tile over eight all-compute waves, on a 2x4 grid.

    This is the T_M sweep's shallow end and, with wavetm1's 1x8/1x4, all that is
    left of it. The axis is how many waves share an N range, which with direct-B
    is exactly how many times each of B's bytes crosses L1, since there is no LDS
    to share it through. T_M=4 was the deep end and is gone: it profiled at 2.06x
    kid158's L1 accesses with its return path 93% busy and its matrix pipe at
    only 74%, and measured 1.14-1.39x the best kid on all six of the large-M
    shapes it was built for -- including 1.14x this family's own 2x4 form at
    g2/m8192/k1024 (31.5 vs 27.6us), which is the sweep's result stated as a
    number. What T_M=4 also could not express is the half-M tile: SCALE_OPSEL
    packs four M subtiles of scale into one op_sel dword, so
    COM_REP_M = B_M/(W_M*T_M) must be a multiple of 4, and 128 rows over four
    M-waves leaves 2. See kid168 below.

    Eight waves rather than four is a register argument, and it holds for every
    kid in this file: a 256x256 fp32 accumulator is 1024 registers per lane-slot
    however it is split, so four waves would spend their entire 512-register file
    on the accumulator plus double-buffered fp8 fragments with nothing left for
    addressing. Eight cut it to 128 per wave, which fits alongside
    single-buffered fragments inside the 256 registers that 2 waves/SIMD allows.

    Direct-B is what makes the tile fit in LDS as well: staging B would cost
    (B_M+B_N)/32 load groups per slot and cap B_M+B_N at 384, while A alone
    leaves the ring three slots deep. Its cost is that B's global latency lands
    inside the K tile it feeds, which the pipeline covers by waiting on B one
    n-repeat at a time.
    """
    inst = OpusGemmInstance(
        512,            # BLOCK_SIZE (8 waves)
        bm, bn, bk,     # BLOCK tile
        2, 4,           # T_M, T_N (name only; traits derive the real 2x4 grid)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, 128, 128,    # GROUP_M=1 (per-token), GROUP_N=GROUP_K=128
        "a8w8_mxscale_bmm_bpreshuffle_wave8n4",
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.preload_sf = True
    inst.xcd_wgm = xcd_wgm
    inst.mpack_sfa = mpack_sfa
    inst.shuffle_scale = shuffle_scale
    return inst


_BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)
    194: (256, 256, 128, 1),
    # kid168: the half-M tile, which this family has never had. Where the tuned
    # M is an awkward multiple (g8/m2560 is 320 workgroups at 256x256, two rounds
    # of 256 CUs with the second a quarter full), halving a tile dimension halves
    # the tail. Halving N also halves B reuse, and at g>=8 that costs more than
    # the tail does; halving M keeps the N range whole. The plain kid159 wins
    # exactly those cells today and is the only reason any mid-M cell still goes
    # to a non-preshuffled kid.
    168: (128, 256, 128, 1),
    # kid175: a tile the row-major family wins M=256..1024 with, which this
    # family had only ever run at 256 columns or more. It clears the wave grid's
    # two divisibility constraints -- COM_REP_M = B_M/(W_M*T_M) = 4 is the
    # multiple of 4 SCALE_OPSEL needs, and COM_REP_N = B_N/(W_N*T_N) stays whole
    # -- so the only question was whether the tile is worth eight waves.
    #
    # kid187 was the reason to doubt it: four waves on a 128x128 tile measured 9%
    # behind the two-consumer form, because splitting a tile further cuts MFMA
    # per wave per K tile without removing any of the waits, and eight waves cut
    # it again to 8 MFMA per wave. It does not repeat here, and the difference is
    # that kid187 shared B through LDS while this does not: the split buys a
    # halved L1 crossing count that kid187 never got, and that is worth more than
    # the thinner per-wave tile costs.
    #
    # This is the one that changes a verdict -- at g2/m1024 it lands under the
    # row-major kid326 (20.45 vs 21.13), the cell that was preshuffle's worst
    # mid-M loss -- and it is the fastest kid here at m<=256 (g8/m256 19.3us).
    175: (128, 64,  256, 1),
    # kid348/kid349: the 128x128 tile at eight waves, which this family has had at
    # 64 columns and at 256 but never between. They earn 4 cells of the m=128..512
    # band by 1.025-1.057x -- kid349 at g8/m512 both K and g16/m256/k4096, kid348 at
    # g16/m256/k1024 -- and below g8 they are far off the pace (g2/m128/k1024 is
    # 9.1us against 4.9), which is what a 128-row eight-wave tile should do on a
    # grid of 16 workgroups. Both clear the wave-grid constraints with only
    # COM_REP_N moved off kid175 (1 -> 2, against kid194's shipped 4), SF_PANEL_ROWS
    # unchanged at 129, and no spill.
    #
    # They were added to test an explanation, and the measurement refuted it. Keep
    # the refutation: aiter's Triton batched_gemm_a8w8, swept over its own config
    # grid, runs g16/m256/k4096 in 31.1us against 40.3us for the best of the 43
    # preshuffle kids the tuner sweeps (8/8 paired draws, output checked against a
    # float reference). Its winning config is a 128x128 tile, K 256 deep, eight
    # warps, and the same tile at four warps measures 40.34us -- kid229's number to
    # three digits. That read as the wave count at a grid of exactly one workgroup
    # per CU (16*2*8 = 256 tiles on 256 CUs), where a 256-thread kid puts one wave
    # on each SIMD with nothing to overlap its global latency against.
    #
    # It is not the wave count. kid349 is that tile at eight waves -- same 128x128,
    # same grid, same two waves per SIMD as the Triton config -- and lands at
    # 39.4us, 1.27x behind it, having taken only 1.025x off kid229. And the K depth
    # ordering inverts: Triton wants 256 over 128 by 32.80 to 37.72, while kid348
    # loses to kid349 by 42.8 to 39.4, so the two are not responding to the same
    # thing.
    #
    # What that leaves is the inner loop, and the scale path is the term with a
    # measured size at this cell: the same 128x128 tile with the scale panel
    # preloaded (kid229) is 40.3us and without it (kid231/kid232) is 51.7-52.3us.
    # Triton is doing none of that work -- int8 with one scalar per row and one per
    # column against e8m0 every 128 elements of K -- so a 1.27x gap at equal
    # geometry is the price of microscaling here rather than a tile this family is
    # missing. Anyone reopening this should start there and not at the tile table.
    348: (128, 128, 256, 1),
    349: (128, 128, 128, 1),
}
a8w8_mxscale_bmm_bpreshuffle_wave8n4_kernels_list = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg)
    for kid, (bm, bn, bk, wg) in _BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_TILES.items()
}

# kid194 + the L2 rasterization that so far only wavetm1 has had. The reason it
# is worth trying on this family now is the per-traits SF_PRELOAD_K_MAX: the
# swizzle is only live at split_k == 1 (the kernel falls back to the linear map
# when the K range is split, since the reduction's workgroups no longer tile the
# output), and before the derived bound kid194 could not reach K=16384 on that
# column at all -- every large-K cell it won, it won at split_k >= 2, where the
# parameter is dead code. Raising its bound to 30848 moved it onto the one column
# where the map matters, and against kid213 there it wins by 3-4%.
#
# It works, and what gates it is the width of the tile grid rather than anything
# about this family. Band 4 against kid194 at k16384, g16/m4096 (6 draws, order
# rotated, bit-exact at every point since the map only moves which workgroup owns
# which tile):
#
#   n     N tiles   band 4    draws won
#   1024      4     +0.4%       4/6      <- the shipped envelope: noise
#   2048      8     +1.0%       6/6
#   4096     16     +2.2%       6/6
#   8192     32     +2.2%       6/6
#
# The reason it needs N is that the reordering's whole effect is the aspect ratio
# of the tile run an XCD walks. At n=1024 there are 4 N tiles and 8 tiles per XCD,
# so the run is 2 columns of A-rows either way and there is nothing to re-shape.
# At n=4096 a 32-tile run is 2x16 under the linear map and 4x8 under band 4, and
# at K=16384 a row block and a column block are 4 MB each: 64+8 MB of operand
# footprint against 16+32, which is the L2's problem stated as a number.
#
# So it changes no shipped cell -- the tuned envelope is n=1024 -- and is kept for
# the wider-N rows this bound now makes reachable at split_k=1.
#
# Band 2 was measured alongside (kid347) and is noise at every width above:
# +0.1% to +0.9%, never separable. Halving the band halves the A footprint but
# doubles the B one, and at these K the two are the same size, so it trades one
# for one. Uncomment to reopen.
_BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_XCD_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU, XCD_WGM)     twin of
    346: (256, 256, 128, 1, 4),               # kid194 + FlyDSL's band height
    # 347: (256, 256, 128, 1, 2),             # kid194 + half of it; noise, see above
}
a8w8_mxscale_bmm_bpreshuffle_wave8n4_kernels_list.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg, xcd_wgm=wgm)
    for kid, (bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_XCD_TILES.items()
})

# The three tiles above again on the shuffle_scale scale layout, one twin each, so that the
# only difference inside a pair is where the A scale comes from: the LDS panel
# filled from a plain (M, K/128) scale, or one dword per (M subtile pair, K tile
# pair) read straight from global. See the kid210 note for what the layout is and
# why its addressing is the cheap one.
#
# These exist because the layout is not a per-kid choice in a serving stack. The
# A scale is written once per call by the quantizer ahead of this GEMM (or offline
# for B), so it arrives in whatever layout that producer emits and every kid the
# tuner can pick for a shape has to read that same one. That makes the question
# not "is kid210 faster than kid205" but "does the envelope move if the producer
# switches", which needs a twin on every tile that owns part of the envelope.
# kid210 alone only answers it for kid205's tile.
#
# sub is T_M*W_M = 32 here against wavetm1's 16, i.e. the dword pairs M subtiles
# two of this grid's rows apart. Nothing else about the tiles moves.
#
# Per pair, shuffle_scale against plain, +% = shuffle_scale slower (_fly_family.py, each cell the min
# of both kid orders because the first kid in a process reads several percent slow
# -- kid194 leads the forward order and shows it, 257 against 232 at g4/m16384).
# The three starred rows are medians of four runs and every run agreed on the
# sign; the rest are single runs:
#
#     shape             | 194->213  168->214  175->215  205->210
#     g16 m32768 k4096  |   +4.4      +11.1      -0.4      -0.5
#   * g4  m16384 k4096  |   +2.8       +8.3      +5.5      +0.2
#   * g8  m2048  k4096  |   +4.2       +4.7      +2.1      +1.1
#   * g2  m1024  k4096  |   +0.3       +4.7      +2.3      -1.9
#     g8  m256   k4096  |   -0.7       +7.0      +0.7      +1.5
#     g8  m64    k4096  |   -0.1       +4.9      +1.3      -0.6
#     g16 m8192  k1024  |   +0.2       +2.4      +1.5      -2.3
#     g16 m4096  k1024  |   -0.8       +0.6      -5.8      -1.3
#     g16 m1024  k1024  |   +0.2       +0.6      -4.4      -1.5
#     g8  m1024  k1024  |   -0.3       +0.9      -4.2      +0.8
#
# The layout is not free per kid, and what it costs tracks how much the tile had
# to amortise the LDS panel over. kid168 pays throughout and pays most where K is
# largest (+11% at g16/m32768): its 128x256 tile has the family's worst ratio of
# MFMA to scale bytes, so the cooperative fill is worth most to it and losing it
# hurts. kid194 pays 3-4% at K=4096 down to m2048 and nothing below that or at
# K=1024. kid175 is the one that reverses -- 4-6% *faster* on shuffle_scale at K=1024 below
# m8192, slower at K=4096 -- which is
# what its B_K=256 tile should do, since that is the geometry where the dword's
# two K blocks land in one tile and op_sel's K bit is just the K repeat, with no
# unrolled loop parity to pay for. kid205/210 is the wash the kid210 note found.
#
# But the envelope is what a producer-side switch actually moves, and it barely
# does, because whichever tile wins a cell has a twin within a percent or two of
# it (best plain kid vs best shuffle_scale kid, us):
#
#     shape             | plain      shuffle_scale        delta
#     g16 m32768 k4096  | 1813.8 205 1804.5 210  -0.5%
#     g4  m16384 k4096  |  232.0 194  238.7 213  +2.9%
#     g8  m2048  k4096  |   58.5 194   59.5 210  +1.7%
#     g2  m1024  k4096  |   18.8 175   19.2 215  +2.1%
#     g8  m256   k4096  |   19.2 175   19.3 215  +0.5%
#     g8  m64    k4096  |   18.8 175   19.0 215  +1.1%
#     g16 m8192  k1024  |  204.3 194  204.7 213  +0.2%
#     g16 m4096  k1024  |   86.3 194   85.6 213  -0.8%
#     g16 m1024  k1024  |   29.2 205   28.8 210  -1.4%
#     g8  m1024  k1024  |   19.0 205   19.1 210  +0.5%
#
# So adopting the layout family-wide is close to free and buys close to nothing:
# K=1024 comes out a wash to 1.4% ahead, K=4096 gives up 1-3% through the mid-M
# band (m1024..m16384) and gets it back at m32768 where kid210 covers the cell.
# The per-kid losses do not reach the envelope because the tile that loses most on
# shuffle_scale (kid168) never owns a cell on this shape set, and the one that gains
# (kid175) owns the small-M end.
#
# The bdirect end was the follow-up to this and is measured under kid216/kid217:
# same crossover, both directions of it. That leaves six twins over three
# families, which is enough to ask the question the way it is actually posed. The
# layout is what an upstream quantiser emits, and that is fixed once per model,
# not chosen per call, so there is no picking the better of the two per shape: a
# model on shuffle_scale scales can run only shuffle_scale-capable kids, and every row-major and
# every plain preshuffle kid is then unusable. The two worlds are disjoint kid
# sets and what has to be compared is envelope against envelope.
#
# Measured on dsv4, the one model table for this op, at its N=1024/K=4096, with
# the old candidate set (the flatmm kids and kid158, which is what the shipped
# table was tuned over) against that set plus the preshuffle families, against
# the shuffle_scale twins alone (us, min of both kid orders, graph-timed):
#
#     shape       | old set      | + bpreshuffle | shuffle_scale only     | shuffle_scale delta
#     g2  m1      |    4.7 311   |    4.7 311    |    6.9 216   | -46.0%
#     g2  m16     |    5.0 311   |    5.0 311    |    7.0 216   | -42.3%
#     g2  m64     |    6.1 311   |    6.1 311    |    7.2 216   | -18.7%
#     g2  m256    |   10.0 324   |   10.0 172    |   10.9 216   |  -9.2%
#     g2  m1024   |   19.4 326   |   17.3 175    |   18.6 215   |  -7.5%
#     g2  m4096   |   55.6 325   |   35.3 168    |   36.6 214   |  -3.6%
#     g2  m16384  |  118.0 158   |  109.7 194    |  109.7 213   |  -0.1%
#     g16 m1      |   12.0 313   |   12.0 313    |   12.6 216   |  -4.3%
#     g16 m16     |   13.2 313   |   13.2 313    |   13.3 216   |  -0.7%
#     g16 m64     |   17.5 653   |   17.3 172    |   16.8 216   |  +2.8%
#     g16 m256    |   35.0 325   |   34.6 175    |   34.8 215   |  -0.6%
#     g16 m1024   |   92.0 158   |   72.4 205    |   72.7 210   |  -0.4%
#     g16 m4096   |  249.7 158   |  239.9 194    |  244.3 213   |  -1.8%
#     g16 m16384  | 1143.6 158   |  928.6 205    |  922.9 210   |  +0.6%
#
# Two separate results in that, and the first of them has since been measured
# properly and came out differently, so take the "+ bpreshuffle" column above only
# as the large-M half of it. Read off that column alone, supporting preshuffled B
# looks worth nothing below m1024 -- kid311 and kid313 hold every small-M cell --
# and 11-36% above it. The small-M half of that is an artifact of the candidate
# set: kid179 is kid311's tile exactly (bdirect 16x32x512) and once it is in the
# pool it takes those cells, 3.99 against 4.73us at g2/m1/K=4096.
#
# The full result, over all 116 shapes of the dsv4 envelope (K=1024 and K=4096,
# g in 2/4/8/16, m from 1 to 32768 -- the shape set of the fp8-vs-bf16 table) with
# the pool split by B layout and each side given every kid that wins anywhere in
# it, best of 3 runs a cell to keep allocation placement out of it:
#
#                          | median | mean  | best  | worst | faster | flat | slower
#     no blds at all       |  +9.4% | +8.7% | +30.2 | -19.2 |     96 |   13 |      7
#     blds, 5 winner tiles |  +9.4% | +9.2% | +30.0 |  -3.5 |    100 |   15 |      1
#     blds, whole family   | +10.6% |+10.9% | +35.6 |  -5.4 |    106 |    7 |      3
#     + the T_N=2 fix      | +10.0% |+10.8% | +35.9 |  -4.3 |    104 |   10 |      2
#     + the pf twins       | +10.2% |+10.8% | +35.5 |  -4.7 |    102 |   13 |      1
#       K=1024             | +11.7% |+13.7% | +35.5 |  +0.2 |     53 |    3 |      0
#       K=4096             |  +4.8% | +8.1% | +34.7 |  -4.7 |     49 |   10 |      1
#
# The T_N=2 row is the same kernels as the one above it: that fix only put the
# B_M=16 twins on the contiguous B path, and none of them is the best preshuffled
# kid at any cell either before or after, so read the difference between those two
# rows as this method's resolution rather than as a change.
#
# The pf-twin row is a real kernel change: four blds twins prefetch the scale and
# until then had no emitted instance at all, for the reason below. They win 8 of
# these 116 cells and 10 of the deployable table's 133 rows, and with them every
# K=1024 cell stops trailing -- worst goes -3.7% -> +0.2%. What is left is one
# K=4096 cell, which is the placement lottery discussed further down.
#
# So preshuffling B pays across the whole envelope rather than only at large M,
# and pays about twice as much at K=1024 as at K=4096: the shorter K gives less
# MFMA to hide B's LDS hop behind, so removing the hop is worth more.
#
# The first three rows differ only by how much of the blds family exists, and that
# is the second thing measured here. With none of it the seven mid-M cells the
# row-major flatmm kids own cost up to 19%, and the reason is not the layout: a
# bdirect kid is its plain kid with B's layout *and* B's path changed at once, and
# the path is what those cells object to. Flipping the layout alone is a wash --
# kid158 against kid196, the one same-pipeline pair, is median -0.2% over these
# same 116 shapes with 100 of them inside +-1%, because the row-major global read
# is already coalesced (the kernel reads B in whatever chunks it likes and fixes
# the order into LDS) so the shuffle buys not a faster read but an unnecessary LDS
# hop. A blds twin -- same tile, same LDS staging, B preshuffled -- then costs
# nothing to add and is consistently 1-2% ahead of its plain twin.
#
# Twinning the whole family rather than the five tiles that happened to win is
# worth the +1.2% median it adds, but the point of it is the row it does not
# change: an untwinned plain tile is a cell a preshuffled deployment has no answer
# for, and which tiles those are is a property of the shape set, not of the
# kernels. The assert on _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF is what keeps that
# from turning back into a judgement call.
#
# One to three cells trail depending on the row, all of them K=4096 on the last
# row, and every K=4096 one of them is an artifact -- which took finding out what
# the table's own error bar is.
#
# Repeating the sweep agrees with itself to 0.8% at the median, and that agreement
# means nothing: at K=4096 a kernel's time depends on where its weight buffer
# landed, and every repeat of one sweep draws the same placement. Measured by
# allocating the operands, timing, then re-allocating after a sweep's worth of
# allocation and graph-capture churn and timing again, at g8/m128/K=4096:
#
#                      fresh   after churn
#     kid324 row-major 20.02      14.86     -26%
#     kid320 row-major 20.20      16.66     -18%
#     kid653 row-major 19.79      16.65     -16%
#     kid326 row-major 18.63      18.35      -2%
#     kid228 blds      15.55      14.15      -9%
#     kid250 blds      22.84      22.78       0%
#
# Row-major B is read with a row stride of K bytes, so at K=4096 every row of a
# load group starts on a fresh 4KB page and 32 of them spread over 32 pages whose
# physical mapping decides the channel spread. The preshuffled read is one
# contiguous chunk_bytes run per issue, so its spread is fixed by the interleave.
# At K=1024, where four B rows share a page, the same churn moves everything by
# under 3.5%. This is the sharpest thing measured here in favour of preshuffling
# B and it is not a mean: the layout takes a 26% placement lottery off the table.
#
# Re-measuring both sides in one allocation state closes all three cells. The
# margin is quoted fresh / after churn:
#
#     g8  m128  K=4096  kid324 vs kid228   +22.4% / +4.8%
#     g4  m256  K=4096  kid324 vs kid228    +1.4% / -0.1%
#     g2  m64   K=4096  kid311 vs kid179     0.0% /  0.0%
#     g2  m2048 K=4096  kid325 vs kid229    +3.8% / +0.5%
#
# So no K=4096 cell on this envelope is slower with B preshuffled; the pool-versus-
# pool numbers above understate it, because the two sides read different buffers
# and each cell compares two independent placement draws. Any single K=4096 margin
# under ~10% is below what this method resolves.
#
# What used to not close was g2/m1024/K=1024, and the reading of it here was that
# preshuffling B flattens prefetch_scale: three tiles each measured a (pf, no-pf)
# plain pair whose two twins came out interchangeable, so pf looked worth 5-17% to
# a plain kid and nothing to a preshuffled one. It was a codegen bug, not an effect.
# name() for this family left prefetch_scale out, and instances are emitted and
# deduplicated by name, so all four pf twins collapsed onto their no-pf siblings
# and were never built: the two "twins" of each tile were one kernel measured twice,
# which is exactly why they were interchangeable. The name-clash assert at the
# bottom of the catalog is there so this cannot recur.
#
# With the four twins actually emitted, the layout and the scale prefetch are
# independent and compose. pf's gain, plain against twin, on the same shape:
#
#                        g16/m2048/K=4096          g8/m128/K=4096
#     64x64x128       +17.0% / +16.8%           +16.4% / +25.7%
#     16x32x512       +12.1% / +11.9%            +2.1% /  +2.4%
#     16x32x256        +4.5% /  +4.7%            +1.7% /  +1.6%
#     128x128x128      +1.4% /  +1.3%            +0.7% /  +0.5%
#
# Nothing is lost anywhere, and on 64x64x128 at m128 the twin gains half again what
# the plain kid does, which puts twin+pf (kid227, 17.08us) 14.1% ahead of plain+pf
# (kid653, 19.89) at a cell the plain kid used to own. Numerics are unchanged:
# errRatio 0.00001 across the four tiles x five shapes, and the table gate passes.
#
# The A-scale relayout on top is a wash wherever the shuffle_scale set has a tile of the
# right size -- within +-2% over the whole large-M end, which is the twin-by-twin
# result restated -- and a rout wherever it does not, up to 46% at g2/m1. All of
# that loss is a missing tile rather than the layout: the smallest shuffle_scale kid is
# kid216's 64x32 and the band below m256 belongs to kid311's 16x32x512 and
# kid313, neither of which has a twin. So a model-wide switch would mean porting
# the flatmm family as well, and the rows where a twin already exists say what
# that would buy, which is nothing. That is the answer to whether every kid needs
# filling in: under a per-model layout it would have to be, which is exactly why
# the switch is not worth making.
#
# The reason it comes out a wash where it is measurable at all is structural. shuffle_scale's
# gain is dropping the LDS scale panel, and a cell only goes to a tile whose ratio
# of MFMA to scale bytes already made that panel cheap, so the layout arrives
# where it has least to win. kid217 is the cleanest instance: a fair twin for the
# tile that owns m512, and it still loses 12-15% at K=4096.
#
# The LDS panel these twins drop is filled from A's scale only, so the panel's
# other tenant (SFB) is unaffected either way. Nor would relayouting SFB help:
# its address carries no lane term, so a K tile's B scales are one broadcast dword
# per N group however they are stored, already a single line per issue. Only A's
# scale is read per lane, one dword from each of 16 rows a stride_sfa apart, and
# coalescing that is what shuffle_scale is for.
#
# One methodology note, because it invalidates part of what is written above and
# in the kid210/216/217 entries. Calling this op from python costs ~11us of host
# time, and a timing loop of plain calls is launch-bound below about 19us: it
# floors there and stays flat from m=1 to m=256 while the shipped table records
# 7.8 to 14.5 over that range. Every number at or under ~19us in the older tables
# in this file is that floor rather than a kernel, and the small-M rows of those
# tables say nothing. The table above is timed under a captured graph, where the
# replay is one launch for the whole loop, and its m=1 numbers move with M as
# they should. The large-M rows elsewhere are unaffected and reproduce the shipped
# table within a few percent.
#
# The kids are kept as the measurement, not as tuner candidates: nothing should
# route to them until a producer emits the layout, and until then their entries
# are what says what that switch would cost.
_BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_SHUFFLE_SCALE_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU)     twin of
    213: (256, 256, 128, 1),         # kid194
    214: (128, 256, 128, 1),         # kid168
    215: (128, 64,  256, 1),         # kid175 -- also the only COM_REP_K=2 shuffle_scale kid
}
a8w8_mxscale_bmm_bpreshuffle_wave8n4_kernels_list.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wave8n4(bm, bn, bk, wg, shuffle_scale=True)
    for kid, (bm, bn, bk, wg)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVE8N4_SHUFFLE_SCALE_TILES.items()
})

# There is no B_K=256 tile in this family, and the shuffle_scale scale layout is why one was
# tried: at B_K=256 its dword's two K blocks land in the same tile, so op_sel's K
# bit is just the K repeat instead of an unrolled loop parity. 128x256x256 fits
# here comfortably (64 accumulator + 64 A + 64 B) where the T_M=1 grid cannot --
# see the B_K=256 note in the wavetm1 section for both measurements and why
# neither is kept. B_K=256 costs 1.6x on its own, whatever reads the scales.





def _a8w8_mxscale_bmm_bpreshuffle_wavetm1(block_size, bm, bn, bk, wg_per_cu,
                                          xcd_wgm=0, mpack_sfa=False,
                                          shuffle_scale=False):
    """The T_M=1 grid at B_M=128, where A stays in registers.

    Same schedule and same reason to want T_M=1 as kid195 -- no wave shares an N
    range, so each of B's bytes crosses L1 once -- without the LDS streaming that
    grid needs at B_M=256. Halving B_M halves one wave's A fragment to 64
    registers, which fits beside the accumulator and B.

    BLOCK_SIZE picks the grid: 512 threads give 1x8, 256 give 1x4. The 1x4 is the
    geometry the reference flydsl kernel runs on this tile, and the one the 4-wave
    allwave_bdirect family cannot express, since its traits fix the grid at 2x2.
    """
    inst = OpusGemmInstance(
        block_size,     # BLOCK_SIZE (512 -> 8 waves / 1x8, 256 -> 4 waves / 1x4)
        bm, bn, bk,     # BLOCK tile
        1, block_size // 64,  # T_M, T_N (name only; traits derive the real grid)
        16, 16, 128,    # W_M, W_N, W_K (MFMA 16x16x128 fp8) -- name only
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, 128, 128,    # GROUP_M=1 (per-token), GROUP_N=GROUP_K=128
        "a8w8_mxscale_bmm_bpreshuffle_wavetm1",
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.preload_sf = True
    inst.xcd_wgm = xcd_wgm
    inst.mpack_sfa = mpack_sfa
    inst.shuffle_scale = shuffle_scale
    return inst


# T_M=1 tiles. Both sit on kid168's 128x256 tile so the only thing moving against
# it is the wave grid: 2x4 there, 1x8 / 1x4 here.
#
# kid202 (1x8) keeps kid168's eight waves and only changes which operand is
# redundant. kid203 (1x4) is the reference flydsl geometry outright -- four waves,
# 32 MFMA each -- and it is the one that pays off. Its register bill is the tight
# one (128 accumulator + 64 A + 32 B, leaving addressing what is left under 256),
# and it lands at 246 VGPR / 2 waves/SIMD with no spill: the same footprint the
# flydsl kernel compiles to on this tile, which is what the geometry costs.
#
# Measured at g16/m32768/n1024/k4096, batch-major C, all five doing an identical
# 16,777,216 MFMA:
#
#     kid203  1x4   1933.6 us   246 VGPR  occ 2
#     kid202  1x8   2486.5 us   161 VGPR  occ 3
#     kid168  2x4   2508.8 us   148 VGPR  occ 3
#     kid201  2x2   3792.2 us   260 VGPR  occ 1
#     flydsl  1x4   1776.8 us   246 VGPR  occ 2
#
# MfmaUtil tracks that order and is the useful diagnostic here: 48.5% for kid203
# against 37.6 / 37.5 / 22.8, with flydsl at 49.8.
#
# Against the 256x256 shelf the tile still decides, so kid203 on its own only
# leads at large M (g16/n1024/k4096, us): m2048 144.2 vs kid194's 126.8, m8192
# 491.5 vs 480.5, m32768 1933.6 vs 1928.1. On its own 128x256 tile it beats kid168
# throughout (141.9 / 601.8 / 2508.8), which is the comparison the grid change is
# about; the tile map swizzle below is what closes the rest.
#
# So T_M, not the tile and not the wave count, is what pinned the 4-wave grid at
# 23%: the 2x2 grid has both M-waves fetch the same B fragment from global, and
# with direct-B there is no LDS to share it through. At m8192 that cost kid201
# 13.6M SQ_INSTS_VMEM and 67.3M TCC_READ against kid203's 7.1M and 52.6M, plus
# 17.8M cycles of L1 read tag-conflict stall that kid203 does not have at all
# (0, same as flydsl). DRAM traffic is within 3% across all of them, which is why
# the earlier round of L2/DRAM counters could not see this.
#
# What is left against flydsl is its XCD workgroup swizzle: the same kernel with
# it off runs 1978.6 us, which kid203 at 1933.6 already matches, so the whole
# residual is locality -- the thing opus measured as a net loss when it last tried
# it (see the swizzle note in the flatmm_splitk pipeline header) and that the kids
# below put back.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU)
    202: (512, 128, 256, 128, 1),
    203: (256, 128, 256, 128, 1),
}
a8w8_mxscale_bmm_bpreshuffle_wavetm1_kernels_list = {
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg)
    for kid, (bs, bm, bn, bk, wg) in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_TILES.items()
}

# kid203 + the L2 rasterization of the workgroup -> tile map, at three band
# heights. This is the one thing left between kid203 and the reference kernel:
# FlyDSL's own measurement of it on this tile is 1978.6us without against 1763.5
# with, i.e. 1.12x, and it only ever tunes the band height as off-or-4.
#
# An earlier round of this on kid188 ran 6% slower while halving HBM traffic, and
# concluded these kids are not bandwidth-bound so cheaper traffic buys nothing
# (see the note in the flatmm_splitk pipeline header). That does not carry over:
# it was measured on a 24%-MfmaUtil pipeline, where the time was going somewhere
# else entirely and a cheaper memory system had nothing to give back. kid203 sits
# at 48.5%, and here the same mapping pays.
#
# Measured (g16/n1024/k4096, batch-major C, us):
#
#     M       kid203   +wgm2    +wgm4    +wgm8   | kid194   flydsl
#     2048     144.2       -    126.7       -   |  126.8    128.0
#     4096     271.4    253.5    255.5    248.4 |  255.9    275.7
#     8192     491.5    478.3    471.6    475.4 |  480.5    507.9
#     32768   1933.6   1841.6   1845.9   1837.2 | 1928.1   1776.8
#
# So it is worth 4-12% at every M, the band height barely matters (2/4/8 land
# within 1% of each other, which is inside this benchmark's run-to-run spread),
# and 4 -- what FlyDSL picked -- is a fine default. With it kid205 ties kid194 at
# small M and pulls ahead from m8192 up, and it beats FlyDSL outright below m32768
# (7% at m4096 and m8192), leaving only m32768 at 1.04x.
#
# Read those columns down, never across runs: this harness has a ~11% clock ramp
# that penalizes whichever kid is measured first in a process (BMM_KIDS=205,194,205
# gives the same kid 273.1 us then 244.7 us). The kid harness and the flydsl one
# both prewarm for 3s before timing now; without that the two are not comparable.
#
# The mechanism is visible at m32768 and absent at m8192, which is also why the
# gain is bigger there. kid203 -> kid205 leaves TCC_READ alone (210.24M vs
# 210.26M -- the same requests, only elsewhere) while L2 hit goes 66.5% -> 82.8%,
# DRAM read requests 73.3M -> 33.5M, SQ_WAIT_ANY 391M -> 323M and MfmaUtil
# 48.4% -> 52.2%. At m8192 the linear map already hits 82.4%, the working set
# having fit anyway, and the swizzle moves the counters by well under a point.
#
# Where kid205 stands against the reference kernel, and what is actually left:
# NOT the inner loop, and NOT the wait structure. At m32768 with identical wave
# and MFMA counts FlyDSL waits *more* (SQ_WAIT_ANY 356.8M against kid205's 322.6M)
# and still wins on time, and kid205's K loop is the leaner of the two -- 129
# instructions per 32 MFMA against 134, 3.03 non-MFMA per MFMA against 3.19, 7
# s_waitcnt against 17. So porting that 17-deep waitcnt ladder would be a
# pessimisation, and the earlier read of the 4-wave problem as a synchronisation
# one was wrong: it was the wave grid (see kid201) and then locality.
#
# A K sweep at m8192/n1024 splits the rest into slope and intercept (us):
#
#     K       1024    2048    4096    8192  | per-K   K-independent
#     kid205   214.5   310.8   492.3   880.6 | 0.0927      ~121
#     kid194   199.5   304.3   478.4   910.6 | 0.0987      ~102
#     flydsl   212.0   288.3   503.3   926.6 | 0.1039       ~76
#
# (Both opus rows predate the dword-store fill_sfa, which took 2-3% off them at
# K>=4096; see the kid208 note below. The reading below is unaffected -- the fill
# change moves both kids the same way and neither toward flydsl.)
#
# kid205's steady state is 11% *better* per unit K -- it beats FlyDSL outright at
# K>=4096 on this shape -- and what it gives back is a K-independent cost, which
# both opus kids carry and kid205 carries most (its half-height tile doubles the
# workgroup count at a given M). An ATT trace of kid205 at K=1024 vs K=8192
# localizes it: vmem_store is hit exactly 256 times at both, i.e. purely fixed,
# and cost 53.5k cycles, 16.1% of the traced CU's total at K=1024.
#
# Part of that was the epilogue writing C in half lines, which the store loop in
# the wave8 pipeline now fixes; the numbers above are all post-fix. VEC_C packs
# along N and lane%16 is the row, so one store instruction covers 16 rows x 16
# columns -- 32 bytes of each row, half of a 64B line, however wide the per-lane
# store is. (FlyDSL writes the same tile with 128 buffer_store_short per lane
# against opus's 32 buffer_store_dwordx2, and is the better-behaved of the two:
# only the coalesced footprint per instruction matters, not the per-lane width.)
# Storing a whole n-repeat at a time put COM_REP_M other stores between the two
# halves of every line and the L2 did not reliably still hold the first half:
# 5,453,904 DRAM writes for a 268.4 MB C tile against the 4,194,304 full lines it
# needs, 1.30x, all of the excess partial. Pairing the halves back to back lands
# on 4,194,304 exactly, and is worth 3% on kid205 at m32768 and 9% on kid194. Both
# write exactly 268.4 MB into the L2 either way, which is why only the DRAM-side
# counter (TCC_EA0_WRREQ_WRITE_DRAM) could see it.
#
# What is left of the intercept is ~45 us and is not the epilogue: the store fix
# moved it by only ~4. The prologue is the remaining suspect -- the LDS scale-panel
# preload and its addressing, plus the per-K-tile s_barrier, which at K=8192 is
# already the largest non-MFMA ATT entry at 146.4k cycles (9.4%).
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_XCD_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    # Band heights 2 and 8 were measured alongside this and are within 1% of it
    # (m32768: 1870.8 / 1827.9 against 1850.4), so only FlyDSL's own band height
    # is kept. The 1x8 form at band 4 (kid207) loses to this 1x4 one at every M
    # past 1024, same as kid202 does to kid203.
    205: (256, 128, 256, 128, 1, 4),   # = kid203 + FlyDSL's band height; best here
}
a8w8_mxscale_bmm_bpreshuffle_wavetm1_kernels_list.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_XCD_TILES.items()
})

# kid208: kid205 with the A scale panel packed along M by the host instead of by
# the prologue, i.e. SFA_MPACK_GLOBAL. Retried on this kid because the verdict
# recorded for kid198 above was measured against kid194, and the two kids fail
# for different reasons. kid198 lost 8.8% by putting one more VMEM instruction
# per K tile onto the pipe holding 89% of kid194's stalls. kid205's problem is
# not the steady state at all -- its per-unit-K cost already beats FlyDSL's --
# it is the ~45 us K-independent intercept the note above pins on the prologue,
# and the LDS scale panel is what this removes from it.
#
# It is a slope-for-intercept trade and it measures as one. A K sweep at
# g16/m8192/n1024 (us), and the fit over it:
#
#     K        1024    2048    4096    8192  | per-K   K-independent
#     kid205  218.9   293.8   474.4   850.2  | 0.0881      ~129
#     kid208  206.1   296.4   490.4   918.3  | 0.0993      ~104
#
# So it takes ~24 us off the intercept -- half of the ~45 us the note above is
# about -- and pays 12.8% more per unit K for it, crossing over at K~2150: 5.8%
# ahead at K=1024, 8.0% behind at K=8192.
#
# Which makes this the answer to whether the A scale layout is where the residual
# against FlyDSL lives: it is not. At the K where FlyDSL still leads, moving the
# panel to the host is a loss, and kid205 already wins per unit K there.
#
# Kept because the win below the crossover is real and broad -- kid208 beats
# kid205 in 11 of 15 rows of a K=1024 sweep over g4/8/16 x m1024..16384, by 1-6%
# -- though the row is usually kid194's or kid203's anyway. It takes two: g16/m1024
# (31.7 against kid205's 32.6, kid194's 33.7, kid203's 36.0) and g16/m8192, where
# it ties kid194 (207.7 vs 207.8) with kid205 4 percent back.
#
# Keep the two halves of the panel's cost apart when reading the fit above. The
# ~24 us is the fixed half, which is what removing the panel gets. The per-unit-K
# half is separate -- the panel is sf_k_scales deep, so filling it grows with K --
# and it is set by the fill's store width, which is dwords rather than bytes since
# this was measured (see fill_sfa in the wave8 pipeline). That store width is
# worth 2-3% at K>=4096 across all three 8-wave families and nothing at K=1024,
# i.e. it moves only the slope; the kid205 row above is post-fix, which is why its
# per-K figure beats the 0.0929 quoted in the K sweep further up.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_MPACK_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    208: (256, 128, 256, 128, 1, 4),
}
a8w8_mxscale_bmm_bpreshuffle_wavetm1_kernels_list.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm,
                                               mpack_sfa=True)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_MPACK_TILES.items()
})

# kid210: kid205 with BOTH scale panels in the reference kernel's own layout,
# read from global -- shuffle_scale_a / _b, i.e. SHUFFLE_SCALE.
#
# This is kid208's experiment done the way FlyDSL does it, and the difference is
# the one axis kid208 got wrong. kid208 keeps our M packing, four M subtiles to a
# dword, which leaves the row axis outside the dword at the panel's K pitch: a
# lane's dwordx2 is one instruction, but 16 lanes land on 16 different 64B lines
# (256 bytes apart at K=4096). FlyDSL instead spends the dword on two M subtiles
# crossed with two K tiles and puts row%16 at a stride of one dword, so a quarter
# wave reads one line. Same bytes, same instruction count per K tile -- two dwords
# either way -- and the only thing that changes is how many lines a wave touches
# to get them.
#
# The cost of that is the K axis inside the dword: scale_op_sel is an immediate,
# so the K loop unrolls by two and the loop body exists twice. It also buys B's
# pack_e8m0x4 back, since B's dword doubles each byte and one op_sel then serves
# both operands, which is what the duplication in the reference layout is for.
#
# Worth knowing that this addressing is cheaper than the reference's own, because
# it means the measurements below are not reading an implementation gap. FlyDSL
# indexes a table of MFMA atoms as scale_atoms[(kh*2+im, kh*2+in_b)] with
# k_halves = BK//128, so at its BK=256 the dword's K bit is just kh and costs
# nothing -- that is the layout's native tile. At BK=128 there is no kh=1 to index
# with, so it keeps the word loaded across the two-tile chunk and slides the
# second tile's bytes down instead: scale_shift = (kt % tiles_per_chunk) * 16 and
# sa_v = [v.shrui(scale_shift)], i.e. one v_lshrrev per scale word on every odd K
# tile (mxscale_preshuffle.py, compute()). This kid unrolls by the same two but
# spends the op_sel immediate on the parity, (KP<<1)|(im&1), so it pays no VALU at
# all. The B side of that table is also what the duplication is for: op_sel_b
# carries the N parity in its low bit, which lands on two equal bytes.
#
# Medians of two runs each (us), noise within a config +/-0.5% except the g4 row
# at +/-1.5%:
#
#     g   m      K     | kid205  kid208  kid210
#     16  32768  4096  |  1935    2046    1942
#      4  16384  4096  |   268     273     277
#      2   8192  4096  |    77      77      78
#     16   8192  1024  |   220     208     220
#      8   8192  1024  |   115     110     117
#      2   8192  1024  |    31    30.5    30.6
#
# So against kid208 the layout does exactly what it was predicted to: at K=4096,
# where the scale panel is largest, it removes essentially all of kid208's 5.6%
# deficit (2046 -> 1942 against kid205's 1935). Reading the same bytes off one
# line per quarter wave instead of sixteen is worth 5% there, which also settles
# what kid208's note left open -- kid198 and kid208 lost on access shape, not on
# the extra instruction they were blamed for.
#
# It buys nothing over kid205 at large M though -- tying it at large K and 3%
# behind at g4/m16384 -- which is the same shape of answer kid208 gave: the A
# scale layout is not where the residual against FlyDSL lives. The LDS panel
# already turns the scattered read into one cooperative fill, and the ~24 us of
# prologue that removing it saves does not come back as steady state.
#
# Where it is kept is the other end of M at K=1024, which it takes off kid208 --
# the corner kid208 itself was kept for. The kid208/kid210 columns below
# are the mean of both running orders, each taken from its warm repeat, because a
# percent at these sizes is position (us, N=1024, K=1024):
#
#     g   m     | kid205  kid194  kid208  kid210
#     16  1024  |  30.2    32.7    29.7    29.4
#     16  2048  |  52.1    50.2    49.3    48.8
#     16  4096  |    --      --    92.1    90.6
#     16  8192  |   217.0  205.9   203.7   213.2
#      8  1024  |  20.2    25.3    19.9    19.3
#      4  1024  |  19.2    22.8    19.7    19.3
#
# So kid210 owns m<=4096 at K=1024 by 0.8-3.0% and kid208 keeps m8192 up by 4.4%,
# crossing between the two; against kid205 the same cells are 2-7%. Both orders
# agree on the sign in every row (g8/m1024 reads 19.3 against 19.9 whichever goes
# first), which is what makes a percent readable here at all.
#
# That split is consistent with what the layout does: kid210's win is the LDS
# panel's fixed cost, so it shows where there is least K and least M to amortise
# it over, and kid208 is the same trade with the coalescing penalty -- which is
# why kid208 needs more M before its own version of the saving nets out.
#
# Two things were tried on top and moved nothing. Keeping the staged B wait on
# the second tile of each pair, which is free -- that tile issues no scale loads
# at all, so the vmcnt accounting the staged wait needs still holds -- measured
# 1942 against 1945 at g16/m32768. And the shuffle_scale path drops both LDS scale panels
# rather than just A's, so the LDS it gives back is not what is limiting here
# either.
_BMM_MXSCALE_BPRESHUFFLE_WAVETM1_SHUFFLE_SCALE_TILES = {
    #   (BLOCK_SIZE, B_M, B_N, B_K, WG_PER_CU, XCD_WGM)
    210: (256, 128, 256, 128, 1, 4),
}
a8w8_mxscale_bmm_bpreshuffle_wavetm1_kernels_list.update({
    kid: _a8w8_mxscale_bmm_bpreshuffle_wavetm1(bs, bm, bn, bk, wg, xcd_wgm=wgm,
                                               shuffle_scale=True)
    for kid, (bs, bm, bn, bk, wg, wgm)
    in _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_SHUFFLE_SCALE_TILES.items()
})

# The shuffle_scale layout at B_K=256, tried twice as kids 211/212 (shuffle_scale, and the same tile
# without it) and removed both times. This is the geometry where the layout costs
# nothing to adopt: its dword always holds two 128-blocks of K, and at B_K=128
# (kid210) those land in two different K tiles, so op_sel's K bit has to come from
# an unrolled loop parity and the K loop body exists twice. At B_K=256 both blocks
# are inside one tile, the bit is just the K repeat, and the loop stays single. So
# this is where kid210's tie with kid205 would come apart, if the doubled body is
# what paid for it.
#
# B_K=256 is a 1.6x tile whatever reads the scales, which ends the question. It
# doubles both fragments, so the register file decides the rest of the tile.
# T_M=1 must drop B_N to 64 (A alone is 128 registers; B_N=128 lands exactly on
# the 224 limit with nothing left for addressing), and T_M=2 halves COM_REP_M
# instead and fits 128x256x256 at 64+64+64. At g16/m32768/k4096, against kid205's
# 1945 us: T_M=1/B_N=64 runs 4746 and T_M=2/B_N=256 runs 3065. Their own controls
# run 4741 and 3126, so the tile is all of it.
#
# The equal-tile pairs are still the cleanest read on the layout itself, since the
# control isolates it (us, two runs, repeatable to 0.1%):
#
#                        T_M=1, B_N=64    T_M=2, B_N=256
#     g   m      K     |  shuffle_scale   control |  shuffle_scale   control
#      2   8192  1024  | 58.7     63.0  | 41.2     41.7
#      8   8192  1024  |  227      239  |  154      152
#     16   8192  1024  |  432      460  |  330      329
#      2   8192  4096  |  162      160  | 96.7     92.6
#      4  16384  4096  |  618      610  |  362      349
#     16  32768  4096  | 4746     4741  | 3065     3126
#
# Which locates what the layout is worth, and it is not op_sel or access shape:
# it is the LDS scale panel's fixed cost, and only on a tile bad enough to keep
# re-paying it. The 5-7% at K=1024 is the B_N=64 column, where a 64-wide tile
# quadruples the workgroups and every one of them fills the panel again; on the
# B_N=256 tile the same comparison is a wash (+2.0% to -4.4%). kid208 got its
# 5.8% at K=1024 from the same place, and kid205 amortises the fill well enough
# that removing it is worth nothing at a competitive tile.
#
# Reaching either needed the est_acc_vgpr fix in the traits (it multiplied the
# accumulator by COM_REP_K, which is reduced, not accumulated over); every kid
# before this had COM_REP_K == 1, so the two forms agreed. The kernel keeps the
# COM_REP_K==2 path and the shuffle keeps its `sub` argument -- the T_M=2 pair
# needs the dword's M axis to be adjacent M subtiles 32 rows apart rather than
# FlyDSL's fixed +16, which the layout generalizes to without losing the
# coalescing it exists for: a wave's 16 lanes still read 16 adjacent dwords, and
# wave_id_m only picks which 16.

# The swizzle above is built on wgid, the dispatch order, so the XCD it undoes the
# round-robin on is the physical one. The reference kernel instead re-encodes the
# tile N-fast first and swizzles that, which makes its XCD id not the physical one
# -- its first step undoes a round-robin that never happened on that index. Built
# and measured as kids 208/209 (band 4 and 8) because it looked like the better
# permutation on paper: counting the distinct A panels a co-resident window of one
# XCD touches at m32768 (256 m-tiles x 4 n-tiles, 38 CUs per XCD),
#
#     linear 32.0 A panels | M-fast band 2/4/8: 11.0 / 12.4 / 15.6
#                          | N-fast band 2/4/8: 19.5 / 10.2 / 11.0
#
# i.e. 10.2 against kid205's 12.4. Not kept: it measured 1879.8 / 1873.8us against
# kid205's 1873.6 and kid206's 1874.5 at m32768, a 0.3% spread, and the same tie
# at m8192. The swizzle saturates -- going from 32 co-resident A panels to ~12 is
# the whole effect and the rest of the way buys nothing -- so the difference
# between the two conventions is not what the reference kernel's better M scaling
# comes from.




def _a8w8_mxscale_bmm_minterleave(bm, bn, bk, wg_per_cu, skip_scale_wait=False):
    """fp8 e8m0 mxscale BATCHED matmul M-tile-interleaved tile.

    Backs opus_bmm_a8w8_mxscale() kids 162/163. The main kernel
    (gemm_a8w8_mxscale_flatmm_minterleave_kernel<Traits, D_OUT, SKIP_SCALE_WAIT>)
    processes MI=2 consecutive M tiles per WG (baked in the launcher, requires
    M % (MI*B_M) == 0); splitK is unused (must be 1). Same locked geometry /
    traits as the flatmm split-K family (BLOCK_SIZE=256, T_M=2/T_N=1, MFMA
    16x16x128, VEC=(16,16,4), GROUP=(1,128,128), fp32 workspace tuple slot).
    """
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256,            # BLOCK_SIZE
        bm, bn, bk,     # BLOCK tile
        t_m, t_n,       # T_M, T_N (name only)
        16, 16, 128,    # W_M, W_N, W_K (name only)
        16, 16, 4,      # VEC_A, VEC_B, VEC_C
        1, 128, 128,    # GROUP_M=1 (per-token), GROUP_N=GROUP_K=128
        "a8w8_mxscale_bmm_minterleave",
        ["fp32_t"],     # single fp32 host stub; body branches on Y.dtype()
        wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    inst.skip_scale_wait = skip_scale_wait
    return inst


# fp8 e8m0 mxscale BMM M-tile-interleaved tiles (kids 162/163). Fixed geometry
# m128n128k128 wg1; the only axis is SKIP_SCALE_WAIT.
_BMM_MXSCALE_MINTERLEAVE_TILES = {
    #   (B_M, B_N, B_K, WG_PER_CU, skip_scale_wait)
    162: (128, 128, 128, 1, False),
    163: (128, 128, 128, 1, True),   # skip per-K-tile scale s_waitcnt
}
a8w8_mxscale_bmm_minterleave_kernels_list = {
    kid: _a8w8_mxscale_bmm_minterleave(bm, bn, bk, wg, skip)
    for kid, (bm, bn, bk, wg, skip) in _BMM_MXSCALE_MINTERLEAVE_TILES.items()
}


def _a8w8_mxscale_bmm_spec(tag, bm, bn, bk, wg_per_cu, **flags):
    """Generic fp8 e8m0 mxscale BMM specialized-pipeline tile builder.

    Same locked geometry/traits family as the flatmm split-K kids (BLOCK_SIZE
    256, MFMA 16x16x128, VEC=(16,16,4), GROUP=(1,128,128), fp32 workspace tuple
    slot). `tag` selects the kernel family (wave8n2 / wave4m2_selfload);
    `flags` sets the family's compile-time axes.
    """
    t_m, t_n = (1, 2) if bm == 16 else (2, 1)
    inst = OpusGemmInstance(
        256, bm, bn, bk, t_m, t_n, 16, 16, 128, 16, 16, 4, 1, 128, 128,
        tag, ["fp32_t"], wg_per_cu,
    )
    inst.name_root = "opus_bmm"
    for key, val in flags.items():
        setattr(inst, key, val)
    return inst


# fused (kid 100): the only fused-reduce path (splitK counter variant). Same
# 256x32x128x128 wg2 traits as standard kid 0/32, so its device symbols resolve
# to the standard family's TUs -> host-only launcher emit.
a8w8_mxscale_bmm_fused_kernels_list = {
    100: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_fused", 32, 128, 128, 2),
}

# pipeline (kids 149/150/151/152/158/159/164): BLOCK_SIZE 512, m{128,256}n{128,256}k128, dual
# bf16/fp32 traits (output dtype baked into the traits tuple), non-splitk scale
# kargs. One of the gemm_a8w8_scale_* kernels selected by flags. The wave
# layout (T_M/T_N/W_*) is derived inside opus_gemm_a8w8_scale_traits_gfx950 from
# BLOCK + <B_M,B_N,B_K>, so only B_M/B_N/B_K matter here (the T_M/T_N passed to
# OpusGemmInstance are cosmetic for this tag).
def _a8w8_mxscale_bmm_pipeline(**flags):
    inst = OpusGemmInstance(
        512, 256, 256, 128, 2, 1, 16, 16, 128, 16, 16, 4, 1, 128, 128,
        "a8w8_mxscale_bmm_pipeline", ["fp32_t"], 1,
    )
    inst.name_root = "opus_bmm"
    for key, val in flags.items():
        setattr(inst, key, val)
    return inst


a8w8_mxscale_bmm_pipeline_kernels_list = {
    # kid 149: B_M=128 plain scale pipeline (m128n256k128). Same gemm_a8w8_scale_
    # kernel as kid 150, just half the M tile -> 2x output tiles -> fills more CUs
    # on batched wo_a shapes. Was a hand-written cross-module adapter delegating
    # to opus_gemm's a8w8_mxscale GEMM launcher; now BMM-native codegen.
    149: _a8w8_mxscale_bmm_pipeline(B_M=128),
    150: _a8w8_mxscale_bmm_pipeline(),
    151: _a8w8_mxscale_bmm_pipeline(k1024_only=True),
    152: _a8w8_mxscale_bmm_pipeline(k1024_lb1=True),
    # kid158: preload BOTH SFA (per-token) and SFB (block) scale panels into LDS.
    158: _a8w8_mxscale_bmm_pipeline(preload_sf_lds=True),
    # kid159: kid158's scale preload at kid149's half-M tile (m128n256k128).
    # Narrow-N wo_a shapes underfill the machine at 256x256: g2/n1024/k4096 puts
    # only 4 N-tiles per batch, so m<=4096 launches <=128 workgroups against 256
    # CUs. kid149 already halves B_M to double the tile count there but loses to
    # kid158 anyway (m4096: 71.4 vs 52.9 us) because it is the one tile in this
    # family without the scale preload, which on its own is worth 31% (kid150 vs
    # kid158 at the same 256x256 tile: 76.8 vs 52.9). This pairs the two so the
    # tile count and the preload stop being a forced choice.
    159: _a8w8_mxscale_bmm_pipeline(B_M=128, preload_sf_lds=True),
    # kid164: kid158 with the N tile halved (m256n128k128). The tile count on
    # narrow-N wo_a comes from N, not M, so this is the one that actually fills
    # the machine there: g2/n1024/k4096/m4096 goes from 128 to 256 workgroups
    # against 256 CUs, where kid159's half-M tile only trades A reuse for the
    # same count. The preshuffled kid193 at this tile beats kid158 by 27-30% on
    # m2560..4096 while its own 256x256 sibling (kid192) *loses* to kid158 by
    # 20%, so the tile is what pays, not the preshuffle or the wave8 schedule.
    # A 64-column quadrant is narrower than a 128-column B scale block, which is
    # what SFB_HALF_STRIDE/SFB_GROUPS_PER_HALF in the traits exist to express.
    164: _a8w8_mxscale_bmm_pipeline(B_N=128, preload_sf_lds=True),
}


def _a8w8_mxscale_bmm_pipeline_bpreshuffle(**flags):
    """kid158 reading a 16x16-preshuffled weight buffer.

    This is the preshuffled kid to reach for at this tile. It changes the
    preshuffle axis alone -- same kernel, tile, 4x2 wave grid, LDS budget and
    four-quadrant schedule as kid158, with only the producer's B global address
    math flipped -- and measures within noise of it (8192^3: 395.2 vs 393.3 us,
    +0.5%, bit-exact). Callers must pass the preshuffled wo_a.

    It was built to price the preshuffle on its own, because the older
    preshuffled kids (kid192/194/195) had also moved B out of LDS into registers
    and swapped the quadrant schedule for a whole-tile one, so their 13-22%
    deficit against kid158 could not be attributed to any one of the three. The
    answer is that none of it is the preshuffle, and that the other two are not
    separable axes: a whole-tile schedule needs all of B_N live per K tile, which
    at 256x256 is 64KB of staging and leaves only 2 prefetch slots against the 3
    the pipeline asserts -- so whole-tile forces B direct-to-register, and direct
    B then pays T_M-fold L1 amplification on B, keeps 64 VGPR of fragment live to
    cover global latency, and (see the prefetch_k_iter note in the wave8 traits)
    caps A's prefetch lead at two tiles because vmcnt is one in-order counter.
    The quadrant schedule stages half a tile, fits 3 slots, and pays none of it.
    """
    inst = OpusGemmInstance(
        512, 256, 256, 128, 2, 1, 16, 16, 128, 16, 16, 4, 1, 128, 128,
        "a8w8_mxscale_bmm_pipeline_bpreshuffle", ["fp32_t"], 1,
    )
    inst.name_root = "opus_bmm"
    for key, val in flags.items():
        setattr(inst, key, val)
    return inst


a8w8_mxscale_bmm_pipeline_bpreshuffle_kernels_list = {
    # kid196: kid158 (preload_sf_lds) with a preshuffled B.
    196: _a8w8_mxscale_bmm_pipeline_bpreshuffle(preload_sf_lds=True),
}

# mouter (kids 131/144) + mouter_tunable (kids 160/161): wg1 m128n128k128,
# 1 bool axis <SKIP_SCALE_WAIT>. Both share gemm_..._mouter_kernel, so the
# tunable variant reuses the mouter device instantiations (host-only emit).
a8w8_mxscale_bmm_mouter_kernels_list = {
    131: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter", 128, 128, 128, 1),
    144: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter", 128, 128, 128, 1, skip_scale_wait=True),
}
a8w8_mxscale_bmm_mouter_tunable_kernels_list = {
    160: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter_tunable", 128, 128, 128, 1),
    161: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_mouter_tunable", 128, 128, 128, 1, skip_scale_wait=True),
}

# wave8n2 (kid 132): wg1 m128n128k128, no compile-time flags (logical B_N = 256).
a8w8_mxscale_bmm_wave8n2_kernels_list = {
    132: _a8w8_mxscale_bmm_spec("a8w8_mxscale_bmm_wave8n2", 128, 128, 128, 1),
}

# wave4m2_selfload (kids 134/142/148): wg1 m128n128k128, 2 bool axes
# <SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND> (logical B_M = 128*2 = 256).
_BMM_WAVE4M2_TILES = {
    #   (ssw,   psod)
    134: (False, False),
    142: (True,  False),
    148: (True,  True),
}
a8w8_mxscale_bmm_wave4m2_selfload_kernels_list = {
    kid: _a8w8_mxscale_bmm_spec(
        "a8w8_mxscale_bmm_wave4m2_selfload", 128, 128, 128, 1,
        skip_scale_wait=ssw, pack_scale_on_demand=psod,
    )
    for kid, (ssw, psod) in _BMM_WAVE4M2_TILES.items()
}


# All name-keyed a8w8_mxscale BMM kernel families (gfx950-only). Kept as a tuple
# of the per-family kid-keyed dicts -- NOT merged into one dict, because int kids
# repeat across families and are deduped downstream by launcher NAME (see
# gen_instances.py). Single source of truth for both consumers there: the codegen
# kdict merge and the BMM int-kid tune-lookup emitter.
a8w8_mxscale_bmm_kernel_lists = (
    a8w8_mxscale_bmm_flatmm_splitk_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_bdirect_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_blds_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_wave8n4_kernels_list,
    a8w8_mxscale_bmm_bpreshuffle_wavetm1_kernels_list,
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
    )


def _a16w16_kbuf2v_bk128_sk_gfx942(bs, bm, bn, bk, tn, wm, wn, wk):
    """SplitK P1 + B_K=128 sub-K decomp."""
    vec = 16 // 2
    return OpusGemmInstance(
        bs, bm, bn, bk, 2, tn, wm, wn, wk, vec, vec, 4, 0, 0, 0,
        "a16w16_kbuf2v_bk128_sk", ["fp32_t"], arch_prefix="gfx942",
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
# (<10000) and gfx942 (50000+). Today only the cluster/TDM split-K (atomic
# fp32 reduction) pipeline is wired (????:fp32 output, no bias).
GFX1250_KID_OFFSET = 20000


def _a16w16_cluster_tdm_splitk_ws_gfx1250(bm, bn, bk, layout, num_slots=3, wg_per_cu=2):
    """Factory for the gfx1250 a16w16 cluster/TDM split-K (workspace + reduce) kid.

    Locked geometry from the kernel base
    (demon_gcn/wmma_opus_rdna4/gemm_a16w16_cluster_tdm_splitk_reduce_4wave.cc):
    BLOCK_SIZE=128 (4 waves x 32 = 2 producer + 2 consumer), MFMA 16x16x32,
    NO-CLUSTER (one WG per B_M x B_N tile). The main kernel WMMA-accumulates in
    fp32 and PLAIN-stores each split's partial into an fp32 workspace; a separate
    reduce kernel sums the split slices, folds bias, and casts to the Y dtype.
    output_dtypes = ["fp32_t"] (only the fp32-workspace main kernel is
    instantiated; Y bf16/fp32 is a runtime decision in the reduce kernel).

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
    # -- ORIGINAL 11 tiles: KEEP THIS ORDER (indices 0..10) -- the C++ heuristic
    #    opus_a16w16_heuristic_kid_gfx1250() hardcodes kids 20000/20024/20032
    #    (16x32/64/128, idx 0/3/4) and 20040/20048/20056 (32x32/64/128, idx
    #    5/6/7), and tuned CSVs reference these numbers. Do NOT reorder/insert.
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

# Kid numbering (clean, contiguous; heuristic / tuned-CSV back-compat dropped):
#   plain (no-cluster) kids occupy [20000, 20100), ONE P=3 kid per tile (P=2 is
#   dropped -- unvalidated). Tiles the picker rejects (>=256-request TDM
#   direct-copy, now FIXED) fall back to P=3, 1 WG/CU so every no-spill tile still
#   emits a plain kid (LDS(P=3) <= 320 KB for this set). The C++ heuristic
#   constants in opus_gemm_heuristic_dispatch_gfx1250.cuh are regenerated to match.
#
# The consumer kExpN stability guard (previously _GFX1250_MAX_KEXPN=8) is removed.
gfx1250_kernels_list = {}
GFX1250_PLAIN_KID_OF = {}   # (B_M,B_N,B_K) -> kid (P=3; for tuner + heuristic regen)
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
# Same 4-wave TDM split-K + fp32 workspace + reduce kernel, but launched as a
# (cluster_wg_m x cluster_wg_n x 1) workgroup CLUSTER: peers co-reside and share
# A/B TDM loads via CLUSTER_LOAD_ASYNC multicast (named-barrier producer/consumer
# handshake, same as the plain base). The host launcher rounds the grid up to the
# cluster dims; the workgroups that round-up adds own no tile and return at their
# cluster-barrier arrival, so no shape needs an exact cluster fill. Distinct kid
# band (20100+) so it never collides with the no-cluster base kids (20000..20099).
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


# -- gfx1250 FUSED single-kernel split-K (a16w16_clusterlaunch_tdm_splitk_fuse) --
# Single kernel: last split WG folds bias + reduces the SplitK-1 partials in-kernel
# (cluster-barrier sync), no separate reduce kernel. SplitK / MClusterWg are
# COMPILE-TIME (cluster dims (SplitK, MClusterWg, 1)); DataWs (bf16/fp32) is a kid
# property. B is TDM-multicast across the MClusterWg M-peers.
#
# CURRENTLY UNREGISTERED (see GFX1250_SPLITK_FUSE_ENABLED below): the pipeline is
# still being fixed, so the family contributes no kid and the [21000, ...) band it
# used to hold is free for another kernel family to take.
def _a16w16_splitk_fuse_gfx1250(
    bm, bn, bk, layout, split_k, m_cluster, ws_dtype="bf16_t",
    num_slots=3, wg_per_cu=2,
):
    from dataclasses import replace

    inst = _a16w16_cluster_tdm_splitk_ws_gfx1250(
        bm, bn, bk, layout, num_slots=num_slots, wg_per_cu=wg_per_cu
    )
    return replace(
        inst,
        kernel_tag="a16w16_clusterlaunch_tdm_splitk_fuse",
        # output_dtypes MUST stay ["fp32_t"] (the split-K lookup invariant): the
        # host launcher is instantiated ONLY as <fp32_t> and opus_gemm.cu forces
        # the fuse band to the <fp32_t> dispatch slot; the launcher then picks the
        # real Y dtype at RUNTIME (if Y.dtype()==bf16 ... else float). Advertising
        # bf16_t here would make gen_a16w16_tune_lookup emit &{name}<bf16_t> in the
        # BF16 tune map -> undefined symbol (that specialization is never built).
        # The tuner exempts fuse kids from the output-dtype narrowing separately.
        output_dtypes=["fp32_t"],
        fuse_split_k=split_k,
        fuse_m_cluster=m_cluster,
        fuse_ws_dtype=ws_dtype,
    )


# Registration switch for the whole fused family. False sweeps no (tile, split_k,
# n_cluster, ws) combination at all, so gfx1250_splitk_fuse_kernels_list stays
# empty: no kid to look up, nothing for the tuner to pick, nothing for the codegen
# to emit, and the kid band below is unclaimed. The factory above, the emitter in
# codegen/gen_instances_gfx1250.py and the device pipeline are all still here --
# flipping this back to True is the only step needed to bring the family back.
GFX1250_SPLITK_FUSE_ENABLED = False

gfx1250_splitk_fuse_kernels_list = {}
# Kid band the family claims WHEN ENABLED. It reaches 22377 at the current sweep;
# while disabled the whole range is free.
GFX1250_SPLITK_FUSE_KID_BASE = 21000
_sf_kid = GFX1250_SPLITK_FUSE_KID_BASE
# (B_M, B_N, B_K, layout, split_k, m_cluster, ws_dtype) -> kid, for the tuner /
# candidate selection to look a fuse kid up by config.
GFX1250_SPLITK_FUSE_KID_OF = {}
# Fuse tiles = the SAME no-spill (B_M, B_N, B_K) set as the clusterlaunch sweep
# (_GFX1250_CLUSTERLAUNCH_TILES), so fuse covers the full tile range. Layout
# follows the base rule (B_M==16 -> tileN, else tileM); wg_per_cu is inherited
# per tile from the clusterlaunch table (the fuse producer shares the same TDM
# request profile, so that wg keeps 2-WG/CU co-residency TDM-budget-safe).
#
# N-DIRECTION MULTICAST: the cluster is (SplitK, n_cluster, 1) where the 2nd dim
# (stored in the fuse_m_cluster field) groups n_cluster N-tile peers that share
# A[M-tile] via TDM multicast (mirrors the clusterlaunch cwn A-multicast that
# wins at small M). n_cluster is swept 1..5 (TDM fan-out <= 5) subject to
# SplitK*n_cluster <= 16 (16-WG cluster budget).
#
# split_k sweep per workspace dtype:
#   * bf16 workspace: split_k 2..15
#   * fp32 workspace: split_k 2..8 (kept conservative; the reduce now stages
#     partials through a bounded LDS RING (kFuseReduceRing in the pipeline), so
#     split_k is NO LONGER LDS-bounded -- this cap could be lifted to 15 too).
# SplitK is capped at 15 (NOT 16): each __cluster_dims__ axis is a 4-bit field.
# SplitK / n_cluster are COMPILE-TIME (cluster dims), so each (tile, split_k,
# n_cluster, ws) is a distinct kid; neither is a runtime knob for fuse.
_FUSE_REDUCE_RING = 3  # must match kFuseReduceRing in the fuse pipeline header
_FUSE_NUM_SLOTS = 3


def _fuse_ring_lds_ok(bm, bn, bk, wg, ws_bytes):
    """Guard: the reduce LDS ring (kFuseReduceRing tiles of B_M*B_N*ws_bytes)
    must fit kLdsTotalBytes. Mirrors the traits LDS formula so we never emit a
    kid that would fail the pipeline's ring static_assert at compile time."""
    pitch = bk + 8
    seg_ab = _FUSE_NUM_SLOTS * (bm + bn) * pitch * 2  # bf16 A/B footprint
    lds_total = (160 * 1024 + 1024) if (wg == 1 and seg_ab <= 160 * 1024) else seg_ab
    return _FUSE_REDUCE_RING * bm * bn * ws_bytes <= lds_total


_FUSE_WS_SWEEP = (("bf16_t", 2, 15), ("fp32_t", 4, 8))  # (ws_dtype, elem_bytes, sk_hi)
# N-direction cluster (A-multicast) fan-out: the fuse_m_cluster field holds the
# cluster's 2nd-dim WG count, which for this pipeline groups N-tile peers sharing
# A. TDM multicast fans out to <= 5 WGs, and the cluster (SplitK, n_cluster, 1)
# must satisfy SplitK*n_cluster <= 16 (16-bit workgroup_mask / 16-WG budget).
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
            continue  # ring wouldn't fit LDS for this (tile, ws) -- skip
        for _nc in range(1, _FUSE_MAX_NCLUSTER + 1):
            for _sk in range(2, _sk_hi + 1):
                if _sk * _nc > 16:  # SplitK * n_cluster <= 16 (cluster budget)
                    continue
                gfx1250_splitk_fuse_kernels_list[_sf_kid] = _a16w16_splitk_fuse_gfx1250(
                    _bm, _bn, _bk, _layout,
                    split_k=_sk, m_cluster=_nc, ws_dtype=_ws, wg_per_cu=_wg,
                )
                GFX1250_SPLITK_FUSE_KID_OF[
                    (_bm, _bn, _bk, _layout, _sk, _nc, _ws)
                ] = _sf_kid
                _sf_kid += 1
assert _sf_kid <= 30000, f"splitk_fuse gfx1250 kids overflow [21000,30000): {_sf_kid}"
GFX1250_SPLITK_FUSE_KIDS = frozenset(gfx1250_splitk_fuse_kernels_list.keys())

# combined list (used by production gen_instances / dispatch)
kernels_list = {
    **a8w8_scale_kernels_list,
    **a8w8_kernels_list,
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

default_kernels_dict = {
    (-1): OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
    (-2): OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0,     "a8w8",       ["fp32_t"]),
    (-3): _a16w16(512, 256, 256, 64, 4, 16, 16, 32),  # same as a16w16 #9
}
# fmt: on


# Instances are emitted -- and deduplicated -- by name(), so two catalog entries
# that differ in configuration but agree on a name do not both get built: one
# silently inherits the other's kernel, compile-time flags and all. The blds family
# shipped that way, because its name left prefetch_scale out and each pf kid
# collapsed onto its no-pf sibling, then measured identically to it for the obvious
# reason. An alias, one configuration under two kids as with kid0/kid32, is fine and
# stays allowed; one name covering two configurations is what this rejects.
_name_owner = {}
_name_clashes = []
for _cat in (*a8w8_mxscale_bmm_kernel_lists, kernels_list):
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


# Subset-compile kid taxonomy (consumed by gen_instances.py for the `HEURISTIC_DEFAULT_KIDS ?

# Splitk kids: a16w16_flatmm_splitk pipeline (kid 200..223 + nooob mirror).
SPLITK_KIDS = (
    frozenset(a16w16_flatmm_splitk_kernels_list.keys())
    | frozenset(a16w16_flatmm_splitk_kernels_list_nooob.keys())
    | frozenset(gfx942_splitk_kernels_list.keys())
    | frozenset(gfx1250_kernels_list.keys())
    | frozenset(gfx1250_clusterlaunch_kernels_list.keys())
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

# Heuristic-dispatch fallback kids (gfx950).
HEURISTIC_DEFAULT_KIDS_GFX950 = frozenset(
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

HEURISTIC_DEFAULT_KIDS_GFX942 = frozenset(
    {
        # gfx942 heuristic dispatcher fallbacks.
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

# gfx1250 has no shape-heuristic dispatch yet (tune-id entry only). This set
# is used purely to keep the kid in the subset-compile set S so the tune-id
# path can always reach it.
# Only the kids the C++ heuristic (opus_a16w16_heuristic_kid_gfx1250) can return
# must be force-compiled as the always-available (M,N,K) fallback. Every other
# plain kid and ALL clusterlaunch kids are compiled on demand by the tuner
# (candidate selection + sidecar expansion), so default builds stay small.
HEURISTIC_DEFAULT_KIDS_GFX1250 = frozenset(
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

HEURISTIC_DEFAULT_KIDS = (
    HEURISTIC_DEFAULT_KIDS_GFX950
    | HEURISTIC_DEFAULT_KIDS_GFX942
    | HEURISTIC_DEFAULT_KIDS_GFX1250
)

HEURISTIC_DEFAULT_KIDS_BY_ARCH = {
    "gfx950": HEURISTIC_DEFAULT_KIDS_GFX950,
    "gfx942": HEURISTIC_DEFAULT_KIDS_GFX942,
    "gfx1250": HEURISTIC_DEFAULT_KIDS_GFX1250,
}


def heuristic_kids_for_arch(arches):
    """Return the heuristic-default kid subset whose arch_prefix matches.

    ``arches`` is an iterable of lowercase arch strings (e.g. ``{"gfx942"}``)
    or ``None`` (caller does not know / multi-arch build) -- in the ``None``
    case the full union is returned so the legacy multi-arch behaviour is
    preserved.
    """
    if arches is None:
        return HEURISTIC_DEFAULT_KIDS
    arches = {a.lower() for a in arches}
    out = frozenset()
    for arch in arches:
        out = out | HEURISTIC_DEFAULT_KIDS_BY_ARCH.get(arch, frozenset())
    return out


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
