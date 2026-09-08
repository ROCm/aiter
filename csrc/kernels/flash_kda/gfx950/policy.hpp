#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "../hip_common.hpp"
#include "../k1_kda_split_kernel.hpp"
#include "../k2_kda_csplit_bt64_kernel.hpp"
#include "k1_kda_bt16_fused_kernel.hpp"
#include "k1_kda_bt32_merge_kernel.hpp"
#include "k1_kda_bt64_merge_kernel.hpp"
#include "k1_kda_postprep_fused_kernel.hpp"
#include "k1_kda_split_solve_kernel.hpp"
#include "k2_kda_context_affine_ab_fused_kernel.hpp"
#include "k2_kda_context_affine_ab_fused_persistent_kernel.hpp"
#include "k2_kda_context_affine_scan_ksplit_kernel.hpp"
#include "k2_kda_context_direct_global_n1_h12_kernel.hpp"
#include "k2_kda_context_direct_ksplit_kernel.hpp"
#include "k2_kda_context_direct_ksplit_tailpipe_kernel.hpp"
#include "k2_kda_context_hybrid_compact_scan_kernel.hpp"
#include "k2_kda_context_hybrid_persistent_replay_kernel.hpp"
#include "k2_kda_context_parallel_kernel.hpp"
#include "k2_kda_csplit_bt64_bv16_nw8_kernel.hpp"
#include "k2_kda_csplit_bt64_plain_kernel.hpp"
#include "k2_kda_csplit_segment_out_kernel.hpp"
#include "k2_kda_vsplit_rs_x32_kernel.hpp"

namespace flashkda_hip::gfx950 {

struct LaunchPolicy {
private:
    static constexpr int kHybridDirectMaxChunks = 64;
    static_assert(
        WorkspaceSizes::kCsplitSegmentA >=
            4 * 128 * int(sizeof(float)),
        "cs_segment_a cannot hold one plain BT64 suffix-decay table");
    static_assert(
        direct_global_n1_h12_detail::kKrGllLdsBytes == 4 * 1024,
        "direct-global Kr GLL launch must reserve exactly 4 KiB LDS");

    struct ContextRouteConfig {
        bool force_context;
        int group_chunks;
        int direct_max_chunks;
        bool automatic_gva_packed_nw4;
        bool automatic_gva_equal_n4_g16;
    };

    enum class ContextDirectKsplitMode : uint8_t {
        disabled,
        waves2,
        waves4,
        tailpipe_waves2,
        tailpipe_waves4,
    };

public:
    static HipLaunchPolicy make(
            const FwdParams& p, const HipDeviceInfo& device) {
        const ContextRouteConfig context_route = resolve_context_route(p);
        const K2DefaultRoute default_k2_route =
            default_route(p, context_route.force_context);
        const bool use_context_direct_prefixless =
            context_direct_prefixless_enabled(
                p, context_route, default_k2_route);
        const bool use_context_equal_dense_n4_g64 =
            context_equal_dense_n4_g64_enabled(
                p, context_route, default_k2_route);
        const bool use_context_persistent = context_persistent_enabled(
            p, device, context_route, default_k2_route);
        const bool use_db = use_double_buffer(p, device);
        const bool use_pad = partial_padding_enabled();
        const bool use_tb3 = use_db && triple_buffer_enabled();
        const bool use_tiled_kr = tiled_kr_enabled();
        const bool use_plain_common_nw8 = plain_common_nw8_enabled();
        const bool use_regb_x32 = scan_regb_x32_enabled();
        const bool use_state_xchg = scan_state_xchg_enabled();
        // Preserve the established x32 exchange when both diagnostic knobs
        // are set; x16 is a separate strict opt-in, never an implicit fallback.
        const bool use_state_xchg_x16 =
            !use_state_xchg && scan_state_xchg_x16_enabled();
        // This representation change is meaningful only after the compact
        // state exchange.  An isolated RHS_FRAGMENT=1 is deliberately inert.
        const bool use_rhs_fragment = scan_rhs_fragment_enabled() &&
            (use_state_xchg || use_state_xchg_x16);
        const bool use_output_x32 = output_x32_enabled();
        const bool disable_output_gll = output_gll_disabled();
        const bool disable_output_gll_sin = output_gll_sin_disabled();
        const bool use_replay_tb3 = output_gll_tb3_enabled();
        const bool use_replay_decay_cache = replay_decay_cache_enabled();
        // Fragment-major cs_sin is a producer/consumer ABI, not an isolated
        // tuning bit.  Expose it only on the automatic plain route with the
        // private GLL replay, and only when both stages execute.  All explicit
        // or skipped-stage diagnostics retain row-major cs_sin on both sides.
        const bool use_sin_fragment = sin_fragment_enabled() &&
            !use_plain_common_nw8 &&
            default_k2_route == K2DefaultRoute::csplit64 &&
            std::getenv("FLASH_KDA_K2") == nullptr &&
            std::getenv("FLASH_KDA_CS_SKIP_SCAN") == nullptr &&
            std::getenv("FLASH_KDA_CS_SKIP_OUT") == nullptr &&
            !use_output_x32 && !disable_output_gll;
        auto select_one_arena_plain = [&]<bool SIN_FRAGMENT>()
                -> Csplit64ScanLauncher {
            if (!use_pad)
                return &launch_plain<
                    1, false, false, false, false, SIN_FRAGMENT>;
            if (use_state_xchg)
                return use_rhs_fragment
                    ? &launch_plain<
                        1, true, false, true, true, SIN_FRAGMENT, true>
                    : &launch_plain<
                        1, true, false, true, true, SIN_FRAGMENT>;
            if (use_state_xchg_x16)
                return use_rhs_fragment
                    ? &launch_plain<
                        1, true, false, false, true, SIN_FRAGMENT, true>
                    : &launch_plain<
                        1, true, false, false, true, SIN_FRAGMENT>;
            if (use_regb_x32)
                return &launch_plain<
                    1, true, false, true, false, SIN_FRAGMENT>;
            return &launch_plain<
                1, true, false, false, false, SIN_FRAGMENT>;
        };
        // Fragment-major cs_sin is independent of the scan's LDS arena
        // count.  In particular, enabling it for DB2/DB3 also removes the
        // replay kernel's 33-KiB row-major state staging allocation, while
        // retaining the exact same BF16 entry-state fragments.
        auto select_multi_arena_plain =
                [&]<int ARENAS, bool SIN_FRAGMENT>()
                -> Csplit64ScanLauncher {
            if (use_pad && use_tiled_kr) {
                if (use_state_xchg)
                    return use_rhs_fragment
                        ? &launch_plain<
                            ARENAS, true, true, true, true,
                            SIN_FRAGMENT, true>
                        : &launch_plain<
                            ARENAS, true, true, true, true, SIN_FRAGMENT>;
                if (use_state_xchg_x16)
                    return use_rhs_fragment
                        ? &launch_plain<
                            ARENAS, true, true, false, true,
                            SIN_FRAGMENT, true>
                        : &launch_plain<
                            ARENAS, true, true, false, true, SIN_FRAGMENT>;
                if (use_regb_x32)
                    return &launch_plain<
                        ARENAS, true, true, true, false, SIN_FRAGMENT>;
            }
            if (use_pad)
                return use_tiled_kr
                    ? &launch_plain<
                        ARENAS, true, true, false, false, SIN_FRAGMENT>
                    : &launch_plain<
                        ARENAS, true, false, false, false, SIN_FRAGMENT>;
            return use_tiled_kr
                ? &launch_plain<
                    ARENAS, false, true, false, false, SIN_FRAGMENT>
                : &launch_plain<
                    ARENAS, false, false, false, false, SIN_FRAGMENT>;
        };
        Csplit64ScanLauncher plain_scan;
        if (use_plain_common_nw8)
            plain_scan = &launch_plain_nw8;
        else if (use_tb3)
            plain_scan = use_sin_fragment
                ? select_multi_arena_plain.template operator()<3, true>()
                : select_multi_arena_plain.template operator()<3, false>();
        else if (use_db)
            plain_scan = use_sin_fragment
                ? select_multi_arena_plain.template operator()<2, true>()
                : select_multi_arena_plain.template operator()<2, false>();
        else
            plain_scan = use_sin_fragment
                ? select_one_arena_plain.template operator()<true>()
                : select_one_arena_plain.template operator()<false>();
        SegmentOutputLauncher segment_output = use_output_x32
            ? &launch_segment_output_x32
            : disable_output_gll
                ? nullptr
                : use_sin_fragment
                    ? (use_replay_tb3
                        ? (use_replay_decay_cache
                            ? &launch_segment_output_gll<false, 3, true, true>
                            : &launch_segment_output_gll<false, 3, false, true>)
                        : &launch_segment_output_gll<false, 2, false, true>)
                : disable_output_gll_sin
                    ? (use_replay_tb3
                        ? &launch_segment_output_gll<false, 3>
                        : &launch_segment_output_gll<false, 2>)
                    : (use_replay_tb3
                        ? (use_replay_decay_cache
                            ? &launch_segment_output_gll<true, 3, true>
                            : &launch_segment_output_gll<true, 3, false>)
                        : &launch_segment_output_gll<true, 2>);
        // CDNA4 global-to-LDS replay is the production default.  Setting
        // OUT_GLL=0 rolls back to the common operator; OUT_X32 explicitly
        // selects the older architecture-private diagnostic instead.
        HipLaunchPolicy policy{16, default_k2_route, false,
                &launch_k6_nw8_x32,
                nullptr,
                plain_scan,
                &launch_plain_nw8,
                segment_output,
                &launch_plain_k1, &launch_vsplit_rs_x32,
                bt16_k1_disabled() ? nullptr : &launch_bt16_k1,
                false, false, nullptr, nullptr, nullptr, nullptr};
        policy.launch_context_parallel = &launch_context_parallel;
        policy.use_bt16_k1_for_plain = true;
        policy.bt16_k1_plain_beta_cache = plain_beta_cache_active();
        policy.plain_k1_suffix_decay_cache =
            plain_suffix_decay_cache_active();
        policy.context_group_chunks = context_route.group_chunks;
        policy.context_direct_max_chunks = context_route.direct_max_chunks;
        policy.context_automatic_gva_packed_nw4 =
            context_route.automatic_gva_packed_nw4;
        policy.context_automatic_gva_equal_n4_g16 =
            context_route.automatic_gva_equal_n4_g16;
        policy.context_direct_prefixless =
            use_context_direct_prefixless;
        policy.context_equal_dense_n4_g64 =
            use_context_equal_dense_n4_g64;
        // The explicit fused-K1 rollback reaches the legacy split producer,
        // which still assumes H_q == H.  Keep GVA on the common producer only
        // for that diagnostic setting; the production fused mode supports it.
        policy.bt16_k1_supports_gva =
            bt16_fused_mode() != Bt16FusedMode::disabled;
        policy.bt16_k1_context_operand_cache =
            context_operand_cache_active();
        policy.plain_csplit_supports_gva =
            policy.bt16_k1_supports_gva &&
            policy.launch_bt16_k1 != nullptr &&
            policy.use_bt16_k1_for_plain &&
            policy.launch_plain_k1 != nullptr;
        if (use_context_persistent) {
            policy.launch_context_prefix = &launch_context_persistent_prefix;
            policy.context_persistent_blocks = device.cu_count;
        }
        return policy;
    }

private:
    enum class Bt16FusedMode {
        disabled,
        vector_x32,
        exact_x32,
        exact_x16,
    };

    template <bool VL, bool EXACT_PREP, bool USE_X32,
              bool CACHE_CONTEXT_OPERANDS, bool PUBLISH_ACTIVATED_BETA,
              bool PACKED_DIRECT_PREFIXLESS,
              bool DENSE_N1_ALL_FULL_C16, bool GVA = false>
    static void launch_bt16_fused(
            const Bt16K1Launch& a, const dim3& grid) {
        static_assert(!GVA ||
                          (!CACHE_CONTEXT_OPERANDS &&
                           !PUBLISH_ACTIVATED_BETA &&
                           !PACKED_DIRECT_PREFIXLESS &&
                           !DENSE_N1_ALL_FULL_C16),
                      "GVA fused K1 is restricted to the ordinary route");
        static_assert(!PACKED_DIRECT_PREFIXLESS || VL,
                      "prefixless fused K1 launch is packed-only");
        static_assert(!DENSE_N1_ALL_FULL_C16 || !VL,
                      "full-C16 fused K1 launch is dense-only");
        static_assert(!DENSE_N1_ALL_FULL_C16 || !EXACT_PREP,
                      "full-C16 fused K1 requires production vector prep");
        static_assert(!DENSE_N1_ALL_FULL_C16 || USE_X32,
                      "full-C16 fused K1 requires the production X32 solve");
        static_assert(!DENSE_N1_ALL_FULL_C16 ||
                          CACHE_CONTEXT_OPERANDS,
                      "full-C16 fused K1 is context-cache-only");
        static_assert(!DENSE_N1_ALL_FULL_C16 ||
                          PUBLISH_ACTIVATED_BETA,
                      "full-C16 fused K1 must publish activated beta");
        auto launch = [&]<bool OPT, bool PADDED_SOLVE,
                          bool EARLY_DENSE_BETA>() {
            static_assert(!PADDED_SOLVE || DENSE_N1_ALL_FULL_C16,
                          "padded solve requires the full-C16 launch proof");
            static_assert(!PADDED_SOLVE || USE_X32,
                          "padded solve requires the X32 launch");
            static_assert(!PADDED_SOLVE || OPT,
                          "padded solve requires the production schedule");
            static_assert(!PADDED_SOLVE || CACHE_CONTEXT_OPERANDS,
                          "padded solve is context-cache-only");
            static_assert(!PADDED_SOLVE || PUBLISH_ACTIVATED_BETA,
                          "padded solve must publish activated beta");
            static_assert(!EARLY_DENSE_BETA ||
                              (DENSE_N1_ALL_FULL_C16 && !EXACT_PREP &&
                               USE_X32 && OPT && CACHE_CONTEXT_OPERANDS &&
                               PUBLISH_ACTIVATED_BETA),
                          "early beta requires the production dense-N1 "
                          "full-C16 contract");
            k1_kda_bt16_fused_kernel<
                VL, EXACT_PREP, USE_X32, OPT, OPT, OPT,
                CACHE_CONTEXT_OPERANDS, PUBLISH_ACTIVATED_BETA,
                PACKED_DIRECT_PREFIXLESS, DENSE_N1_ALL_FULL_C16,
                PADDED_SOLVE, EARLY_DENSE_BETA, GVA>
                <<<grid, 256, 0, a.stream>>>(
                    a.q, a.k, a.g, a.beta, a.A_log, a.dt_bias,
                    a.scale, a.gate_scale, a.T_seq, a.H, a.H_q,
                    a.kd, a.qd, a.kr, a.gt, a.kinv, a.inv, a.mqk,
                    PUBLISH_ACTIVATED_BETA ? a.beta_cache : nullptr,
                    VL ? a.cu_seqlens : nullptr,
                    VL && !PACKED_DIRECT_PREFIXLESS
                        ? a.tile_prefix : nullptr,
                    a.N, a.total_tiles);
        };
        // The selector already proved the production fused schedule.  Compile
        // only that candidate symbol; the diagnostic OPT=false specialization
        // retains the established tenth and eleventh bits below.  Padding and
        // early beta are independent exact opt-ins under the same shape proof.
        if constexpr (DENSE_N1_ALL_FULL_C16) {
            const bool padded_solve =
                bt16_dense_n1_padded_solve_enabled();
            auto launch_dense = [&]<bool EARLY_DENSE_BETA>() {
                if (padded_solve)
                    launch.template operator()<
                        true, true, EARLY_DENSE_BETA>();
                else
                    launch.template operator()<
                        true, false, EARLY_DENSE_BETA>();
            };
            if (bt16_dense_n1_early_beta_enabled())
                launch_dense.template operator()<true>();
            else
                launch_dense.template operator()<false>();
        } else {
            const bool opt = bt16_fused_opt_enabled();
            if (opt)
                launch.template operator()<true, false, false>();
            else
                launch.template operator()<false, false, false>();
        }
    }

    static bool bt16_dense_n1_all_full_c16_enabled(
            const Bt16K1Launch& a,
            Bt16FusedMode fused,
            bool cache_context_operands) {
        // This removes all dense mapping/tail predicates from K1.  Keep it a
        // strict, whole-shape proof on the candidate single-sequence context
        // buckets: malformed environment values, unnormalized multi-sequence
        // packed inputs, equal-dense N4, partial tiles, and non-production
        // fused schedules all retain the established tenth template argument
        // (false).  The 1K/2K extension changes only this host admission: the
        // device kernel is already tile-parallel and receives the same full-C16
        // proof for every block.  Public/raw-v2 packed N=1 is intentionally
        // normalized to this byte-equivalent dense geometry before policy
        // construction.
        if (!env_exact(
                "FLASH_KDA_GFX950_BT16_DENSE_N1_ALL_FULL_C16", "1"))
            return false;
        return cache_context_operands && a.cache_context_operands &&
            !a.is_varlen && a.N == 1 &&
            (a.T_seq == 256 || a.T_seq == 512 ||
             a.T_seq == 1024 || a.T_seq == 2048) &&
            a.T_seq % WorkspaceSizes::CHUNK == 0 &&
            a.NT == a.T_seq / WorkspaceSizes::CHUNK &&
            a.total_tiles == a.NT &&
            fused == Bt16FusedMode::vector_x32 &&
            bt16_fused_opt_enabled();
    }

    static bool bt16_fused_opt_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_BT16_FUSED_OPT");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool bt16_dense_n1_padded_solve_enabled() {
        // This is deliberately not a default-on tuning axis.  The caller only
        // consults it inside the compile-time dense-N1/full-C16 branch, whose
        // selector already proves vector-x32, OPT, cached operands, activated
        // beta publication, and one of the exact candidate sequence buckets.
        return env_exact(
            "FLASH_KDA_GFX950_BT16_DENSE_N1_PADDED_SOLVE", "1");
    }

    static bool bt16_dense_n1_early_beta_enabled() {
        // This is an independent A/B axis under the same whole-shape proof as
        // padded solve.  Unset and every noncanonical value retain the late
        // wave-0 beta calculation and therefore preserve the zero-env path.
        return env_exact(
            "FLASH_KDA_GFX950_BT16_DENSE_N1_EARLY_BETA", "1");
    }

    static void launch_bt16_k1(const Bt16K1Launch& a) {
        const dim3 grid = a.is_varlen ? dim3(a.total_tiles, a.H)
                                      : dim3(a.NT, a.N * a.H);
        const Bt16FusedMode fused = bt16_fused_mode();
        if (fused != Bt16FusedMode::disabled) {
            if (a.H_q != a.H) {
                auto launch_gva = [&]<bool VL>() {
                    if (fused == Bt16FusedMode::exact_x16)
                        launch_bt16_fused<
                            VL, true, false, false, false, false, false,
                            true>(a, grid);
                    else if (fused == Bt16FusedMode::exact_x32)
                        launch_bt16_fused<
                            VL, true, true, false, false, false, false,
                            true>(a, grid);
                    else
                        launch_bt16_fused<
                            VL, false, true, false, false, false, false,
                            true>(a, grid);
                };
                if (a.is_varlen)
                    launch_gva.template operator()<true>();
                else
                    launch_gva.template operator()<false>();
                return;
            }
            // Common dispatch has already matched this request to the policy
            // capability published by make().  Trust the per-launch bit so K1
            // and K2 cannot disagree after independently re-reading env.
            const bool cache_context_operands = a.cache_context_operands;
            const bool publish_plain_beta =
                a.publish_activated_beta && plain_beta_cache_enabled();
            const bool dense_n1_all_full_c16 =
                bt16_dense_n1_all_full_c16_enabled(
                    a, fused, cache_context_operands);
            auto launch = [&]<bool VL, bool CACHE_CONTEXT_OPERANDS,
                              bool PUBLISH_ACTIVATED_BETA,
                              bool PACKED_DIRECT_PREFIXLESS,
                              bool DENSE_N1_ALL_FULL_C16>() {
                if constexpr (DENSE_N1_ALL_FULL_C16)
                    launch_bt16_fused<
                        VL, false, true, CACHE_CONTEXT_OPERANDS,
                        PUBLISH_ACTIVATED_BETA,
                        PACKED_DIRECT_PREFIXLESS, true>(a, grid);
                else if (fused == Bt16FusedMode::exact_x16)
                    launch_bt16_fused<
                        VL, true, false, CACHE_CONTEXT_OPERANDS,
                        PUBLISH_ACTIVATED_BETA,
                        PACKED_DIRECT_PREFIXLESS,
                        DENSE_N1_ALL_FULL_C16>(a, grid);
                else if (fused == Bt16FusedMode::exact_x32)
                    launch_bt16_fused<
                        VL, true, true, CACHE_CONTEXT_OPERANDS,
                        PUBLISH_ACTIVATED_BETA,
                        PACKED_DIRECT_PREFIXLESS,
                        DENSE_N1_ALL_FULL_C16>(a, grid);
                else
                    launch_bt16_fused<
                        VL, false, true, CACHE_CONTEXT_OPERANDS,
                        PUBLISH_ACTIVATED_BETA,
                        PACKED_DIRECT_PREFIXLESS,
                        DENSE_N1_ALL_FULL_C16>(a, grid);
            };
            auto dispatch = [&]<bool VL, bool PACKED_DIRECT_PREFIXLESS,
                                bool DENSE_N1_ALL_FULL_C16>() {
                if constexpr (DENSE_N1_ALL_FULL_C16) {
                    launch.template operator()<
                        VL, true, true, PACKED_DIRECT_PREFIXLESS, true>();
                } else {
                    if (cache_context_operands)
                        launch.template operator()<
                            VL, true, true, PACKED_DIRECT_PREFIXLESS, false>();
                    else if (publish_plain_beta)
                        launch.template operator()<
                            VL, false, true, PACKED_DIRECT_PREFIXLESS, false>();
                    else
                        launch.template operator()<
                            VL, false, false, PACKED_DIRECT_PREFIXLESS, false>();
                }
            };
            if (a.is_varlen) {
                if (a.packed_direct_prefixless)
                    dispatch.template operator()<true, true, false>();
                else
                    dispatch.template operator()<true, false, false>();
            } else if (dense_n1_all_full_c16) {
                dispatch.template operator()<false, false, true>();
            } else {
                dispatch.template operator()<false, false, false>();
            }
            return;
        }
        if (a.is_varlen) {
            k1_kda_split_prep_kernel<true><<<grid, 256, 0, a.stream>>>(
                a.q, a.k, a.g, a.A_log, a.dt_bias, a.scale, a.gate_scale,
                a.T_seq, a.H, a.kd, a.qd, a.kr, a.gt, a.kinv,
                reinterpret_cast<float*>(a.mqk), a.cu_seqlens,
                a.tile_prefix, a.N, a.total_tiles,
                nullptr, nullptr, nullptr, 0);
            k1_kda_split_solve_kernel<true><<<grid, 64, 0, a.stream>>>(
                a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                a.cu_seqlens, a.tile_prefix, a.N, a.total_tiles,
                a.T_seq, a.H);
        } else {
            k1_kda_split_prep_kernel<false><<<grid, 256, 0, a.stream>>>(
                a.q, a.k, a.g, a.A_log, a.dt_bias, a.scale, a.gate_scale,
                a.T_seq, a.H, a.kd, a.qd, a.kr, a.gt, a.kinv,
                reinterpret_cast<float*>(a.mqk), nullptr, nullptr,
                a.N, a.total_tiles, nullptr, nullptr, nullptr, 0);
            k1_kda_split_solve_kernel<false><<<grid, 64, 0, a.stream>>>(
                a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                nullptr, nullptr, a.N, a.total_tiles, a.T_seq, a.H);
        }
    }

    static bool bt16_k1_disabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_BT16_K1");
        return value != nullptr && value[0] == '0' && value[1] == '\0';
    }

    static bool context_operand_cache_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool context_operand_cache_active() {
        return context_operand_cache_enabled() && !bt16_k1_disabled() &&
            bt16_fused_mode() != Bt16FusedMode::disabled;
    }

    static bool plain_beta_cache_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_PLAIN_BETA_CACHE");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool plain_beta_cache_active() {
        return plain_beta_cache_enabled() && !bt16_k1_disabled() &&
            bt16_fused_mode() != Bt16FusedMode::disabled;
    }

    static bool plain_suffix_decay_cache_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_SCAN_DECAY_CACHE");
        // Unlike production-default tuning axes, this experiment is strict
        // opt-in.  Unset, "0", and every noncanonical value roll back.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool plain_suffix_decay_cache_active() {
        return plain_suffix_decay_cache_enabled() && !bt16_k1_disabled() &&
            fused_k1_enabled();
    }

    static Bt16FusedMode bt16_fused_mode() {
        const char* value = std::getenv("FLASH_KDA_GFX950_BT16_FUSED");
        // Exact preparation restores the monolithic kernel's normalization
        // and decay rounding order, while CDNA4 K32 contraction is bit-exact
        // to the X16 solve for this operator.  Keep both the preservation
        // split and diagnostic template axes available without changing the
        // architecture-neutral route.
        if (value == nullptr || std::strcmp(value, "1") == 0 ||
            std::strcmp(value, "vector_x32") == 0)
            return Bt16FusedMode::vector_x32;
        if (std::strcmp(value, "exact_x32") == 0)
            return Bt16FusedMode::exact_x32;
        if (std::strcmp(value, "0") == 0)
            return Bt16FusedMode::disabled;
        if (std::strcmp(value, "exact_x16") == 0)
            return Bt16FusedMode::exact_x16;
        return Bt16FusedMode::vector_x32;
    }

    template <bool VL, bool PADDED, bool PRE_SOLVED,
              bool BETA_ACTIVATED, bool PUBLISH_SUFFIX_DECAY,
              bool ELIDE_DEAD_STORES, bool MERGE_PRE_SOLVED_LOADS,
              bool FRAGMENT_FORWARD>
    static void launch_fused_plain_k1(
            const PlainCsplit64K1Launch& a, const dim3& grid) {
        k1_kda_postprep_fused_kernel<
            VL, true, true, PADDED, PRE_SOLVED, BETA_ACTIVATED,
            PUBLISH_SUFFIX_DECAY, ELIDE_DEAD_STORES,
            MERGE_PRE_SOLVED_LOADS, FRAGMENT_FORWARD>
            <<<grid, 256, 0, a.stream>>>(
                BETA_ACTIVATED ? a.beta_cache : a.beta,
                a.kd, a.qd, a.kr, a.gt, a.kinv, a.inv,
                a.mqk, a.cross32, a.cross64,
                PUBLISH_SUFFIX_DECAY ? a.suffix_decay : nullptr,
                VL ? a.cu_seqlens : nullptr,
                VL ? a.tile_prefix : nullptr,
                VL ? a.pair_prefix : nullptr,
                VL ? a.segment_prefix : nullptr,
                a.N, a.total_tiles, a.total_pairs, a.total_segments,
                a.T_seq, a.H, a.NT);
    }

    template <bool VL, bool PADDED, bool ELIDE_DEAD_STORES,
              bool MERGE_PRE_SOLVED_LOADS, bool FRAGMENT_FORWARD>
    static void launch_fused_plain_k1_specialization(
            const PlainCsplit64K1Launch& a, const dim3& grid) {
        if (!a.pre_solved) {
            launch_fused_plain_k1<
                VL, PADDED, false, false, false, false, false, false>(
                    a, grid);
            return;
        }
        if (a.beta_activated) {
            if (a.publish_suffix_decay)
                launch_fused_plain_k1<
                    VL, PADDED, true, true, true, ELIDE_DEAD_STORES,
                    MERGE_PRE_SOLVED_LOADS, FRAGMENT_FORWARD>(a, grid);
            else
                launch_fused_plain_k1<
                    VL, PADDED, true, true, false, ELIDE_DEAD_STORES,
                    MERGE_PRE_SOLVED_LOADS, FRAGMENT_FORWARD>(a, grid);
        } else if (a.publish_suffix_decay) {
            launch_fused_plain_k1<
                VL, PADDED, true, false, true, ELIDE_DEAD_STORES,
                MERGE_PRE_SOLVED_LOADS, FRAGMENT_FORWARD>(a, grid);
        } else {
            launch_fused_plain_k1<
                VL, PADDED, true, false, false, ELIDE_DEAD_STORES,
                MERGE_PRE_SOLVED_LOADS, FRAGMENT_FORWARD>(a, grid);
        }
    }

    template <bool VL, bool PADDED>
    static void launch_fused_plain_k1_dispatch(
            const PlainCsplit64K1Launch& a, const dim3& grid) {
        const bool elide_dead_stores =
            postprep_dead_stores_enabled();
        const bool merge_pre_solved_loads =
            postprep_merged_loads_enabled();
        if constexpr (PADDED) {
            if (elide_dead_stores && merge_pre_solved_loads &&
                postprep_fragment_forward_enabled()) {
                launch_fused_plain_k1_specialization<
                    VL, PADDED, true, true, true>(a, grid);
                return;
            }
        }
        if (elide_dead_stores) {
            if (merge_pre_solved_loads)
                launch_fused_plain_k1_specialization<
                    VL, PADDED, true, true, false>(a, grid);
            else
                launch_fused_plain_k1_specialization<
                    VL, PADDED, true, false, false>(a, grid);
        } else if (merge_pre_solved_loads) {
            launch_fused_plain_k1_specialization<
                VL, PADDED, false, true, false>(a, grid);
        } else {
            launch_fused_plain_k1_specialization<
                VL, PADDED, false, false, false>(a, grid);
        }
    }

    static void launch_plain_k1(const PlainCsplit64K1Launch& a) {
        if (fused_k1_enabled()) {
            const dim3 grid = a.is_varlen
                ? dim3(a.total_segments, a.H)
                : dim3((a.NT + 3) / 4, a.N * a.H);
            const bool padded = fused_k1_padded_enabled();
            if (a.is_varlen) {
                if (padded)
                    launch_fused_plain_k1_dispatch<true, true>(a, grid);
                else
                    launch_fused_plain_k1_dispatch<true, false>(a, grid);
            } else {
                if (padded)
                    launch_fused_plain_k1_dispatch<false, true>(a, grid);
                else
                    launch_fused_plain_k1_dispatch<false, false>(a, grid);
            }
            return;
        }

        if (a.is_varlen) {
            if (!a.pre_solved)
                k1_kda_split_solve_kernel<true>
                    <<<dim3(a.total_tiles, a.H), 64, 0, a.stream>>>(
                        a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                        a.cu_seqlens, a.tile_prefix, a.N, a.total_tiles,
                        a.T_seq, a.H);
            k1_kda_bt32_merge_kernel<true>
                <<<dim3(a.total_pairs, a.H), 64, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cu_seqlens, a.tile_prefix, a.pair_prefix, a.N,
                    a.total_tiles, a.total_pairs, a.T_seq, a.H, a.NT);
            k1_kda_bt64_merge_kernel<true>
                <<<dim3(a.total_segments, a.H), 256, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cross64, a.cu_seqlens, a.tile_prefix, a.pair_prefix,
                    a.segment_prefix, a.N, a.total_tiles, a.total_pairs,
                    a.total_segments, a.T_seq, a.H, a.NT);
        } else {
            if (!a.pre_solved)
                k1_kda_split_solve_kernel<false>
                    <<<dim3(a.NT, a.N * a.H), 64, 0, a.stream>>>(
                        a.beta, a.kd, a.qd, a.kinv, a.inv, a.mqk,
                        nullptr, nullptr, a.N, a.total_tiles, a.T_seq, a.H);
            k1_kda_bt32_merge_kernel<false>
                <<<dim3((a.NT + 1) / 2, a.N * a.H), 64, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    nullptr, nullptr, nullptr, a.N, a.total_tiles,
                    a.total_pairs, a.T_seq, a.H, a.NT);
            k1_kda_bt64_merge_kernel<false>
                <<<dim3((a.NT + 3) / 4, a.N * a.H), 256, 0, a.stream>>>(
                    a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cross64, nullptr, nullptr, nullptr, nullptr, a.N,
                    a.total_tiles, a.total_pairs, a.total_segments,
                    a.T_seq, a.H, a.NT);
        }
    }

    // Retain the former gfx950-private three-launch path as an exact A/B and
    // rollback hook.  Explicit common routes remain controlled by FLASH_KDA_K2.
    static bool fused_k1_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_FUSED_K1");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool fused_k1_padded_enabled() {
        // The padded gfx950 operator is the production default.  Keep the
        // former 38-KiB layout as an exact architecture-private A/B rollback.
        const char* value = std::getenv("FLASH_KDA_GFX950_FUSED_K1_PADDED");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool postprep_dead_stores_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_POSTPREP_DEAD_STORES");
        // This experiment is strict opt-in: only canonical "1" selects the
        // store-elided specialization; unset, "0", and every other spelling
        // retain the established instruction stream.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool postprep_merged_loads_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_POSTPREP_MERGED_LOADS");
        // The merged schedule is legal only inside the PRE_SOLVED kernel
        // specialization selected above.  Keep the environment contract
        // canonical so inherited diagnostic values cannot enable it.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool postprep_fragment_forward_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_POSTPREP_FRAGMENT_FORWARD");
        // The fragment route is a third strict opt-in layered only on the
        // canonical dead-store + merged-load production specialization.
        // Every noncanonical spelling retains one of the four P0/P1 axes
        // above, preventing accidental specialization growth in inherited
        // benchmark environments.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool output_x32_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_X32");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool output_gll_disabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_GLL");
        return value != nullptr && value[0] == '0' && value[1] == '\0';
    }

    static bool output_gll_sin_disabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_GLL_SIN");
        return value != nullptr && value[0] == '0' && value[1] == '\0';
    }

    static bool sin_fragment_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_SIN_FRAGMENT");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    // Three arenas fit two CTAs/CU even with the padded incoming-state cache
    // and remove one BT64 rendezvous.  Keep the former two-arena schedule as
    // an exact architecture-private A/B and rollback hook.
    static bool output_gll_tb3_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_OUT_GLL_TB3");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    // Exact replay-only optimization.  Each DMA publication converts the
    // 128 raw gate totals to FP32 decay once in LDS before the existing CTA
    // barrier, instead of reissuing the same exp2 in every V-row wave.
    static bool replay_decay_cache_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_REPLAY_DECAY_CACHE");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    // A third LDS arena is the gfx950 low-grid default.  Delaying slot reuse by
    // two complete segments removes the double-buffer path's segment-end reuse
    // barrier without changing any numerical operation.  `0` is the exact
    // two-arena rollback used by paired performance and PMC checks.
    static bool triple_buffer_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_SCAN_TB3");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    // gfx950 can form both state-carry MFMA operands with native LDS
    // transpose reads when kr is published in K16-major tiles.  Keep the
    // established row-major publication as an exact A/B and rollback path.
    static bool tiled_kr_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_SCAN_KR_TR");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool plain_common_nw8_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_SCAN_COMMON_NW8");
        // Reuse the already-instantiated common NW8 scan behind a strict
        // production-route diagnostic.  Unlike FLASH_KDA_K2=csplit64nw8,
        // this keeps the fused BT16/post-preparation producer unchanged so
        // the scan topology can be measured in isolation.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool use_double_buffer(
            const FwdParams& p, const HipDeviceInfo& device) {
        const char* value = std::getenv("FLASH_KDA_GFX950_SCAN_DB");
        if (value != nullptr) {
            if (value[0] == '0' && value[1] == '\0') return false;
            if (value[0] == '1' && value[1] == '\0') return true;
        }
        // Every multi-arena specialization admits one CTA/CU.  Select one only
        // when the complete V16 scan grid already fits in one device wave, so
        // the extra LDS cannot reduce residency, and only after enough BT64
        // segments exist to amortize the larger prologue.
        const int64_t scan_ctas = int64_t(p.N) * p.H * (128 / 16);
        const int T_seq = p.T_total / p.N;
        return T_seq >= 1024 && scan_ctas <= device.cu_count;
    }

    // Pitch-20 FP32 partial rows remove the half-wave pitch-16 bank alias.
    // Keep an exact rollback for PMC and regression A/B.
    static bool partial_padding_enabled() {
        const char* value = std::getenv("FLASH_KDA_GFX950_PARTIAL_PAD");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }

    static bool scan_regb_x32_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_SCAN_REGB_X32");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool scan_state_xchg_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_SCAN_STATE_XCHG");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool scan_state_xchg_x16_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_SCAN_STATE_XCHG_X16");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool scan_rhs_fragment_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_SCAN_RHS_FRAGMENT");
        // Strict opt-in: unset, "0", and every noncanonical spelling retain
        // the established row-major corrected-RHS / umat-output dataflow.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_tight_scan_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN");
        // This changes only the hybrid affine-scan launch/mapping.  Keep the
        // experiment strict opt-in so unset, "0", and noncanonical values use
        // the established sequence-indexed N*H scan byte for byte.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_scan_b_stream_enabled(int group_chunks) {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM");
        // Three-seed gfx950 measurements graduate streamed-b for G8/G16: it
        // shortens the live FP32 b state without changing the reduction tree.
        // An explicit value remains authoritative, and every noncanonical
        // spelling conservatively selects the established scan.
        if (value != nullptr)
            return value[0] == '1' && value[1] == '\0';
        return group_chunks == 8 || group_chunks == 16;
    }

    static bool context_scan_a_gll_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL");
        // This experiment is valid only for the NW2 streamed-b scan.  Keep
        // inherited, unset, and noncanonical values on the established A
        // publication path; launch dispatch enforces the remaining guards.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_scan_b_phased_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED");
        // This HI=false/NW2 experiment is layered below streamed-b and below
        // A-GLL in dispatch precedence.  All noncanonical values retain the
        // already-selected scan specialization byte for byte.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_scan_ksplit_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT");
        // The K64+K64 partial merge deliberately changes the FP32 reduction
        // tree.  Keep it behind a canonical opt-in; every inherited, unset,
        // zero, or noncanonical value retains the established scan symbols.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_scan_ksplit_prefetch_b_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT_PREFETCH_B");
        // Keep the early-b experiment independent of the established
        // K-split symbol.  Only canonical "1" selects the G64 candidate;
        // unset, zero, and every other spelling retain the original kernel.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_u_forward_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_U_FORWARD");
        // Register-forwarding is the graduated production transport.  An exact
        // "0" disables only this axis for rollback/A-B work; unset and every
        // other spelling retain the production default.  Set both U and V to
        // exact "0" to recover the original LDS publication/read path.
        return value == nullptr ||
            !(value[0] == '0' && value[1] == '\0');
    }

    static bool context_v_forward_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_V_FORWARD");
        // Register-forwarding is the graduated production vnew transport into
        // INV@vnew.  An exact "0" disables only this axis; unset and every
        // other spelling keep it enabled.  Both axes must be exact "0" to
        // select the original U/V LDS transport in full.
        return value == nullptr ||
            !(value[0] == '0' && value[1] == '\0');
    }

    static bool context_nw8_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_NW8");
        // One 512-thread CTA covers all eight V16 slices and shares the
        // V-independent publication once.  Keep it strict opt-in until the
        // larger workgroup clears gfx950 correctness and performance gates.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_tail_first_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_TAIL_FIRST");
        // Rotate only the direct sequence-slot mapping while retaining the
        // established wide-X 2D launch.  Keep the experiment behind an exact
        // opt-in so every unset or noncanonical spelling is a strict rollback.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_nw1_flat_tail_first_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW1_FLAT_TAIL_FIRST");
        // Flattening the NW1 V16 axis into grid.x is a distinct scheduling
        // experiment.  Only canonical "1" enables it; unset, exact rollback,
        // and malformed inherited values retain the established 2D grid.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_dense_all_full_c16_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_DENSE_ALL_FULL_C16");
        // Paired gfx950 measurements establish this specialization across the
        // aligned dense N=1 NW1-flat buckets.  Exact "0" remains the complete
        // rollback; every other spelling keeps the graduated default.
        return value == nullptr ||
            !(value[0] == '0' && value[1] == '\0');
    }

    static bool context_direct_dense_n1_h12_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_DENSE_N1_H12");
        // This specialization removes runtime geometry checks and address
        // axes.  Only the canonical opt-in may enable it; unset, zero, and
        // malformed inherited values retain the established flat symbol.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_paired_state_products_x32_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_PAIRED_STATE_X32");
        // This deliberately lengthens the Qd@state result lifetime while
        // sharing its state conversion with Kd@state.  Keep it strict opt-in
        // until the VGPR/latency tradeoff clears the gfx950 performance gate.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_nw1_wave_barrier_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW1_WAVE_BARRIER");
        // The specialization changes only synchronization within a single
        // 64-lane workgroup.  Retain a strict opt-in until real gfx950 timing
        // confirms that replacing s_barrier outweighs its scheduling effect.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static ContextDirectKsplitMode context_direct_ksplit_mode() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_KSPLIT");
        // K-split changes both the reduction order and the workgroup shape.
        // Keep every specialization behind one exact spelling so an inherited
        // or malformed environment cannot silently alter production numerics.
        if (value == nullptr || std::strcmp(value, "0") == 0)
            return ContextDirectKsplitMode::disabled;
        if (std::strcmp(value, "2") == 0)
            return ContextDirectKsplitMode::waves2;
        if (std::strcmp(value, "4") == 0)
            return ContextDirectKsplitMode::waves4;
        if (std::strcmp(value, "tail2") == 0)
            return ContextDirectKsplitMode::tailpipe_waves2;
        if (std::strcmp(value, "tail4") == 0)
            return ContextDirectKsplitMode::tailpipe_waves4;
        return ContextDirectKsplitMode::disabled;
    }

    static bool context_direct_ksplit_tail_mqk_prefetch_enabled() {
        // This changes only the tail4 Mqk scheduling point.  Keep the nested
        // experiment behind one canonical spelling so unset, rollback, and
        // malformed inherited values retain the established tail4 schedule.
        return env_exact(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_KSPLIT_TAIL_MQK_PREFETCH",
            "1");
    }

    static bool context_direct_ksplit_long_n1_h12_enabled() {
        // This is a nested reachability bit for the tail4 template, not an
        // independent route request.  Only canonical "1" admits the isolated
        // 1K/2K candidate; unset, rollback, and malformed values preserve the
        // established 256/512 symbols and all zero-environment routing.
        return env_exact(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_KSPLIT_LONG_N1_H12",
            "1");
    }

    static bool context_direct_global_n1_h12_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_GLOBAL_N1_H12");
        // This removes the complete LDS publication/consumption ABI from the
        // short dense replay.  Admit only an explicit canonical opt-in while
        // correctness and cache behavior are measured on real gfx950.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_global_kr_gll_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_GLOBAL_KR_GLL");
        // This is a transport submode of the strict direct-global route, not
        // an independent route request.  Only canonical "1" selects it;
        // unset, zero, and every malformed spelling retain the control path.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_global_kq_gll_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_GLOBAL_KQ_GLL");
        // Kd/Qd GLL is another transport submode of direct-global.  Keep it
        // independent from Kr-GLL so the 8 KiB candidate and their 12 KiB
        // composition can be measured without changing the route request.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_direct_prefixless_enabled(
            const FwdParams& p,
            const ContextRouteConfig& route,
            K2DefaultRoute default_k2_route) {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_DIRECT_PREFIXLESS");
        const bool explicit_prefixless =
            value != nullptr && value[0] == '1' && value[1] == '\0';
        // Fifteen one-token decodes plus one 1K prefill are the measured ATOM
        // boundary batch.  Its aggregate raw-v2 signature is N=16,
        // total==bound+15, total_tiles=81.  The bound is a caller promise, not
        // a device-verified maximum, so other packed distributions can share
        // that signature; both prefixless mappings remain general for every
        // N<=16 distribution.  NW1-flat plus the prefixless full-chunk
        // specialization wins all four decode-first/prefill-first 1024/1025
        // cases.  Keep this graduation narrow: equal 16x1K is materially
        // faster on the established NW4 topology.
        const bool automatic_mixed_boundary_prefixless =
            value == nullptr && p.cu_seqlens != nullptr && p.N == 16 &&
            (p.max_seqlen_upper_bound == 1024 ||
             p.max_seqlen_upper_bound == 1025) &&
            p.T_total == p.max_seqlen_upper_bound + 15 &&
            p.total_tiles == 81 &&
            std::getenv("FLASH_KDA_GFX950_CONTEXT_DIRECT") == nullptr &&
            std::getenv("FLASH_KDA_GFX950_CONTEXT_AFFINE") == nullptr &&
            std::getenv("FLASH_KDA_GFX950_CONTEXT_HYBRID") == nullptr &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS") == nullptr &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW") == nullptr &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW1_FLAT_TAIL_FIRST") ==
                nullptr;
        // This removes a graph node, so it is deliberately stricter than a
        // kernel-local tuning bit.  Outside the exact production graduation,
        // only an exact opt-in plus a complete packed pure-direct fused-K1/K2
        // route may publish the capability to the architecture-neutral
        // launcher.  Every malformed, inherited, or partial recipe retains
        // the established prefix topology.
        if (!explicit_prefixless && !automatic_mixed_boundary_prefixless)
            return false;
        return p.H_q == p.H && p.cu_seqlens != nullptr && p.N > 0 &&
            p.N <= kPackedDirectPrefixlessMaxSequences && p.H > 0 &&
            default_k2_route == K2DefaultRoute::context_parallel &&
            route.group_chunks == 0 && route.direct_max_chunks == 0 &&
            std::getenv("FLASH_KDA_K2") == nullptr &&
            !bt16_k1_disabled() &&
            bt16_fused_mode() != Bt16FusedMode::disabled;
    }

    static bool context_equal_dense_n4_g64_enabled(
            const FwdParams& p,
            const ContextRouteConfig& route,
            K2DefaultRoute default_k2_route) {
        // Unlike a kernel-local schedule bit, this candidate removes the
        // packed prefix launch and changes every downstream workspace index.
        // Publish one capability only after the complete K1 -> affine AB ->
        // K-split scan -> replay recipe has matched.  Common dispatch repeats
        // the geometry proof before replacing packed metadata with the dense
        // N=4 layout.
        if (!env_exact(
                "FLASH_KDA_GFX950_CONTEXT_EQUAL_DENSE_N4_G64", "1"))
            return false;

        constexpr int kSequences = 4;
        constexpr int kSequenceTokens = 4096;
        constexpr int kChunksPerSequence =
            kSequenceTokens / WorkspaceSizes::CHUNK;
        constexpr int kDenseTiles = kSequences * kChunksPerSequence;
        constexpr int kPackedTiles = kDenseTiles + kSequences;
        static_assert(kChunksPerSequence == 256 && kDenseTiles == 1024 &&
                      kPackedTiles == 1028,
                      "equal dense N4/G64 geometry changed");
        if (p.H_q != p.H || p.cu_seqlens == nullptr ||
            p.N != kSequences || p.H <= 0 ||
            p.T_total != kSequences * kSequenceTokens ||
            p.max_seqlen_upper_bound != kSequenceTokens ||
            p.total_tiles != kPackedTiles ||
            default_k2_route != K2DefaultRoute::context_parallel ||
            !route.force_context || route.group_chunks != 64 ||
            route.direct_max_chunks != 0 ||
            std::getenv("FLASH_KDA_K2") != nullptr ||
            bt16_k1_disabled() ||
            bt16_fused_mode() == Bt16FusedMode::disabled)
            return false;

        const char* const bt16_fused =
            std::getenv("FLASH_KDA_GFX950_BT16_FUSED");
        const bool canonical_bt16_fused = bt16_fused == nullptr ||
            std::strcmp(bt16_fused, "1") == 0 ||
            std::strcmp(bt16_fused, "vector_x32") == 0 ||
            std::strcmp(bt16_fused, "exact_x32") == 0 ||
            std::strcmp(bt16_fused, "exact_x16") == 0;
        if (!canonical_bt16_fused ||
            !env_unset_or_exact("FLASH_KDA_GFX950_BT16_K1", "1"))
            return false;

        // Select the automatic packed N4/G64 control recipe first.  Exact
        // rollback spellings remain valid controls, while malformed inherited
        // values or orthogonal context routes cannot silently enter the
        // metadata-free graph.
        if (!env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_DIRECT", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_AFFINE", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_HYBRID", "0") ||
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS") != nullptr ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_DIRECT_PREFIXLESS", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_PERSISTENT", "0"))
            return false;

        // The exact-N4 producer is the established NW4/P0 cached U/V-forward
        // fused recurrence.  Stage-early may independently select its sibling
        // symbol, but only canonical 0/1 spellings are accepted.
        const char* const stage_early = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_STAGE_EARLY");
        const bool canonical_stage_early = stage_early == nullptr ||
            std::strcmp(stage_early, "0") == 0 ||
            std::strcmp(stage_early, "1") == 0;
        if (!canonical_stage_early ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE", "1") ||
            !context_operand_cache_active() ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_U_FORWARD", "1") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_V_FORWARD", "1") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_FUSED", "1") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_NW8", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_B", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_A", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_REPLAY", "0"))
            return false;

        // Reuse the established dense K-split scan and dense replay.  Reject
        // every axis that would substitute a different scan topology or
        // producer/consumer ABI; the packed graph remains the full fallback.
        const char* const scan_nw = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_SCAN_NW");
        const char* const scan_ksplit = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT");
        const bool ksplit_selected =
            (scan_ksplit != nullptr &&
             std::strcmp(scan_ksplit, "1") == 0) ||
            (scan_nw == nullptr && scan_ksplit == nullptr);
        return env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_SCAN_NW", "2") &&
            env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT", "1") &&
            ksplit_selected &&
            env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT_PREFETCH_B", "0") &&
            env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM", "0") &&
            env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL", "0") &&
            env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED", "0") &&
            env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN", "0");
    }

    static bool context_affine_ab_fused_enabled(int group_chunks) {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_FUSED");
        // G8/G16 are the measured dense-N1 production routes and default to
        // the fused B/A producer.  Exact "0" (and every malformed inherited
        // spelling) preserves the two-launch rollback; other group sizes stay
        // opt-in until they clear their own performance gate.
        if (value != nullptr)
            return value[0] == '1' && value[1] == '\0';
        return group_chunks == 8 || group_chunks == 16;
    }

    static bool context_affine_ab_stage_early_enabled() {
        const char* value = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_STAGE_EARLY");
        // This changes only the schedule inside an already-selected fused
        // producer.  Keep it strict opt-in: unset, zero, inherited malformed
        // spellings, and all orthogonal producer recipes retain the established
        // packed/dense symbols byte for byte.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_lds_pipeline_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE");
        // The dual-arena transport is intentionally a fifth specialization:
        // only U=V=true may select P1.  Unset, "0", noncanonical spellings,
        // and every other U/V combination retain the established P0 kernel.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_lds_pipeline_b_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_B");
        // Per-pass experiments are strict additions to the legacy all-pass
        // switch.  Every noncanonical spelling retains the selected P0/P1
        // behavior of FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE.
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_lds_pipeline_a_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_A");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool context_lds_pipeline_replay_enabled() {
        const char* value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_REPLAY");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static bool env_unset_or_exact(
            const char* name, const char* expected) {
        const char* value = std::getenv(name);
        return value == nullptr || std::strcmp(value, expected) == 0;
    }

    static bool env_exact(const char* name, const char* expected) {
        const char* value = std::getenv(name);
        return value != nullptr && std::strcmp(value, expected) == 0;
    }

    static bool context_persistent_established_ab_enabled() {
        // This is a nested experiment, not an independent route selector.
        // The caller consults it only after the complete packed-hybrid G64
        // persistent graph has passed its existing all-or-nothing guard.
        return env_exact(
            "FLASH_KDA_GFX950_CONTEXT_PERSISTENT_ESTABLISHED_AB", "1");
    }

    static bool context_persistent_enabled(
            const FwdParams& p,
            const HipDeviceInfo& device,
            const ContextRouteConfig& route,
            K2DefaultRoute default_k2_route) {
        // This experiment replaces the complete packed prefix + affine B/A +
        // scan + replay topology.  Resolve the contract once while building
        // the policy: if any prerequisite is absent or any orthogonal context
        // experiment is enabled, both the prefix callback and persistent block
        // cap stay null/zero and every established symbol is retained.
        if (!env_exact(
                "FLASH_KDA_GFX950_CONTEXT_PERSISTENT", "1"))
            return false;
        if (p.H_q != p.H || p.cu_seqlens == nullptr ||
            p.N <= 0 || p.H <= 0 ||
            device.cu_count <= 0 ||
            default_k2_route != K2DefaultRoute::context_parallel ||
            route.group_chunks != kHybridCompactGroupChunks ||
            route.direct_max_chunks != kHybridCompactDirectMaxChunks ||
            std::getenv("FLASH_KDA_K2") != nullptr)
            return false;

        // Require an explicit packed-hybrid G64 recipe.  Canonical rollback
        // values and unset optional axes are accepted; malformed inherited
        // spellings cannot accidentally select the new graph.
        if (!env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_DIRECT", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_AFFINE", "0") ||
            !env_exact(
                "FLASH_KDA_GFX950_CONTEXT_HYBRID", "1") ||
            !env_exact(
                "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS", "64") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW", "4") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_NW", "2"))
            return false;

        if (!env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_NW8", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_TIGHT_SCAN", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_B", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_A", "0") ||
            !env_unset_or_exact(
                "FLASH_KDA_GFX950_CONTEXT_LDS_PIPELINE_REPLAY", "0"))
            return false;

        // The new symbols are the canonical P0/NW4 cached recurrence with U
        // and V register forwarding and fused affine maps.  The producer must
        // also be the matched fused BT16 specialization that actually
        // publishes the activated beta/decay cache consumed below.
        return env_unset_or_exact(
                   "FLASH_KDA_GFX950_CONTEXT_OPERAND_CACHE", "1") &&
            context_operand_cache_active() &&
            env_exact(
                "FLASH_KDA_GFX950_CONTEXT_U_FORWARD", "1") &&
            env_exact(
                "FLASH_KDA_GFX950_CONTEXT_V_FORWARD", "1") &&
            env_exact(
                "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_FUSED", "1");
    }

    static ContextRouteConfig resolve_context_route(const FwdParams& p) {
        const char* direct_value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_DIRECT");
        const char* affine_value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_AFFINE");
        const char* hybrid_value =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_HYBRID");
        const bool force_direct =
            direct_value != nullptr && direct_value[0] == '1' &&
            direct_value[1] == '\0';
        const bool force_affine =
            affine_value != nullptr && affine_value[0] == '1' &&
            affine_value[1] == '\0';
        const bool force_hybrid =
            hybrid_value != nullptr && hybrid_value[0] == '1' &&
            hybrid_value[1] == '\0';
        const bool is_varlen = p.cu_seqlens != nullptr;
        const bool is_gva = p.H_q != p.H;
        const bool has_length_hint =
            is_varlen && p.max_seqlen_upper_bound > 0;
        const int64_t hinted_bound = has_length_hint
            ? int64_t(p.max_seqlen_upper_bound) : 0;
        const int64_t T_seq = p.T_total / p.N;
        const char* group_env =
            std::getenv("FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS");

        // A single 256/512-token K3 sequence is faster as one fused-K1 plus
        // pure-direct replay pair than through the four-launch plain C-split
        // topology.  This covers both dense input and the one-sequence packed
        // ABI used by ATOM.  Keep the 128-token V-split endpoint unchanged,
        // and retain explicit route knobs as authoritative diagnostics.
        const bool automatic_short_single_direct =
            p.N == 1 &&
            (T_seq == 256 || T_seq == 512) &&
            !force_direct && !force_affine && !force_hybrid;

        // G8 doubles the affine producer/replay grid at the two measured
        // single-sequence boundary buckets.  The dense-N1 arena role swap is
        // exact there, and three-seed paired measurements establish G8 over
        // both the short direct path and G16.  Do not interpolate this route
        // to unmeasured lengths; explicit route/group knobs remain primary.
        const bool automatic_dense_single_g8 =
            !is_varlen && p.N == 1 &&
            (T_seq == 1024 || T_seq == 2048) && group_env == nullptr &&
            !force_direct && !force_affine && !force_hybrid;

        // A dense single sequence from 2K onward no longer has enough
        // independent plain-scan CTAs to fill a 256-CU gfx950.  The affine
        // topology exposes the time axis as independent context groups and
        // wins decisively at and beyond the measured boundary.  G8 takes the
        // exact 2K bucket above; G16/G32/G64 retain the remaining lengths.
        const bool automatic_dense_single_affine =
            !is_varlen && p.N == 1 && T_seq >= 2048 &&
            !force_direct && !force_affine && !force_hybrid;

        // The raw-v2 bound is a host-routing promise.  Ordinary routes never
        // replace device-prefix geometry with it; the one exact-equality
        // proof below may additionally select an equivalent compact dense
        // mapping in common dispatch.  Two cases have enough static
        // information to remove otherwise-empty affine passes:
        //
        // * a large packed batch whose every sequence is at most C16*64+1;
        // * an equal-length batch proven by max_bound*N == total_tokens,
        //   inside the K3 token-budget range covered by the boundary screen.
        //
        // Explicit diagnostic routes retain precedence, and a zero hint is
        // byte-for-byte the legacy policy decision.
        const bool hinted_equal_lengths = has_length_hint &&
            hinted_bound * int64_t(p.N) == int64_t(p.T_total);
        // The production resume bucket is four equal 4K sequences.  Its
        // direct NW4 recurrence leaves the time axis serial for 256 chunks;
        // measured G64 affine replay exposes sixteen independent ranges and
        // wins by a wide margin.  Keep the automatic graduation exact so the
        // raw-v2 bound remains a routing proof rather than an interpolation.
        const bool automatic_equal_n4_g64 =
            !is_gva && hinted_equal_lengths && p.N == 4 &&
            hinted_bound == 4096 &&
            p.T_total == 4 * 4096 && group_env == nullptr &&
            !force_direct && !force_affine && !force_hybrid;
        // GVA has twice the value/state-head parallelism of the measured
        // Hq=2/Hv=4 resume shape.  Its generic uncached graph is fastest with
        // four G16 affine ranges per sequence and an NW4 map scan; do not
        // reuse the equal-head G64/cache/metadata-elision specialization.
        const bool automatic_gva_equal_n4_g16 =
            is_gva && hinted_equal_lengths && p.N == 4 &&
            hinted_bound == 4096 && p.T_total == 4 * 4096 &&
            group_env == nullptr &&
            !force_direct && !force_affine && !force_hybrid;
        const bool hinted_short_many = has_length_hint &&
            p.N >= 16 && hinted_bound <= kHybridDirectMaxChunks * 16 + 1;
        const bool hinted_equal_budget = hinted_equal_lengths &&
            p.T_total <= 32768 &&
            ((!is_gva && p.N == 4 && hinted_bound >= 2048 &&
              !automatic_equal_n4_g64) ||
             (p.N >= 9 && hinted_bound >= 1024));
        const bool hinted_direct =
            !force_direct && !force_affine && !force_hybrid &&
            (hinted_short_many || hinted_equal_budget);

        // All three explicit route knobs must bypass the ordinary C-split /
        // V-split threshold policy.  In particular, a forced hybrid probe is
        // useful on shapes that would not otherwise select context-parallel;
        // leaving it out here silently made CONTEXT_HYBRID ineffective and
        // allowed CSPLIT64_MIN_T to win instead.
        const bool force_context =
            force_direct || force_affine || (force_hybrid && is_varlen) ||
            hinted_direct || automatic_short_single_direct ||
            automatic_dense_single_g8 || automatic_dense_single_affine ||
            automatic_equal_n4_g64 || automatic_gva_equal_n4_g16;

        const int requested_group = group_env ? std::atoi(group_env) : 0;
        const bool requested_g8 = env_exact(
            "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS", "8");
        const bool requested_g16 = env_exact(
            "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS", "16");
        const int NT = p.total_tiles / p.N;
        // G8 uses a dense-N1-only arena role swap: BF16 affine A occupies
        // cs_u while FP32 affine b occupies cs_sin.  Both arenas fill exactly
        // only for complete groups, so keep this diagnostic route stricter
        // than the ordinary affine scratch calculation below.
        const bool g8_alias_safe =
            (requested_g8 || automatic_dense_single_g8) && !is_varlen &&
            p.N == 1 && NT > 0 && p.total_tiles == NT && (NT % 8) == 0;
        // G16 exactly fills the aliased affine-b arena, so graduate it only
        // for aligned dense single sequences.  It is the measured winner at
        // 2K and 4K; beyond that point G32 scales better.  Any explicit group
        // setting remains authoritative.
        const bool automatic_dense_g16 =
            automatic_dense_single_affine && group_env == nullptr &&
            !automatic_dense_single_g8 && T_seq <= 4096 && NT > 0 &&
            (NT % 16) == 0;
        // N=8 long-ragged K3 batches are too small to enter the established
        // hybrid threshold, but the caller-provided maximum proves enough
        // context depth for G64.  Three-seed paired measurements select G64
        // over G32/G128.  A bucketed graph hint remains conservative here.
        const bool hinted_n8_g64 = !is_gva && has_length_hint && p.N == 8 &&
            hinted_bound >= 4096;
        // Use the exact group that the former launch-time scratch guard used.
        // Invalid overrides conservatively guard as G32, while the normal
        // automatic long-context selection guards as G64.
        const int guard_group = requested_g8 || automatic_dense_single_g8
            ? 8
            : requested_g16 || automatic_dense_g16 ||
                    automatic_gva_equal_n4_g16
            ? 16
            : requested_group == 128
            ? 128
            : requested_group == 64 || automatic_equal_n4_g64 ||
                (requested_group == 0 &&
                 (T_seq >= 12288 || hinted_n8_g64))
                ? 64
                : 32;

        // affine_b aliases the ordinary cs_u arena.  One FP32 KxV map consumes
        // the bytes of sixteen C16 BF16 tiles, so an arbitrary forced-affine
        // shape is not automatically safe.  Dense NT is exact.  For packed
        // metadata, sum_i ceil(chunks_i / G) is bounded by adding G-1 for
        // every sequence before dividing the known total tile count.
        const int64_t hinted_equal_chunks = hinted_equal_lengths
            ? (hinted_bound + 15) / 16 : 0;
        const int64_t affine_group_upper = is_varlen
            ? (hinted_equal_lengths
                ? int64_t(p.N) *
                    ((hinted_equal_chunks + guard_group - 1) / guard_group)
                : (int64_t(p.total_tiles) +
                   int64_t(p.N) * (guard_group - 1)) / guard_group)
            : int64_t(p.N) *
                ((int64_t(NT) + guard_group - 1) / guard_group);
        const int64_t affine_group_capacity = p.total_tiles / 16;
        const bool ordinary_scratch_safe =
            affine_group_upper <= affine_group_capacity;
        const bool scratch_safe =
            g8_alias_safe || ordinary_scratch_safe;
        // G16 exactly fills the cs_u alias for aligned dense N=1 and for the
        // proven equal 4x4K diagnostic.  Do not broaden the ordinary group
        // whitelist: an unproven packed ceil per sequence can otherwise
        // overrun that arena.
        const bool eligible_equal_n4_g16 =
            is_varlen && hinted_equal_lengths && p.N == 4 &&
            hinted_bound == 4096 && hinted_equal_chunks > 0 &&
            (hinted_equal_chunks % 16) == 0;
        const bool eligible_g16 =
            (requested_g16 || automatic_dense_g16 ||
             automatic_gva_equal_n4_g16) &&
            ((!is_varlen && p.N == 1 && NT > 0) ||
             eligible_equal_n4_g16) && scratch_safe;

        // Packed serving may mix many short resumed requests with a deep
        // prefill.  Hybrid dispatch keeps the short sequences register-local
        // and forms affine maps only for the long contexts.
        const bool hybrid = is_varlen && !hinted_direct &&
            ((!force_affine &&
              (force_hybrid || (!force_direct && p.N >= 9))) ||
             (force_affine && !scratch_safe));
        const bool direct = !hybrid &&
            (automatic_short_single_direct || hinted_direct ||
             (force_affine && !scratch_safe) ||
             (!force_affine &&
              (force_direct || (p.N >= 16 && T_seq <= 1024))));

        int group_chunks;
        if (direct) {
            group_chunks = 0;
        } else if (g8_alias_safe) {
            group_chunks = 8;
        } else if (eligible_g16) {
            group_chunks = 16;
        } else if (automatic_equal_n4_g64) {
            group_chunks = 64;
        } else if (requested_group == 32 || requested_group == 64 ||
                   requested_group == 128) {
            group_chunks = requested_group;
        } else {
            group_chunks =
                hybrid || T_seq >= 12288 || hinted_n8_g64 ? 64 : 32;
        }
        const bool automatic_gva_packed_nw4 =
            is_gva && is_varlen && !direct && !hybrid &&
            p.N >= 4 && p.N <= 8 && group_env == nullptr &&
            !force_direct && !force_affine && !force_hybrid;
        return {force_context, group_chunks,
                hybrid ? kHybridDirectMaxChunks : 0,
                automatic_gva_packed_nw4,
                automatic_gva_equal_n4_g16};
    }

    static K2DefaultRoute default_route(
            const FwdParams& p, bool force_context) {
        // Paired gfx950 measurements across the Kimi-K3 N x average-T plane
        // put the plain C-split crossover at 256 tokens.  Only the 128-token
        // bucket stays on register-state V-split; C-split wins single-256 and
        // the previously misrouted 16x1K/64x256 cases as well as long inputs.
        const long T_seq = p.T_total / p.N;
        if (force_context)
            return K2DefaultRoute::context_parallel;
        const char* value = std::getenv("FLASH_KDA_GFX950_CSPLIT64_MIN_T");
        if (value != nullptr && *value != '\0') {
            errno = 0;
            char* end = nullptr;
            const long threshold = std::strtol(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' ||
                threshold < 0 || threshold > INT_MAX)
                return K2DefaultRoute::vsplit_rs;
            if (threshold == 0)
                return K2DefaultRoute::vsplit_rs;
            return T_seq >= threshold ? K2DefaultRoute::csplit64
                                      : K2DefaultRoute::vsplit_rs;
        }
        // A single K3 sequence at 8K+ leaves the serial C-split scan at only
        // H*8=96 CTAs on a 256-CU MI355X.  Split it into affine context ranges
        // while retaining the established short/batched routes.
        if (p.N == 1 && T_seq >= 8192)
            return K2DefaultRoute::context_parallel;
        // Multi-sequence batches expose enough independent work for the NW4
        // direct replay, while the low-N long/ragged cases benefit from the
        // same affine segmentation used by a single deep context.
        if ((p.N >= 16 && T_seq <= 1024) ||
            (p.N >= 8 && p.N < 16 && T_seq >= 1024))
            return K2DefaultRoute::context_parallel;
        return T_seq >= 256 ? K2DefaultRoute::csplit64
                            : K2DefaultRoute::vsplit_rs;
    }

    static void launch_context_persistent_prefix(
            const PersistentPrefixLaunch& a) {
        k1_build_tile_prefix_hybrid_g64_compact_kernel
            <<<1, 64, 0, a.stream>>>(
                a.cu_seqlens, a.N, a.tile_prefix, a.pair_prefix,
                a.segment_prefix, a.sequence_worklist, a.sequence_count);
    }

    static void launch_context_parallel_persistent(
            const ContextParallelLaunch& a) {
        const dim3 persistent_grid(a.context_persistent_blocks);
        const bool use_established_ab =
            context_persistent_established_ab_enabled();
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            static_assert(VL,
                "gfx950 context persistent topology is packed-only");

            // The direct pass remains the sole owner of short and empty
            // sequences.  Long sequences return before recurrence and are
            // handled by the affine topology below.
            k2_kda_context_parallel_nw4_kernel<
                1, KdaContextMode::kReplay, HO, FP, true, true, 4,
                kHybridCompactDirectMaxChunks, true, true, true, false>
                <<<dim3(a.N * a.H, 2), 256, 0, a.stream>>>(
                    a.v, a.beta_cache, a.out, a.kd, a.qd, a.kr, a.gt,
                    a.inv, a.mqk, nullptr, nullptr, a.init_state,
                    a.final_state, a.cu_seqlens, a.tile_prefix, nullptr,
                    a.N, a.total_tiles, a.T_seq, a.H, a.NT);

            if (use_established_ab) {
                // Reuse the established one-task fused producer while the
                // compact device prefix remains the source of truth.  The
                // host-visible upper is the same conservative packed-hybrid
                // G64 bound used by the established route; trailing CTAs are
                // rejected by context_prefix[N] inside the existing kernel.
                const int max_affine_sequences = std::min(
                    a.N,
                    a.total_tiles /
                        (kHybridCompactDirectMaxChunks + 1));
                const int64_t upper =
                    (int64_t(a.total_tiles) +
                     int64_t(max_affine_sequences) *
                         (kHybridCompactGroupChunks - 1)) /
                    kHybridCompactGroupChunks;
                const int context_upper = std::max(1, int(upper));
                const dim3 established_ab_grid(context_upper * a.H, 2);
                k2_kda_context_affine_ab_fused_nw4_kernel<
                    kHybridCompactGroupChunks>
                    <<<established_ab_grid, 256, 0, a.stream>>>(
                        a.v, a.beta_cache, a.kd, a.kr, a.gt, a.inv,
                        a.affine_b, a.affine_a, a.cu_seqlens,
                        a.tile_prefix, a.context_prefix, a.N,
                        a.total_tiles, a.H);
            } else {
                k2_kda_context_affine_ab_fused_persistent_g64_nw4_kernel
                    <<<persistent_grid, 256, 0, a.stream>>>(
                        a.v, a.beta_cache, a.kd, a.kr, a.gt, a.inv,
                        a.affine_b, a.affine_a, a.cu_seqlens,
                        a.tile_prefix, a.context_prefix, a.N,
                        a.total_tiles, a.H);
            }

            k2_kda_context_affine_scan_hybrid_g64_compact_grid_stride_nw2_kernel<
                HI, HO, FP>
                <<<persistent_grid, 128, 0, a.stream>>>(
                    a.affine_a, a.affine_b, a.init_state, a.final_state,
                    a.cu_seqlens, a.context_prefix, a.sequence_worklist,
                    a.sequence_count, a.N, a.H);

            k2_kda_context_replay_hybrid_g64_grid_stride_nw4_kernel<HO, FP>
                <<<persistent_grid, 256, 0, a.stream>>>(
                    a.v, a.beta_cache, a.out, a.kd, a.qd, a.kr, a.gt,
                    a.inv, a.mqk, a.affine_b, a.final_state,
                    a.cu_seqlens, a.tile_prefix, a.context_prefix, a.N,
                    a.total_tiles, a.H);
        };
        dispatch_state_mode<true>(
            a.has_state_in, a.has_state_out, a.state_fp32, launch);
    }

    static void launch_context_parallel(const ContextParallelLaunch& a) {
        const int group_chunks = a.context_group_chunks;
        const int direct_max_chunks = a.context_direct_max_chunks;
        if (!a.is_gva && a.context_persistent_blocks > 0 && a.is_varlen &&
            group_chunks == kHybridCompactGroupChunks &&
            direct_max_chunks == kHybridCompactDirectMaxChunks &&
            a.sequence_worklist != nullptr && a.sequence_count != nullptr) {
            launch_context_parallel_persistent(a);
            return;
        }
        const bool direct = group_chunks == 0;
        const bool hybrid = direct_max_chunks > 0;
        // The proven equal packed 4x4K bucket reaches this callback as a G64
        // affine route.  Its measured recipe uses the fused producer and the
        // K-split scan.  The strict whole-graph candidate reaches the same
        // callback after common dispatch has normalized all four stages to
        // dense indexing; keep that fact explicit instead of inferring it
        // from null metadata.
        const bool packed_automatic_equal_n4_g64 =
            !a.is_gva && a.is_varlen && a.N == 4 && a.T_seq == 4096 &&
            group_chunks == 64 && direct_max_chunks == 0 &&
            !env_exact("FLASH_KDA_GFX950_CONTEXT_DIRECT", "1") &&
            !env_exact("FLASH_KDA_GFX950_CONTEXT_AFFINE", "1") &&
            !env_exact("FLASH_KDA_GFX950_CONTEXT_HYBRID", "1") &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS") == nullptr;
        const bool equal_dense_n4_g64 =
            !a.is_gva && a.equal_dense_n4_g64 && !a.is_varlen && a.N == 4 &&
            a.T_seq == 4096 && a.NT == 256 && a.total_tiles == 1024 &&
            group_chunks == 64 && direct_max_chunks == 0 &&
            a.cu_seqlens == nullptr && a.tile_prefix == nullptr &&
            a.context_prefix == nullptr;
        const bool automatic_equal_n4_g64 =
            packed_automatic_equal_n4_g64 || equal_dense_n4_g64;
        // Consume the exact policy-time route facts rather than reconstructing
        // them after common dispatch has normalized launch metadata.
        const bool automatic_gva_packed_nw4 =
            a.automatic_gva_packed_nw4 && a.is_gva && a.is_varlen &&
            !direct && !hybrid;
        const bool automatic_gva_equal_n4_g16 =
            a.automatic_gva_equal_n4_g16 &&
            automatic_gva_packed_nw4 && group_chunks == 16;
        // Consume the exact K1 publication fact supplied by common dispatch.
        // The cache is value-head-major, so the same ABI covers GVA after K1
        // has applied the grouped q/k-head mapping to its raw inputs.
        const bool cache_context_operands =
            a.context_operands_cached && !a.is_gva;
        const bool forward_u = context_u_forward_enabled();
        const bool forward_v = context_v_forward_enabled();
        const bool context_nw8 = context_nw8_enabled() &&
            cache_context_operands && forward_u && forward_v;
        const bool pipeline_lds_global = context_lds_pipeline_enabled();
        const bool pipeline_lds_b = forward_u && forward_v &&
            (pipeline_lds_global || context_lds_pipeline_b_enabled());
        const bool pipeline_lds_a = forward_u && forward_v &&
            (pipeline_lds_global || context_lds_pipeline_a_enabled());
        const bool pipeline_lds_replay = forward_u && forward_v &&
            (pipeline_lds_global || context_lds_pipeline_replay_enabled());
        const bool fuse_affine_ab =
            (context_affine_ab_fused_enabled(group_chunks) ||
             (automatic_equal_n4_g64 &&
              std::getenv(
                  "FLASH_KDA_GFX950_CONTEXT_AFFINE_AB_FUSED") == nullptr)) &&
            !direct &&
            (a.is_varlen || (!a.is_varlen && a.N == 1) ||
             equal_dense_n4_g64) && !context_nw8 &&
            cache_context_operands && forward_u && forward_v &&
            !pipeline_lds_b && !pipeline_lds_a;
        const bool affine_ab_stage_early =
            fuse_affine_ab && context_affine_ab_stage_early_enabled() &&
            (group_chunks == 64 || (!a.is_varlen && group_chunks == 16));
        if (direct) {
            const char* direct_nw_value =
                std::getenv("FLASH_KDA_GFX950_CONTEXT_DIRECT_NW");
            const int requested_direct_nw = direct_nw_value
                ? std::atoi(direct_nw_value) : 4;
            const bool direct_tail_first =
                context_direct_tail_first_enabled();
            const bool requested_nw1_flat_tail_first =
                context_direct_nw1_flat_tail_first_enabled();
            const bool dense_all_full_c16_enabled =
                context_direct_dense_all_full_c16_enabled();
            const bool paired_state_products_x32 =
                context_direct_paired_state_products_x32_enabled();
            const bool nw1_wave_barrier =
                context_direct_nw1_wave_barrier_enabled();
            const bool direct_dense_n1_h12_requested =
                context_direct_dense_n1_h12_enabled();
            const bool direct_global_n1_h12_requested =
                context_direct_global_n1_h12_enabled();
            const bool direct_global_kr_gll_requested =
                context_direct_global_kr_gll_enabled();
            const bool direct_global_kq_gll_requested =
                context_direct_global_kq_gll_enabled();
            const ContextDirectKsplitMode direct_ksplit_mode =
                context_direct_ksplit_mode();
            const bool direct_ksplit_tail_mqk_prefetch =
                context_direct_ksplit_tail_mqk_prefetch_enabled();
            const bool direct_ksplit_long_n1_h12_requested =
                context_direct_ksplit_long_n1_h12_enabled();
            const char* nw1_flat_value = std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW1_FLAT_TAIL_FIRST");
            // The automatic 256/512 route must select the measured recipe,
            // not merely the context-parallel family: NW4 was 48/75 us while
            // NW1-flat all-full was 34/50 us.  Any explicit route, NW, or flat
            // scheduling request retains authority for diagnostic A/B work.
            const bool automatic_short_single_nw1_flat =
                a.N == 1 && (a.T_seq == 256 || a.T_seq == 512) &&
                !env_exact("FLASH_KDA_GFX950_CONTEXT_DIRECT", "1") &&
                !env_exact("FLASH_KDA_GFX950_CONTEXT_AFFINE", "1") &&
                !env_exact("FLASH_KDA_GFX950_CONTEXT_HYBRID", "1") &&
                direct_nw_value == nullptr && nw1_flat_value == nullptr;
            // Match the exact host-side mixed-boundary graduation above.  In
            // this callback T_seq is the packed average, hence 64/65 for a
            // 1024/1025-token prefill plus fifteen one-token decodes.  The
            // conservative packed tile upper bound is 81 for both maxima,
            // whereas equal 16x1K reaches 1040 and must retain NW4.
            // Requiring PREFIXLESS itself to be unset preserves the existing
            // explicit-axis contract: PREFIXLESS=1 alone changes only mapping,
            // while the zero-environment production recipe changes both
            // mapping and schedule as one measured unit.
            const bool automatic_mixed_boundary_nw1_flat =
                a.packed_direct_prefixless && a.is_varlen && a.N == 16 &&
                a.total_tiles == 81 &&
                (a.T_seq == 64 || a.T_seq == 65) &&
                std::getenv(
                    "FLASH_KDA_GFX950_CONTEXT_DIRECT_PREFIXLESS") ==
                    nullptr &&
                std::getenv("FLASH_KDA_GFX950_CONTEXT_DIRECT") == nullptr &&
                std::getenv("FLASH_KDA_GFX950_CONTEXT_AFFINE") == nullptr &&
                std::getenv("FLASH_KDA_GFX950_CONTEXT_HYBRID") == nullptr &&
                std::getenv(
                    "FLASH_KDA_GFX950_CONTEXT_GROUP_CHUNKS") == nullptr &&
                direct_nw_value == nullptr && nw1_flat_value == nullptr;
            const bool use_nw1_flat_tail_first =
                requested_nw1_flat_tail_first ||
                automatic_short_single_nw1_flat ||
                automatic_mixed_boundary_nw1_flat;
            // NW1-flat is best for the short/many-sequence direct batches it
            // was introduced for, but leaves a deep equal-length N=4 resume
            // under-filled.  Preserve explicit NW1-2D diagnostics; adapt only
            // the flat recipe, whose measured gfx950 crossover strongly
            // selects the established NW4 replay at 2K and above.
            const bool use_deep_n4_nw4 =
                use_nw1_flat_tail_first && a.N == 4 &&
                a.T_seq >= 2048 &&
                env_unset_or_exact(
                    "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW", "1");
            const int direct_nw = use_deep_n4_nw4
                ? 4 : requested_direct_nw;
            // Flatten all eight NW1 V16 CTAs of one (sequence, head) next to
            // each other.  Form the product in 64 bits and reject the
            // specialization unless dim3.x remains representable by the
            // signed index decode in the device kernel.
            const uint64_t direct_flat_blocks_per_sequence = a.H > 0
                ? uint64_t(a.H) * uint64_t(8) : uint64_t(0);
            const bool direct_flat_grid_safe = a.N > 0 &&
                direct_flat_blocks_per_sequence > 0 &&
                uint64_t(a.N) <=
                    uint64_t(std::numeric_limits<int32_t>::max()) /
                        direct_flat_blocks_per_sequence;
            const uint32_t direct_flat_blocks = direct_flat_grid_safe
                ? uint32_t(
                    uint64_t(a.N) * direct_flat_blocks_per_sequence)
                : uint32_t(0);
            const bool direct_nw1_flat_tail_first =
                use_nw1_flat_tail_first && !use_deep_n4_nw4 &&
                direct_max_chunks == 0 && !direct_tail_first &&
                env_unset_or_exact(
                    "FLASH_KDA_GFX950_CONTEXT_DIRECT_NW", "1") &&
                !context_nw8 && cache_context_operands && forward_u &&
                forward_v && !pipeline_lds_replay && direct_flat_grid_safe;
            // This symbol deliberately has no generic runtime dispatch axis:
            // its template contract fixes the complete production NW1-flat
            // tuple, and this host proof fixes every geometry/address fact it
            // removes.  In particular, packed metadata and the paired/wave
            // experiments cannot reach it.
            const bool direct_dense_n1_h12 =
                !a.is_gva && direct_dense_n1_h12_requested &&
                !direct_global_n1_h12_requested &&
                direct_ksplit_mode == ContextDirectKsplitMode::disabled &&
                direct_nw1_flat_tail_first && !a.is_varlen &&
                a.N == 1 && a.H == 12 &&
                (a.T_seq == 256 || a.T_seq == 512) &&
                a.NT == a.T_seq / 16 && a.total_tiles == a.NT &&
                a.cu_seqlens == nullptr && a.tile_prefix == nullptr &&
                a.context_prefix == nullptr &&
                !a.packed_direct_prefixless && dense_all_full_c16_enabled &&
                !paired_state_products_x32 && !nw1_wave_barrier &&
                direct_flat_blocks == 12u * 8u;
            // Direct global fragments preserve the established reduction tree
            // but remove the K2 LDS arena and every workgroup barrier.  Keep it
            // mutually exclusive with both other dense-H12 experiments so a
            // contaminated environment falls back to the production symbol.
            const bool direct_global_n1_h12 =
                !a.is_gva && direct_global_n1_h12_requested &&
                !direct_dense_n1_h12_requested &&
                direct_ksplit_mode == ContextDirectKsplitMode::disabled &&
                direct_nw1_flat_tail_first && !a.is_varlen &&
                a.N == 1 && a.H == 12 &&
                (a.T_seq == 256 || a.T_seq == 512) &&
                a.NT == a.T_seq / 16 && a.total_tiles == a.NT &&
                a.cu_seqlens == nullptr && a.tile_prefix == nullptr &&
                a.context_prefix == nullptr &&
                !a.packed_direct_prefixless && dense_all_full_c16_enabled &&
                !paired_state_products_x32 && !nw1_wave_barrier &&
                direct_flat_blocks == 12u * 8u;
            // The K-split symbols own one dense (head,V16) slab per CTA and
            // divide K128 across two or four waves.  This is deliberately a
            // complete host proof rather than a generic route: packed inputs,
            // partial C16 tiles, other head counts, diagnostic transport axes,
            // and non-production schedules retain the established NW1 kernel.
            const bool direct_ksplit_short_shape =
                !direct_ksplit_long_n1_h12_requested &&
                (a.T_seq == 256 || a.T_seq == 512);
            const bool direct_ksplit_long_shape =
                direct_ksplit_long_n1_h12_requested &&
                direct_ksplit_mode ==
                    ContextDirectKsplitMode::tailpipe_waves4 &&
                (a.T_seq == 1024 || a.T_seq == 2048);
            const bool direct_ksplit_eligible =
                !a.is_gva &&
                direct_ksplit_mode != ContextDirectKsplitMode::disabled &&
                !direct_dense_n1_h12_requested &&
                !direct_global_n1_h12_requested &&
                direct_nw1_flat_tail_first && !a.is_varlen &&
                a.N == 1 && a.H == 12 &&
                (direct_ksplit_short_shape || direct_ksplit_long_shape) &&
                a.NT == a.T_seq / 16 && a.total_tiles == a.NT &&
                a.cu_seqlens == nullptr && a.tile_prefix == nullptr &&
                a.context_prefix == nullptr &&
                !a.packed_direct_prefixless && dense_all_full_c16_enabled &&
                !paired_state_products_x32 && !nw1_wave_barrier &&
                direct_flat_blocks == 12u * 8u;
            auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
                (void)HI;
                if (direct_nw1_flat_tail_first) {
                    if constexpr (!VL) {
                        if (direct_global_n1_h12) {
                            auto launch_direct_global =
                                    [&]<bool KR_GLL, bool KQ_GLL>() {
                                static_assert(!KR_GLL || HO,
                                              "Kr GLL requires HO=true");
                                constexpr int shared_bytes =
                                    (KR_GLL
                                         ? direct_global_n1_h12_detail::
                                               kKrGllLdsBytes
                                         : 0) +
                                    (KQ_GLL
                                         ? direct_global_n1_h12_detail::
                                               kKqGllLdsBytes
                                         : 0);
                                k2_kda_context_direct_global_n1_h12_kernel<
                                    HO, FP, KR_GLL, KQ_GLL>
                                    <<<dim3(12 * 8, 1, 1), 64, shared_bytes,
                                       a.stream>>>(
                                        a.v, a.beta_cache, a.out, a.kd, a.qd,
                                        a.kr, a.gt, a.inv, a.mqk,
                                        a.init_state, a.final_state, a.NT);
                            };
                            if constexpr (HO) {
                                if (direct_global_kr_gll_requested) {
                                    if (direct_global_kq_gll_requested)
                                        launch_direct_global
                                            .template operator()<true, true>();
                                    else
                                        launch_direct_global
                                            .template operator()<true, false>();
                                } else if (
                                        direct_global_kq_gll_requested) {
                                    launch_direct_global
                                        .template operator()<false, true>();
                                } else {
                                    launch_direct_global
                                        .template operator()<false, false>();
                                }
                            } else {
                                // HO=false never instantiates or selects Kr,
                                // but can still measure the Kd/Qd transport.
                                if (direct_global_kq_gll_requested)
                                    launch_direct_global
                                        .template operator()<false, true>();
                                else
                                    launch_direct_global
                                        .template operator()<false, false>();
                            }
                            return;
                        }
                        if (direct_ksplit_eligible) {
                            auto launch_ksplit =
                                    [&]<int KSPLIT_WAVES,
                                        bool TAILPIPE,
                                        bool TAIL_MQK_PREFETCH,
                                        bool LONG_SEQUENCE>() {
                                if constexpr (TAILPIPE) {
                                    k2_kda_context_direct_ksplit_tailpipe_n1_h12_kernel<
                                        KSPLIT_WAVES, HO, FP,
                                        TAIL_MQK_PREFETCH, LONG_SEQUENCE>
                                        <<<dim3(12 * 8, 1, 1),
                                           KSPLIT_WAVES * 64, 0, a.stream>>>(
                                            a.v, a.beta_cache, a.out, a.kd,
                                            a.qd, a.kr, a.gt, a.inv, a.mqk,
                                            a.init_state, a.final_state,
                                            a.T_seq, a.NT);
                                } else {
                                    static_assert(!TAIL_MQK_PREFETCH,
                                                  "Mqk prefetch requires tailpipe");
                                    static_assert(!LONG_SEQUENCE,
                                                  "long sequence requires tailpipe");
                                    k2_kda_context_direct_ksplit_n1_h12_kernel<
                                        KSPLIT_WAVES, HO, FP>
                                        <<<dim3(12 * 8, 1, 1),
                                           KSPLIT_WAVES * 64, 0, a.stream>>>(
                                            a.v, a.beta_cache, a.out, a.kd,
                                            a.qd, a.kr, a.gt, a.inv, a.mqk,
                                            a.init_state, a.final_state,
                                            a.T_seq, a.NT);
                                }
                            };
                            if (direct_ksplit_long_shape) {
                                if (direct_ksplit_tail_mqk_prefetch) {
                                    launch_ksplit.template operator()<
                                        4, true, true, true>();
                                } else {
                                    launch_ksplit.template operator()<
                                        4, true, false, true>();
                                }
                            } else if (direct_ksplit_mode ==
                                ContextDirectKsplitMode::waves2) {
                                launch_ksplit.template operator()<
                                    2, false, false, false>();
                            } else if (direct_ksplit_mode ==
                                       ContextDirectKsplitMode::waves4) {
                                launch_ksplit.template operator()<
                                    4, false, false, false>();
                            } else if (direct_ksplit_mode ==
                                       ContextDirectKsplitMode::tailpipe_waves2) {
                                launch_ksplit.template operator()<
                                    2, true, false, false>();
                            } else if (direct_ksplit_tail_mqk_prefetch) {
                                launch_ksplit.template operator()<
                                    4, true, true, false>();
                            } else {
                                launch_ksplit.template operator()<
                                    4, true, false, false>();
                            }
                            return;
                        }
                        if (direct_dense_n1_h12) {
                            k2_kda_context_parallel_nw4_kernel<
                                1, KdaContextMode::kReplay, HO, FP, false,
                                true, 1, 0, true, true, true, false, true,
                                false, true, RegBX32, TiledKrCarryX16,
                                false, false, true>
                                <<<dim3(12 * 8, 1, 1), 64, 0, a.stream>>>(
                                    a.v, a.beta_cache, a.out, a.kd, a.qd,
                                    a.kr, a.gt, a.inv, a.mqk, nullptr,
                                    nullptr, a.init_state, a.final_state,
                                    nullptr, nullptr, nullptr, 1, a.NT,
                                    a.T_seq, 12, a.NT);
                            return;
                        }
                    }
                    auto launch_flat = [&]<bool PACKED_DIRECT_PREFIXLESS,
                                           bool DENSE_ALL_FULL_C16,
                                           bool PAIRED_STATE_PRODUCTS_X32,
                                           bool NW1_WAVE_BARRIER>() {
                        k2_kda_context_parallel_nw4_kernel<
                            1, KdaContextMode::kReplay, HO, FP, VL, true,
                            1, 0, true, true, true, false, true,
                            PACKED_DIRECT_PREFIXLESS, DENSE_ALL_FULL_C16,
                            RegBX32, TiledKrCarryX16,
                            PAIRED_STATE_PRODUCTS_X32, NW1_WAVE_BARRIER>
                            <<<dim3(direct_flat_blocks, 1, 1), 64, 0,
                               a.stream>>>(
                                a.v, a.beta_cache, a.out, a.kd, a.qd, a.kr,
                                a.gt, a.inv, a.mqk, nullptr, nullptr,
                                a.init_state, a.final_state,
                                VL ? a.cu_seqlens : nullptr,
                                VL && !PACKED_DIRECT_PREFIXLESS
                                    ? a.tile_prefix : nullptr,
                                nullptr, a.N, a.total_tiles, a.T_seq, a.H,
                                a.NT);
                    };
                    auto dispatch_pair =
                            [&]<bool PACKED_DIRECT_PREFIXLESS,
                                bool DENSE_ALL_FULL_C16,
                                bool NW1_WAVE_BARRIER>() {
                        if (paired_state_products_x32)
                            launch_flat.template operator()<
                                PACKED_DIRECT_PREFIXLESS,
                                DENSE_ALL_FULL_C16, true,
                                NW1_WAVE_BARRIER>();
                        else
                            launch_flat.template operator()<
                                PACKED_DIRECT_PREFIXLESS,
                                DENSE_ALL_FULL_C16, false,
                                NW1_WAVE_BARRIER>();
                    };
                    auto dispatch_flat =
                            [&]<bool PACKED_DIRECT_PREFIXLESS,
                                bool DENSE_ALL_FULL_C16>() {
                        if (nw1_wave_barrier)
                            dispatch_pair.template operator()<
                                PACKED_DIRECT_PREFIXLESS,
                                DENSE_ALL_FULL_C16, true>();
                        else
                            dispatch_pair.template operator()<
                                PACKED_DIRECT_PREFIXLESS,
                                DENSE_ALL_FULL_C16, false>();
                    };
                    if constexpr (VL) {
                        if (a.packed_direct_prefixless)
                            dispatch_flat.template operator()<true, false>();
                        else
                            dispatch_flat.template operator()<false, false>();
                    } else {
                        if (dense_all_full_c16_enabled && a.N == 1 &&
                            (a.T_seq & 15) == 0)
                            dispatch_flat.template operator()<false, true>();
                        else
                            dispatch_flat.template operator()<false, false>();
                    }
                    return;
                }
                auto launch_nw = [&]<int NW, bool CACHED_OPERANDS,
                                     bool U_FORWARD, bool V_FORWARD,
                                     bool LDS_PIPELINE>() {
                    const float* const beta = CACHED_OPERANDS
                        ? a.beta_cache : a.beta;
                    auto launch_kernel = [&]<bool DIRECT_TAIL_FIRST,
                                             bool PACKED_DIRECT_PREFIXLESS>() {
                        k2_kda_context_parallel_nw4_kernel<
                            1, KdaContextMode::kReplay, HO, FP, VL, true,
                            NW, 0, CACHED_OPERANDS, U_FORWARD, V_FORWARD,
                            LDS_PIPELINE, DIRECT_TAIL_FIRST,
                            PACKED_DIRECT_PREFIXLESS>
                            <<<dim3(a.N * a.H, 8 / NW), NW * 64, 0,
                               a.stream>>>(
                                a.v, beta, a.out, a.kd, a.qd, a.kr, a.gt,
                                a.inv, a.mqk, nullptr, nullptr, a.init_state,
                                a.final_state,
                                VL ? a.cu_seqlens : nullptr,
                                VL && !PACKED_DIRECT_PREFIXLESS
                                    ? a.tile_prefix : nullptr,
                                nullptr, a.N,
                                a.total_tiles, a.T_seq, a.H, a.NT);
                    };
                    auto launch_mapping = [&]<bool DIRECT_TAIL_FIRST>() {
                        if constexpr (VL) {
                            if (a.packed_direct_prefixless)
                                launch_kernel.template operator()<
                                    DIRECT_TAIL_FIRST, true>();
                            else
                                launch_kernel.template operator()<
                                    DIRECT_TAIL_FIRST, false>();
                        } else {
                            launch_kernel.template operator()<
                                DIRECT_TAIL_FIRST, false>();
                        }
                    };
                    if constexpr (NW == 4 && CACHED_OPERANDS && U_FORWARD &&
                                  V_FORWARD && !LDS_PIPELINE) {
                        if (direct_tail_first)
                            launch_mapping.template operator()<true>();
                        else
                            launch_mapping.template operator()<false>();
                    } else {
                        launch_mapping.template operator()<false>();
                    }
                };
                auto dispatch_forward = [&]<int NW, bool CACHED_OPERANDS>() {
                    if (forward_u) {
                        if (forward_v) {
                            if (pipeline_lds_replay)
                                launch_nw.template operator()<
                                    NW, CACHED_OPERANDS, true, true, true>();
                            else
                                launch_nw.template operator()<
                                    NW, CACHED_OPERANDS, true, true, false>();
                        } else {
                            launch_nw.template operator()<
                                NW, CACHED_OPERANDS, true, false, false>();
                        }
                    } else if (forward_v) {
                        launch_nw.template operator()<
                            NW, CACHED_OPERANDS, false, true, false>();
                    } else {
                        launch_nw.template operator()<
                            NW, CACHED_OPERANDS, false, false, false>();
                    }
                };
                auto dispatch_cache = [&]<int NW>() {
                    if (cache_context_operands)
                        dispatch_forward.template operator()<NW, true>();
                    else
                        dispatch_forward.template operator()<NW, false>();
                };
                if (context_nw8 ||
                    (direct_nw == 8 && cache_context_operands &&
                     forward_u && forward_v)) {
                    if (pipeline_lds_replay)
                        launch_nw.template operator()<
                            8, true, true, true, true>();
                    else
                        launch_nw.template operator()<
                            8, true, true, true, false>();
                    return;
                }
                if (direct_nw == 1)
                    dispatch_cache.template operator()<1>();
                else if (direct_nw == 2)
                    dispatch_cache.template operator()<2>();
                else
                    dispatch_cache.template operator()<4>();
            };
            dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                                a.state_fp32, launch);
            return;
        }
        const char* scan_nw_env = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_SCAN_NW");
        // Explicit NW2-only scan experiments retain their established default
        // when SCAN_NW itself is unset; an automatic NW4 route must not make a
        // requested K-split/A-GLL/B-phased probe silently inert.
        const bool automatic_gva_scan_nw4 =
            automatic_gva_packed_nw4 && scan_nw_env == nullptr &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT") == nullptr &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_A_GLL") == nullptr &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_B_PHASED") == nullptr;
        const int scan_nw = scan_nw_env
            ? std::atoi(scan_nw_env)
            : automatic_gva_scan_nw4 ? 4 : 2;
        const char* scan_b_stream_env = std::getenv(
            "FLASH_KDA_GFX950_CONTEXT_SCAN_B_STREAM");
        const bool automatic_gva_disable_b_stream =
            automatic_gva_equal_n4_g16 && automatic_gva_scan_nw4 &&
            scan_b_stream_env == nullptr;
        const bool scan_b_stream = !automatic_gva_disable_b_stream &&
            context_scan_b_stream_enabled(group_chunks);
        const bool scan_a_gll = context_scan_a_gll_enabled();
        const bool scan_b_phased = context_scan_b_phased_enabled();
        // K-split is intentionally limited to dense N=1 and packed
        // pure-affine launches.  Hybrid prefixes currently create many empty
        // sequence-indexed scan CTAs, where doubling the workgroup would make
        // the launch tax worse until a compact persistent worklist exists.
        // Keep every other scan experiment orthogonal: requesting one of
        // those axes must not silently turn it into a K-split comparison.
        const bool automatic_equal_n4_g64_ksplit =
            automatic_equal_n4_g64 && scan_nw_env == nullptr &&
            std::getenv(
                "FLASH_KDA_GFX950_CONTEXT_SCAN_KSPLIT") == nullptr;
        const bool scan_ksplit =
            (context_scan_ksplit_enabled() ||
             automatic_equal_n4_g64_ksplit) &&
            !direct && !hybrid && scan_nw == 2 && !scan_b_stream &&
            !scan_a_gll && !scan_b_phased &&
            (a.is_varlen || (!a.is_varlen && a.N == 1) ||
             equal_dense_n4_g64);
        const bool scan_ksplit_prefetch_b =
            scan_ksplit && context_scan_ksplit_prefetch_b_enabled();
        auto launch_group = [&]<int GROUP_CHUNKS>() {
            // Hybrid packed serving keeps up to 1024-token requests on the
            // direct register-state path and builds affine maps only for
            // longer sequences.  Requiring at least 65 C16 tiles per affine
            // sequence also guarantees that the existing cs_u arena can hold
            // one FP32 KxV map per selected context group.
            const int groups_per_sequence =
                (a.NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
            int context_upper;
            if (!a.is_varlen) {
                context_upper = a.N * groups_per_sequence;
            } else if (hybrid) {
                // Only sequences longer than the direct threshold contribute
                // affine groups.  If L such sequences exist, then
                //   sum ceil(chunks_i / G)
                //     <= floor((total_chunks + L*(G-1)) / G).
                // The workspace tile count is a conservative upper bound on
                // total_chunks, while every selected sequence consumes at
                // least DIRECT_MAX_CHUNKS+1 chunks.  This avoids launching the
                // previous +N worth of guaranteed early-return CTAs in mixed
                // decode/prefill batches without reading the device prefix.
                const int max_affine_sequences = std::min(
                    a.N,
                    a.total_tiles / (direct_max_chunks + 1));
                const int64_t upper =
                    (int64_t(a.total_tiles) +
                     int64_t(max_affine_sequences) * (GROUP_CHUNKS - 1)) /
                    GROUP_CHUNKS;
                // A zero-sized HIP grid is invalid.  The single fallback
                // context immediately returns when all sequences were direct.
                context_upper = std::max(1, int(upper));
            } else {
                const int64_t upper =
                    (int64_t(a.total_tiles) +
                     int64_t(a.N) * (GROUP_CHUNKS - 1)) /
                    GROUP_CHUNKS;
                context_upper = std::max(1, int(upper));
            }
            // A filtered hybrid prefix has no entry for short or empty
            // sequences; the direct pass below remains their sole owner.
            // Tighten only when the host upper actually removes scan CTAs.
            const bool tight_hybrid_scan =
                hybrid && context_upper < a.N &&
                context_tight_scan_enabled();
            auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
                auto launch_cached = [&]<bool CACHED_OPERANDS,
                                        bool U_FORWARD, bool V_FORWARD,
                                        bool LDS_PIPELINE_B,
                                        bool LDS_PIPELINE_A,
                                        bool LDS_PIPELINE_REPLAY>() {
                    const float* const beta = CACHED_OPERANDS
                        ? a.beta_cache : a.beta;
                    auto launch_recurrence = [&]<int CONTEXT_NW>() {
                    const dim3 context_grid(
                        context_upper * a.H, 8 / CONTEXT_NW);
                    if (hybrid) {
                        k2_kda_context_parallel_nw4_kernel<
                            1, KdaContextMode::kReplay, HO, FP, VL, true,
                            CONTEXT_NW,
                            kHybridDirectMaxChunks, CACHED_OPERANDS,
                            U_FORWARD, V_FORWARD, LDS_PIPELINE_REPLAY>
                            <<<dim3(a.N * a.H, 8 / CONTEXT_NW),
                               CONTEXT_NW * 64, 0, a.stream>>>(
                                a.v, beta, a.out, a.kd, a.qd, a.kr, a.gt,
                                a.inv, a.mqk, nullptr, nullptr, a.init_state,
                                a.final_state,
                                VL ? a.cu_seqlens : nullptr,
                                VL ? a.tile_prefix : nullptr, nullptr, a.N,
                                a.total_tiles, a.T_seq, a.H, a.NT);
                    }
                    auto launch_affine_b = [&]() {
                        k2_kda_context_parallel_nw4_kernel<
                            GROUP_CHUNKS, KdaContextMode::kAffineB,
                            false, false, VL, false, CONTEXT_NW, 0,
                            CACHED_OPERANDS,
                            U_FORWARD, V_FORWARD, LDS_PIPELINE_B>
                            <<<context_grid, CONTEXT_NW * 64, 0, a.stream>>>(
                                a.v, beta, nullptr, a.kd, nullptr, a.kr, a.gt,
                                a.inv, nullptr, a.affine_b, nullptr, nullptr,
                                nullptr, VL ? a.cu_seqlens : nullptr,
                                VL ? a.tile_prefix : nullptr,
                                VL ? a.context_prefix : nullptr, a.N,
                                a.total_tiles, a.T_seq, a.H, a.NT);
                    };
                    auto launch_affine_a = [&]() {
                        k2_kda_context_parallel_nw4_kernel<
                            GROUP_CHUNKS, KdaContextMode::kAffineA,
                            false, false, VL, false, CONTEXT_NW, 0,
                            CACHED_OPERANDS,
                            U_FORWARD, V_FORWARD, LDS_PIPELINE_A>
                            <<<context_grid, CONTEXT_NW * 64, 0, a.stream>>>(
                                nullptr, beta, nullptr, a.kd, nullptr, a.kr,
                                a.gt, a.inv, nullptr, nullptr, a.affine_a,
                                nullptr, nullptr,
                                VL ? a.cu_seqlens : nullptr,
                                VL ? a.tile_prefix : nullptr,
                                VL ? a.context_prefix : nullptr, a.N,
                                a.total_tiles, a.T_seq, a.H, a.NT);
                    };
                    if constexpr (
                            CONTEXT_NW == 4 && CACHED_OPERANDS &&
                            U_FORWARD && V_FORWARD && !LDS_PIPELINE_B &&
                            !LDS_PIPELINE_A) {
                        if (fuse_affine_ab) {
                            if constexpr (VL) {
                                auto launch_established_fused = [&]() {
                                    k2_kda_context_affine_ab_fused_nw4_kernel<
                                        GROUP_CHUNKS>
                                        <<<context_grid, 256, 0, a.stream>>>(
                                            a.v, beta, a.kd, a.kr, a.gt, a.inv,
                                            a.affine_b, a.affine_a,
                                            a.cu_seqlens, a.tile_prefix,
                                            a.context_prefix, a.N,
                                            a.total_tiles, a.H);
                                };
                                if constexpr (GROUP_CHUNKS == 64) {
                                    if (affine_ab_stage_early) {
                                        k2_kda_context_affine_ab_fused_stage_early_g64_nw4_kernel
                                            <<<context_grid, 256, 0, a.stream>>>(
                                                a.v, beta, a.kd, a.kr, a.gt,
                                                a.inv, a.affine_b, a.affine_a,
                                                a.cu_seqlens, a.tile_prefix,
                                                a.context_prefix, a.N,
                                                a.total_tiles, a.H);
                                    } else
                                        launch_established_fused();
                                } else {
                                    launch_established_fused();
                                }
                            } else {
                                auto launch_established_fused = [&]() {
                                    k2_kda_context_affine_ab_fused_dense_nw4_kernel<
                                        GROUP_CHUNKS>
                                        <<<context_grid, 256, 0, a.stream>>>(
                                            a.v, beta, a.kd, a.kr, a.gt, a.inv,
                                            a.affine_b, a.affine_a, a.T_seq,
                                            a.H, a.NT);
                                };
                                auto launch_dense_stage_early = [&]() {
                                    if constexpr (
                                            GROUP_CHUNKS == 16 ||
                                            GROUP_CHUNKS == 64) {
                                        k2_kda_context_affine_ab_fused_dense_stage_early_nw4_kernel<
                                            GROUP_CHUNKS>
                                            <<<context_grid, 256, 0,
                                               a.stream>>>(
                                                a.v, beta, a.kd, a.kr, a.gt,
                                                a.inv, a.affine_b, a.affine_a,
                                                a.T_seq, a.H, a.NT);
                                    }
                                };
                                if constexpr (GROUP_CHUNKS == 64) {
                                    if (equal_dense_n4_g64) {
                                        if (affine_ab_stage_early) {
                                            k2_kda_context_affine_ab_fused_equal_n4_g64_stage_early_nw4_kernel
                                                <<<context_grid, 256, 0,
                                                   a.stream>>>(
                                                    a.v, beta, a.kd, a.kr,
                                                    a.gt, a.inv, a.affine_b,
                                                    a.affine_a, a.H);
                                        } else {
                                            k2_kda_context_affine_ab_fused_equal_n4_g64_nw4_kernel
                                                <<<context_grid, 256, 0,
                                                   a.stream>>>(
                                                    a.v, beta, a.kd, a.kr,
                                                    a.gt, a.inv, a.affine_b,
                                                    a.affine_a, a.H);
                                        }
                                    } else if (affine_ab_stage_early) {
                                        launch_dense_stage_early();
                                    } else {
                                        launch_established_fused();
                                    }
                                } else if constexpr (GROUP_CHUNKS == 16) {
                                    if (affine_ab_stage_early) {
                                        launch_dense_stage_early();
                                    } else {
                                        launch_established_fused();
                                    }
                                } else {
                                    launch_established_fused();
                                }
                            }
                        } else {
                            launch_affine_b();
                            launch_affine_a();
                        }
                    } else {
                        launch_affine_b();
                        launch_affine_a();
                    }
                    auto launch_scan = [&]<int NW, bool TIGHT_VL_GRID>() {
                        const int scan_contexts =
                            TIGHT_VL_GRID ? context_upper : a.N;
                        if constexpr (NW == 2 && !TIGHT_VL_GRID) {
                            // The runtime guard has already excluded all
                            // orthogonal scan axes; this compile-time guard
                            // additionally excludes the tight hybrid mapping.
                            if (scan_ksplit) {
                                if constexpr (GROUP_CHUNKS == 64) {
                                    if (scan_ksplit_prefetch_b) {
                                        k2_kda_context_affine_scan_ksplit_prefetch_b_g64_wg4_kernel<
                                            HI, HO, FP, VL>
                                            <<<dim3(scan_contexts * a.H, 4),
                                               256, 0, a.stream>>>(
                                                a.affine_a, a.affine_b,
                                                a.init_state, a.final_state,
                                                VL ? a.cu_seqlens : nullptr,
                                                VL ? a.context_prefix : nullptr,
                                                a.T_seq, a.H, a.NT);
                                        return;
                                    }
                                }
                                k2_kda_context_affine_scan_ksplit_wg4_kernel<
                                    GROUP_CHUNKS, HI, HO, FP, VL>
                                    <<<dim3(scan_contexts * a.H, 4),
                                       256, 0, a.stream>>>(
                                        a.affine_a, a.affine_b,
                                        a.init_state, a.final_state,
                                        VL ? a.cu_seqlens : nullptr,
                                        VL ? a.context_prefix : nullptr,
                                        a.T_seq, a.H, a.NT);
                                return;
                            }
                        }
                        if (scan_b_stream) {
                            if constexpr (NW == 2) {
                                if (scan_a_gll) {
                                    k2_kda_context_affine_scan_b_stream_a_gll_nw2_kernel<
                                        GROUP_CHUNKS, HI, HO, FP, VL,
                                        TIGHT_VL_GRID>
                                        <<<dim3(scan_contexts * a.H, 4),
                                           128, 0, a.stream>>>(
                                            a.affine_a, a.affine_b,
                                            a.init_state, a.final_state,
                                            VL ? a.cu_seqlens : nullptr,
                                            VL ? a.context_prefix : nullptr,
                                            TIGHT_VL_GRID ? a.N : a.T_seq,
                                            a.H, a.NT);
                                    return;
                                }
                                if constexpr (!HI) {
                                    if (scan_b_phased) {
                                        k2_kda_context_affine_scan_b_stream_b_phased_nw2_kernel<
                                            GROUP_CHUNKS, HO, FP, VL,
                                            TIGHT_VL_GRID>
                                            <<<dim3(scan_contexts * a.H, 4),
                                               128, 0, a.stream>>>(
                                                a.affine_a, a.affine_b,
                                                a.init_state, a.final_state,
                                                VL ? a.cu_seqlens : nullptr,
                                                VL ? a.context_prefix : nullptr,
                                                TIGHT_VL_GRID ? a.N : a.T_seq,
                                                a.H, a.NT);
                                        return;
                                    }
                                }
                            }
                            k2_kda_context_affine_scan_b_stream_nw4_kernel<
                                GROUP_CHUNKS, NW, HI, HO, FP, VL,
                                TIGHT_VL_GRID>
                                <<<dim3(scan_contexts * a.H, 8 / NW),
                                   NW * 64, 0, a.stream>>>(
                                    a.affine_a, a.affine_b, a.init_state,
                                    a.final_state,
                                    VL ? a.cu_seqlens : nullptr,
                                    VL ? a.context_prefix : nullptr,
                                    TIGHT_VL_GRID ? a.N : a.T_seq,
                                    a.H, a.NT);
                        } else {
                            k2_kda_context_affine_scan_nw4_kernel<
                                GROUP_CHUNKS, NW, HI, HO, FP, VL,
                                TIGHT_VL_GRID>
                                <<<dim3(scan_contexts * a.H, 8 / NW),
                                   NW * 64, 0, a.stream>>>(
                                    a.affine_a, a.affine_b, a.init_state,
                                    a.final_state,
                                    VL ? a.cu_seqlens : nullptr,
                                    VL ? a.context_prefix : nullptr,
                                    TIGHT_VL_GRID ? a.N : a.T_seq,
                                    a.H, a.NT);
                        }
                    };
                    auto dispatch_scan = [&]<int NW>() {
                        if constexpr (VL) {
                            if (tight_hybrid_scan) {
                                launch_scan.template operator()<NW, true>();
                                return;
                            }
                        }
                        launch_scan.template operator()<NW, false>();
                    };
                    if (scan_nw == 4)
                        dispatch_scan.template operator()<4>();
                    else if (scan_nw == 2)
                        dispatch_scan.template operator()<2>();
                    else
                        dispatch_scan.template operator()<1>();
                    k2_kda_context_parallel_nw4_kernel<
                        GROUP_CHUNKS, KdaContextMode::kReplay, HO, FP, VL,
                        false, CONTEXT_NW, 0, CACHED_OPERANDS, U_FORWARD,
                        V_FORWARD, LDS_PIPELINE_REPLAY>
                        <<<context_grid, CONTEXT_NW * 64, 0, a.stream>>>(
                            a.v, beta, a.out, a.kd, a.qd, a.kr, a.gt,
                            a.inv, a.mqk, a.affine_b, nullptr, nullptr,
                            a.final_state,
                            VL ? a.cu_seqlens : nullptr,
                            VL ? a.tile_prefix : nullptr,
                            VL ? a.context_prefix : nullptr, a.N,
                            a.total_tiles, a.T_seq, a.H, a.NT);
                    };
                    if constexpr (CACHED_OPERANDS &&
                                  U_FORWARD && V_FORWARD) {
                        if (context_nw8) {
                            launch_recurrence.template operator()<8>();
                            return;
                        }
                    }
                    launch_recurrence.template operator()<4>();
                };
                auto dispatch_forward = [&]<bool CACHED_OPERANDS>() {
                    if (forward_u) {
                        if (forward_v) {
                            // These host combinations select only the existing
                            // true/false LDS_PIPELINE specialization of each
                            // independently-templated context mode.
                            auto dispatch_replay =
                                [&]<bool LDS_PIPELINE_B,
                                    bool LDS_PIPELINE_A>() {
                                if (pipeline_lds_replay)
                                    launch_cached.template operator()<
                                        CACHED_OPERANDS, true, true,
                                        LDS_PIPELINE_B, LDS_PIPELINE_A, true>();
                                else
                                    launch_cached.template operator()<
                                        CACHED_OPERANDS, true, true,
                                        LDS_PIPELINE_B, LDS_PIPELINE_A, false>();
                            };
                            auto dispatch_a = [&]<bool LDS_PIPELINE_B>() {
                                if (pipeline_lds_a)
                                    dispatch_replay.template operator()<
                                        LDS_PIPELINE_B, true>();
                                else
                                    dispatch_replay.template operator()<
                                        LDS_PIPELINE_B, false>();
                            };
                            if (pipeline_lds_b)
                                dispatch_a.template operator()<true>();
                            else
                                dispatch_a.template operator()<false>();
                        } else {
                            launch_cached.template operator()<
                                CACHED_OPERANDS, true, false,
                                false, false, false>();
                        }
                    } else if (forward_v) {
                        launch_cached.template operator()<
                            CACHED_OPERANDS, false, true,
                            false, false, false>();
                    } else {
                        launch_cached.template operator()<
                            CACHED_OPERANDS, false, false,
                            false, false, false>();
                    }
                };
                if (cache_context_operands)
                    dispatch_forward.template operator()<true>();
                else
                    dispatch_forward.template operator()<false>();
            };
            dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                                a.state_fp32, launch);
        };
        if (group_chunks == 8)
            launch_group.template operator()<8>();
        else if (group_chunks == 16)
            launch_group.template operator()<16>();
        else if (group_chunks == 128)
            launch_group.template operator()<128>();
        else if (group_chunks == 64)
            launch_group.template operator()<64>();
        else
            launch_group.template operator()<32>();
    }

    static void launch_k6_nw8_x32(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt64_bv16_nw8_x32_kernel<HI, HO, FP, VL>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.decay, a.inv,
                    a.cross32, a.cross64, a.u, a.sin, a.init_state,
                    a.final_state, a.cu_seqlens, a.tile_prefix,
                    a.pair_prefix, a.segment_prefix, a.total_tiles,
                    a.total_pairs, a.total_segments, a.T_seq, a.H, a.NT,
                    a.scan_flags);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    template <typename RegBGemm, typename KrCarry>
    static void launch_vsplit_rs_variant(const VsplitRsLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_vsplit_rs_kernel<
                16, HI, HO, FP, VL, RegBGemm, KrCarry>
                <<<a.grid, 64, 0, a.stream>>>(
                    a.v, a.beta, a.out, a.kd, a.qd, a.kr, a.gt, a.inv,
                    a.mqk, a.init_state, a.final_state, a.cu_seqlens,
                    a.tile_prefix, a.total_tiles, a.T_seq, a.H, a.NT);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    // Short-sequence production path: compose the native K32 register-state
    // contraction with tiled kr transpose reads.  The two operators stay
    // independent template axes so the common recurrence contains no ISA or
    // architecture selection.
    static void launch_vsplit_rs_x32(const VsplitRsLaunch& a) {
        launch_vsplit_rs_variant<RegBX32, TiledKrCarryX16>(a);
    }

    // Production gfx950 plain BT64 scan.  PAD removes the FP32 reduction
    // arena's half-wave bank alias.  Multi-arena variants are selected only
    // when the complete scan grid fits in one device wave.  Keeping the arena
    // count compile-time leaves every hot loop free of policy predicates.
    // For STATE_XCHG, REGB_X32 selects the native x32 consumer; false selects
    // the bit-exact pair of original x16 MFMAs over the same packed exchange.
    template <int ARENAS, bool PAD, bool TILED_KR,
              bool REGB_X32 = false, bool STATE_XCHG = false,
              bool SIN_FRAGMENT = false,
              bool RHS_FRAGMENT_XCHG = false>
    static void launch_plain(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            auto launch_cache = [&]<bool BETA_ACTIVATED,
                                    bool DECAY_CACHED>() {
                k2_kda_csplit_bt64_plain_kernel<
                    4, HI, HO, FP, VL, ARENAS, PAD, TILED_KR,
                    REGB_X32, STATE_XCHG, SIN_FRAGMENT,
                    BETA_ACTIVATED, DECAY_CACHED, RHS_FRAGMENT_XCHG>
                    <<<a.grid, 256, 0, a.stream>>>(
                        a.v, a.beta, a.kd, a.kr, a.gt,
                        DECAY_CACHED ? a.decay : nullptr,
                        a.inv, a.cross32,
                        a.cross64, a.u, a.sin, a.init_state, a.final_state,
                        a.cu_seqlens, a.tile_prefix, a.pair_prefix,
                        a.segment_prefix, a.total_tiles, a.total_pairs,
                        a.total_segments, a.T_seq, a.H, a.NT);
            };
            const bool beta_cached = (a.scan_flags & (1u << 1)) != 0;
            const bool decay_cached = (a.scan_flags & (1u << 2)) != 0;
            if (beta_cached && decay_cached)
                launch_cache.template operator()<true, true>();
            else if (beta_cached)
                launch_cache.template operator()<true, false>();
            else if (decay_cached)
                launch_cache.template operator()<false, true>();
            else
                launch_cache.template operator()<false, false>();
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    // Diagnostic hook (`FLASH_KDA_K2=csplit64nw8`) for the plain pipeline.
    // It is intentionally architecture-private so a gfx950 x32/BV8 operator
    // can replace this implementation without touching common dispatch.
    static void launch_plain_nw8(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt64_scan_kernel<8, HI, HO, FP, VL>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.inv, a.cross32,
                    a.cross64, a.u, a.sin, a.init_state, a.final_state,
                    a.cu_seqlens, a.tile_prefix, a.pair_prefix,
                    a.segment_prefix, a.total_tiles, a.total_pairs,
                    a.total_segments, a.T_seq, a.H, a.NT);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    static void launch_segment_output_x32(
            const CsplitSegmentOutputLaunch& a) {
        if (a.is_varlen) {
            k2_kda_csplit_segment_out_x32_kernel<true>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                    a.N, a.total_tiles, a.total_segments,
                    a.T_seq, a.H, a.NT);
        } else {
            k2_kda_csplit_segment_out_x32_kernel<false>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    nullptr, nullptr, nullptr, a.N, a.total_tiles,
                    a.total_segments, a.T_seq, a.H, a.NT);
        }
    }

    template <bool StageSin, int Arenas, bool CACHE_DECAY_LDS = false,
              bool SIN_FRAGMENT = false>
    static void launch_segment_output_gll(
            const CsplitSegmentOutputLaunch& a) {
        if (a.is_varlen) {
            k2_kda_csplit_segment_out_gll_kernel<
                true, StageSin, Arenas, CACHE_DECAY_LDS, SIN_FRAGMENT>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                    a.N, a.total_tiles, a.total_segments,
                    a.T_seq, a.H, a.NT);
        } else {
            k2_kda_csplit_segment_out_gll_kernel<
                false, StageSin, Arenas, CACHE_DECAY_LDS, SIN_FRAGMENT>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.u, a.sin, a.qd, a.kr, a.gt, a.mqk, a.out,
                    nullptr, nullptr, nullptr, a.N, a.total_tiles,
                    a.total_segments, a.T_seq, a.H, a.NT);
        }
    }
};

}  // namespace flashkda_hip::gfx950
