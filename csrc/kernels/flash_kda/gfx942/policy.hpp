#pragma once

#include <cstdlib>
#include <cstring>

#include "../hip_common.hpp"
#include "k1_kda_bt64_aqk_producer_kernel.hpp"
#include "k1_build_prefix_persistent_worklist_kernel.hpp"
#include "k1_kda_bt64_fused_prepare_neumann_kernel.hpp"
#include "../k2_kda_csplit_bt64_bv16_nw8_kernel.hpp"
#include "k2_kda_csplit_bt64_bv16_nw8_prefetch_kernel.hpp"
#include "k2_kda_csplit_bt64_bv16_nw8_persistent_kernel.hpp"
#include "k2_kda_csplit_bt64_fused_nw8_kernel.hpp"
#include "k2_kda_csplit_bt64_out_aqk_v2_kernel.hpp"
#include "kda_local_hybrid_launch_experimental.hpp"
#include "../k2_kda_csplit_bt64_out_bk32_kernel.hpp"

namespace flashkda_hip::gfx942 {

struct LaunchPolicy {
    static HipLaunchPolicy make(
            const FwdParams& p, const HipDeviceInfo& device) {
        // Preserve the measured gfx942 occupancy policy exactly.
        const int nh = p.N * p.H;
        const int bv = device.cu_count > 0 && nh * 4 >= device.cu_count * 2
            ? 32 : 16;
        const char* const k2_route = std::getenv("FLASH_KDA_K2");
        const bool automatic_policy = k2_route == nullptr;
        const bool automatic_large_packed_rs_mw_k6 = automatic_policy &&
            p.cu_seqlens != nullptr && p.N >= 4 && p.T_total >= 16384;
        const bool use_vsplit_rs_mw_k6 =
            automatic_large_packed_rs_mw_k6 ||
            (k2_route != nullptr &&
             std::strcmp(k2_route, "vsplit_rs_mw_k6") == 0);

        // Single-sequence gfx942 defaults are bucketed only when the user has
        // not selected a K2 route.  The promoted hybrid route below uses an
        // explicit tri-state so unset/0/1 mean automatic/rollback/opt-in.
        const bool automatic_single = automatic_policy && p.N == 1;
        const bool automatic_p3_prefetch =
            automatic_single && p.T_total >= 512;
        const bool automatic_hybrid_local = automatic_single &&
            p.T_total >= 1024 && p.T_total < 8192;
        const bool automatic_p3_p4_pipeline =
            automatic_single && p.T_total >= 8192;
        const bool automatic_fused_prepare = automatic_single &&
            ((p.T_total >= 512 && p.T_total < 1024) ||
             p.T_total >= 8192);
        const bool use_p3_prefetch = tuning_enabled(
            "FLASH_KDA_GFX942_P3_PREFETCH", automatic_p3_prefetch);
        const bool hybrid_local_stage_guards =
            std::getenv("FLASH_KDA_CS_SKIP_K1_PREP") == nullptr &&
            std::getenv("FLASH_KDA_CS_SKIP_K1_SOLVE") == nullptr;
        // The matched local-Mqk producer plus range P3/P4 route is the gfx942
        // default for the validated single-sequence H12 long-context regime.
        // An explicit 0 rolls back to the established P3/P4 pipeline, while
        // an exact 1 remains an opt-in even when FLASH_KDA_K2 disables the
        // normal automatic policy.
        const char* const hybrid_local_pipeline_setting = std::getenv(
            "FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE");
        const bool hybrid_local_pipeline_eligible =
            p.N == 1 && p.H == 12 && p.T_total >= 8192;
        const bool use_hybrid_local_pipeline =
            hybrid_local_pipeline_eligible && hybrid_local_stage_guards &&
            (hybrid_local_pipeline_setting == nullptr
                ? automatic_policy
                : env_is_exactly_one(
                    "FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE"));
        // Exact-local P4 is a matched producer/consumer pair.  Skip-stage
        // diagnostics retain the original path so the consumer can never
        // observe an unpublished segment_a arena.
        const bool use_hybrid_local_output =
            (tuning_enabled(
                 "FLASH_KDA_GFX942_HYBRID_LOCAL_OUT",
                 automatic_hybrid_local) ||
             use_hybrid_local_pipeline) &&
            hybrid_local_stage_guards;
        const bool use_aqk_overlap =
            !use_hybrid_local_output &&
            tuning_enabled("FLASH_KDA_GFX942_AQK_OVERLAP", false);
        const bool use_fused_output =
            !use_hybrid_local_output &&
            tuning_enabled("FLASH_KDA_GFX942_FUSED_OUT", false);
        const bool use_p3_p4_pipeline =
            (use_hybrid_local_pipeline ||
             tuning_enabled(
                 "FLASH_KDA_GFX942_P3_P4_PIPELINE",
                 automatic_p3_p4_pipeline)) &&
            (use_hybrid_local_pipeline || !use_hybrid_local_output) &&
            !use_aqk_overlap && !use_fused_output;
        // Benchmark-only escape hatch for the matched Aqk producer/consumer
        // overlap study.  Production PERSISTENT_MIXED remains mutually
        // exclusive with AQK_OVERLAP unless this second exact opt-in is set.
        const bool use_persistent_aqk_experiment = use_aqk_overlap &&
            env_is_exactly_one(
                "FLASH_KDA_GFX942_P3_PERSISTENT_AQK_EXPERIMENT");
        // Strict production opt-in for ragged packed batches.  Keep uniform
        // resume (N=4), 16x1K, and 64x256 on their direct schedules; the two
        // uniform batches regressed 4.5% and 6.9% in the alternating A/B.
        // Do not combine this scheduler with the two auxiliary-stream output
        // experiments. HYBRID_LOCAL_OUT is legal because its segment_a arena
        // is disjoint from the prefix ABI.
        const bool use_p3_persistent =
            env_is_exactly_one("FLASH_KDA_GFX942_P3_PERSISTENT_MIXED") &&
            p.cu_seqlens != nullptr && p.N >= 8 &&
            p.T_total / p.N >= 1536 &&
            !use_p3_p4_pipeline && !use_fused_output &&
            (!use_aqk_overlap || use_persistent_aqk_experiment);
        const bool use_fused_prepare = use_vsplit_rs_mw_k6 ||
            use_hybrid_local_output ||
            tuning_enabled(
                "FLASH_KDA_GFX942_FUSED_PREP_NEUMANN",
                automatic_fused_prepare);
        const K2DefaultRoute default_route = automatic_large_packed_rs_mw_k6
            ? K2DefaultRoute::vsplit_rs_mw_k6
            : K2DefaultRoute::csplit64_k6;
        return {bv, default_route, true,
                use_p3_prefetch ? &launch_k6_nw8_prefetch : &launch_k6_nw8,
                use_fused_output ? &launch_k6_fused : nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                tuning_enabled("FLASH_KDA_GFX942_DPP_NORM", true),
                tuning_enabled(
                    "FLASH_KDA_GFX942_PREP_BETA", p.T_total >= 2048),
                use_fused_prepare
                    ? (use_vsplit_rs_mw_k6
                        ? &launch_fused_prepare_local_mqk_c16
                        : (use_hybrid_local_output
                        ? &launch_fused_prepare_local_mqk
                        : &launch_fused_prepare_neumann))
                    : nullptr,
                use_hybrid_local_output && !use_hybrid_local_pipeline
                    ? &launch_k6_hybrid_local_noop
                    : (use_aqk_overlap ? &launch_k6_aqk_producer : nullptr),
                use_hybrid_local_output && !use_hybrid_local_pipeline
                    ? &launch_k6_hybrid_local_output
                    : (use_aqk_overlap ? &launch_k6_aqk_output : nullptr),
                use_p3_p4_pipeline
                    ? (use_hybrid_local_pipeline
                        ? (use_p3_prefetch
                            ? &launch_k6_p3_p4_pipeline_prefetch_hybrid_local
                            : &launch_k6_p3_p4_pipeline_hybrid_local)
                        : (use_p3_prefetch
                            ? &launch_k6_p3_p4_pipeline_prefetch
                            : &launch_k6_p3_p4_pipeline))
                    : nullptr,
                use_p3_persistent ? &launch_k6_persistent : nullptr,
                use_p3_persistent ? &launch_persistent_prefix : nullptr};
    }

private:
    static bool tuning_enabled(const char* name, bool default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr)
            return default_value;
        return !(value[0] == '0' && value[1] == '\0');
    }

    static bool env_is_exactly_one(const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }

    static void launch_persistent_prefix(const PersistentPrefixLaunch& a) {
        k1_build_prefix_persistent_worklist_kernel<<<
            1, 64, 0, a.stream>>>(
                a.cu_seqlens, a.N, a.tile_prefix, a.pair_prefix,
                a.segment_prefix, a.sequence_worklist, a.sequence_count,
                a.task_counter);
    }

    static void launch_fused_prepare_local_mqk(
            const FusedPrepareNeumannLaunch& a) {
        experimental::launch_fused_prepare_local_mqk(a);
    }

    static void launch_fused_prepare_local_mqk_c16(
            const FusedPrepareNeumannLaunch& a) {
        experimental::launch_fused_prepare_local_mqk_c16(a);
    }

    static void launch_k6_hybrid_local_noop(
            const Csplit64K6OutputLaunch&) {}

    static void launch_k6_hybrid_local_output(
            const Csplit64K6OutputLaunch& a) {
        experimental::launch_k6_hybrid_local_output(a);
    }

    static void launch_k6_nw8(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt64_bv16_nw8_kernel<HI, HO, FP, VL>
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

    static void launch_k6_nw8_prefetch(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt64_bv16_nw8_prefetch_kernel<HI, HO, FP, VL>
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

    static void launch_k6_persistent(const Csplit64ScanLaunch& a) {
        dim3 grid = a.grid;
        if (env_is_exactly_one(
                "FLASH_KDA_GFX942_P3_PERSISTENT_AQK_EXPERIMENT")) {
            const char* value = std::getenv(
                "FLASH_KDA_GFX942_P3_PERSISTENT_BLOCKS");
            const int blocks = value == nullptr ? 0 : std::atoi(value);
            if (blocks == 160 || blocks == 240)
                grid = dim3(blocks);
        }
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            (void)VL;
            k2_kda_csplit_bt64_bv16_nw8_persistent_worklist_kernel<
                HI, HO, FP><<<grid, 512, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.decay, a.inv,
                    a.cross32, a.cross64, a.u, a.sin, a.init_state,
                    a.final_state, a.cu_seqlens, a.tile_prefix,
                    a.pair_prefix, a.segment_prefix, a.sequence_worklist,
                    a.sequence_count, 0, a.task_counter, a.total_tiles,
                    a.total_pairs, a.total_segments, a.T_seq, a.H, a.NT,
                    a.scan_flags);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    static constexpr int kP3P4BatchSegments = 56;
    static constexpr int kP3P4MaxBatches = 64;

    struct P3P4AuxResource {
        int device = -1;
        hipStream_t caller = nullptr;
        hipStream_t stream = nullptr;
        hipEvent_t done = nullptr;
        hipEvent_t ready[kP3P4MaxBatches]{};
        int ready_count = 0;

        ~P3P4AuxResource() {
            for (int i = 0; i < ready_count; ++i) {
                if (ready[i] != nullptr)
                    (void)hipEventDestroy(ready[i]);
            }
            if (done != nullptr)
                (void)hipEventDestroy(done);
            if (stream != nullptr)
                (void)hipStreamDestroy(stream);
        }
    };

    static P3P4AuxResource* p3_p4_aux_resource(
            hipStream_t caller, int batch_count) {
        constexpr int kResourceSlots = 8;
        int device = -1;
        if (batch_count <= 0 || batch_count > kP3P4MaxBatches ||
            hipGetDevice(&device) != hipSuccess || device < 0)
            return nullptr;

        thread_local P3P4AuxResource resources[kResourceSlots];
        P3P4AuxResource* empty = nullptr;
        P3P4AuxResource* resource = nullptr;
        for (auto& candidate : resources) {
            if (candidate.stream != nullptr && candidate.device == device &&
                candidate.caller == caller) {
                resource = &candidate;
                break;
            }
            if (empty == nullptr && candidate.stream == nullptr)
                empty = &candidate;
        }
        if (resource == nullptr)
            resource = empty;
        if (resource == nullptr)
            return nullptr;

        if (resource->stream == nullptr) {
            int priority = 0;
            if (caller != nullptr &&
                hipStreamGetPriority(caller, &priority) != hipSuccess)
                priority = 0;
            if (hipStreamCreateWithPriority(
                    &resource->stream, hipStreamNonBlocking, priority) !=
                hipSuccess)
                return nullptr;
            if (hipEventCreateWithFlags(
                    &resource->done, hipEventDisableTiming) != hipSuccess) {
                (void)hipStreamDestroy(resource->stream);
                resource->stream = nullptr;
                return nullptr;
            }
            resource->device = device;
            resource->caller = caller;
        }

        while (resource->ready_count < batch_count) {
            hipEvent_t event = nullptr;
            if (hipEventCreateWithFlags(
                    &event, hipEventDisableTiming) != hipSuccess)
                return nullptr;
            resource->ready[resource->ready_count++] = event;
        }
        return resource;
    }

    template <bool PREFETCH, bool HI, bool HO, bool STATE_IN_FP32,
              bool STATE_OUT_FP32, bool VL>
    static void launch_k6_p3_range(
            const Csplit64ScanLaunch& a, const void* state_in,
            void* state_out, int segment_begin, int segment_count) {
        if constexpr (PREFETCH) {
            k2_kda_csplit_bt64_bv16_nw8_prefetch_range_kernel<
                HI, HO, STATE_IN_FP32, STATE_OUT_FP32, VL>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.decay, a.inv,
                    a.cross32, a.cross64, a.u, a.sin, state_in, state_out,
                    VL ? a.cu_seqlens : nullptr,
                    VL ? a.tile_prefix : nullptr,
                    VL ? a.pair_prefix : nullptr,
                    VL ? a.segment_prefix : nullptr,
                    a.total_tiles, a.total_pairs, a.total_segments,
                    a.T_seq, a.H, a.NT, a.scan_flags,
                    segment_begin, segment_count);
        } else {
            k2_kda_csplit_bt64_bv16_nw8_range_kernel<
                HI, HO, STATE_IN_FP32, STATE_OUT_FP32, VL>
                <<<a.grid, 512, 0, a.stream>>>(
                    a.v, a.beta, a.kd, a.kr, a.gt, a.decay, a.inv,
                    a.cross32, a.cross64, a.u, a.sin, state_in, state_out,
                    VL ? a.cu_seqlens : nullptr,
                    VL ? a.tile_prefix : nullptr,
                    VL ? a.pair_prefix : nullptr,
                    VL ? a.segment_prefix : nullptr,
                    a.total_tiles, a.total_pairs, a.total_segments,
                    a.T_seq, a.H, a.NT, a.scan_flags,
                    segment_begin, segment_count);
        }
    }

    template <bool HYBRID_LOCAL, bool VL>
    static void launch_k6_p4_range(
            const Csplit64K6OutputLaunch& a, hipStream_t stream,
            int segment_begin, int segment_count) {
        if constexpr (HYBRID_LOCAL) {
            experimental::launch_k6_hybrid_local_output_range(
                a, stream, segment_begin, segment_count);
        } else {
            k2_kda_csplit_bt64_out_bk32_range_kernel<VL>
                <<<dim3(segment_count, a.H), 512, 0, stream>>>(
                    a.cs_u, a.cs_sin, a.qd, a.kr, a.gt, a.out,
                    VL ? a.cu_seqlens : nullptr,
                    VL ? a.tile_prefix : nullptr,
                    VL ? a.segment_prefix : nullptr,
                    a.N, a.total_tiles, a.total_segments,
                    a.T_seq, a.H, a.NT, segment_begin);
        }
    }

    template <bool PREFETCH, bool HYBRID_LOCAL>
    static bool launch_k6_p3_p4_segment_pipeline_impl(
            const Csplit64ScanLaunch& scan,
            const Csplit64K6OutputLaunch& output) {
        // The current range contract uses one sequence-local offset for all
        // scan CTAs, so only N=1 is legal.  H12/8K+ is the measured MI308X
        // regime; smaller or different-head shapes retain the stable path.
        const int segment_count = (scan.T_seq + 63) / 64;
        int batch_segments = segment_count <= 128
            ? 48
            : (HYBRID_LOCAL ? 72 : kP3P4BatchSegments);
        if constexpr (HYBRID_LOCAL) {
            // Runtime sweep knob for the exact hybrid route only.  The
            // established old-P4 pipeline keeps its measured 48/56 constants
            // unchanged.
            const char* value = std::getenv(
                "FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE_BATCH");
            const int requested = value == nullptr ? 0 : std::atoi(value);
            if (requested >= 8 && requested <= 128 &&
                (requested & 7) == 0)
                batch_segments = requested;
        }
        const int batch_count =
            (segment_count + batch_segments - 1) / batch_segments;
        if (scan.N != 1 || output.N != 1 || scan.H != 12 ||
            scan.H != output.H ||
            scan.total_segments != output.total_segments ||
            scan.T_seq < 8192 || segment_count <= batch_segments ||
            batch_count > kP3P4MaxBatches ||
            scan.is_varlen != output.is_varlen)
            return false;

        constexpr int kC = WorkspaceSizes::CHUNK;
        constexpr int kD = WorkspaceSizes::D;
        constexpr int kLocalTiles = 4;
        constexpr int kSegmentATiles = 10;
        constexpr int64_t kTileElems = int64_t(kC) * kC;
        static_assert(
            WorkspaceSizes::kCsplitSegmentA ==
                kSegmentATiles * kTileElems * sizeof(__bf16));
        static_assert(
            (kLocalTiles * kTileElems * sizeof(__bf16)) %
                alignof(float) == 0);
        if constexpr (HYBRID_LOCAL) {
            // Fused K1 stores its compact local Mqk as one contiguous
            // four-tile region across every [head,segment].  The remaining
            // six tiles per record form a disjoint arena for the FP32 carry.
            const int64_t arena_bytes = int64_t(output.H) *
                output.total_segments * WorkspaceSizes::kCsplitSegmentA;
            const int64_t local_bytes = int64_t(output.H) *
                output.total_segments * kLocalTiles * kTileElems *
                sizeof(__bf16);
            const int64_t carry_bytes = int64_t(scan.N) * scan.H *
                kD * kD * sizeof(float);
            if (output.segment_a == nullptr || local_bytes > arena_bytes ||
                carry_bytes > arena_bytes - local_bytes)
                return false;
        }

        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(scan.stream, &capture_status) != hipSuccess ||
            capture_status != hipStreamCaptureStatusNone)
            return false;

        P3P4AuxResource* resource =
            p3_p4_aux_resource(scan.stream, batch_count);
        if (resource == nullptr)
            return false;

        auto launch_first = [&](int count, float* scratch) {
            if (scan.is_varlen) {
                if (!scan.has_state_in)
                    launch_k6_p3_range<
                        PREFETCH, false, true, true, true, true>(
                        scan, nullptr, scratch, 0, count);
                else if (scan.state_fp32)
                    launch_k6_p3_range<
                        PREFETCH, true, true, true, true, true>(
                        scan, scan.init_state, scratch, 0, count);
                else
                    launch_k6_p3_range<
                        PREFETCH, true, true, false, true, true>(
                        scan, scan.init_state, scratch, 0, count);
            } else if (!scan.has_state_in) {
                launch_k6_p3_range<
                    PREFETCH, false, true, true, true, false>(
                    scan, nullptr, scratch, 0, count);
            } else if (scan.state_fp32) {
                launch_k6_p3_range<
                    PREFETCH, true, true, true, true, false>(
                    scan, scan.init_state, scratch, 0, count);
            } else {
                launch_k6_p3_range<
                    PREFETCH, true, true, false, true, false>(
                    scan, scan.init_state, scratch, 0, count);
            }
        };
        auto launch_middle = [&](int begin, int count, float* scratch) {
            if (scan.is_varlen)
                launch_k6_p3_range<
                    PREFETCH, true, true, true, true, true>(
                    scan, scratch, scratch, begin, count);
            else
                launch_k6_p3_range<
                    PREFETCH, true, true, true, true, false>(
                    scan, scratch, scratch, begin, count);
        };
        auto launch_last = [&](int begin, int count, float* scratch) {
            if (scan.is_varlen) {
                if (!scan.has_state_out)
                    launch_k6_p3_range<
                        PREFETCH, true, false, true, true, true>(
                        scan, scratch, nullptr, begin, count);
                else if (scan.state_fp32)
                    launch_k6_p3_range<
                        PREFETCH, true, true, true, true, true>(
                        scan, scratch, scan.final_state, begin, count);
                else
                    launch_k6_p3_range<
                        PREFETCH, true, true, true, false, true>(
                        scan, scratch, scan.final_state, begin, count);
            } else if (!scan.has_state_out) {
                launch_k6_p3_range<
                    PREFETCH, true, false, true, true, false>(
                    scan, scratch, nullptr, begin, count);
            } else if (scan.state_fp32) {
                launch_k6_p3_range<
                    PREFETCH, true, true, true, true, false>(
                    scan, scratch, scan.final_state, begin, count);
            } else {
                launch_k6_p3_range<
                    PREFETCH, true, true, true, false, false>(
                    scan, scratch, scan.final_state, begin, count);
            }
        };

        // Ordinary range P4 does not consume segment_a and can reuse it from
        // byte zero.  Hybrid P4 must retain K1's compact four-tile local-Mqk
        // prefix, so place the FP32 carry at the first disjoint aligned byte.
        auto* scratch_base = output.segment_a;
        if constexpr (HYBRID_LOCAL) {
            scratch_base += int64_t(output.H) * output.total_segments *
                kLocalTiles * kTileElems;
        }
        auto* scratch = reinterpret_cast<float*>(scratch_base);
        auto launch_p4 = [&](hipStream_t stream, int begin, int count) {
            if (scan.is_varlen)
                launch_k6_p4_range<HYBRID_LOCAL, true>(
                    output, stream, begin, count);
            else
                launch_k6_p4_range<HYBRID_LOCAL, false>(
                    output, stream, begin, count);
        };
        for (int batch = 0, begin = 0; begin < segment_count;
             ++batch, begin += batch_segments) {
            const int count = min(batch_segments, segment_count - begin);
            const bool first = begin == 0;
            const bool last = begin + count == segment_count;
            if (first)
                launch_first(count, scratch);
            else if (last)
                launch_last(begin, count, scratch);
            else
                launch_middle(begin, count, scratch);

            if (hipEventRecord(resource->ready[batch], scan.stream) !=
                    hipSuccess ||
                hipStreamWaitEvent(
                    resource->stream, resource->ready[batch], 0) !=
                    hipSuccess) {
                // A runtime event failure is exceptional.  Complete all work
                // already issued before returning to preserve caller order.
                (void)hipStreamSynchronize(scan.stream);
                (void)hipStreamSynchronize(resource->stream);
                launch_p4(scan.stream, begin, count);
                for (int next = begin + batch_segments;
                     next < segment_count; next += batch_segments) {
                    const int next_count =
                        min(batch_segments, segment_count - next);
                    if (next + next_count == segment_count)
                        launch_last(next, next_count, scratch);
                    else
                        launch_middle(next, next_count, scratch);
                    launch_p4(scan.stream, next, next_count);
                }
                return true;
            }
            launch_p4(resource->stream, begin, count);
        }

        if (hipEventRecord(resource->done, resource->stream) != hipSuccess ||
            hipStreamWaitEvent(scan.stream, resource->done, 0) != hipSuccess)
            (void)hipStreamSynchronize(resource->stream);
        return true;
    }

    template <bool PREFETCH, bool HYBRID_LOCAL>
    static bool launch_k6_p3_p4_pipeline_impl(
            const Csplit64ScanLaunch& scan,
            const Csplit64K6OutputLaunch& output) {
        return launch_k6_p3_p4_segment_pipeline_impl<
            PREFETCH, HYBRID_LOCAL>(scan, output);
    }

    static bool launch_k6_p3_p4_pipeline(
            const Csplit64ScanLaunch& scan,
            const Csplit64K6OutputLaunch& output) {
        return launch_k6_p3_p4_pipeline_impl<false, false>(scan, output);
    }

    static bool launch_k6_p3_p4_pipeline_prefetch(
            const Csplit64ScanLaunch& scan,
            const Csplit64K6OutputLaunch& output) {
        return launch_k6_p3_p4_pipeline_impl<true, false>(scan, output);
    }

    static bool launch_k6_p3_p4_pipeline_hybrid_local(
            const Csplit64ScanLaunch& scan,
            const Csplit64K6OutputLaunch& output) {
        return launch_k6_p3_p4_pipeline_impl<false, true>(scan, output);
    }

    static bool launch_k6_p3_p4_pipeline_prefetch_hybrid_local(
            const Csplit64ScanLaunch& scan,
            const Csplit64K6OutputLaunch& output) {
        return launch_k6_p3_p4_pipeline_impl<true, true>(scan, output);
    }

    static void launch_fused_prepare_neumann(
            const FusedPrepareNeumannLaunch& a) {
        const dim3 grid = a.is_varlen
            ? dim3(a.total_segments, a.H)
            : dim3((a.NT + 3) / 4, a.N * a.H);
        auto launch = [&]<bool VL, bool USE_DPP>() {
            k1_kda_bt64_fused_prepare_neumann_kernel<VL, USE_DPP>
                <<<grid, 256, 0, a.stream>>>(
                    a.q, a.k, a.g, a.beta, a.A_log, a.dt_bias,
                    a.scale, a.gate_scale, a.kd, a.qd, a.kr, a.gt,
                    a.decay, a.inv, a.cross32, a.cross64, a.beta_cache,
                    VL ? a.cu_seqlens : nullptr,
                    VL ? a.tile_prefix : nullptr,
                    VL ? a.pair_prefix : nullptr,
                    VL ? a.segment_prefix : nullptr,
                    a.N, a.total_tiles, a.total_pairs, a.total_segments,
                    a.T_seq, a.H, a.NT);
        };
        if (a.is_varlen) {
            if (a.use_dpp)
                launch.template operator()<true, true>();
            else
                launch.template operator()<true, false>();
        } else if (a.use_dpp) {
            launch.template operator()<false, true>();
        } else {
            launch.template operator()<false, false>();
        }
    }

    static void launch_k6_fused(const Csplit64ScanLaunch& a) {
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            const dim3 producer_grid = VL
                ? dim3(a.total_segments, a.H)
                : dim3((a.NT + 3) / 4, a.N * a.H);
            // Diagnostic stage-isolation knobs are intentionally scoped to
            // this opt-in route.  They leave production behavior unchanged
            // when absent and are never valid for correctness runs.
            if (std::getenv(
                    "FLASH_KDA_GFX942_FUSED_OUT_SKIP_PRODUCER") == nullptr)
                k1_kda_bt64_aqk_producer_kernel<VL>
                    <<<producer_grid, 256, 0, a.stream>>>(
                        a.qd, a.kr, a.gt, a.sin, a.cu_seqlens,
                        a.tile_prefix, a.segment_prefix, a.N, a.total_tiles,
                        a.total_segments, a.T_seq, a.H, a.NT);
            if (std::getenv(
                    "FLASH_KDA_GFX942_FUSED_OUT_SKIP_SCAN") == nullptr)
                k2_kda_csplit_bt64_fused_nw8_kernel<HI, HO, FP, VL>
                    <<<a.grid, 512, 0, a.stream>>>(
                        a.v, a.beta, a.out, a.kd, a.qd, a.kr, a.gt,
                        a.decay, a.inv, a.cross32, a.cross64, a.sin,
                        a.init_state, a.final_state, a.cu_seqlens,
                        a.tile_prefix, a.pair_prefix, a.segment_prefix,
                        a.total_tiles, a.total_pairs, a.total_segments,
                        a.T_seq, a.H, a.NT, a.scan_flags);
        };
        dispatch_state_mode(a.is_varlen, a.has_state_in, a.has_state_out,
                            a.state_fp32, launch);
    }

    struct AqkAuxResource {
        hipStream_t stream = nullptr;
        hipEvent_t ready = nullptr;
        hipEvent_t done = nullptr;
        bool pending = false;

        ~AqkAuxResource() {
            if (ready != nullptr)
                (void)hipEventDestroy(ready);
            if (done != nullptr)
                (void)hipEventDestroy(done);
            if (stream != nullptr)
                (void)hipStreamDestroy(stream);
        }
    };

    static AqkAuxResource* aqk_aux_resource() {
        constexpr int kMaxDevices = 64;
        int device = -1;
        if (hipGetDevice(&device) != hipSuccess ||
            device < 0 || device >= kMaxDevices)
            return nullptr;
        thread_local AqkAuxResource resources[kMaxDevices];
        AqkAuxResource& resource = resources[device];
        if (resource.stream != nullptr)
            return &resource;

        hipStream_t stream = nullptr;
        hipEvent_t ready = nullptr;
        hipEvent_t done = nullptr;
        if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) !=
            hipSuccess)
            return nullptr;
        if (hipEventCreateWithFlags(&ready, hipEventDisableTiming) !=
            hipSuccess) {
            (void)hipStreamDestroy(stream);
            return nullptr;
        }
        if (hipEventCreateWithFlags(&done, hipEventDisableTiming) !=
            hipSuccess) {
            (void)hipEventDestroy(ready);
            (void)hipStreamDestroy(stream);
            return nullptr;
        }
        resource.stream = stream;
        resource.ready = ready;
        resource.done = done;
        return &resource;
    }

    static void launch_aqk_producer_on(
            const Csplit64K6OutputLaunch& a, hipStream_t stream) {
        const dim3 grid = a.is_varlen
            ? dim3(a.total_segments, a.H)
            : dim3((a.NT + 3) / 4, a.N * a.H);
        if (a.is_varlen) {
            k1_kda_bt64_aqk_producer_kernel<true>
                <<<grid, 256, 0, stream>>>(
                    a.qd, a.kr, a.gt, a.segment_a, a.cu_seqlens,
                    a.tile_prefix, a.segment_prefix, a.N, a.total_tiles,
                    a.total_segments, a.T_seq, a.H, a.NT);
        } else {
            k1_kda_bt64_aqk_producer_kernel<false>
                <<<grid, 256, 0, stream>>>(
                    a.qd, a.kr, a.gt, a.segment_a, nullptr, nullptr,
                    nullptr, a.N, a.total_tiles, a.total_segments,
                    a.T_seq, a.H, a.NT);
        }
    }

    static void launch_k6_aqk_producer(
            const Csplit64K6OutputLaunch& a) {
        // Diagnostic lower bound for a future K1-inline full-A producer.
        // The matched output intentionally consumes stale scratch when this
        // knob is set, so it is valid for timing only, never correctness.
        if (std::getenv("FLASH_KDA_GFX942_AQK_SKIP_PRODUCER") != nullptr)
            return;
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        const bool capturing =
            hipStreamIsCapturing(a.stream, &capture_status) != hipSuccess ||
            capture_status != hipStreamCaptureStatusNone;
        AqkAuxResource* resource = capturing ? nullptr : aqk_aux_resource();
        if (resource == nullptr || resource->pending) {
            launch_aqk_producer_on(a, a.stream);
            return;
        }

        if (hipEventRecord(resource->ready, a.stream) != hipSuccess ||
            hipStreamWaitEvent(resource->stream, resource->ready, 0) !=
                hipSuccess) {
            launch_aqk_producer_on(a, a.stream);
            return;
        }
        launch_aqk_producer_on(a, resource->stream);
        if (hipEventRecord(resource->done, resource->stream) != hipSuccess) {
            (void)hipStreamSynchronize(resource->stream);
            return;
        }
        resource->pending = true;
    }

    static void launch_k6_aqk_output(
            const Csplit64K6OutputLaunch& a) {
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        const bool capturing =
            hipStreamIsCapturing(a.stream, &capture_status) != hipSuccess ||
            capture_status != hipStreamCaptureStatusNone;
        AqkAuxResource* resource = capturing ? nullptr : aqk_aux_resource();
        if (resource != nullptr && resource->pending) {
            if (hipStreamWaitEvent(a.stream, resource->done, 0) != hipSuccess)
                (void)hipStreamSynchronize(resource->stream);
            resource->pending = false;
        }
        // Timing-only stage isolation: join the auxiliary producer onto the
        // caller stream, then omit the consumer without exposing stale output
        // to a correctness run.
        if (std::getenv(
                "FLASH_KDA_GFX942_AQK_SKIP_CONSUMER") != nullptr)
            return;

        const dim3 grid = a.is_varlen
            ? dim3(a.total_segments, a.H)
            : dim3((a.NT + 3) / 4, a.N * a.H);
        const bool use_persistent_bk64 = env_is_exactly_one(
                "FLASH_KDA_GFX942_P3_PERSISTENT_MIXED") &&
            env_is_exactly_one(
                "FLASH_KDA_GFX942_P3_PERSISTENT_AQK_EXPERIMENT");
        if (a.is_varlen) {
            if (use_persistent_bk64)
                k2_kda_csplit_bt64_out_aqk_v2_kernel<true, 64, false>
                    <<<grid, 512, 0, a.stream>>>(
                        a.cs_u, a.cs_sin, a.qd, a.gt, a.segment_a, a.out,
                        a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                        a.N, a.total_tiles, a.total_segments,
                        a.T_seq, a.H, a.NT);
            else
                k2_kda_csplit_bt64_out_aqk_v2_kernel<true>
                    <<<grid, 512, 0, a.stream>>>(
                        a.cs_u, a.cs_sin, a.qd, a.gt, a.segment_a, a.out,
                        a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                        a.N, a.total_tiles, a.total_segments,
                        a.T_seq, a.H, a.NT);
        } else {
            if (use_persistent_bk64)
                k2_kda_csplit_bt64_out_aqk_v2_kernel<false, 64, false>
                    <<<grid, 512, 0, a.stream>>>(
                        a.cs_u, a.cs_sin, a.qd, a.gt, a.segment_a, a.out,
                        nullptr, nullptr, nullptr, a.N, a.total_tiles,
                        a.total_segments, a.T_seq, a.H, a.NT);
            else
                k2_kda_csplit_bt64_out_aqk_v2_kernel<false>
                    <<<grid, 512, 0, a.stream>>>(
                        a.cs_u, a.cs_sin, a.qd, a.gt, a.segment_a, a.out,
                        nullptr, nullptr, nullptr, a.N, a.total_tiles,
                        a.total_segments, a.T_seq, a.H, a.NT);
        }
    }
};

}  // namespace flashkda_hip::gfx942
