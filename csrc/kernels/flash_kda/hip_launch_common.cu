// FlashKDA HIP/MFMA backend — architecture-neutral launcher implementation.
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <hip/hip_runtime.h>  // hipified under ROCm
#include "hip_common.hpp"
#include "k1_kda_bt16_kernel.hpp"
#include "k1_kda_split_kernel.hpp"
#include "k1_kda_bt32_merge_kernel.hpp"
#include "k1_kda_bt64_merge_kernel.hpp"
#include "k1_kda_bt64_neumann_c_kernel.hpp"
#include "k2_kda_baseline_kernel.hpp"
#include "k2_kda_vsplit_kernel.hpp"
#include "k2_kda_vsplit_db_kernel.hpp"
#include "k2_kda_vsplit_lds_kernel.hpp"
#include "k2_kda_vsplit_db2_kernel.hpp"
#include "k2_kda_vsplit_mw_kernel.hpp"
#include "k2_kda_vsplit_rs_kernel.hpp"
#include "k2_kda_splitscan_kernel.hpp"
#include "k2_kda_wusplit_kernel.hpp"
#include "k2_kda_csplit_kernel.hpp"
#include "k2_kda_csplit_bt32_kernel.hpp"
#include "k2_kda_csplit_bt64_kernel.hpp"
#include "k2_kda_csplit_bt64_stream_kernel.hpp"
#include "k2_kda_csplit_bt64_wide_kernel.hpp"
#include "k2_kda_csplit_bt64_bv16_kernel.hpp"
#include "k2_kda_csplit_bt64_out_kernel.hpp"
#include "k2_kda_csplit_bt64_out_bk32_kernel.hpp"

namespace flashkda_hip {

// Shared ABI for the final scalar scan-mode argument in the native BV16
// kernels.  Keeping beta source selection in this word lets cached and legacy
// routes share the existing beta pointer without adding another 64-bit arg.
constexpr unsigned kCs64ScanUseDecayTable = 1u << 0;
constexpr unsigned kCs64ScanBetaActivated = 1u << 1;
constexpr unsigned kCs64ScanSuffixDecayCached = 1u << 2;

void launch_fwd_common(
        const FwdParams& p,
        const HipDeviceInfo& device,
        const HipLaunchPolicy& policy) {
    const void* q_ptr = p.q_ptr;
    const void* k_ptr = p.k_ptr;
    const void* v_ptr = p.v_ptr;
    const void* g_ptr = p.g_ptr;
    const void* beta_ptr = p.beta_ptr;
    const float scale = p.scale;
    void* out_ptr = p.out_ptr;
    const float* A_log_ptr = p.A_log_ptr;
    const float* dt_bias_ptr = p.dt_bias_ptr;
    const float gate_scale = p.gate_scale;
    const int total_tiles = p.total_tiles;
    const int H_q = p.H_q;
    const int H = p.H;
    const bool is_gva = H_q != H;
    const int N = p.N;
    const void* init_state = p.init_state;
    void* final_state = p.final_state;
    const bool has_state_in = p.has_state_in;
    const bool has_state_out = p.has_state_out;
    const bool state_fp32 = p.state_fp32;
    const int32_t* cu_seqlens = p.cu_seqlens;
    const hipStream_t stream = p.stream;

    const LaunchShape shape = make_launch_shape(p);
    const bool is_varlen = shape.is_varlen;
    const int NT = shape.NT;
    const int T_seq = shape.T_seq;
    const int total_segments = shape.total_segments;
    const int total_pairs = shape.total_pairs;
    const int64_t n_ht = shape.n_ht;

    // Most C-split producers still assume one Q/K head per value head.  GVA may
    // use only a policy-advertised fused BT16 producer, followed by either the
    // register-state K2, the generic uncached context graph, or a separately
    // advertised whole plain C-split graph.  Keep route overrides from
    // selecting any other producer/consumer ABI.
    const char* k2env = is_gva ? nullptr : getenv("FLASH_KDA_K2");
    const bool cs_skip_k1_solve =
        getenv("FLASH_KDA_CS_SKIP_K1_SOLVE") != nullptr;
    const bool cs_skip_k1_prep =
        getenv("FLASH_KDA_CS_SKIP_K1_PREP") != nullptr;
    // Matched gfx942 route: the fused BT64 producer publishes the exact C16
    // local-Mqk/activated-beta ABI consumed by the register-state scan.  The
    // default and explicit spellings share the same callback/skip guard.
    const bool default_vsplit_rs_mw_k6 = !is_gva && !k2env &&
        policy.default_k2_route == K2DefaultRoute::vsplit_rs_mw_k6;
    const bool request_vsplit_rs_mw_k6 =
        default_vsplit_rs_mw_k6 ||
        (k2env && strcmp(k2env, "vsplit_rs_mw_k6") == 0);
    const bool use_vsplit_rs_mw_k6 = request_vsplit_rs_mw_k6 &&
        policy.launch_fused_prepare_neumann != nullptr &&
        !cs_skip_k1_prep && !cs_skip_k1_solve;
    const bool use_csplit32 = k2env && strcmp(k2env, "csplit32") == 0;
    const bool use_csplit64_stream =
        k2env && strcmp(k2env, "csplit64stream") == 0;
    const bool use_csplit64_wide =
        k2env && strcmp(k2env, "csplit64wide") == 0;
    const bool use_csplit64_plain_nw8 = policy.launch_plain_nw8 &&
        k2env && strcmp(k2env, "csplit64nw8") == 0;
    // Close the complete GVA plain producer/post-preparation handshake before
    // admitting C-split.  In particular, either K1 skip knob would otherwise
    // expose the legacy split producer, whose raw q/k indexing assumes H_q==H.
    // A disabled fused BT16 mode is reflected in bt16_k1_supports_gva and must
    // likewise retain the established V-split fallback.
    const bool use_gva_plain_csplit = is_gva &&
        policy.plain_csplit_supports_gva &&
        policy.bt16_k1_supports_gva &&
        policy.launch_bt16_k1 != nullptr &&
        policy.use_bt16_k1_for_plain &&
        policy.launch_plain_k1 != nullptr &&
        !cs_skip_k1_prep && !cs_skip_k1_solve;
    const bool use_default_csplit64 = !k2env &&
        policy.default_k2_route == K2DefaultRoute::csplit64 &&
        (!is_gva || use_gva_plain_csplit);
    const bool use_context_parallel = !k2env &&
        policy.default_k2_route == K2DefaultRoute::context_parallel &&
        policy.launch_context_parallel != nullptr &&
        (!is_gva || (policy.launch_bt16_k1 != nullptr &&
                     policy.bt16_k1_supports_gva));
    // Close the producer/consumer cache ABI before either callback runs.  GVA
    // deliberately stays on raw beta/ws_gt: keeping a single uncached GVA K1
    // specialization avoids a large compile-time/code-size expansion.
    const bool request_context_operand_cache =
        use_context_parallel && !is_gva &&
        policy.launch_bt16_k1 != nullptr &&
        policy.bt16_k1_context_operand_cache;
    // The raw-v2 maximum is a caller promise: every packed sequence is no
    // longer than max_seqlen_upper_bound.  If their sum is exactly N times
    // that bound, every sequence must therefore have the same length.  The
    // measured N=4/G64 affine bucket can use the cheaper dense K1/K2 index
    // mapping and omit the prefix node entirely.  The architecture policy
    // publishes this whole-graph capability only for a canonical exact-"1"
    // recipe; repeat every geometry check here before erasing metadata.
    const int equal_dense_nt = p.max_seqlen_upper_bound > 0
        ? int((int64_t(p.max_seqlen_upper_bound) +
               WorkspaceSizes::CHUNK - 1) / WorkspaceSizes::CHUNK)
        : 0;
    const int64_t equal_dense_total_tiles =
        int64_t(N) * int64_t(equal_dense_nt);
    const bool use_context_equal_dense_n4_g64 =
        !is_gva && policy.context_equal_dense_n4_g64 &&
        use_context_parallel && is_varlen && N == 4 && H > 0 &&
        p.max_seqlen_upper_bound == 4096 &&
        int64_t(p.max_seqlen_upper_bound) * int64_t(N) ==
            int64_t(p.T_total) &&
        policy.launch_bt16_k1 != nullptr &&
        policy.context_group_chunks == 64 &&
        policy.context_direct_max_chunks == 0 && equal_dense_nt > 0 &&
        equal_dense_total_tiles == int64_t(4 * 256) &&
        int64_t(total_tiles) == equal_dense_total_tiles + N &&
        !policy.context_direct_prefixless &&
        policy.launch_context_prefix == nullptr &&
        policy.context_persistent_blocks == 0;
    // A prefix node may be removed only as a matched K1/K2 topology.  The
    // gfx950 policy capability is already strict opt-in; repeat the structural
    // guards here before suppressing architecture-neutral work so a partial or
    // future policy cannot expose uninitialized prefix storage.
    const bool use_context_direct_prefixless =
        use_context_parallel && !is_gva && is_varlen &&
        !use_context_equal_dense_n4_g64 &&
        policy.context_direct_prefixless &&
        policy.launch_bt16_k1 != nullptr &&
        policy.context_group_chunks == 0 &&
        policy.context_direct_max_chunks == 0 &&
        N > 0 && N <= 16;
    // The packed-hybrid grid-stride candidate is a matched prefix/K2 graph.
    // A partial policy must retain the established prefix and every established
    // context kernel rather than silently enabling only half of the topology.
    const bool use_context_persistent = use_context_parallel && !is_gva &&
        is_varlen &&
        policy.launch_context_prefix != nullptr &&
        policy.context_persistent_blocks > 0;
    const bool use_csplit64_k6_auto = !is_gva &&
        ((!k2env &&
         (policy.default_k2_route == K2DefaultRoute::csplit64_k6 ||
          (default_vsplit_rs_mw_k6 && !use_vsplit_rs_mw_k6))) ||
        (k2env && strcmp(k2env, "csplit64rtpk6auto") == 0));
    // Persistent P3 is exposed only as a matched builder/consumer pair.  The
    // architecture policy applies its strict environment and shape guards;
    // explicit non-auto K2 selections continue to take precedence.
    const bool use_csplit64_k6_persistent = use_csplit64_k6_auto &&
        policy.launch_k6_persistent != nullptr &&
        policy.launch_persistent_prefix != nullptr;
    // A single K3 sequence exposes only H*8=96 BV16 scan CTAs on MI308X.
    // Double the waves/CTA until the natural V-split grid can sustain two
    // resident workgroups per CU.  The NW8 phase layout keeps the serial scan
    // at four CTA barriers per BT64; larger batched/ragged grids retain NW4.
    const bool use_csplit64_k6_nw8 =
        (k2env && strcmp(k2env, "csplit64rtpk6bk32nw8") == 0) ||
        use_csplit64_k6_persistent ||
        (use_csplit64_k6_auto && device.cu_count > 0 &&
         (policy.launch_k6_fused != nullptr ||
          int64_t(N) * H * 8 < int64_t(device.cu_count) * 2));
    const bool use_csplit64_k6_bk32 = use_csplit64_k6_auto ||
        use_csplit64_k6_nw8 ||
        (k2env && strcmp(k2env, "csplit64rtpk6bk32") == 0);
    const bool use_csplit64_k6 = use_csplit64_k6_bk32 ||
        (k2env && strcmp(k2env, "csplit64rtpk6") == 0);
    const bool use_csplit64_rtp = use_csplit64_k6 ||
        (k2env && strcmp(k2env, "csplit64rtp") == 0);
    const bool use_csplit64_bv16 = use_csplit64_rtp ||
        (k2env && strcmp(k2env, "csplit64bv16") == 0);
    const bool use_csplit64_native_k1 = use_csplit64_rtp ||
        use_vsplit_rs_mw_k6 ||
        (k2env && strcmp(k2env, "csplit64k1") == 0);
    const bool use_csplit64 = use_default_csplit64 ||
        use_csplit64_plain_nw8 || use_csplit64_stream || use_csplit64_wide ||
        use_csplit64_bv16 || use_csplit64_native_k1 ||
        (k2env && strcmp(k2env, "csplit64") == 0);
    const bool use_bt32_factor = use_csplit32 ||
        (use_csplit64 && !use_csplit64_native_k1);
    // Native BT64 K1 normally needs split preparation.  Its FP16 Neumann/DAG
    // construction bypasses the legacy BT32 merge route; an architecture
    // callback may later replace both stages as one raw-input launch.
    const bool use_csplit = use_bt32_factor || use_csplit64_native_k1 ||
                            (k2env && strcmp(k2env, "csplit") == 0);
    const bool use_split_prep_dpp =
        policy.use_split_prep_dpp && use_csplit && !cs_skip_k1_prep;
    // Producer-side beta is valid only when P1 and native direct-RTP P2 both
    // execute.  Any skip-stage diagnostic falls back to P2's legacy producer
    // instead of consuming stale workspace.
    const bool use_split_prep_beta_cache =
        policy.use_split_prep_beta_cache && use_csplit64_k6 &&
        !cs_skip_k1_prep && !cs_skip_k1_solve;
    // Architecture-private raw-input fusion is opt-in through the policy.  It
    // is legal only for the direct-RTP K6 contract whose P2 consumes a decay
    // table and whose scan consumes the activated beta cache.  Skip-stage
    // diagnostics deliberately retain the original independently skippable
    // P1/P2 launches.
    const bool use_fused_prepare_neumann =
        (use_csplit64_k6 || use_vsplit_rs_mw_k6) &&
        policy.launch_fused_prepare_neumann != nullptr &&
        !cs_skip_k1_prep && !cs_skip_k1_solve;
    const bool use_csplit_bt16_k1 = use_default_csplit64 &&
        policy.launch_bt16_k1 != nullptr &&
        policy.use_bt16_k1_for_plain &&
        (!is_gva || use_gva_plain_csplit) &&
        !cs_skip_k1_prep && !cs_skip_k1_solve;
    // Only the architecture-selected production plain route may replace all
    // three K1 factorization stages.  Explicit routes and skip-stage timing
    // knobs retain the shared implementation as a stable diagnostic baseline.
    const bool use_plain_k1_callback = use_default_csplit64 &&
        policy.launch_plain_k1 != nullptr &&
        !cs_skip_k1_prep && !cs_skip_k1_solve;
    // Only the production K6 specialization publishes activated beta.  The
    // legacy BV16 route and the skip-K1 timing knob must read logits.
    const bool use_csplit64_k6_beta_cache =
        use_csplit64_k6 && !cs_skip_k1_solve;
    // Plain scan consumers are enabled only after the common launcher has
    // selected the solved BT16 producer and the policy has promised that this
    // exact callback specialization publishes activated beta.  This is kept
    // separate from the context-only decay/cache contract.
    const bool use_plain_beta_cache =
        use_csplit_bt16_k1 && use_plain_k1_callback &&
        policy.bt16_k1_plain_beta_cache && !is_gva;
    // The suffix cache is legal only when this exact automatic plain route
    // launches the PRE_SOLVED fused post-preparation callback that publishes
    // it.  Explicit, skipped, fallback, K6, context and other-architecture
    // routes therefore cannot observe stale cs_segment_a contents.
    const bool use_plain_suffix_decay_cache =
        use_csplit_bt16_k1 && use_plain_k1_callback &&
        policy.plain_k1_suffix_decay_cache;
    const WorkspaceLayout ws = carve_workspace(p.workspace_ptr, shape, H, N);
    auto* ws_kd = ws.kd;
    auto* ws_qd = ws.qd;
    auto* ws_kr = ws.kr;
    auto* ws_gt = ws.gt;
    auto* ws_inv = ws.inv;
    auto* ws_mqk = ws.mqk;
    auto* cs_u = ws.cs_u;
    auto* cs_sin = ws.cs_sin;
    auto* cs_cross_inv = ws.cs_cross_inv;
    auto* cs_cross64 = ws.cs_cross64;
    auto* cs_beta = ws.cs_beta;
    auto* cs_segment_a = ws.cs_segment_a;

    // Varlen: build the per-sequence chunk-tile prefix sum in the trailing
    // workspace region reserved by get_workspace_size_hip.
    int* tile_prefix = ws.tile_prefix;
    int* pair_prefix = ws.pair_prefix;
    int* segment_prefix = ws.segment_prefix;
    int* sequence_worklist = ws.sequence_worklist;
    int* sequence_count = ws.sequence_count;
    unsigned int* task_counter = ws.task_counter;
    if (is_varlen && !use_context_direct_prefixless &&
        !use_context_equal_dense_n4_g64) {
        if (use_csplit64_k6_persistent) {
            const PersistentPrefixLaunch args{
                stream, cu_seqlens, N, tile_prefix, pair_prefix,
                segment_prefix, sequence_worklist, sequence_count,
                task_counter};
            policy.launch_persistent_prefix(args);
        } else if (use_context_persistent) {
            const PersistentPrefixLaunch args{
                stream, cu_seqlens, N, tile_prefix, pair_prefix,
                segment_prefix, sequence_worklist, sequence_count,
                task_counter};
            policy.launch_context_prefix(args);
        } else {
            const int context_group_chunks = use_context_parallel
                ? policy.context_group_chunks : 0;
            const int context_direct_max_chunks = use_context_parallel
                ? policy.context_direct_max_chunks : 0;
            // Plain/common C-split scan kernels retain an owner CTA for an
            // empty sequence but return before publishing its final state.
            // Fold the recurrent identity (copy initial state, or write zero)
            // into this already-required prefix launch so no hot K2
            // specialization or launch topology changes.
            // gfx942's vsplit_rs_mw_k6 reuses C-split preparation but its
            // V-split consumer already publishes the empty-sequence state.
            // Keep that established default free of this extra state walk.
            const bool initialize_empty_state =
                use_csplit && !use_vsplit_rs_mw_k6;
            k1_build_tile_prefix<<<1, 64, 0, stream>>>(
                cu_seqlens, N, tile_prefix, pair_prefix, segment_prefix,
                context_group_chunks, context_direct_max_chunks,
                init_state, final_state, H, initialize_empty_state,
                has_state_in, has_state_out, state_fp32);
        }
    }
    if (use_csplit) {
        // Shared split preparation publishes the architecture-neutral
        // workspace ABI consumed by either the common or policy K1 stages.
        constexpr int kSplitPrepThreads = 256;
        constexpr int kSplitSolveThreads = 64;
        constexpr int kBt64NativeThreads = 256;
        // Direct RTP reconstructs qd/ki from bounded operands and never
        // consumes ws_mqk. Native BT64 K1 is the only C producer, so the
        // production K6 route must omit the legacy per-BT16 triangular solve.
        const bool need_split_solve =
            !(use_csplit64_k6 || use_vsplit_rs_mw_k6);
        auto launch_split_prep =
            [&]<bool VL, bool USE_DPP, bool PREP_BETA>(const dim3& grid) {
                k1_kda_split_prep_kernel<VL, USE_DPP, PREP_BETA>
                    <<<grid, kSplitPrepThreads, 0, stream>>>(
                        reinterpret_cast<const __bf16*>(q_ptr),
                        reinterpret_cast<const __bf16*>(k_ptr),
                        reinterpret_cast<const __bf16*>(g_ptr),
                        A_log_ptr, dt_bias_ptr, scale, gate_scale, T_seq, H,
                        ws_kd, ws_qd, ws_kr, ws_gt, cs_u,
                        reinterpret_cast<float*>(ws_mqk),
                        VL ? cu_seqlens : nullptr,
                        VL ? tile_prefix : nullptr, N, total_tiles,
                        reinterpret_cast<const float*>(beta_ptr), cs_beta,
                        VL ? segment_prefix : nullptr, total_segments);
            };
        auto dispatch_split_prep = [&]<bool VL>(const dim3& grid) {
            if (use_split_prep_dpp) {
                if (use_split_prep_beta_cache)
                    launch_split_prep.template operator()<VL, true, true>(grid);
                else
                    launch_split_prep.template operator()<VL, true, false>(grid);
            } else if (use_split_prep_beta_cache) {
                launch_split_prep.template operator()<VL, false, true>(grid);
            } else {
                launch_split_prep.template operator()<VL, false, false>(grid);
            }
        };
        if (is_varlen) {
            dim3 grid(total_tiles, H);
            if (!cs_skip_k1_prep && !use_fused_prepare_neumann &&
                !use_csplit_bt16_k1)
                dispatch_split_prep.template operator()<true>(grid);
            if (!cs_skip_k1_solve && need_split_solve &&
                !use_plain_k1_callback) k1_kda_split_solve_kernel<true><<<grid, kSplitSolveThreads, 0, stream>>>(
                reinterpret_cast<const float*>(beta_ptr),
                ws_kd, ws_qd, cs_u, ws_inv, ws_mqk,
                cu_seqlens, tile_prefix, N, total_tiles, T_seq, H);
        } else {
            dim3 grid(NT, N * H);
            if (!cs_skip_k1_prep && !use_fused_prepare_neumann &&
                !use_csplit_bt16_k1)
                dispatch_split_prep.template operator()<false>(grid);
            if (!cs_skip_k1_solve && need_split_solve &&
                !use_plain_k1_callback) k1_kda_split_solve_kernel<false><<<grid, kSplitSolveThreads, 0, stream>>>(
                reinterpret_cast<const float*>(beta_ptr),
                ws_kd, ws_qd, cs_u, ws_inv, ws_mqk,
                nullptr, nullptr, N, total_tiles, T_seq, H);
        }
        if (use_csplit_bt16_k1) {
            const Bt16K1Launch args{
                stream,
                reinterpret_cast<const __bf16*>(q_ptr),
                reinterpret_cast<const __bf16*>(k_ptr),
                reinterpret_cast<const __bf16*>(g_ptr),
                reinterpret_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr, scale, gate_scale,
                ws_kd, ws_qd, ws_kr, ws_gt, cs_u, ws_inv, ws_mqk,
                cs_beta,
                cu_seqlens, tile_prefix, N, total_tiles, T_seq, H, H_q, NT,
                is_varlen, false, use_plain_beta_cache, false};
            policy.launch_bt16_k1(args);
        }
        if (use_fused_prepare_neumann) {
            // The ordinary K6 producer stores its per-token decay table in
            // ws_mqk.  C16 register-state K2 instead consumes raw ws_gt and
            // needs ws_mqk for the compact per-tile local Mqk matrices.  Its
            // unused decay output is safely redirected to the larger cs_u
            // arena, keeping the public workspace ABI unchanged.
            auto* const fused_decay = use_vsplit_rs_mw_k6
                ? reinterpret_cast<float*>(cs_u)
                : reinterpret_cast<float*>(ws_mqk);
            auto* const fused_local_mqk = use_vsplit_rs_mw_k6
                ? ws_mqk : cs_segment_a;
            const FusedPrepareNeumannLaunch args{
                stream,
                reinterpret_cast<const __bf16*>(q_ptr),
                reinterpret_cast<const __bf16*>(k_ptr),
                reinterpret_cast<const __bf16*>(g_ptr),
                reinterpret_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr, scale, gate_scale,
                ws_kd, ws_qd, ws_kr, ws_gt,
                fused_decay, ws_inv,
                cs_cross_inv, cs_cross64, cs_beta, fused_local_mqk,
                cu_seqlens, tile_prefix, pair_prefix, segment_prefix,
                N, total_tiles, total_pairs, total_segments,
                T_seq, H, NT, use_split_prep_dpp, is_varlen};
            policy.launch_fused_prepare_neumann(args);
        } else if (use_plain_k1_callback) {
            const PlainCsplit64K1Launch args{
                stream, reinterpret_cast<const float*>(beta_ptr), cs_beta,
                reinterpret_cast<float*>(cs_segment_a),
                ws_kd,
                ws_qd, ws_kr, ws_gt, cs_u, ws_inv, ws_mqk,
                cs_cross_inv, cs_cross64, cu_seqlens, tile_prefix,
                pair_prefix, segment_prefix, N, total_tiles, total_pairs,
                total_segments, T_seq, H, NT, is_varlen,
                use_csplit_bt16_k1, use_plain_beta_cache,
                use_plain_suffix_decay_cache};
            policy.launch_plain_k1(args);
        } else if (use_bt32_factor && !cs_skip_k1_solve) {
            if (is_varlen) {
                k1_kda_bt32_merge_kernel<true><<<dim3(total_pairs, H), 64, 0, stream>>>(
                    reinterpret_cast<const float*>(beta_ptr), ws_kd, ws_kr,
                    ws_gt, ws_inv, cs_cross_inv, cu_seqlens,
                    tile_prefix, pair_prefix, N, total_tiles, total_pairs,
                    T_seq, H, NT);
            } else {
                k1_kda_bt32_merge_kernel<false><<<dim3((NT + 1) / 2, N * H), 64, 0, stream>>>(
                    reinterpret_cast<const float*>(beta_ptr), ws_kd, ws_kr,
                    ws_gt, ws_inv, cs_cross_inv, nullptr, nullptr, nullptr,
                    N, total_tiles, total_pairs, T_seq, H, NT);
            }
        }
        // USE_DECAY_TABLE KKT prefetches each four-component decay factor as
        // aligned f32x4 vectors before folding it in ascending chunk order.
        if (!use_fused_prepare_neumann && !use_plain_k1_callback &&
            use_csplit64_native_k1 &&
            !cs_skip_k1_solve) {
            if (is_varlen) {
                if (use_csplit64_k6) {
                    if (use_split_prep_beta_cache)
                        k1_kda_bt64_neumann_c_kernel<true, true, true><<<
                            dim3(total_segments, H), kBt64NativeThreads,
                            kK1Bt64NeumannCSmemBytes, stream>>>(
                            cs_beta, ws_kd, ws_kr, cs_u, ws_gt,
                            reinterpret_cast<const float*>(ws_mqk), ws_inv,
                            cs_cross_inv, cs_cross64, cu_seqlens, tile_prefix,
                            pair_prefix, segment_prefix, N, total_tiles,
                            total_pairs, total_segments, T_seq, H, NT);
                    else
                        k1_kda_bt64_neumann_c_kernel<true, true, false><<<
                            dim3(total_segments, H), kBt64NativeThreads,
                            kK1Bt64NeumannCSmemBytes, stream>>>(
                            reinterpret_cast<const float*>(beta_ptr), ws_kd,
                            ws_kr, cs_u, ws_gt,
                            reinterpret_cast<const float*>(ws_mqk), ws_inv,
                            cs_cross_inv, cs_cross64, cu_seqlens, tile_prefix,
                            pair_prefix, segment_prefix, N, total_tiles,
                            total_pairs, total_segments, T_seq, H, NT);
                } else
                    k1_kda_bt64_neumann_c_kernel<true, false><<<
                        dim3(total_segments, H), kBt64NativeThreads,
                        kK1Bt64NeumannCSmemBytes, stream>>>(
                        reinterpret_cast<const float*>(beta_ptr), ws_kd, ws_kr,
                        cs_u, ws_gt, reinterpret_cast<const float*>(ws_mqk),
                        ws_inv, cs_cross_inv, cs_cross64, cu_seqlens,
                        tile_prefix, pair_prefix, segment_prefix, N, total_tiles,
                        total_pairs, total_segments, T_seq, H, NT);
            } else {
                if (use_csplit64_k6) {
                    if (use_split_prep_beta_cache)
                        k1_kda_bt64_neumann_c_kernel<false, true, true><<<
                            dim3((NT + 3) / 4, N * H), kBt64NativeThreads,
                            kK1Bt64NeumannCSmemBytes, stream>>>(
                            cs_beta, ws_kd, ws_kr, cs_u, ws_gt,
                            reinterpret_cast<const float*>(ws_mqk), ws_inv,
                            cs_cross_inv, cs_cross64, nullptr, nullptr,
                            nullptr, nullptr, N, total_tiles, total_pairs,
                            total_segments, T_seq, H, NT);
                    else
                        k1_kda_bt64_neumann_c_kernel<false, true, false><<<
                            dim3((NT + 3) / 4, N * H), kBt64NativeThreads,
                            kK1Bt64NeumannCSmemBytes, stream>>>(
                            reinterpret_cast<const float*>(beta_ptr), ws_kd,
                            ws_kr, cs_u, ws_gt,
                            reinterpret_cast<const float*>(ws_mqk), ws_inv,
                            cs_cross_inv, cs_cross64, nullptr, nullptr,
                            nullptr, nullptr, N, total_tiles, total_pairs,
                            total_segments, T_seq, H, NT);
                } else
                    k1_kda_bt64_neumann_c_kernel<false, false><<<
                        dim3((NT + 3) / 4, N * H), kBt64NativeThreads,
                        kK1Bt64NeumannCSmemBytes, stream>>>(
                        reinterpret_cast<const float*>(beta_ptr), ws_kd, ws_kr,
                        cs_u, ws_gt, reinterpret_cast<const float*>(ws_mqk),
                        ws_inv, cs_cross_inv, cs_cross64, nullptr, nullptr,
                        nullptr, nullptr, N, total_tiles, total_pairs,
                        total_segments, T_seq, H, NT);
            }
        } else if (!use_fused_prepare_neumann &&
                   !use_plain_k1_callback && use_csplit64 &&
                   !cs_skip_k1_solve) {
            if (is_varlen) {
                k1_kda_bt64_merge_kernel<true><<<dim3(total_segments, H), 256, 0, stream>>>(
                    reinterpret_cast<const float*>(beta_ptr), ws_kd, ws_kr, ws_gt,
                    ws_inv, cs_cross_inv, cs_cross64, cu_seqlens, tile_prefix,
                    pair_prefix, segment_prefix, N, total_tiles, total_pairs,
                    total_segments, T_seq, H, NT);
            } else {
                k1_kda_bt64_merge_kernel<false><<<dim3((NT + 3) / 4, N * H), 256, 0, stream>>>(
                    reinterpret_cast<const float*>(beta_ptr), ws_kd, ws_kr, ws_gt,
                    ws_inv, cs_cross_inv, cs_cross64, nullptr, nullptr, nullptr,
                    nullptr, N, total_tiles, total_pairs, total_segments,
                    T_seq, H, NT);
            }
        }
    }

    const bool use_bt16_k1_callback = !use_csplit && !k2env &&
        policy.launch_bt16_k1 != nullptr &&
        (!is_gva || policy.bt16_k1_supports_gva);
    const bool context_operands_cached =
        request_context_operand_cache && use_bt16_k1_callback;
    if (use_bt16_k1_callback) {
        const bool k1_is_varlen =
            is_varlen && !use_context_equal_dense_n4_g64;
        const int k1_total_tiles = use_context_equal_dense_n4_g64
            ? int(equal_dense_total_tiles) : total_tiles;
        const int k1_nt = use_context_equal_dense_n4_g64
            ? equal_dense_nt : NT;
        const Bt16K1Launch args{
            stream,
            reinterpret_cast<const __bf16*>(q_ptr),
            reinterpret_cast<const __bf16*>(k_ptr),
            reinterpret_cast<const __bf16*>(g_ptr),
            reinterpret_cast<const float*>(beta_ptr),
            A_log_ptr, dt_bias_ptr, scale, gate_scale,
            ws_kd, ws_qd, ws_kr, ws_gt, cs_u, ws_inv, ws_mqk,
            cs_beta,
            k1_is_varlen ? cu_seqlens : nullptr,
            use_context_direct_prefixless ? nullptr
                                          : (k1_is_varlen ? tile_prefix
                                                          : nullptr),
            N, k1_total_tiles, T_seq, H, H_q, k1_nt,
            k1_is_varlen, context_operands_cached, false,
            use_context_direct_prefixless};
        policy.launch_bt16_k1(args);
    } else if (!use_csplit && is_varlen) {
        // Grid over the global tile upper bound; each block resolves its
        // (seq, chunk) via tile_prefix and drops gap tiles.
        dim3 grid(total_tiles, H);
        if (is_gva)
            k1_kda_bt16_kernel<true, true><<<grid, 64, 0, stream>>>(
                reinterpret_cast<const __bf16*>(q_ptr),
                reinterpret_cast<const __bf16*>(k_ptr),
                reinterpret_cast<const __bf16*>(g_ptr),
                reinterpret_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr,
                scale, gate_scale, T_seq, H, H_q,
                ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                cu_seqlens, tile_prefix, N, total_tiles);
        else
            k1_kda_bt16_kernel<true, false><<<grid, 64, 0, stream>>>(
                reinterpret_cast<const __bf16*>(q_ptr),
                reinterpret_cast<const __bf16*>(k_ptr),
                reinterpret_cast<const __bf16*>(g_ptr),
                reinterpret_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr,
                scale, gate_scale, T_seq, H, H,
                ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                cu_seqlens, tile_prefix, N, total_tiles);
    } else if (!use_csplit) {
        dim3 grid(NT, N * H);
        if (is_gva)
            k1_kda_bt16_kernel<false, true><<<grid, 64, 0, stream>>>(
                reinterpret_cast<const __bf16*>(q_ptr),
                reinterpret_cast<const __bf16*>(k_ptr),
                reinterpret_cast<const __bf16*>(g_ptr),
                reinterpret_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr,
                scale, gate_scale, T_seq, H, H_q,
                ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                nullptr, nullptr, N, total_tiles);
        else
            k1_kda_bt16_kernel<false, false><<<grid, 64, 0, stream>>>(
                reinterpret_cast<const __bf16*>(q_ptr),
                reinterpret_cast<const __bf16*>(k_ptr),
                reinterpret_cast<const __bf16*>(g_ptr),
                reinterpret_cast<const float*>(beta_ptr),
                A_log_ptr, dt_bias_ptr,
                scale, gate_scale, T_seq, H, H,
                ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                nullptr, nullptr, N, total_tiles);
    }

    // ---- K2: recurrence. Default V-split (one block per (seq,head,V-group));
    // baseline (one block per (seq,head)) kept behind runtime dispatch for A/B.
    // Env knobs for measurement: FLASH_KDA_K2=baseline|vsplit, FLASH_KDA_BV=16|32|64.
    auto* v_bf   = reinterpret_cast<const __bf16*>(v_ptr);
    auto* beta_f = reinterpret_cast<const float*>(beta_ptr);
    const bool use_csplit64_scan_beta_cache =
        use_csplit64_k6_beta_cache || use_plain_beta_cache;
    auto* scan_beta_f = use_csplit64_scan_beta_cache ? cs_beta : beta_f;
    auto* scan_decay_f = use_plain_suffix_decay_cache
        ? reinterpret_cast<const float*>(cs_segment_a)
        : reinterpret_cast<const float*>(ws_mqk);
    auto* out_bf = reinterpret_cast<__bf16*>(out_ptr);
    const unsigned csplit64_scan_flags =
        (use_csplit64_k6 ? kCs64ScanUseDecayTable : 0u) |
        (use_csplit64_scan_beta_cache ? kCs64ScanBetaActivated : 0u) |
        (use_plain_suffix_decay_cache
            ? kCs64ScanSuffixDecayCached : 0u);

    const bool use_baseline = k2env && strcmp(k2env, "baseline") == 0;
    const bool use_splitscan = k2env && strcmp(k2env, "splitscan") == 0;
    const bool use_wusplit = k2env && strcmp(k2env, "wusplit") == 0;
    // FLASH_KDA_K2=vsplit forces the original (non-pipelined) V-split for A/B;
    // vsplit_db exposes the software-pipelined LDS-state route, while the
    // register-state variant is the production default.
    const bool force_vsplit_old = k2env && strcmp(k2env, "vsplit") == 0;
    const bool use_vsplit_db = k2env && strcmp(k2env, "vsplit_db") == 0;
    const bool use_vsplit_lds = k2env && strcmp(k2env, "vsplit_lds") == 0;
    const bool use_vsplit_db2 = k2env && strcmp(k2env, "vsplit_db2") == 0;
    const bool use_vsplit_mw = k2env && strcmp(k2env, "vsplit_mw") == 0;
    const bool use_vsplit_rs = k2env && strcmp(k2env, "vsplit_rs") == 0;
    const bool use_vsplit_rs_mw =
        (k2env && strcmp(k2env, "vsplit_rs_mw") == 0) ||
        use_vsplit_rs_mw_k6;
    const bool use_default_vsplit_rs = is_gva ||
        (!k2env && policy.default_k2_route == K2DefaultRoute::vsplit_rs);

    // Plain C-split segment output is architecture-neutral by default.  A
    // policy may substitute an architecture-private kernel through the typed
    // callback without making this TU include or branch on private ISA.
    auto launch_segment_output = [&]() {
        const dim3 grid = is_varlen
            ? dim3(total_segments, H)
            : dim3((NT + 3) / 4, N * H);
        const CsplitSegmentOutputLaunch args{
            grid, stream, cs_u, cs_sin, ws_qd, ws_kr, ws_gt, ws_mqk,
            out_bf, cu_seqlens, tile_prefix, segment_prefix, N,
            total_tiles, total_segments, T_seq, H, NT, is_varlen};
        if (policy.launch_segment_output != nullptr) {
            policy.launch_segment_output(args);
        } else if (is_varlen) {
            k2_kda_csplit_segment_out_kernel<true><<<
                grid, 512, 0, stream>>>(
                cs_u, cs_sin, ws_qd, ws_kr, ws_gt, ws_mqk, out_bf,
                cu_seqlens, tile_prefix, segment_prefix, N, total_tiles,
                total_segments, T_seq, H, NT);
        } else {
            k2_kda_csplit_segment_out_kernel<false><<<
                grid, 512, 0, stream>>>(
                cs_u, cs_sin, ws_qd, ws_kr, ws_gt, ws_mqk, out_bf,
                nullptr, nullptr, nullptr, N, total_tiles, total_segments,
                T_seq, H, NT);
        }
    };

    // The production V-split route may substitute an architecture-private
    // register-state operator.  Explicit K2/BV selections retain the common
    // kernel as a stable cross-architecture correctness and performance A/B.
    auto launch_private_vsplit_rs = [&](dim3 grid) {
        if (!use_default_vsplit_rs || policy.launch_vsplit_rs == nullptr ||
            getenv("FLASH_KDA_BV") != nullptr)
            return false;
        const VsplitRsLaunch args{
            grid, stream, v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr,
            ws_gt, ws_inv, ws_mqk, init_state, final_state, cu_seqlens,
            tile_prefix, total_tiles, T_seq, H, NT, has_state_in,
            has_state_out, state_fp32, is_varlen};
        policy.launch_vsplit_rs(args);
        return true;
    };

    // State (M4) / varlen (M4): only the register-resident rs kernel carries
    // initial/final state AND handles variable-length sequences. Either takes
    // priority over experimental env selection. gfx942 uses the same
    // occupancy-aware BV=16/32 policy as dense; gfx950 retains its tuned BV=16
    // configuration, including its state-aware context-parallel callback.
    if (use_context_parallel) {
        // The ordinary affine ABI stores BF16 A in cs_sin and FP32 b in
        // cs_u. At G8 cs_u is too small for b, while an aligned dense single
        // sequence admits an exact role swap:
        //
        //   cs_u   : NT C16 tiles * 4 KiB == (NT/8) A maps * 32 KiB
        //   cs_sin : (NT/4) C64 slots * 32 KiB
        //          == (NT/8) b maps * 64 KiB.
        //
        // Repeat every structural policy guard here because these pointers
        // are an architecture-neutral ABI. In particular, a partial G8 tail
        // would need complete extra A/b maps; the policy must reject it, and
        // common must not silently apply the shape-specific alias contract.
        static_assert(
            8 * WorkspaceSizes::kCsplitU ==
                WorkspaceSizes::D * WorkspaceSizes::D *
                    int(sizeof(__bf16)),
            "G8 cs_u/affine_a capacity contract changed");
        static_assert(
            2 * WorkspaceSizes::kCsplitSin ==
                WorkspaceSizes::D * WorkspaceSizes::D * int(sizeof(float)),
            "G8 cs_sin/affine_b capacity contract changed");
        const bool swap_g8_dense_single_affine_arenas =
            policy.context_group_chunks == 8 &&
            policy.context_direct_max_chunks == 0 &&
            !is_varlen && N == 1 && NT > 0 && total_tiles == NT &&
            (NT % 8) == 0;
        auto* const context_affine_a = swap_g8_dense_single_affine_arenas
            ? cs_u : cs_sin;
        auto* const context_affine_b = swap_g8_dense_single_affine_arenas
            ? reinterpret_cast<float*>(cs_sin)
            : reinterpret_cast<float*>(cs_u);
        const bool context_is_varlen =
            is_varlen && !use_context_equal_dense_n4_g64;
        const int context_total_tiles = use_context_equal_dense_n4_g64
            ? int(equal_dense_total_tiles) : total_tiles;
        const int context_nt = use_context_equal_dense_n4_g64
            ? equal_dense_nt : NT;
        const ContextParallelLaunch args{
            stream, v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt,
            ws_inv, ws_mqk, cs_beta, context_affine_a, context_affine_b,
            init_state, final_state,
            context_is_varlen ? cu_seqlens : nullptr,
            use_context_direct_prefixless ? nullptr
                                          : (context_is_varlen ? tile_prefix
                                                               : nullptr),
            context_is_varlen ? segment_prefix : nullptr,
            context_is_varlen ? sequence_worklist : nullptr,
            context_is_varlen ? sequence_count : nullptr,
            context_total_tiles, T_seq, H, N, context_nt,
            has_state_in, has_state_out, state_fp32, context_is_varlen,
            policy.context_group_chunks,
            policy.context_direct_max_chunks,
            use_context_persistent ? policy.context_persistent_blocks : 0,
            use_context_direct_prefixless,
            use_context_equal_dense_n4_g64,
            context_operands_cached,
            is_gva,
            policy.context_automatic_gva_packed_nw4,
            policy.context_automatic_gva_equal_n4_g16};
        policy.launch_context_parallel(args);
    } else if (use_vsplit_rs_mw) {
        // Strict opt-in probe: pack two or four independent register-state V16
        // waves into one CTA so they share the V-independent chunk workspace.
        // NW1 intentionally calls the established kernel as the A/B control.
        // The matched K6 route defaults to NW4 only when the natural
        // sequence/head grid is already large enough to cover the device;
        // low-N and ragged tails retain NW2.  FLASH_KDA_RS_MW remains an
        // exact A/B override for all explicit register-state routes.
        const char* rs_mw_env = getenv("FLASH_KDA_RS_MW");
        const int rs_mw_requested = rs_mw_env
            ? atoi(rs_mw_env)
            : (use_vsplit_rs_mw_k6 ? (N >= 8 ? 4 : 2) : 1);
        const int rs_mw = (rs_mw_requested == 2 || rs_mw_requested == 4)
            ? rs_mw_requested : 1;
        auto launch = [&]<bool HI, bool HO, bool FP, bool VL>() {
            auto launch_mw = [&]<int NW, bool ACTIVATED_BETA>() {
                const float* const scan_beta = ACTIVATED_BETA
                    ? cs_beta : beta_f;
                k2_kda_vsplit_rs_mw_kernel<
                    NW, HI, HO, FP, VL, ACTIVATED_BETA><<<
                        dim3(N * H, WorkspaceSizes::D / (NW * 16)),
                        NW * 64, 0, stream>>>(
                        v_bf, scan_beta, out_bf, ws_kd, ws_qd, ws_kr,
                        ws_gt, ws_inv, ws_mqk, init_state, final_state,
                        cu_seqlens, tile_prefix, total_tiles, T_seq, H, NT);
            };
            if (rs_mw == 4) {
                if (use_vsplit_rs_mw_k6)
                    launch_mw.template operator()<4, true>();
                else
                    launch_mw.template operator()<4, false>();
            } else if (rs_mw == 2) {
                if (use_vsplit_rs_mw_k6)
                    launch_mw.template operator()<2, true>();
                else
                    launch_mw.template operator()<2, false>();
            } else if (use_vsplit_rs_mw_k6) {
                // Keep the matched producer/consumer contract under the NW1
                // diagnostic override as well; ordinary NW1 routes retain the
                // established single-wave raw-beta control below.
                launch_mw.template operator()<1, true>();
            } else {
                k2_kda_vsplit_rs_kernel<16, HI, HO, FP, VL><<<
                    dim3(N * H, WorkspaceSizes::D / 16), 64, 0, stream>>>(
                    v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt,
                    ws_inv, ws_mqk, init_state, final_state, cu_seqlens,
                    tile_prefix, total_tiles, T_seq, H, NT);
            }
        };
        dispatch_state_mode(is_varlen, has_state_in, has_state_out,
                            state_fp32, launch);
    } else if (use_csplit64) {
        constexpr int D = WorkspaceSizes::D;
        const bool cs_skip_scan = getenv("FLASH_KDA_CS_SKIP_SCAN") != nullptr;
        const bool cs_skip_out = getenv("FLASH_KDA_CS_SKIP_OUT") != nullptr;
        const bool use_csplit64_k6_fused = use_csplit64_k6_nw8 &&
            policy.launch_k6_fused != nullptr &&
            !cs_skip_scan && !cs_skip_out;
        const bool use_csplit64_k6_aqk_output =
            use_csplit64_k6_bk32 && !use_csplit64_k6_fused &&
            policy.launch_k6_aqk_producer != nullptr &&
            policy.launch_k6_aqk_output != nullptr;
        const bool use_csplit64_k6_pipeline =
            use_csplit64_k6_nw8 && !use_csplit64_k6_fused &&
            !use_csplit64_k6_aqk_output && !cs_skip_scan && !cs_skip_out &&
            policy.launch_k6_pipeline != nullptr;
        const int persistent_blocks = device.cu_count > 0
            ? device.cu_count * 2 : 160;
        dim3 scan_grid = use_csplit64_k6_persistent
            ? dim3(persistent_blocks)
            : dim3(N * H, D / (use_csplit64_wide ? 32 : 16));
        const Csplit64ScanLaunch private_scan_args{
            scan_grid, stream, v_bf, scan_beta_f, ws_qd, out_bf, ws_kd,
            ws_kr, ws_gt,
            scan_decay_f, ws_inv, cs_cross_inv,
            cs_cross64, cs_u, cs_sin, init_state, final_state, cu_seqlens,
            tile_prefix, pair_prefix, segment_prefix, sequence_worklist,
            sequence_count, task_counter, total_tiles, total_pairs,
            total_segments, T_seq, H, N, NT, csplit64_scan_flags,
            has_state_in, has_state_out, state_fp32, is_varlen};
        const Csplit64K6OutputLaunch aqk_output_args{
            stream, ws_qd, ws_kr, ws_gt, cs_u, cs_sin, cs_segment_a,
            out_bf, cu_seqlens, tile_prefix, segment_prefix, N,
            total_tiles, total_segments, T_seq, H, NT, is_varlen};

        auto launch_scan = [&]<bool HI, bool HO, bool FP, bool VL>() {
            // Architecture-private scan hooks keep ISA-specific kernels out of
            // this TU while preserving the common state/output pipeline.
            if (use_csplit64_k6_fused) {
                Csplit64ScanLaunch args = private_scan_args;
                args.beta = scan_beta_f;
                args.has_state_in = HI;
                args.has_state_out = HO;
                args.state_fp32 = FP;
                args.is_varlen = VL;
                policy.launch_k6_fused(args);
            } else if (use_csplit64_k6_persistent) {
                Csplit64ScanLaunch args = private_scan_args;
                args.beta = scan_beta_f;
                args.has_state_in = HI;
                args.has_state_out = HO;
                args.state_fp32 = FP;
                args.is_varlen = VL;
                policy.launch_k6_persistent(args);
            } else if (use_csplit64_k6_nw8) {
                Csplit64ScanLaunch args = private_scan_args;
                args.beta = scan_beta_f;
                args.has_state_in = HI;
                args.has_state_out = HO;
                args.state_fp32 = FP;
                args.is_varlen = VL;
                policy.launch_k6_nw8(args);
            } else if (use_default_csplit64 &&
                       policy.launch_plain_default != nullptr) {
                Csplit64ScanLaunch args = private_scan_args;
                args.has_state_in = HI;
                args.has_state_out = HO;
                args.state_fp32 = FP;
                args.is_varlen = VL;
                policy.launch_plain_default(args);
            } else if (use_csplit64_plain_nw8) {
                Csplit64ScanLaunch args = private_scan_args;
                args.has_state_in = HI;
                args.has_state_out = HO;
                args.state_fp32 = FP;
                args.is_varlen = VL;
                policy.launch_plain_nw8(args);
            } else if (use_csplit64_bv16) {
                k2_kda_csplit_bt64_bv16_kernel<HI, HO, FP, VL>
                    <<<scan_grid, 256, 0, stream>>>(
                        v_bf, scan_beta_f, ws_kd, ws_kr, ws_gt,
                        reinterpret_cast<const float*>(ws_mqk), ws_inv,
                        cs_cross_inv, cs_cross64, cs_u, cs_sin, init_state,
                        final_state, cu_seqlens, tile_prefix, pair_prefix,
                        segment_prefix, total_tiles, total_pairs,
                        total_segments, T_seq, H, NT, csplit64_scan_flags);
            } else if (use_csplit64_wide) {
                k2_kda_csplit_bt64_wide_kernel<HI, HO, FP, VL>
                    <<<scan_grid, 256, 0, stream>>>(
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv,
                        cs_cross_inv, cs_cross64, cs_u, cs_sin, init_state,
                        final_state, cu_seqlens, tile_prefix, pair_prefix,
                        segment_prefix, total_tiles, total_pairs,
                        total_segments, T_seq, H, NT);
            } else if (use_csplit64_stream) {
                k2_kda_csplit_bt64_stream_kernel<HI, HO, FP, VL>
                    <<<scan_grid, 256, 0, stream>>>(
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv,
                        cs_cross_inv, cs_cross64, cs_u, cs_sin, init_state,
                        final_state, cu_seqlens, tile_prefix, pair_prefix,
                        segment_prefix, total_tiles, total_pairs,
                        total_segments, T_seq, H, NT);
            } else {
                k2_kda_csplit_bt64_scan_kernel<4, HI, HO, FP, VL>
                    <<<scan_grid, 256, 0, stream>>>(
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv,
                        cs_cross_inv, cs_cross64, cs_u, cs_sin, init_state,
                        final_state, cu_seqlens, tile_prefix, pair_prefix,
                        segment_prefix, total_tiles, total_pairs,
                        total_segments, T_seq, H, NT);
            }
        };
        bool k6_pipeline_handled = false;
        if (use_csplit64_k6_pipeline) {
            Csplit64ScanLaunch args = private_scan_args;
            args.beta = scan_beta_f;
            k6_pipeline_handled =
                policy.launch_k6_pipeline(args, aqk_output_args);
        }
        if (!k6_pipeline_handled) {
            // The architecture callback may enqueue compact-A production on
            // an auxiliary stream.  Its matched output callback joins that
            // work only after P3 has been enqueued on the main stream.
            if (use_csplit64_k6_aqk_output && !cs_skip_out)
                policy.launch_k6_aqk_producer(aqk_output_args);
            if (!cs_skip_scan)
                dispatch_state_mode(is_varlen, has_state_in, has_state_out,
                                    state_fp32, launch_scan);
        }

        // The RTP K6 consumes only cs_u/cs_sin and bounded preparation
        // operands, with per-chunk-pair decay scales cached once per CTA;
        // unlike the replay output it has no ws_mqk dependency.
        constexpr int kBt64OutputThreads = 512;
        if (k6_pipeline_handled) {
            // The architecture callback enqueued both range P3 and old P4.
        } else if (use_csplit64_k6_fused) {
            // Output is produced in the architecture-private recurrent scan.
        } else if (!cs_skip_out && use_csplit64_k6_aqk_output) {
            policy.launch_k6_aqk_output(aqk_output_args);
        } else if (!cs_skip_out && use_csplit64_k6_bk32 && is_varlen) {
            k2_kda_csplit_bt64_out_bk32_kernel<true><<<
                dim3(total_segments, H), kBt64OutputThreads, 0, stream>>>(
                cs_u, cs_sin, ws_qd, ws_kr, ws_gt, out_bf,
                cu_seqlens, tile_prefix, segment_prefix, N, total_tiles,
                total_segments, T_seq, H, NT);
        } else if (!cs_skip_out && use_csplit64_k6_bk32) {
            k2_kda_csplit_bt64_out_bk32_kernel<false><<<
                dim3((NT + 3) / 4, N * H), kBt64OutputThreads, 0, stream>>>(
                cs_u, cs_sin, ws_qd, ws_kr, ws_gt, out_bf,
                nullptr, nullptr, nullptr, N, total_tiles, total_segments,
                T_seq, H, NT);
        } else if (!cs_skip_out && use_csplit64_k6 && is_varlen) {
            k2_kda_csplit_bt64_out_kernel<true><<<
                dim3(total_segments, H), kBt64OutputThreads, 0, stream>>>(
                cs_u, cs_sin, ws_qd, ws_kr, ws_gt, out_bf,
                cu_seqlens, tile_prefix, segment_prefix, N, total_tiles,
                total_segments, T_seq, H, NT);
        } else if (!cs_skip_out && use_csplit64_k6) {
            k2_kda_csplit_bt64_out_kernel<false><<<
                dim3((NT + 3) / 4, N * H), kBt64OutputThreads, 0, stream>>>(
                cs_u, cs_sin, ws_qd, ws_kr, ws_gt, out_bf,
                nullptr, nullptr, nullptr, N, total_tiles, total_segments,
                T_seq, H, NT);
        } else if (!cs_skip_out) {
            launch_segment_output();
        }
    } else if (use_csplit32) {
        constexpr int D = WorkspaceSizes::D;
        const bool cs_skip_scan = getenv("FLASH_KDA_CS_SKIP_SCAN") != nullptr;
        const bool cs_skip_out = getenv("FLASH_KDA_CS_SKIP_OUT") != nullptr;
        dim3 scan_grid(N * H, D / 16);
        auto launch_scan = [&]<bool HI, bool HO, bool FP, bool VL>() {
            k2_kda_csplit_bt32_scan_kernel<HI, HO, FP, VL>
                <<<scan_grid, 256, 0, stream>>>(
                v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv, cs_cross_inv,
                cs_u, cs_sin, init_state, final_state, cu_seqlens, tile_prefix,
                pair_prefix, segment_prefix, total_tiles, total_pairs,
                total_segments,
                T_seq, H, NT);
        };
        if (!cs_skip_scan)
            dispatch_state_mode(is_varlen, has_state_in, has_state_out,
                                state_fp32, launch_scan);

        if (!cs_skip_out)
            launch_segment_output();
    } else if (use_csplit) {
        constexpr int D = WorkspaceSizes::D;
        const char* bvenv = getenv("FLASH_KDA_BV");
        int scan_bv = bvenv ? atoi(bvenv) : policy.default_k2_bv;
        if (scan_bv != 16 && scan_bv != 32) scan_bv = 16;
        const char* out_bvenv = getenv("FLASH_KDA_OUT_BV");
        int out_bv = out_bvenv ? atoi(out_bvenv) : 64;
        if (out_bv != 16 && out_bv != 32 && out_bv != 64) out_bv = 64;
        const char* mwenv = getenv("FLASH_KDA_CS_SCAN_MW");
        const int scan_mw = mwenv ? atoi(mwenv) : 1;

        auto launch_scan = [&](auto BVc) {
            constexpr int BV = decltype(BVc)::value;
            dim3 scan_grid(N * H, D / BV);
            #define CS_SCAN(HI, HO, FP, VL) do { \
                if constexpr (BV == 16) { \
                  if (scan_mw == 2) \
                    k2_kda_csplit_scan_mw_kernel<2, HI, HO, FP, VL><<<scan_grid, 128, 0, stream>>>( \
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv, cs_u, cs_sin, \
                        init_state, final_state, cu_seqlens, tile_prefix, segment_prefix, \
                        total_tiles, total_segments, T_seq, H, NT); \
                  else if (scan_mw == 4) \
                    k2_kda_csplit_scan_mw_kernel<4, HI, HO, FP, VL><<<scan_grid, 256, 0, stream>>>( \
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv, cs_u, cs_sin, \
                        init_state, final_state, cu_seqlens, tile_prefix, segment_prefix, \
                        total_tiles, total_segments, T_seq, H, NT); \
                  else if (scan_mw == 8) \
                    k2_kda_csplit_scan_mw_kernel<8, HI, HO, FP, VL><<<scan_grid, 512, 0, stream>>>( \
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv, cs_u, cs_sin, \
                        init_state, final_state, cu_seqlens, tile_prefix, segment_prefix, \
                        total_tiles, total_segments, T_seq, H, NT); \
                  else \
                    k2_kda_csplit_scan_kernel<BV, HI, HO, FP, VL><<<scan_grid, 64, 0, stream>>>( \
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv, cs_u, cs_sin, \
                        init_state, final_state, cu_seqlens, tile_prefix, segment_prefix, \
                        total_tiles, total_segments, T_seq, H, NT); \
                } else \
                    k2_kda_csplit_scan_kernel<BV, HI, HO, FP, VL><<<scan_grid, 64, 0, stream>>>( \
                        v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv, cs_u, cs_sin, \
                        init_state, final_state, cu_seqlens, tile_prefix, segment_prefix, \
                        total_tiles, total_segments, T_seq, H, NT); \
            } while (0)
            auto launch_state_mode = [&]<
                    bool HI, bool HO, bool FP, bool VL>() {
                CS_SCAN(HI, HO, FP, VL);
            };
            dispatch_state_mode(is_varlen, has_state_in, has_state_out,
                                state_fp32, launch_state_mode);
            #undef CS_SCAN
        };
        const bool cs_skip_scan = getenv("FLASH_KDA_CS_SKIP_SCAN") != nullptr;
        const bool cs_skip_out = getenv("FLASH_KDA_CS_SKIP_OUT") != nullptr;
        if (!cs_skip_scan) {
            if (scan_bv == 32) launch_scan(std::integral_constant<int, 32>{});
            else               launch_scan(std::integral_constant<int, 16>{});
        }

        // Production K6 shape: one BT64 segment and all eight V16 tiles per
        // 8-wave CTA. FLASH_KDA_OUT_BV is retained only for the old diagnostic
        // kernel and intentionally does not alter this launch geometry.
        if (!cs_skip_out)
            launch_segment_output();
    } else if (has_state_in || has_state_out || is_varlen) {
        constexpr int D = WorkspaceSizes::D;
        if (!launch_private_vsplit_rs(dim3(N * H, D / 16))) {
            auto launch_state = [&](auto BVc) {
                constexpr int BV = decltype(BVc)::value;
                dim3 g2(N * H, D / BV);
                #define RS_STATE(HI, HO, FP, VL) \
                    k2_kda_vsplit_rs_kernel<BV, HI, HO, FP, VL><<<g2, 64, 0, stream>>>( \
                        v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, \
                        init_state, final_state, cu_seqlens, tile_prefix, total_tiles, \
                        T_seq, H, NT)
                auto launch_state_mode = [&]<
                        bool HI, bool HO, bool FP, bool VL>() {
                    RS_STATE(HI, HO, FP, VL);
                };
                dispatch_state_mode(is_varlen, has_state_in, has_state_out,
                                    state_fp32, launch_state_mode);
                #undef RS_STATE
            };
            const int state_bv = policy.default_k2_bv;
            if (state_bv == 32)
                launch_state(std::integral_constant<int, 32>{});
            else
                launch_state(std::integral_constant<int, 16>{});
        }
    } else if (use_wusplit) {
        // GDN-style WU + output-split. Three passes: chunk-parallel WU factors
        // (u_bar/w_bar) -> serial state carry (only pass on the critical path)
        // -> chunk-parallel output. Scratch (u_bar, w_bar, S_in snapshot) is
        // cached static across calls. BV via FLASH_KDA_BV (default 16).
        constexpr int C = WorkspaceSizes::CHUNK;
        constexpr int D = WorkspaceSizes::D;
        const char* bvenv = getenv("FLASH_KDA_BV");
        int bv = bvenv ? atoi(bvenv) : 16;
        if (bv != 16 && bv != 32 && bv != 64) bv = 16;

        static void*  wu_buf = nullptr;
        static size_t wu_cap = 0;
        const size_t bytes_ub  = (size_t)n_ht * C * D * sizeof(__bf16);  // u_bar
        const size_t bytes_wb  = (size_t)n_ht * C * D * sizeof(__bf16);  // w_bar
        const size_t bytes_sin = (size_t)n_ht * D * D * sizeof(__bf16);  // S_in
        const size_t need = bytes_ub + bytes_wb + bytes_sin;
        if (need > wu_cap) {
            if (wu_buf) hipFree(wu_buf);
            hipMalloc(&wu_buf, need);
            wu_cap = need;
        }
        auto* wu_ubar = reinterpret_cast<__bf16*>(reinterpret_cast<char*>(wu_buf));
        auto* wu_wbar = reinterpret_cast<__bf16*>(reinterpret_cast<char*>(wu_buf) + bytes_ub);
        auto* wu_sin  = reinterpret_cast<__bf16*>(reinterpret_cast<char*>(wu_buf) + bytes_ub + bytes_wb);

        k2_wu_prep_kernel<<<dim3(NT, N * H), 64, 0, stream>>>(
            v_bf, beta_f, ws_kd, ws_inv, wu_ubar, wu_wbar, T_seq, H, NT);

        auto launch_carry_out = [&](auto BVc) {
            constexpr int BV = decltype(BVc)::value;
            k2_wu_carry_kernel<BV><<<dim3(N * H, D / BV), 64, 0, stream>>>(
                wu_ubar, wu_wbar, ws_kr, ws_gt, wu_sin, T_seq, H, NT);
            k2_wu_out_kernel<BV><<<dim3(N * H, NT, D / BV), 64, 0, stream>>>(
                wu_ubar, wu_wbar, wu_sin, ws_qd, ws_mqk, out_bf, T_seq, H, NT);
        };
        if (bv == 32)      launch_carry_out(std::integral_constant<int, 32>{});
        else if (bv == 64) launch_carry_out(std::integral_constant<int, 64>{});
        else               launch_carry_out(std::integral_constant<int, 16>{});
    } else if (use_splitscan) {
        // Split the time dimension into nseg segments to fill the GPU for small
        // N*H. Auto-pick L (chunks/segment) so N*H*nseg ~ TARGET blocks; env
        // FLASH_KDA_SS_L overrides. Scratch (M_seg, Sloc, Sin) cached across calls.
        constexpr int D = WorkspaceSizes::D;
        const int nh = N * H;
        int L;
        const char* lenv = getenv("FLASH_KDA_SS_L");
        if (lenv) {
            L = atoi(lenv); if (L < 1) L = 1;
        } else {
            // Bound the serial scan length: the phase-2 scan runs nseg serial
            // [K,K]@[K,V] steps in one wave, so many segments (small L) is
            // catastrophic. Target ~256/nh segments, capped at 32, floored so
            // segments are never tiny. Empirically L~32 is the sweet spot;
            // split-scan only beats vsplit for very small nh with long NT.
            int target_nseg = 256 / nh;
            if (target_nseg > 32) target_nseg = 32;
            if (target_nseg < 2)  target_nseg = 2;
            if (target_nseg > NT) target_nseg = NT;
            L = (NT + target_nseg - 1) / target_nseg;
            if (L < 8) L = 8;
        }
        const int nseg = (NT + L - 1) / L;

        static void*  ss_buf = nullptr;
        static size_t ss_cap = 0;
        const size_t one = (size_t)nh * nseg * D * D * sizeof(__bf16);
        const size_t need = 3 * one;   // mseg, sloc, sin
        if (need > ss_cap) {
            if (ss_buf) hipFree(ss_buf);
            hipMalloc(&ss_buf, need);
            ss_cap = need;
        }
        auto* ss_mseg = reinterpret_cast<__bf16*>(reinterpret_cast<char*>(ss_buf));
        auto* ss_sloc = reinterpret_cast<__bf16*>(reinterpret_cast<char*>(ss_buf) + one);
        auto* ss_sin  = reinterpret_cast<__bf16*>(reinterpret_cast<char*>(ss_buf) + 2 * one);

        dim3 gseg(nh, nseg);
        k2_ss_mseg_kernel<<<gseg, 64, 0, stream>>>(
            ws_kd, ws_kr, ws_gt, ws_inv, beta_f, ss_mseg, T_seq, H, NT, L, nseg);
        k2_ss_sloc_kernel<<<gseg, 64, 0, stream>>>(
            v_bf, beta_f, ws_kd, ws_kr, ws_gt, ws_inv, ss_sloc, T_seq, H, NT, L, nseg);
        k2_ss_scan_kernel<<<dim3(nh), 64, 0, stream>>>(
            ss_mseg, ss_sloc, ss_sin, NT, L, nseg);
        k2_ss_apply_kernel<<<gseg, 64, 0, stream>>>(
            v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            ss_sin, T_seq, H, NT, L, nseg);
    } else if (use_vsplit_db) {
        // Explicit A/B route for the depth-1 register-prefetch pipeline.  This
        // used to fall through to the default register-state kernel at BV=16,
        // making FLASH_KDA_K2=vsplit_db silently benchmark the wrong kernel.
        const char* bvenv = getenv("FLASH_KDA_BV");
        int bv = bvenv ? atoi(bvenv) : 16;
        constexpr int D = WorkspaceSizes::D;
        if (bv == 32)      k2_kda_vsplit_db_kernel<32><<<dim3(N * H, D / 32), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
        else if (bv == 64) k2_kda_vsplit_db_kernel<64><<<dim3(N * H, D / 64), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
        else               k2_kda_vsplit_db_kernel<16><<<dim3(N * H, D / 16), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
    } else if (use_vsplit_lds) {
        // Step E: direct global->LDS (global_load_lds) double-buffered prefetch.
        // BV via FLASH_KDA_BV (default 16, the tr-read fast path).
        const char* bvenv = getenv("FLASH_KDA_BV");
        int bv = bvenv ? atoi(bvenv) : 16;
        constexpr int D = WorkspaceSizes::D;
        if (bv == 32)      k2_kda_vsplit_lds_kernel<32><<<dim3(N * H, D / 32), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
        else if (bv == 64) k2_kda_vsplit_lds_kernel<64><<<dim3(N * H, D / 64), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
        else               k2_kda_vsplit_lds_kernel<16><<<dim3(N * H, D / 16), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
    } else if (use_vsplit_db2) {
        // Step E-alt: depth-2 register prefetch (two chunks' HBM loads in flight).
        const char* bvenv = getenv("FLASH_KDA_BV");
        int bv = bvenv ? atoi(bvenv) : 16;
        constexpr int D = WorkspaceSizes::D;
        if (bv == 32)      k2_kda_vsplit_db2_kernel<32><<<dim3(N * H, D / 32), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
        else if (bv == 64) k2_kda_vsplit_db2_kernel<64><<<dim3(N * H, D / 64), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
        else               k2_kda_vsplit_db2_kernel<16><<<dim3(N * H, D / 16), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
    } else if (use_vsplit_mw) {
        // Occupancy axis: multi-wave V-split. NW V-groups per block share the
        // V-independent workspace (loaded once) and co-reside as NW waves/CU.
        // BV fixed 16 (tr path); NW must divide D/BV = 8.
        constexpr int D = WorkspaceSizes::D, BV = 16;
        const char* mwenv = getenv("FLASH_KDA_MW");
        int nw = mwenv ? atoi(mwenv) : 4;
        if (nw != 1 && nw != 2 && nw != 4 && nw != 8) nw = 4;
        auto launch_mw = [&](auto NWc) {
            constexpr int NW = decltype(NWc)::value;
            k2_kda_vsplit_mw_kernel<BV, NW><<<dim3(N * H, (D / BV) / NW), NW * 64, 0, stream>>>(
                v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, T_seq, H, NT);
        };
        if (nw == 1)      launch_mw(std::integral_constant<int, 1>{});
        else if (nw == 2) launch_mw(std::integral_constant<int, 2>{});
        else if (nw == 8) launch_mw(std::integral_constant<int, 8>{});
        else              launch_mw(std::integral_constant<int, 4>{});
    } else if (use_vsplit_rs) {
        // M2b: fp32 register-resident state. Same data flow / SW-pipeline as the
        // default vsplit_db, but the recurrence state lives in fp32 VGPRs (cast to
        // bf16 only for MFMA operands), so the carry never rounds to bf16 between
        // chunks. Accuracy candidate (measured vs fp32-state oracle). BV=16 keeps
        // the state at 32 fp32/lane; larger BV risks spill so falls back to 16.
        const char* bvenv = getenv("FLASH_KDA_BV");
        int bv = bvenv ? atoi(bvenv) : 16;
        constexpr int D = WorkspaceSizes::D;
        if (bv == 32)      k2_kda_vsplit_rs_kernel<32><<<dim3(N * H, D / 32), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, nullptr, nullptr, nullptr, total_tiles, T_seq, H, NT);
        else if (bv == 64) k2_kda_vsplit_rs_kernel<64><<<dim3(N * H, D / 64), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, nullptr, nullptr, nullptr, total_tiles, T_seq, H, NT);
        else               k2_kda_vsplit_rs_kernel<16><<<dim3(N * H, D / 16), 64, 0, stream>>>(
                               v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, nullptr, nullptr, nullptr, total_tiles, T_seq, H, NT);
    } else if (use_baseline) {
        k2_kda_baseline_kernel<<<dim3(N * H), 64, 0, stream>>>(
            v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            T_seq, H, NT);
    } else {
        const char* bvenv = getenv("FLASH_KDA_BV");
        int bv = bvenv ? atoi(bvenv) : policy.default_k2_bv;
        constexpr int D = WorkspaceSizes::D;
        if (bv == 16 && !force_vsplit_old) {
            // Default recurrence family: fp32 register-resident state on top of
            // the software-pipelined V-split data flow.  An architecture policy
            // may replace its MFMA/LDS operators through the callback; otherwise
            // this common K16 specialization remains the fallback.
            if (!launch_private_vsplit_rs(dim3(N * H, D / 16)))
                k2_kda_vsplit_rs_kernel<16><<<
                    dim3(N * H, D / 16), 64, 0, stream>>>(
                    v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt,
                    ws_inv, ws_mqk, nullptr, nullptr, nullptr, nullptr,
                    total_tiles, T_seq, H, NT);
        } else if (bv == 32 && policy.use_default_rs_bv32 && !bvenv) {
            // gfx942 occupancy-aware default: register-state BV32 once the
            // reduced V-grid still supplies two waves/CU.
            k2_kda_vsplit_rs_kernel<32><<<dim3(N * H, D / 32), 64, 0, stream>>>(
                v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                nullptr, nullptr, nullptr, nullptr, total_tiles, T_seq, H, NT);
        } else if (bv == 32) {
            k2_kda_vsplit_kernel<32><<<dim3(N * H, D / 32), 64, 0, stream>>>(
                v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                T_seq, H, NT);
        } else if (bv == 64) {
            k2_kda_vsplit_kernel<64><<<dim3(N * H, D / 64), 64, 0, stream>>>(
                v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                T_seq, H, NT);
        } else {
            // force_vsplit_old at bv=16, or any unrecognized bv
            k2_kda_vsplit_kernel<16><<<dim3(N * H, D / 16), 64, 0, stream>>>(
                v_bf, beta_f, out_bf, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                T_seq, H, NT);
        }
    }
}

}  // namespace flashkda_hip
