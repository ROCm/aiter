#pragma once

#include <hip/hip_runtime.h>  // hipified -> <hip/hip_runtime.h> under ROCm
#include <cstdint>
#include <type_traits>

#include "flash_kda.h"

namespace flashkda_hip {

// Keep architecture identity explicit.  An unknown CDNA target must never
// silently inherit tuning (or ISA assumptions) from a supported GPU.
enum class HipArchitecture : uint8_t {
    gfx942,
    gfx950,
    unsupported,
};

struct HipDeviceInfo {
    HipArchitecture architecture = HipArchitecture::unsupported;
    int cu_count = 0;
};

// Stable host-side ABI shared by the dispatcher and architecture launchers.
// Kernel-private choices belong in the gfx942/gfx950 policy, not in this
// argument bundle.
struct FwdParams {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* g_ptr;
    const void* beta_ptr;
    float scale;
    void* out_ptr;
    void* workspace_ptr;
    const float* A_log_ptr;
    const float* dt_bias_ptr;
    float gate_scale;
    int total_tiles;
    int T_total;
    // Q/K heads may be shared by an integer group of value/gate/state heads.
    // H remains the value-head count used by every workspace and K2 kernel.
    int H_q;
    int H;
    int N;
    const void* init_state;
    void* final_state;
    bool has_state_in;
    bool has_state_out;
    bool state_fp32;
    const int32_t* cu_seqlens;
    hipStream_t stream;
    // Optional caller-provided upper bound used only by host route policy.
    // Zero preserves the legacy packed policy.  Execution geometry continues
    // to come from T_total/N and device prefix metadata.
    int max_seqlen_upper_bound = 0;
};

enum class K2DefaultRoute : uint8_t {
    vsplit_rs,
    csplit64,
    csplit64_k6,
    // Low-head, deep-sequence route.  Architecture-private launchers form
    // affine maps for independent context ranges, scan those short maps, then
    // replay the ranges from their true incoming state.
    context_parallel,
    // Matched gfx942 route: a C16 fused K1 producer publishes local Mqk and
    // activated beta for the multi-wave register-state K2 consumer.
    vsplit_rs_mw_k6,
};

// Host callback ABI for architecture-private BT64 scan kernels.  Keeping the
// callback outside the common kernel TU avoids both ISA coupling and duplicate
// HIP symbols when gfx942 and gfx950 are built into one extension.
struct Csplit64ScanLaunch {
    dim3 grid;
    hipStream_t stream;
    const __bf16* v;
    const float* beta;
    const __bf16* qd;
    __bf16* out;
    const __bf16* kd;
    const __bf16* kr;
    const float* gt;
    const float* decay;
    const __bf16* inv;
    const __bf16* cross32;
    const __bf16* cross64;
    __bf16* u;
    __bf16* sin;
    const void* init_state;
    void* final_state;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    const int* pair_prefix;
    const int* segment_prefix;
    const int* sequence_worklist;
    const int* sequence_count;
    unsigned int* task_counter;
    int total_tiles;
    int total_pairs;
    int total_segments;
    int T_seq;
    int H;
    int N;
    int NT;
    unsigned scan_flags;
    bool has_state_in;
    bool has_state_out;
    bool state_fp32;
    bool is_varlen;
};

using Csplit64ScanLauncher = void (*)(const Csplit64ScanLaunch&);

// Same-stream builder for the packed prefix/worklist ABI consumed by an
// architecture-private persistent scan.  A policy exposes this only together
// with its matching persistent consumer.
struct PersistentPrefixLaunch {
    hipStream_t stream;
    const int32_t* cu_seqlens;
    int N;
    int* tile_prefix;
    int* pair_prefix;
    int* segment_prefix;
    int* sequence_worklist;
    int* sequence_count;
    unsigned int* task_counter;
};

using PersistentPrefixLauncher = void (*)(const PersistentPrefixLaunch&);

// Host callback ABI for the independent BT16 preparation used by V-split.
// The common launcher owns routing/workspace packing; a policy may replace the
// complete K1 launch sequence without putting architecture tests in this TU.
struct Bt16K1Launch {
    hipStream_t stream;
    const __bf16* q;
    const __bf16* k;
    const __bf16* g;
    const float* beta;
    const float* A_log;
    const float* dt_bias;
    float scale;
    float gate_scale;
    __bf16* kd;
    __bf16* qd;
    __bf16* kr;
    float* gt;
    __bf16* kinv;
    __bf16* inv;
    __bf16* mqk;
    // Optional tile-major [H * total_tiles, C] activated-beta cache.  Context
    // and plain C-split reuse the existing C64 beta arena so their consumers
    // do not repeat the producer's sigmoid work.
    float* beta_cache;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    int N;
    int total_tiles;
    int T_seq;
    int H;
    int H_q;
    int NT;
    bool is_varlen;
    bool cache_context_operands;
    // The common launcher sets this only after the selected architecture
    // policy has promised that its fused producer will publish activated beta
    // into beta_cache.  It is independent of the context-only ws_gt decay ABI.
    bool publish_activated_beta;
    // Matched packed-direct experiment: resolve the compact C16 tile mapping
    // from cu_seqlens inside the fused gfx950 producer.  The common launcher
    // sets this only when it also skips the prefix node and requests the
    // matching direct K2 specialization.
    bool packed_direct_prefixless = false;
};

using Bt16K1Launcher = void (*)(const Bt16K1Launch&);

// Host callback ABI for the three plain C-split K1 factorization stages.
// Split preparation remains shared; an architecture policy may replace the
// solve/BT32/BT64 sequence without making the common launcher depend on
// architecture-private MFMA or LDS intrinsics.
struct PlainCsplit64K1Launch {
    hipStream_t stream;
    const float* beta;
    const float* beta_cache;
    // Optional segment-major [H * total_segments, D, 4] FP32 suffix-decay
    // cache backed by the otherwise-idle cs_segment_a arena on plain gfx950.
    float* suffix_decay;
    __bf16* kd;
    const __bf16* qd;
    const __bf16* kr;
    const float* gt;
    const __bf16* kinv;
    __bf16* inv;
    __bf16* mqk;
    __bf16* cross32;
    __bf16* cross64;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    const int* pair_prefix;
    const int* segment_prefix;
    int N;
    int total_tiles;
    int total_pairs;
    int total_segments;
    int T_seq;
    int H;
    int NT;
    bool is_varlen;
    // True only when the common launcher actually ran a producer that
    // published the per-C16 inv/mqk factors.  Architecture callbacks must
    // consume this fact instead of re-deriving it from process environment.
    bool pre_solved;
    // True only when beta_cache contains the activated values published by
    // the exact producer invocation that established pre_solved.
    bool beta_activated;
    // True only for the matched PRE_SOLVED fused post-preparation producer.
    // A false value forbids both publication and the matching scan consumer.
    bool publish_suffix_decay;
};

using PlainCsplit64K1Launcher =
    void (*)(const PlainCsplit64K1Launch&);

// Host callback ABI for an architecture-private direct-RTP K1 producer that
// replaces both split preparation and the native BT64 factorization.  The
// common launcher still owns route selection and workspace packing; the
    // callback must publish the same persistent kd/qd/kr/gt/decay/factor/beta
// arenas as the shared two-launch sequence.
struct FusedPrepareNeumannLaunch {
    hipStream_t stream;
    const __bf16* q;
    const __bf16* k;
    const __bf16* g;
    const float* beta;
    const float* A_log;
    const float* dt_bias;
    float scale;
    float gate_scale;
    __bf16* kd;
    __bf16* qd;
    __bf16* kr;
    float* gt;
    float* decay;
    __bf16* inv;
    __bf16* cross32;
    __bf16* cross64;
    float* beta_cache;
    // Per-segment scratch shared with an optional K6 output callback.
    // gfx942's exact-local route writes four packed 16x16 BF16 Mqk tiles at
    // the front of this arena during fused preparation.
    __bf16* segment_a;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    const int* pair_prefix;
    const int* segment_prefix;
    int N;
    int total_tiles;
    int total_pairs;
    int total_segments;
    int T_seq;
    int H;
    int NT;
    bool use_dpp;
    bool is_varlen;
};

using FusedPrepareNeumannLauncher =
    void (*)(const FusedPrepareNeumannLaunch&);

// Host callback ABI for the register-state V-split recurrence.  The common
// launcher owns route/state dispatch and workspace packing; an architecture
// policy may instantiate the same recurrence body with private MFMA operators
// without exposing those ISA helpers to this translation unit.
struct VsplitRsLaunch {
    dim3 grid;
    hipStream_t stream;
    const __bf16* v;
    const float* beta;
    __bf16* out;
    const __bf16* kd;
    const __bf16* qd;
    const __bf16* kr;
    const float* gt;
    const __bf16* inv;
    const __bf16* mqk;
    const void* init_state;
    void* final_state;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    int total_tiles;
    int T_seq;
    int H;
    int NT;
    bool has_state_in;
    bool has_state_out;
    bool state_fp32;
    bool is_varlen;
};

using VsplitRsLauncher = void (*)(const VsplitRsLaunch&);

// Host callback ABI for an architecture-private context-parallel recurrence.
// The common launcher owns routing, the ordinary C16 K1 workspace, and two
// otherwise-unused C-split arenas repurposed as affine scratch.  A successful
// callback produces output/final state completely; no C-split replay follows.
struct ContextParallelLaunch {
    hipStream_t stream;
    const __bf16* v;
    const float* beta;
    __bf16* out;
    const __bf16* kd;
    const __bf16* qd;
    const __bf16* kr;
    const float* gt;
    const __bf16* inv;
    const __bf16* mqk;
    const float* beta_cache;
    __bf16* affine_a;
    float* affine_b;
    const void* init_state;
    void* final_state;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    int* context_prefix;
    const int* sequence_worklist;
    const int* sequence_count;
    int total_tiles;
    int T_seq;
    int H;
    int N;
    int NT;
    bool has_state_in;
    bool has_state_out;
    bool state_fp32;
    bool is_varlen;
    // Resolved once by the architecture policy before the common prefix
    // launch.  A zero group count selects direct replay; a nonzero direct
    // threshold selects the packed hybrid partition.
    int context_group_chunks;
    int context_direct_max_chunks;
    // Nonzero only for the matched packed-hybrid prefix/K2 topology.  Each
    // affine phase launches this many deterministic grid-stride CTAs; zero
    // retains every established context symbol and launch shape.
    int context_persistent_blocks;
    // True only for the matched packed-direct two-node topology.  Consumers
    // must derive their compact C16 workspace base from cu_seqlens and must
    // not dereference tile_prefix when this is set.
    bool packed_direct_prefixless = false;
    // True only after common dispatch has proved the packed N=4, 4K-each
    // promise and normalized the complete K1/producer/scan/replay graph to
    // the equivalent dense layout.  This is a whole-graph ABI bit: no kernel
    // may infer it from the now-null packed metadata in isolation.
    bool equal_dense_n4_g64 = false;
    // True only when common dispatch selected the matching BT16 producer and
    // that exact invocation was asked to publish activated beta plus the
    // context decay operands.  Consumers must use this launch-time handshake
    // instead of independently re-reading process environment.
    bool context_operands_cached = false;
    // Whole-launch geometry bit used to keep grouped-value attention on the
    // generic uncached context graph.  Context K2 itself is value-head based,
    // but cache-dependent and metadata-eliding specializations are not yet
    // part of the GVA producer/consumer contract.
    bool is_gva = false;
    // Policy-resolved routing facts.  Carry these across common dispatch so
    // the architecture callback does not reconstruct an automatic route from
    // an ambiguous shape after metadata or environment normalization.
    bool automatic_gva_packed_nw4 = false;
    bool automatic_gva_equal_n4_g16 = false;
};

using ContextParallelLauncher = void (*)(const ContextParallelLaunch&);

// Host callback ABI for the plain C-split segment output replay.  The common
// launcher owns route selection and workspace layout; an architecture policy
// may replace only the kernel launch without exposing private ISA headers to
// the common or other-architecture translation units.
struct CsplitSegmentOutputLaunch {
    dim3 grid;
    hipStream_t stream;
    const __bf16* u;
    const __bf16* sin;
    const __bf16* qd;
    const __bf16* kr;
    const float* gt;
    const __bf16* mqk;
    __bf16* out;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    const int* segment_prefix;
    int N;
    int total_tiles;
    int total_segments;
    int T_seq;
    int H;
    int NT;
    bool is_varlen;
};

using SegmentOutputLauncher = void (*)(const CsplitSegmentOutputLaunch&);

// Shared ABI for the optional compact-A producer and its K6 output consumer.
// The producer may enqueue work on an architecture-private auxiliary stream;
// the output callback must join that work onto `stream` before consuming
// segment_a.  The common launcher owns the arena and invokes both callbacks as
// one matched pair around P3.
struct Csplit64K6OutputLaunch {
    hipStream_t stream;
    const __bf16* qd;
    const __bf16* kr;
    const float* gt;
    const __bf16* cs_u;
    const __bf16* cs_sin;
    __bf16* segment_a;
    __bf16* out;
    const int32_t* cu_seqlens;
    const int* tile_prefix;
    const int* segment_prefix;
    int N;
    int total_tiles;
    int total_segments;
    int T_seq;
    int H;
    int NT;
    bool is_varlen;
};

using Csplit64K6OutputLauncher =
    void (*)(const Csplit64K6OutputLaunch&);

// Matched range pipeline for a recurrent scan and the established K6 output
// replay.  Returning false guarantees that no work was enqueued, allowing the
// common launcher to fall back to its original sequential P3/P4 path (for
// example during graph capture or on an unsupported shape).
using Csplit64K6PipelineLauncher = bool (*)(
    const Csplit64ScanLaunch&, const Csplit64K6OutputLaunch&);

struct HipLaunchPolicy {
    int default_k2_bv;
    K2DefaultRoute default_k2_route;
    bool use_default_rs_bv32;
    Csplit64ScanLauncher launch_k6_nw8;
    Csplit64ScanLauncher launch_k6_fused;
    Csplit64ScanLauncher launch_plain_default;
    Csplit64ScanLauncher launch_plain_nw8;
    SegmentOutputLauncher launch_segment_output;
    PlainCsplit64K1Launcher launch_plain_k1;
    VsplitRsLauncher launch_vsplit_rs;
    Bt16K1Launcher launch_bt16_k1;
    bool use_split_prep_dpp;
    bool use_split_prep_beta_cache;
    FusedPrepareNeumannLauncher launch_fused_prepare_neumann;
    Csplit64K6OutputLauncher launch_k6_aqk_producer;
    Csplit64K6OutputLauncher launch_k6_aqk_output;
    Csplit64K6PipelineLauncher launch_k6_pipeline;
    Csplit64ScanLauncher launch_k6_persistent = nullptr;
    PersistentPrefixLauncher launch_persistent_prefix = nullptr;
    ContextParallelLauncher launch_context_parallel = nullptr;
    // The architecture callback publishes a complete solved C16 workspace and
    // may replace common split preparation on the automatic plain C-split
    // route.  Explicit diagnostic routes retain the shared stage boundaries.
    bool use_bt16_k1_for_plain = false;
    // Host-side producer capability used to close the plain C-split beta-cache
    // handshake before any consumer is allowed to select the cache.
    bool bt16_k1_plain_beta_cache = false;
    // Separate strict opt-in capability for the PRE_SOLVED fused post-prep
    // suffix-decay publisher.  Common dispatch still closes the route/stage
    // handshake before setting the scan flag.
    bool plain_k1_suffix_decay_cache = false;
    // Architecture-private context routing is resolved while constructing the
    // policy, then consumed by both the common prefix builder and K2 callback.
    // Keep defaults at the aggregate tail so gfx942 initializers retain their
    // existing ABI and ordinary routes retain the standard segment prefix.
    int context_group_chunks = 0;
    int context_direct_max_chunks = 0;
    // Strict matched pair for the gfx950 packed-hybrid persistent candidate.
    // Keeping both fields at the aggregate tail preserves gfx942 aggregate
    // initialization and makes a zero/null policy an exact whole-graph
    // fallback rather than a prefix-only or K2-only experiment.
    PersistentPrefixLauncher launch_context_prefix = nullptr;
    int context_persistent_blocks = 0;
    // Strict whole-graph capability selected by gfx950 policy construction.
    // The common launcher still rechecks the packed pure-direct shape before
    // it removes k1_build_tile_prefix.
    bool context_direct_prefixless = false;
    // Strict whole-graph capability for the packed equal-length N=4/G64
    // dense-normalization candidate.  Appending this field preserves every
    // existing aggregate initializer, including gfx942's shorter prefix.
    bool context_equal_dense_n4_g64 = false;
    // The architecture BT16 callback may consume grouped-value q/k heads and
    // publish the ordinary per-value-head workspace.  Downstream routes still
    // need their own whole-graph capability; keep this false by default so a
    // raw-input producer that assumes H_q == H cannot be selected for GVA.
    bool bt16_k1_supports_gva = false;
    // Host-side producer capability for the context operand cache.  Common
    // dispatch combines it with the actually selected BT16 callback and
    // forwards the resulting per-launch fact to the context K2 consumer.
    bool bt16_k1_context_operand_cache = false;
    // Whole-graph capability for the automatic plain C-split route.  GVA may
    // select it only when the architecture supplies both the grouped-head
    // BT16 raw-input producer and the workspace-only plain post-preparation
    // callback.  Common dispatch still closes skip-stage/cache handshakes for
    // each launch before enabling the route.
    bool plain_csplit_supports_gva = false;
    // Exact automatic GVA context recipes resolved while the original packed
    // length bound and route environment are still available.
    bool context_automatic_gva_packed_nw4 = false;
    bool context_automatic_gva_equal_n4_g16 = false;
};

struct LaunchShape {
    bool is_varlen;
    int NT;
    int T_seq;
    int total_segments;
    int total_pairs;
    int64_t n_ht;
};

inline LaunchShape make_launch_shape(const FwdParams& p) {
    const bool is_varlen = p.cu_seqlens != nullptr;
    const int NT = p.total_tiles / p.N;
    const int T_seq = p.T_total / p.N;
    return {
        is_varlen,
        NT,
        T_seq,
        is_varlen ? (p.T_total + 63) / 64 + p.N
                  : p.N * ((NT + 3) / 4),
        is_varlen ? (p.T_total + 31) / 32 + p.N
                  : p.N * ((NT + 1) / 2),
        int64_t(p.H) * p.total_tiles,
    };
}

// Typed view over the flat workspace.  Both architecture implementations use
// this exact packing, which is also the ABI consumed by the K1/K2 kernels.
struct WorkspaceLayout {
    __bf16* kd;
    __bf16* qd;
    __bf16* kr;
    float* gt;
    __bf16* inv;
    __bf16* mqk;
    __bf16* cs_u;
    __bf16* cs_sin;
    __bf16* cs_cross_inv;
    __bf16* cs_cross64;
    float* cs_beta;
    __bf16* cs_segment_a;
    int* tile_prefix;
    int* pair_prefix;
    int* segment_prefix;
    int* sequence_worklist;
    int* sequence_count;
    unsigned int* task_counter;
};

inline WorkspaceLayout carve_workspace(
        void* workspace_ptr,
        const LaunchShape& shape,
        int H,
        int N) {
    char* base = reinterpret_cast<char*>(workspace_ptr);
    int64_t offset = 0;
    auto* kd = reinterpret_cast<__bf16*>(base + offset);
    offset += shape.n_ht * WorkspaceSizes::kKDecayed;
    auto* qd = reinterpret_cast<__bf16*>(base + offset);
    offset += shape.n_ht * WorkspaceSizes::kQDecayed;
    auto* kr = reinterpret_cast<__bf16*>(base + offset);
    offset += shape.n_ht * WorkspaceSizes::kKRestored;
    auto* gt = reinterpret_cast<float*>(base + offset);
    offset += shape.n_ht * WorkspaceSizes::kGTotal;
    auto* inv = reinterpret_cast<__bf16*>(base + offset);
    offset += shape.n_ht * WorkspaceSizes::kINV;
    auto* mqk = reinterpret_cast<__bf16*>(base + offset);

    const int64_t base_bytes = shape.n_ht * WorkspaceSizes::kPerTile;
    const int64_t prefix_bytes = WorkspaceSizes::prefix_bytes(N);
    auto* cs_u = reinterpret_cast<__bf16*>(base + base_bytes + prefix_bytes);
    auto* cs_sin = reinterpret_cast<__bf16*>(
        reinterpret_cast<char*>(cs_u) +
        shape.n_ht * WorkspaceSizes::kCsplitU);
    auto* cs_cross_inv = reinterpret_cast<__bf16*>(
        reinterpret_cast<char*>(cs_sin) + int64_t(H) *
        shape.total_segments * WorkspaceSizes::kCsplitSin);
    auto* cs_cross64 = reinterpret_cast<__bf16*>(
        reinterpret_cast<char*>(cs_cross_inv) + int64_t(H) *
        shape.total_pairs * WorkspaceSizes::kCsplitCross);
    auto* cs_beta = reinterpret_cast<float*>(
        reinterpret_cast<char*>(cs_cross64) + int64_t(H) *
        shape.total_segments * WorkspaceSizes::kCsplitCross64);
    auto* cs_segment_a = reinterpret_cast<__bf16*>(
        reinterpret_cast<char*>(cs_beta) + int64_t(H) *
        shape.total_segments * WorkspaceSizes::kCsplitBeta);

    int* tile_prefix = nullptr;
    int* pair_prefix = nullptr;
    int* segment_prefix = nullptr;
    int* sequence_worklist = nullptr;
    int* sequence_count = nullptr;
    unsigned int* task_counter = nullptr;
    if (shape.is_varlen) {
        tile_prefix = reinterpret_cast<int*>(
            base + shape.n_ht * WorkspaceSizes::kPerTile);
        pair_prefix = tile_prefix + (N + 1);
        segment_prefix = pair_prefix + (N + 1);
        sequence_worklist = segment_prefix + (N + 1);
        sequence_count = sequence_worklist + N;
        task_counter = reinterpret_cast<unsigned int*>(sequence_count + 1);
    }
    return {kd, qd, kr, gt, inv, mqk, cs_u, cs_sin, cs_cross_inv,
            cs_cross64, cs_beta, cs_segment_a,
            tile_prefix, pair_prefix, segment_prefix, sequence_worklist,
            sequence_count, task_counter};
}

// Centralize the seven valid state combinations and dense/varlen template
// dimension.  Callers provide a C++20 templated lambda with
// <HasIn, HasOut, StateFP32, IsVarlen>.
template <bool IsVarlen, typename Launcher>
inline void dispatch_state_mode(
        bool has_state_in,
        bool has_state_out,
        bool state_fp32,
        Launcher&& launch) {
    if (!has_state_in && !has_state_out)
        launch.template operator()<false, false, false, IsVarlen>();
    else if (has_state_in && has_state_out && state_fp32)
        launch.template operator()<true, true, true, IsVarlen>();
    else if (has_state_in && has_state_out)
        launch.template operator()<true, true, false, IsVarlen>();
    else if (!has_state_in && state_fp32)
        launch.template operator()<false, true, true, IsVarlen>();
    else if (!has_state_in)
        launch.template operator()<false, true, false, IsVarlen>();
    else if (state_fp32)
        launch.template operator()<true, false, true, IsVarlen>();
    else
        launch.template operator()<true, false, false, IsVarlen>();
}

template <typename Launcher>
inline void dispatch_state_mode(
        bool is_varlen,
        bool has_state_in,
        bool has_state_out,
        bool state_fp32,
        Launcher&& launch) {
    if (is_varlen)
        dispatch_state_mode<true>(has_state_in, has_state_out, state_fp32,
                                  static_cast<Launcher&&>(launch));
    else
        dispatch_state_mode<false>(has_state_in, has_state_out, state_fp32,
                                   static_cast<Launcher&&>(launch));
}

}  // namespace flashkda_hip
