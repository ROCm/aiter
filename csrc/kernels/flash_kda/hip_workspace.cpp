#include "flash_kda.h"

namespace flashkda_hip {

int64_t get_workspace_size_hip(int64_t T_total, int64_t H, int64_t N) {
    constexpr int CHUNK = WorkspaceSizes::CHUNK;
    // Upper bounds include one partial tile/segment per sequence.  Preserve
    // the original over-allocation contract used by dense and packed callers.
    const int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;
    const int64_t total_segments = (T_total + 63) / 64 + N;
    const int64_t total_pairs = (T_total + 31) / 32 + N;
    const int64_t prefix_bytes = WorkspaceSizes::prefix_bytes(N);
    return H * total_tiles * WorkspaceSizes::kPerTile + prefix_bytes +
           H * total_tiles * WorkspaceSizes::kCsplitU +
           H * total_segments * WorkspaceSizes::kCsplitSin +
           H * total_pairs * WorkspaceSizes::kCsplitCross +
           H * total_segments * WorkspaceSizes::kCsplitCross64 +
           H * total_segments * WorkspaceSizes::kCsplitBeta +
           H * total_segments * WorkspaceSizes::kCsplitSegmentA;
}

}  // namespace flashkda_hip
