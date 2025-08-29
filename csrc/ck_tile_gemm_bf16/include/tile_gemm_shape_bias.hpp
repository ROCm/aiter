// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename BlockTile_,
          typename BlockWarps_,
          typename WarpTile_,
          bool PermuteA_ = false,
          bool PermuteB_ = false>
struct TileGemmShape_bias
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpTile   = remove_cvref_t<WarpTile_>;

    static constexpr index_t NumWarps = reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});

    static constexpr index_t kM = BlockTile::at(number<0>{});
    static constexpr index_t kN = BlockTile::at(number<1>{});
    static constexpr index_t kK = BlockTile::at(number<2>{});

    static constexpr index_t w_kM = BlockWarps::at(number<0>{});
    static constexpr index_t w_kN = BlockWarps::at(number<1>{});
    static constexpr index_t w_kK = BlockWarps::at(number<2>{});

    static constexpr bool PermuteA = PermuteA_;
    static constexpr bool PermuteB = PermuteB_;

    static constexpr index_t flatNPerWarp  = BlockWarps::at(number<1>{});
    static constexpr index_t flatKPerWarp  = WarpTile::at(number<2>{}) * WarpTile::at(number<1>{});
    static constexpr index_t flatKPerBlock = flatKPerWarp * kK / WarpTile::at(number<2>{});

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "tile_gemm_shape",
                      concat('x', kM, kN, kK, NumWarps),
                      concat('x', BlockWarps::at(number<0>{}), BlockWarps::at(number<1>{}), BlockWarps::at(number<2>{})),
                      concat('x', (WarpTile::at(number<0>{})), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})));
        // clang-format on
    }
};

} // namespace ck_tile
