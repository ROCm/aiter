// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <stdexcept>

namespace aiter::f4gemm
{
constexpr int TILE_M        = 256;
constexpr int TILE_N        = 256;
constexpr int PERSISTENT_TG = 256;

struct LaunchGeometry
{
    int cluster_x; // N direction; supplied through the HIP cluster attribute / ttmp6
    int cluster_y; // M direction
    int grid_x;    // Cluster grid, not the HIP workgroup grid
    int grid_y;
};

constexpr bool is_power_of_two(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

inline unsigned int log2_power_of_two(int value)
{
    unsigned int result = 0;
    for(; value > 1; value >>= 1)
        ++result;
    return result;
}

inline LaunchGeometry make_launch_geometry(int cluster_x, int cluster_y, int grid_y)
{
    // 1x1 takes the unclustered path in AiterAsmKernel. This shader requires
    // ttmp6 and cluster barriers even when the problem contains only one tile.
    // Limit each axis to 8: a single axis of 16 hung the current gfx1250 stack
    // despite fitting the ttmp6 size field. The automatic path uses <= 4.
    if(!is_power_of_two(cluster_x) || !is_power_of_two(cluster_y) ||
       cluster_x > 8 || cluster_y > 8 || cluster_x * cluster_y < 2 ||
       cluster_x * cluster_y > 16)
        throw std::invalid_argument("f4gemm: cluster axes must be powers of two <= 8, "
                                    "with 2..16 TG per cluster");

    const int clusters = PERSISTENT_TG / (cluster_x * cluster_y);
    if(!is_power_of_two(grid_y) || grid_y > clusters)
        throw std::invalid_argument("f4gemm: grid_y must be a power-of-two divisor "
                                    "of the persistent cluster count");
    return {cluster_x, cluster_y, clusters / grid_y, grid_y};
}

inline LaunchGeometry select_launch_geometry(int M, int N)
{
    if(M < 0 || N < 0)
        throw std::invalid_argument("f4gemm: M and N must be nonnegative");
    const int tiles_m = M / TILE_M + (M % TILE_M != 0);
    const int tiles_n = N / TILE_N + (N % TILE_N != 0);

    // Preserve the registered 4x4 preference on aligned shapes. Shrink an
    // axis when its cluster would introduce whole phantom tiles. Do not grow
    // the other axis: larger multicast groups are not automatically faster.
    int cluster_x = 4;
    int cluster_y = 4;
    while(cluster_x > 1 && tiles_n % cluster_x != 0)
        cluster_x >>= 1;
    while(cluster_y > 1 && tiles_m % cluster_y != 0)
        cluster_y >>= 1;

    if(cluster_x * cluster_y == 1)
    {
        // Both tile counts are odd. A two-TG cluster is the smallest legal
        // launch; padding the longer axis wastes the fewest whole tiles.
        if(tiles_n >= tiles_m)
            cluster_x = 2;
        else
            cluster_y = 2;
    }

    const int cluster_rows = (tiles_m + cluster_y - 1) / cluster_y;
    int grid_y = 1;
    while(grid_y < 4 && grid_y < cluster_rows)
        grid_y <<= 1;
    return make_launch_geometry(cluster_x, cluster_y, grid_y);
}
} // namespace aiter::f4gemm
