// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

template <ck_tile::index_t kM, ck_tile::index_t kN, ck_tile::index_t MaxK>
struct GemmTileShapeConfig // 
{
    static constexpr ck_tile::index_t M_Tile = 128; 
    static constexpr ck_tile::index_t N_Tile = 128; 
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

   
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;
};

template <ck_tile::index_t kN, ck_tile::index_t MaxK>
struct GemmTileShapeConfig<128, kN, MaxK> //
{
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 32; 
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

  
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;
};



template <ck_tile::index_t kN, ck_tile::index_t MaxK>
struct GemmTileShapeConfig<1024, kN, MaxK> 
{
    static constexpr ck_tile::index_t M_Tile = 128; 
    static constexpr ck_tile::index_t N_Tile = 128; 
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

  
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;
};

template <ck_tile::index_t kN, ck_tile::index_t MaxK>
struct GemmTileShapeConfig<512, kN, MaxK> //
{
    static constexpr ck_tile::index_t M_Tile = 64; 
    static constexpr ck_tile::index_t N_Tile = 64; 
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;
};


