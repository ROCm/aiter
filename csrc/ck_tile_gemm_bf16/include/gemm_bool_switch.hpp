// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#define BOOL_SWITCH(COND1, CONST_NAME1, ...)    \
    [&] {                                       \
        if(COND1)                               \
        {                                       \
            constexpr bool CONST_NAME1 = true;  \
            __VA_ARGS__();                      \
        }                                       \
        else                                    \
        {                                       \
            constexpr bool CONST_NAME1 = false; \
            __VA_ARGS__();                      \
        }                                       \
    }()

#define BOOL_SWITCH_2(COND1, CONST_NAME1, COND2, CONST_NAME2, ...) \
    [&] {                                                          \
        if(COND1)                                                  \
        {                                                          \
            constexpr bool CONST_NAME1 = true;                     \
            BOOL_SWITCH(COND2, CONST_NAME2, ##__VA_ARGS__);        \
        }                                                          \
        else                                                       \
        {                                                          \
            constexpr bool CONST_NAME1 = false;                    \
            BOOL_SWITCH(COND2, CONST_NAME2, ##__VA_ARGS__);        \
        }                                                          \
    }()

#define BOOL_SWITCH_3(COND1, CONST_NAME1, COND2, CONST_NAME2, COND3, CONST_NAME3, ...) \
    [&] {                                                                              \
        if(COND1)                                                                      \
        {                                                                              \
            constexpr bool CONST_NAME1 = true;                                         \
            BOOL_SWITCH_2(COND2, CONST_NAME2, COND3, CONST_NAME3, ##__VA_ARGS__);      \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            constexpr bool CONST_NAME1 = false;                                        \
            BOOL_SWITCH_2(COND2, CONST_NAME2, COND3, CONST_NAME3, ##__VA_ARGS__);      \
        }                                                                              \
    }()

#define BOOL_SWITCH_4(                                                                      \
    COND1, CONST_NAME1, COND2, CONST_NAME2, COND3, CONST_NAME3, COND4, CONST_NAME4, ...)    \
    [&] {                                                                                   \
        if(COND1)                                                                           \
        {                                                                                   \
            constexpr bool CONST_NAME1 = true;                                              \
            BOOL_SWITCH_3(                                                                  \
                COND2, CONST_NAME2, COND3, CONST_NAME3, COND4, CONST_NAME4, ##__VA_ARGS__); \
        }                                                                                   \
        else                                                                                \
        {                                                                                   \
            constexpr bool CONST_NAME1 = false;                                             \
            BOOL_SWITCH_3(                                                                  \
                COND2, CONST_NAME2, COND3, CONST_NAME3, COND4, CONST_NAME4, ##__VA_ARGS__); \
        }                                                                                   \
    }()

#define KK_SWITCH(KK_SZ, CONST_NAME, ...)                              \
    [&] {                                                              \
        if(KK_SZ <= 512)                                               \
        {                                                              \
            constexpr ck_tile::index_t CONST_NAME = 512;               \
            __VA_ARGS__();                                             \
        }                                                              \
        else if(KK_SZ <= 2048)                                         \
        {                                                              \
            constexpr ck_tile::index_t CONST_NAME = 2048;              \
            __VA_ARGS__();                                             \
        }                                                              \
        else                                                           \
        {                                                              \
            constexpr ck_tile::index_t CONST_NAME = 7168;              \
            __VA_ARGS__();                                             \
        }                                                              \
    }()

#define KN_KK_SWITCH(KN_SZ, CONST_NAME1, KK_SZ, CONST_NAME2, ...)            \
    [&] {                                                                    \
        if(KN_SZ <= 7168)                                                    \
        {                                                                    \
            constexpr ck_tile::index_t CONST_NAME1 = 7168;                   \
            KK_SWITCH(KK_SZ, CONST_NAME2, ##__VA_ARGS__);                    \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            constexpr ck_tile::index_t CONST_NAME1 = 4096;                   \
            KK_SWITCH(KK_SZ, CONST_NAME2, ##__VA_ARGS__);                    \
        }                                                                    \
    }()

#define KM_KN_KK_SWITCH(                                                                    \
    KM_SZ, CONST_NAME1, KN_SZ, CONST_NAME2, KK_SZ, CONST_NAME3, ...)                        \
    [&] {                                                                                   \
        if(KM_SZ <= 128)                                                                     \
        {                                                                                   \
            constexpr ck_tile::index_t CONST_NAME1 = 128;                                    \
            KN_KK_SWITCH(KN_SZ, CONST_NAME2, KK_SZ, CONST_NAME3, ##__VA_ARGS__);            \
        }                                                                                   \
        else if(KM_SZ <= 512)                                                               \
        {                                                                                   \
            constexpr ck_tile::index_t CONST_NAME1 = 512;                                   \
            KN_KK_SWITCH(KN_SZ, CONST_NAME2, KK_SZ, CONST_NAME3, ##__VA_ARGS__);            \
        }                                                                                   \
        else                                                                                \
        {                                                                                   \
            constexpr ck_tile::index_t CONST_NAME1 = 1024;                                   \
            KN_KK_SWITCH(KN_SZ, CONST_NAME2, KK_SZ, CONST_NAME3, ##__VA_ARGS__);            \
        }                                                                                   \
    }()
