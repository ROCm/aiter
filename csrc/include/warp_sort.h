// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"

namespace aiter {

template<typename T, int dpp_i, int row_mask=0xf, int bank_mask =0xf, bool bound_ctrl=true>
__device__ __inline__ T mov_dpp_(T x, ck_tile::number<dpp_i>,
                                      ck_tile::number<row_mask> = {},
                                      ck_tile::number<bank_mask> = {},
                                      ck_tile::bool_constant<bound_ctrl> = {}) {
    static_assert(sizeof(T) == 4);
    return __builtin_bit_cast(T,
                        // __builtin_amdgcn_update_dpp(0,__builtin_bit_cast(int, x),
                        __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x),
                                    dpp_i,
                                    row_mask,
                                    bank_mask,
                                    bound_ctrl));
}

template<typename T>
__device__ __inline__ T dev_max_(const T&a, const T&b)
{
    return a > b ? a : b;
}

template<>
__device__ __inline__ float dev_max_<float>(const float&a, const float&b)
{
    return __builtin_fmaxf(a, b);
}

template<typename T>
__device__ __inline__ T dev_min_(const T&a, const T&b)
{
    return a > b ? b : a;
}

template<>
__device__ __inline__ float dev_min_<float>(const float&a, const float&b)
{
    return __builtin_fminf(a, b);
}

template<typename V, int remote = ck_tile::vector_traits<ck_tile::remove_cvref_t<V>>::vector_size>
__device__ __inline__ auto mov_dpp_vec_from_(const V& v, ck_tile::number<remote> = {})
{
    using base_type = typename ck_tile::vector_traits<ck_tile::remove_cvref_t<V>>::scalar_type;
    constexpr int vector_size = ck_tile::vector_traits<ck_tile::remove_cvref_t<V>>::vector_size;
    static_assert(sizeof(base_type) == 4);

    V r;

    if constexpr (remote == 1){
        ck_tile::static_for<0, vector_size, 1>{}([&](auto i_){
            r[i_.value] = mov_dpp_(v[i_.value],  ck_tile::number<0xb1>{});  /*quad_perm:[1,0,3,2]*/
        });
    }
    else if constexpr (remote == 2){
        ck_tile::static_for<0, vector_size, 1>{}([&](auto i_){
            r[i_.value] = mov_dpp_(v[i_.value],  ck_tile::number<0x4e>{});  /*quad_perm:[2,3,0,1]*/
        });
    }
    else if constexpr (remote == 4){
        ck_tile::static_for<0, vector_size, 1>{}([&](auto i_){
            r[i_.value] = mov_dpp_(v[i_.value],  ck_tile::number<0x104>{}); /* row_shl:4 */
        });
    }
    else if constexpr (remote == 8){
        ck_tile::static_for<0, vector_size, 1>{}([&](auto i_){
            r[i_.value] = mov_dpp_(v[i_.value],  ck_tile::number<0x108>{}); /* row_shl:8 */
        });
    }
    else if constexpr (remote == 16 || remote == 32){
         int src_lane = __lane_id() ^ remote;
         ck_tile::static_for<0, vector_size, 1>{}([&](auto i_){
            auto local = v[i_.value];
            r[i_.value] = __builtin_bit_cast(base_type,
                         __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, local)));
         });
    }
    return r;
}

#define DPP_MERGE_2_CMP_(x_, y_)                        \
    using vec2_t = ck_tile::ext_vector_t<T, 2>;         \
    vec2_t res2;                                        \
    res2[0] = dev_max_(x_, y_);                         \
    res2[1] = dev_min_(x_, y_);

#define DPP_MERGE_2_DPP_()                                  \
    T res1_r = mov_dpp_(res1, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/

#define DPP_ARG_MERGE_2_CMP_(x_, y_, ax_, ay_)          \
    using vec2_t = ck_tile::ext_vector_t<T, 2>;         \
    using aec2_t = ck_tile::ext_vector_t<V, 2>;         \
    vec2_t res2;                                        \
    aec2_t arg2;                                        \
    res2[0] = x_ > y_? x_ : y_;                         \
    res2[1] = x_ > y_? y_ : x_;                         \
    arg2[0] = x_ > y_? ax_ : ay_;                       \
    arg2[1] = x_ > y_? ay_ : ax_;

#define DPP_ARG_MERGE_2_DPP_()                          \
    T res1_r = mov_dpp_(res1, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/ \
    V arg1_r = mov_dpp_(arg1, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/

#define DPP_MERGE_4_CMP_(x_, y_)                    \
    using vec4_t = ck_tile::ext_vector_t<T, 4>;     \
    vec4_t res4;                                    \
                                                    \
    res4[0] = dev_max_(x_[0],  y_[0]);              \
    T m_1 = dev_min_(x_[0],  y_[0]);                \
                                                    \
    T m_2 = dev_max_(x_[1],  y_[1]);                \
    res4[3] = dev_min_(x_[1],  y_[1]);              \
                                                    \
    res4[1] = dev_max_(m_1, m_2);                   \
    res4[2] = dev_min_(m_1, m_2);

#define DPP_MERGE_4_DPP_()                                  \
    vec2_t res2_r;                                          \
    res2_r[0] = mov_dpp_(res2[0],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    res2_r[1] = mov_dpp_(res2[1],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/

#define DPP_ARG_MERGE_4_CMP_(x_, y_, ax_, ay_)      \
    using vec4_t = ck_tile::ext_vector_t<T, 4>;     \
    using aec4_t = ck_tile::ext_vector_t<V, 4>;     \
    vec4_t res4;                                    \
    aec4_t arg4;                                    \
                                                    \
    res4[0] = x_[0] > y_[0] ? x_[0] : y_[0];        \
    T m_1   = x_[0] > y_[0] ? y_[0] : x_[0];        \
    arg4[0] = x_[0] > y_[0] ? ax_[0] : ay_[0];      \
    V am_1  = x_[0] > y_[0] ? ay_[0] : ax_[0];      \
                                                    \
    T m_2 = x_[1] > y_[1] ? x_[1] : y_[1];          \
    res4[3] = x_[1] > y_[1] ? y_[1] : x_[1];        \
    V am_2 = x_[1] > y_[1] ? ax_[1] : ay_[1];       \
    arg4[3] = x_[1] > y_[1] ? ay_[1] : ax_[1];      \
                                                    \
    res4[1] = m_1 > m_2 ? m_1 : m_2;                \
    res4[2] = m_1 > m_2 ? m_2 : m_1;                \
    arg4[1] = m_1 > m_2 ? am_1 : am_2;              \
    arg4[2] = m_1 > m_2 ? am_2 : am_1;

#define DPP_ARG_MERGE_4_DPP_()                              \
    vec2_t res2_r;                                          \
    aec2_t arg2_r;                                          \
    res2_r[0] = mov_dpp_(res2[0],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    res2_r[1] = mov_dpp_(res2[1],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    arg2_r[0] = mov_dpp_(arg2[0],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    arg2_r[1] = mov_dpp_(arg2[1],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/

#define DPP_MERGE_8_CMP_(x_, y_)                            \
        using vec8_t = ck_tile::ext_vector_t<T, 8>;     \
        vec8_t res8;                                    \
                                                        \
        res8[0]      = dev_max_(x_[0], y_[0]);      \
        T res8_4_tmp = dev_min_(x_[0], y_[0]);      \
                                                    \
        T res8_1_tmp = dev_max_(x_[1], y_[1]);      \
        T res8_5_tmp = dev_min_(x_[1], y_[1]);      \
                                                    \
        T res8_2_tmp = dev_max_(x_[2], y_[2]);      \
        T res8_6_tmp = dev_min_(x_[2], y_[2]);      \
                                                    \
        T res8_3_tmp = dev_max_(x_[3], y_[3]);      \
        res8[7]      = dev_min_(x_[3], y_[3]);      \
                                                            \
        T res8_2_tmp_r = dev_max_(res8_2_tmp, res8_4_tmp);  \
        T res8_4_tmp_r = dev_min_(res8_2_tmp, res8_4_tmp);  \
                                                            \
        T res8_3_tmp_r = dev_max_(res8_3_tmp, res8_5_tmp);  \
        T res8_5_tmp_r = dev_min_(res8_3_tmp, res8_5_tmp);  \
                                                            \
        res8[1] = dev_max_(res8_1_tmp, res8_2_tmp_r);       \
        res8[2] = dev_min_(res8_1_tmp, res8_2_tmp_r);       \
                                                            \
        res8[3] = dev_max_(res8_3_tmp_r, res8_4_tmp_r);     \
        res8[4] = dev_min_(res8_3_tmp_r, res8_4_tmp_r);     \
                                                            \
        res8[5] = dev_max_(res8_5_tmp_r, res8_6_tmp);       \
        res8[6] = dev_min_(res8_5_tmp_r, res8_6_tmp);

#define DPP_MERGE_8_DPP_()                              \
        vec4_t res4_r;                                  \
                                                        \
        /* only lane 0,1,2,3 contain valid data */      \
        res4_r[0] = mov_dpp_(res4[0],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[1] = mov_dpp_(res4[1],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[2] = mov_dpp_(res4[2],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[3] = mov_dpp_(res4[3],  ck_tile::number<0x104>{}); /* row_shl:4 */

#define DPP_ARG_MERGE_8_CMP_(x_, y_, ax_, ay_)              \
        using vec8_t = ck_tile::ext_vector_t<T, 8>;         \
        using aec8_t = ck_tile::ext_vector_t<V, 8>;         \
        vec8_t res8;                                        \
        aec8_t arg8;                                        \
                                                            \
        res8[0]      = x_[0] > y_[0] ? x_[0] : y_[0];       \
        T res8_4_tmp = x_[0] > y_[0] ? y_[0] : x_[0];       \
        arg8[0]      = x_[0] > y_[0] ? ax_[0] : ay_[0];     \
        V arg8_4_tmp = x_[0] > y_[0] ? ay_[0] : ax_[0];     \
                                                            \
        T res8_1_tmp = x_[1] > y_[1] ? x_[1] : y_[1];       \
        T res8_5_tmp = x_[1] > y_[1] ? y_[1] : x_[1];       \
        V arg8_1_tmp = x_[1] > y_[1] ? ax_[1] : ay_[1];     \
        V arg8_5_tmp = x_[1] > y_[1] ? ay_[1] : ax_[1];     \
                                                            \
        T res8_2_tmp = x_[2] > y_[2] ? x_[2] : y_[2];       \
        T res8_6_tmp = x_[2] > y_[2] ? y_[2] : x_[2];       \
        V arg8_2_tmp = x_[2] > y_[2] ? ax_[2] : ay_[2];     \
        V arg8_6_tmp = x_[2] > y_[2] ? ay_[2] : ax_[2];     \
                                                            \
        T res8_3_tmp = x_[3] > y_[3] ? x_[3] : y_[3];       \
        res8[7]      = x_[3] > y_[3] ? y_[3] : x_[3];       \
        V arg8_3_tmp = x_[3] > y_[3] ? ax_[3] : ay_[3];     \
        arg8[7]      = x_[3] > y_[3] ? ay_[3] : ax_[3];     \
                                                            \
        T res8_2_tmp_r = res8_2_tmp > res8_4_tmp ? res8_2_tmp :res8_4_tmp;  \
        T res8_4_tmp_r = res8_2_tmp > res8_4_tmp ? res8_4_tmp :res8_2_tmp;  \
        V arg8_2_tmp_r = res8_2_tmp > res8_4_tmp ? arg8_2_tmp :arg8_4_tmp;  \
        V arg8_4_tmp_r = res8_2_tmp > res8_4_tmp ? arg8_4_tmp :arg8_2_tmp;  \
                                                                            \
        T res8_3_tmp_r = res8_3_tmp > res8_5_tmp ? res8_3_tmp : res8_5_tmp; \
        T res8_5_tmp_r = res8_3_tmp > res8_5_tmp ? res8_5_tmp : res8_3_tmp; \
        V arg8_3_tmp_r = res8_3_tmp > res8_5_tmp ? arg8_3_tmp : arg8_5_tmp; \
        V arg8_5_tmp_r = res8_3_tmp > res8_5_tmp ? arg8_5_tmp : arg8_3_tmp; \
                                                                            \
        res8[1] = res8_1_tmp > res8_2_tmp_r ? res8_1_tmp : res8_2_tmp_r;  \
        res8[2] = res8_1_tmp > res8_2_tmp_r ? res8_2_tmp_r : res8_1_tmp;  \
        arg8[1] = res8_1_tmp > res8_2_tmp_r ? arg8_1_tmp : arg8_2_tmp_r;  \
        arg8[2] = res8_1_tmp > res8_2_tmp_r ? arg8_2_tmp_r : arg8_1_tmp;  \
                                                            \
        res8[3] = res8_3_tmp_r > res8_4_tmp_r ? res8_3_tmp_r : res8_4_tmp_r;  \
        res8[4] = res8_3_tmp_r > res8_4_tmp_r ? res8_4_tmp_r : res8_3_tmp_r;  \
        arg8[3] = res8_3_tmp_r > res8_4_tmp_r ? arg8_3_tmp_r : arg8_4_tmp_r;  \
        arg8[4] = res8_3_tmp_r > res8_4_tmp_r ? arg8_4_tmp_r : arg8_3_tmp_r;  \
                                                            \
        res8[5] = res8_5_tmp_r > res8_6_tmp ? res8_5_tmp_r: res8_6_tmp;    \
        res8[6] = res8_5_tmp_r > res8_6_tmp ? res8_6_tmp: res8_5_tmp_r;    \
        arg8[5] = res8_5_tmp_r > res8_6_tmp ? arg8_5_tmp_r: arg8_6_tmp;    \
        arg8[6] = res8_5_tmp_r > res8_6_tmp ? arg8_6_tmp: arg8_5_tmp_r; 

#define DPP_ARG_MERGE_8_DPP_()                          \
        vec4_t res4_r;                                  \
        aec4_t arg4_r;                                  \
                                                        \
        /* only lane 0,1,2,3 contain valid data */      \
        res4_r[0] = mov_dpp_(res4[0],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[1] = mov_dpp_(res4[1],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[2] = mov_dpp_(res4[2],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[3] = mov_dpp_(res4[3],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[0] = mov_dpp_(arg4[0],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[1] = mov_dpp_(arg4[1],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[2] = mov_dpp_(arg4[2],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[3] = mov_dpp_(arg4[3],  ck_tile::number<0x104>{}); /* row_shl:4 */

#define DPP_MERGE_16_CMP_(x_, y_)                               \
        using vec16_t = ck_tile::ext_vector_t<T, 16>;       \
        vec16_t res16;                                      \
                                                        \
        res16[0]      = dev_max_(x_[0], y_[0]);         \
        T res16_8_tmp = dev_min_(x_[0], y_[0]);         \
                                                        \
        T res16_1_tmp = dev_max_(x_[1], y_[1]);         \
        T res16_9_tmp = dev_min_(x_[1], y_[1]);         \
                                                        \
        T res16_2_tmp  = dev_max_(x_[2], y_[2]);        \
        T res16_10_tmp = dev_min_(x_[2], y_[2]);        \
                                                        \
        T res16_3_tmp  = dev_max_(x_[3], y_[3]);        \
        T res16_11_tmp = dev_min_(x_[3], y_[3]);        \
                                                        \
        T res16_4_tmp  = dev_max_(x_[4], y_[4]);        \
        T res16_12_tmp = dev_min_(x_[4], y_[4]);        \
                                                        \
        T res16_5_tmp  = dev_max_(x_[5], y_[5]);        \
        T res16_13_tmp = dev_min_(x_[5], y_[5]);        \
                                                        \
        T res16_6_tmp  = dev_max_(x_[6], y_[6]);        \
        T res16_14_tmp = dev_min_(x_[6], y_[6]);        \
                                                        \
        T res16_7_tmp  = dev_max_(x_[7], y_[7]);        \
        res16[15]      = dev_min_(x_[7], y_[7]);        \
                                                            \
                                                            \
        T res16_4_tmp_x = dev_max_(res16_4_tmp, res16_8_tmp);       \
        T res16_8_tmp_x = dev_min_(res16_4_tmp, res16_8_tmp);       \
                                                                    \
        T res16_5_tmp_x = dev_max_(res16_5_tmp, res16_9_tmp);       \
        T res16_9_tmp_x = dev_min_(res16_5_tmp, res16_9_tmp);       \
                                                                    \
        T res16_6_tmp_x  = dev_max_(res16_6_tmp, res16_10_tmp);     \
        T res16_10_tmp_x = dev_min_(res16_6_tmp, res16_10_tmp);     \
                                                                    \
        T res16_7_tmp_x  = dev_max_(res16_7_tmp, res16_11_tmp);     \
        T res16_11_tmp_x = dev_min_(res16_7_tmp, res16_11_tmp);     \
                                                                    \
                                                                    \
        T res16_2_tmp_x  = dev_max_(res16_2_tmp, res16_4_tmp_x);    \
        T res16_4_tmp_xx = dev_min_(res16_2_tmp, res16_4_tmp_x);    \
                                                                    \
        T res16_3_tmp_x  = dev_max_(res16_3_tmp, res16_5_tmp_x);    \
        T res16_5_tmp_xx = dev_min_(res16_3_tmp, res16_5_tmp_x);    \
                                                                    \
        T res16_6_tmp_xx = dev_max_(res16_6_tmp_x, res16_8_tmp_x);  \
        T res16_8_tmp_xx = dev_min_(res16_6_tmp_x, res16_8_tmp_x);  \
                                                                    \
        T res16_7_tmp_xx = dev_max_(res16_7_tmp_x, res16_9_tmp_x);  \
        T res16_9_tmp_xx = dev_min_(res16_7_tmp_x, res16_9_tmp_x);  \
                                                                    \
        T res16_10_tmp_xx = dev_max_(res16_10_tmp_x, res16_12_tmp);  \
        T res16_12_tmp_xx = dev_min_(res16_10_tmp_x, res16_12_tmp);  \
                                                                     \
        T res16_11_tmp_xx = dev_max_(res16_11_tmp_x, res16_13_tmp);  \
        T res16_13_tmp_xx = dev_min_(res16_11_tmp_x, res16_13_tmp);  \
                                                                    \
        res16[1]  = dev_max_(res16_1_tmp, res16_2_tmp_x);           \
        res16[2]  = dev_min_(res16_1_tmp, res16_2_tmp_x);           \
                                                                    \
        res16[3]  = dev_max_(res16_3_tmp_x, res16_4_tmp_xx);        \
        res16[4]  = dev_min_(res16_3_tmp_x, res16_4_tmp_xx);        \
                                                                    \
        res16[5]  = dev_max_(res16_5_tmp_xx, res16_6_tmp_xx);       \
        res16[6]  = dev_min_(res16_5_tmp_xx, res16_6_tmp_xx);       \
                                                                    \
        res16[7]  = dev_max_(res16_7_tmp_xx, res16_8_tmp_xx);       \
        res16[8]  = dev_min_(res16_7_tmp_xx, res16_8_tmp_xx);       \
                                                                    \
        res16[9]  = dev_max_(res16_9_tmp_xx, res16_10_tmp_xx);      \
        res16[10] = dev_min_(res16_9_tmp_xx, res16_10_tmp_xx);      \
                                                                    \
        res16[11] = dev_max_(res16_11_tmp_xx, res16_12_tmp_xx);     \
        res16[12] = dev_min_(res16_11_tmp_xx, res16_12_tmp_xx);     \
                                                                    \
        res16[13] = dev_max_(res16_13_tmp_xx, res16_14_tmp);        \
        res16[14] = dev_min_(res16_13_tmp_xx, res16_14_tmp);

#define DPP_MERGE_16_DPP_()                                         \
        vec8_t res8_r;                                          \
        /* only lane 0,1,2,3 contain valid data */              \
        res8_r[0] = mov_dpp_(res8[0],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[1] = mov_dpp_(res8[1],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[2] = mov_dpp_(res8[2],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[3] = mov_dpp_(res8[3],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[4] = mov_dpp_(res8[4],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[5] = mov_dpp_(res8[5],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[6] = mov_dpp_(res8[6],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[7] = mov_dpp_(res8[7],  ck_tile::number<0x108>{}); /* row_shl:8 */

// https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
// TODO: this is assuming descending order sort
// result store to smem :)
template <typename T, int lanegroup_size = ck_tile::get_warp_size()>
__device__ __inline__ void warp_merge_sort_to_smem(T* smem, const T& x, ck_tile::number<lanegroup_size> = {})
{
    static_assert(sizeof(T) == 4);
    int lane_id = threadIdx.x % lanegroup_size;
    int group_id = threadIdx.x / lanegroup_size;
    T res1 = x;

    if constexpr (lanegroup_size == 2) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);

        if(lane_id == 0) {
            reinterpret_cast<vec2_t*>(smem)[group_id] = res2;
        }
    } else if constexpr (lanegroup_size == 4) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2, res2_r);

        if(lane_id == 0) {
            reinterpret_cast<vec4_t*>(smem)[group_id] = res4;
        }
    } else if constexpr (lanegroup_size == 8) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2, res2_r);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4, res4_r);

        if(lane_id == 0) {
            union {
                struct  {
                    vec4_t x;
                    vec4_t y;
                };
                vec8_t value;
            } _tmp;
            _tmp.value = res8;
            reinterpret_cast<vec4_t*>(smem)[group_id * 2] = _tmp.x;
            reinterpret_cast<vec4_t*>(smem)[group_id * 2 + 1] = _tmp.y;
        }
    } else if constexpr (lanegroup_size == 16) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2, res2_r);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4, res4_r);
        DPP_MERGE_16_DPP_();
        DPP_MERGE_16_CMP_(res8, res8_r);

        if(lane_id == 0) {
#if 0
            union {
                struct {
                    vec4_t x;
                    vec4_t y;
                    vec4_t z;
                    vec4_t w;
                };
                vec16_t value;
            } _tmp;
            _tmp.value = res16;
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 0] = _tmp.x;
            __syncthreads();
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 1] = _tmp.y;
            __syncthreads();
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 2] = _tmp.z;
            __syncthreads();
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 3] = _tmp.w;
#else
            reinterpret_cast<vec16_t*>(smem)[group_id] = res16;
#endif
        }
    }
}

template <typename T, int lanegroup_size = ck_tile::get_warp_size()>
__device__ __inline__ auto warp_merge_sort_to_reg(const T& x, ck_tile::number<lanegroup_size> = {})
{
    static_assert(sizeof(T) == 4);
    T res1 = x;

    if constexpr (lanegroup_size == 2) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        return res2;
    } else if constexpr (lanegroup_size == 4) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2, res2_r);
        return res4;
    } else if constexpr (lanegroup_size == 8) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2, res2_r);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4, res4_r);
        // TODO: only lane:1,2,3,4 within 8 lanes does not have correct result !
        return res8;
    } else if constexpr (lanegroup_size == 16) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2, res2_r);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4, res4_r);
        DPP_MERGE_16_DPP_();
        DPP_MERGE_16_CMP_(res8, res8_r);
        // TODO: only lane:1,2,3,4 within 16 lanes does not have correct result !
        return res16;
    } else {
        return 0;
    }
}

// sort based on x, and sort v
template <typename T, typename V, int lanegroup_size = ck_tile::get_warp_size()>
__device__ __inline__ auto warp_arg_merge_sort_to_reg(const T& x, const V& v, ck_tile::number<lanegroup_size> = {})
{
    static_assert(sizeof(T) == 4);
    T res1 = x;
    V arg1 = v;

    if constexpr (lanegroup_size == 2) {
        DPP_ARG_MERGE_2_DPP_();
        DPP_ARG_MERGE_2_CMP_(res1, res1_r, arg1, arg1_r);
        return ck_tile::make_tuple(res2, arg2);
    } else if constexpr (lanegroup_size == 4) {
        DPP_ARG_MERGE_2_DPP_();
        DPP_ARG_MERGE_2_CMP_(res1, res1_r, arg1, arg1_r);
        DPP_ARG_MERGE_4_DPP_();
        DPP_ARG_MERGE_4_CMP_(res2, res2_r, arg2, arg2_r);
        return ck_tile::make_tuple(res4, arg4);
    } else if constexpr (lanegroup_size == 8) {
        DPP_ARG_MERGE_2_DPP_();
        DPP_ARG_MERGE_2_CMP_(res1, res1_r, arg1, arg1_r);
        DPP_ARG_MERGE_4_DPP_();
        DPP_ARG_MERGE_4_CMP_(res2, res2_r, arg2, arg2_r);
        DPP_ARG_MERGE_8_DPP_();
        DPP_ARG_MERGE_8_CMP_(res4, res4_r, arg4, arg4_r);
        // TODO: only lane:1,2,3,4 within 8 lanes does not have correct result !
        return ck_tile::make_tuple(res8, arg8);
    }
#if 0
    else if constexpr (lanegroup_size == 16) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2, res2_r);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4, res4_r);
        DPP_MERGE_16_DPP_();
        DPP_MERGE_16_CMP_(res8, res8_r);
        // TODO: only lane:1,2,3,4 within 16 lanes does not have correct result !
        return res16;
    } else {
        return 0;
    }
#endif
}

// combine 2 register and sort together, the other register buffer is from current lane
template <typename T_vec, int lanegroup_size = ck_tile::get_warp_size()>
__device__ __inline__ auto warp_merge_sort_combine2(const T_vec& x, const T_vec& y, ck_tile::number<lanegroup_size> = {})
{
    using T = typename ck_tile::vector_traits<ck_tile::remove_cvref_t<T_vec>>::scalar_type;
    static_assert(sizeof(T) == 4);

    if constexpr (lanegroup_size == 2) {
        DPP_MERGE_2_CMP_(x, y);
        return res2;
    } else if constexpr (lanegroup_size == 4) {
        DPP_MERGE_4_CMP_(x, y);
        return res4;
    } else if constexpr (lanegroup_size == 8) {
        DPP_MERGE_8_CMP_(x, y);
        // TODO: only lane:1,2,3,4 within 8 lanes does not have correct result !
        return res8;
    } else if constexpr (lanegroup_size == 16) {
        DPP_MERGE_16_CMP_(x, y);
        // TODO: only lane:1,2,3,4 within 16 lanes does not have correct result !
        return res16;
    } else {
        return 0;
    }
}

#undef DPP_MERGE_2_DPP_
#undef DPP_MERGE_2_CMP_
#undef DPP_MERGE_4_DPP_
#undef DPP_MERGE_4_CMP_
#undef DPP_MERGE_8_DPP_
#undef DPP_MERGE_8_CMP_
#undef DPP_MERGE_16_DPP_
#undef DPP_MERGE_16_CMP_
#undef DPP_ARG_MERGE_2_DPP_
#undef DPP_ARG_MERGE_2_CMP_
#undef DPP_ARG_MERGE_4_DPP_
#undef DPP_ARG_MERGE_4_CMP_
#undef DPP_ARG_MERGE_8_DPP_
#undef DPP_ARG_MERGE_8_CMP_

// [a, b, c, d....] -> [a, a+b, a+b+c, a+b+c+d, ....]
// NOTE: wave_size need at least be 16!! dpp 16 is one row
template <typename data_t, int warp_size = ck_tile::get_warp_size()>
__device__ inline void warp_cumsum(data_t& thread_data, ck_tile::number<warp_size> = {})
{
    // warp_size must be power of 2
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = true;   // ! out-of-bound is zero !
    auto reduce_op = [&](auto x_, auto y_) { return x_ + y_; };

    if constexpr(warp_size > 1)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x111,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl))); // row_shr:1
    }

    if constexpr(warp_size > 2)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x112,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl))); // row_shr:2
    }
    if constexpr(warp_size > 4)
    {
        thread_data =
            reduce_op(thread_data,
                    __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                    0x114,
                                                                    row_mask,
                                                                    bank_mask,
                                                                    bound_ctrl))); // row_shr:4
    }
    if constexpr(warp_size == 8) {
        
        // wave-size=8 need one extra shift
        thread_data =
            reduce_op(thread_data,
                    __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                    0x118,
                                                                    row_mask,
                                                                    bank_mask,
                                                                    bound_ctrl))); // row_shr:8
#if 0
        constexpr int bank_mask_0_7 = 0b1100;
        auto reduce_op_r = [&](auto x_, auto y_) { return x_ - y_; };
        thread_data = reduce_op_r(thread_data, __builtin_bit_cast(data_t,
                                                __builtin_amdgcn_update_dpp(0, /* old value */
                                                    __builtin_bit_cast(int, thread_data),
                                                    0x157,
                                                    row_mask,
                                                    bank_mask_0_7,
                                                    bound_ctrl))// row_newbcast:7
                                                    );
#else
        data_t xxx =__builtin_bit_cast(data_t, 
                        __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                    0x157,
                                                    row_mask,
                                                    bank_mask,
                                                    bound_ctrl)); // row_newbcast:7
        
        data_t yyy = (__lane_id() / 8) % 2 == 0 ? 0 : xxx;
        thread_data = thread_data - yyy;
#endif
        
    }
    if constexpr(warp_size > 8)
    {
        thread_data =
            reduce_op(thread_data,
                    __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                    0x118,
                                                                    row_mask,
                                                                    bank_mask,
                                                                    bound_ctrl))); // row_shr:8
    }

    if constexpr(warp_size > 16)
    {
        // now row-0, row-0+row-1, row-1+row-2, row-2+row-3
        int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 1) << 2, __builtin_bit_cast(int, thread_data));
        v_remote_tmp = __lane_id() >= 16 ? v_remote_tmp : 0;
        thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
    }

    if constexpr(warp_size > 32)
    {
        // lane-id 48...63->31
        int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 17) << 2, __builtin_bit_cast(int, thread_data));
        v_remote_tmp = __lane_id() >= 32 ? v_remote_tmp : 0;
        thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
    }
}
} // namespace aiter
