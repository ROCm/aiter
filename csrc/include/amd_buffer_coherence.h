// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>

namespace aiter {

// GPU 架构枚举
enum class GpuArch : int32_t
{
    GFX90A  = 0,  // MI200 系列
    GFX908  = 1,  // MI100 系列
    GFX940  = 2,  // MI300A (early)
    GFX941  = 3,  // MI300A
    GFX942  = 4,  // MI300A
    GFX950  = 5,  // MI300X
    GFX1100 = 6,  // RDNA3
    GFX12XX = 7,  // GFX12 系列
    UNKNOWN = -1
};

// Host 侧的 buffer coherence 枚举（独立于架构）
// 这些是抽象的语义化标识，会在 dispatch 时映射到具体架构的值
enum class BufferCoherenceType : int32_t
{
    // 通用类型
    DEFAULT = 0,       // 默认，期望 temporal reuse
    
    // Non-temporal 类型（按作用域）
    WAVE_NT    = 1,    // Wave/Wavefront 级别 non-temporal
    GROUP_NT   = 2,    // Workgroup/CU 级别 non-temporal
    DEVICE_NT  = 3,    // Device 级别 non-temporal
    SYSTEM_NT  = 4,    // System 级别 non-temporal
    
    // 组合类型
    GLC        = 5,    // Global Load Cache (device level)
    SLC        = 6,    // System Level Cache
    GLC_SLC    = 7,    // 两者组合，最大缓存穿透
    
    // 高级提示（主要用于 GFX12）
    HIGH_PRIORITY = 8, // 高优先级 temporal
    LAST_USE      = 9, // 最后一次使用（load op）
};

// =============================================================================
// 各架构的 buffer coherence 值定义（静态常量，用于映射）
// =============================================================================

// GFX90A, GFX908 (CDNA1/CDNA2) - MI100, MI200 系列
namespace gfx90a_coherence {
    constexpr int32_t DEFAULT   = 0;
    constexpr int32_t GLC       = 1;  // glc
    constexpr int32_t SLC       = 2;  // slc
    constexpr int32_t GLC_SLC   = 3;  // glc + slc
    
    // 映射到通用名称（为了兼容性）
    constexpr int32_t DEVICE_NT = GLC;
    constexpr int32_t SYSTEM_NT = SLC;
    constexpr int32_t WAVE_NT   = 0;
    constexpr int32_t GROUP_NT  = 0;
}

// GFX940, GFX941, GFX942, GFX950 (CDNA3) - MI300 系列
// bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
// SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system
// NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse
namespace gfx942_coherence {
    // Scope 基础值
    constexpr int32_t WAVE   = 0;
    constexpr int32_t GROUP  = 1;
    constexpr int32_t DEVICE = 16;
    constexpr int32_t SYSTEM = 17;
    
    // Temporal hints
    constexpr int32_t NT0 = 0;  // temporal reuse expected
    constexpr int32_t NT1 = 2;  // non-temporal
    
    // 组合值
    constexpr int32_t WAVE_NT0   = NT0 | WAVE;   // 0
    constexpr int32_t WAVE_NT1   = NT1 | WAVE;   // 2
    constexpr int32_t GROUP_NT0  = NT0 | GROUP;  // 1
    constexpr int32_t GROUP_NT1  = NT1 | GROUP;  // 3
    constexpr int32_t DEVICE_NT0 = NT0 | DEVICE; // 16
    constexpr int32_t DEVICE_NT1 = NT1 | DEVICE; // 18
    constexpr int32_t SYSTEM_NT0 = NT0 | SYSTEM; // 17
    constexpr int32_t SYSTEM_NT1 = NT1 | SYSTEM; // 19
    
    // 兼容性别名
    constexpr int32_t DEFAULT   = GROUP_NT0;
    constexpr int32_t WAVE_NT   = WAVE_NT1;
    constexpr int32_t GROUP_NT  = GROUP_NT1;
    constexpr int32_t DEVICE_NT = DEVICE_NT1;
    constexpr int32_t SYSTEM_NT = SYSTEM_NT1;
    constexpr int32_t GLC       = DEVICE_NT1;
    constexpr int32_t SLC       = SYSTEM_NT1;
    constexpr int32_t GLC_SLC   = DEVICE_NT1 | SYSTEM_NT1;
}

// GFX12XX (RDNA3+) - 未来架构
namespace gfx12_coherence {
    // Temporal hints
    constexpr int32_t RT    = 0;  // regular temporal
    constexpr int32_t NT    = 1;  // non temporal
    constexpr int32_t HT    = 2;  // high priority temporal
    constexpr int32_t LU    = 3;  // last use (load op)
    constexpr int32_t WB    = 3;  // write-back (store op)
    constexpr int32_t NT_RT = 4;
    constexpr int32_t RT_NT = 5;
    constexpr int32_t NT_HT = 6;
    constexpr int32_t NT_WB = 7;
    
    // Scope
    constexpr int32_t CU     = 0;
    constexpr int32_t SE     = 8;
    constexpr int32_t DEVICE = 16;
    constexpr int32_t SYSTEM = 24;
    
    // 常用组合
    constexpr int32_t CU_NT     = NT | CU;       // 1
    constexpr int32_t SE_NT     = NT | SE;       // 9
    constexpr int32_t DEVICE_NT = NT | DEVICE;   // 17
    constexpr int32_t SYSTEM_NT = NT | SYSTEM;   // 25
    constexpr int32_t CU_HT     = HT | CU;       // 2
    constexpr int32_t DEVICE_HT = HT | DEVICE;   // 18
    
    // 兼容性别名
    constexpr int32_t DEFAULT   = RT;
    constexpr int32_t WAVE_NT   = CU_NT;
    constexpr int32_t GROUP_NT  = CU_NT;
    constexpr int32_t GLC       = DEVICE_NT;
    constexpr int32_t SLC       = SYSTEM_NT;
    constexpr int32_t GLC_SLC   = DEVICE_NT | SYSTEM_NT;
}

} // namespace aiter

