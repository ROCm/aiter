// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "amd_buffer_coherence.h"
#include "ck_tile/core/arch/amd_buffer_coherence.hpp"
#include <cstdlib>
#include <string>

namespace aiter {

// =============================================================================
// BufferCoherenceMapper: 将 aiter 的抽象枚举映射到 ck_tile 的具体架构值
// =============================================================================

class BufferCoherenceMapper
{
public:
    // 将 aiter 的 BufferCoherenceType 映射到指定架构的实际值
    static constexpr int32_t map_to_arch_value(BufferCoherenceType type, GpuArch arch)
    {
        switch(arch)
        {
        case GpuArch::GFX90A:
        case GpuArch::GFX908:
            return map_to_gfx90a(type);
        
        case GpuArch::GFX940:
        case GpuArch::GFX941:
        case GpuArch::GFX942:
        case GpuArch::GFX950:
            return map_to_gfx942(type);
        
        case GpuArch::GFX1100:
        case GpuArch::GFX12XX:
            return map_to_gfx12(type);
        
        default:
            return map_to_gfx90a(type); // 默认使用最保守的映射
        }
    }

    // 直接返回 ck_tile 的枚举值（用于传递给 kernel）
    static constexpr ck_tile::amd_buffer_coherence_enum to_ck_tile_enum(
        BufferCoherenceType type, GpuArch arch)
    {
        return static_cast<ck_tile::amd_buffer_coherence_enum>(map_to_arch_value(type, arch));
    }

private:
    // GFX90A/GFX908 映射
    static constexpr int32_t map_to_gfx90a(BufferCoherenceType type)
    {
        switch(type)
        {
        case BufferCoherenceType::DEFAULT:       return gfx90a_coherence::DEFAULT;
        case BufferCoherenceType::WAVE_NT:       return gfx90a_coherence::WAVE_NT;
        case BufferCoherenceType::GROUP_NT:      return gfx90a_coherence::GROUP_NT;
        case BufferCoherenceType::DEVICE_NT:     return gfx90a_coherence::DEVICE_NT;
        case BufferCoherenceType::SYSTEM_NT:     return gfx90a_coherence::SYSTEM_NT;
        case BufferCoherenceType::GLC:           return gfx90a_coherence::GLC;
        case BufferCoherenceType::SLC:           return gfx90a_coherence::SLC;
        case BufferCoherenceType::GLC_SLC:       return gfx90a_coherence::GLC_SLC;
        case BufferCoherenceType::HIGH_PRIORITY: return gfx90a_coherence::DEFAULT;
        case BufferCoherenceType::LAST_USE:      return gfx90a_coherence::DEFAULT;
        default:                                 return gfx90a_coherence::DEFAULT;
        }
    }

    // GFX942/GFX950 (CDNA3) 映射
    static constexpr int32_t map_to_gfx942(BufferCoherenceType type)
    {
        switch(type)
        {
        case BufferCoherenceType::DEFAULT:       return gfx942_coherence::DEFAULT;
        case BufferCoherenceType::WAVE_NT:       return gfx942_coherence::WAVE_NT;
        case BufferCoherenceType::GROUP_NT:      return gfx942_coherence::GROUP_NT;
        case BufferCoherenceType::DEVICE_NT:     return gfx942_coherence::DEVICE_NT;
        case BufferCoherenceType::SYSTEM_NT:     return gfx942_coherence::SYSTEM_NT;
        case BufferCoherenceType::GLC:           return gfx942_coherence::GLC;
        case BufferCoherenceType::SLC:           return gfx942_coherence::SLC;
        case BufferCoherenceType::GLC_SLC:       return gfx942_coherence::GLC_SLC;
        case BufferCoherenceType::HIGH_PRIORITY: return gfx942_coherence::GROUP_NT0; // temporal
        case BufferCoherenceType::LAST_USE:      return gfx942_coherence::GROUP_NT;
        default:                                 return gfx942_coherence::DEFAULT;
        }
    }

    // GFX12 映射
    static constexpr int32_t map_to_gfx12(BufferCoherenceType type)
    {
        switch(type)
        {
        case BufferCoherenceType::DEFAULT:       return gfx12_coherence::DEFAULT;
        case BufferCoherenceType::WAVE_NT:       return gfx12_coherence::WAVE_NT;
        case BufferCoherenceType::GROUP_NT:      return gfx12_coherence::GROUP_NT;
        case BufferCoherenceType::DEVICE_NT:     return gfx12_coherence::DEVICE_NT;
        case BufferCoherenceType::SYSTEM_NT:     return gfx12_coherence::SYSTEM_NT;
        case BufferCoherenceType::GLC:           return gfx12_coherence::GLC;
        case BufferCoherenceType::SLC:           return gfx12_coherence::SLC;
        case BufferCoherenceType::GLC_SLC:       return gfx12_coherence::GLC_SLC;
        case BufferCoherenceType::HIGH_PRIORITY: return gfx12_coherence::CU_HT;
        case BufferCoherenceType::LAST_USE:      return gfx12_coherence::LU;
        default:                                 return gfx12_coherence::DEFAULT;
        }
    }
};

// =============================================================================
// BufferCoherenceDispatcher: Host 侧的高级 dispatch 接口
// =============================================================================

class BufferCoherenceDispatcher
{
public:
    // 从架构字符串解析 GpuArch 枚举
    static GpuArch parse_arch(const std::string& arch_str)
    {
        if(arch_str.find("gfx90a") != std::string::npos)
            return GpuArch::GFX90A;
        if(arch_str.find("gfx908") != std::string::npos)
            return GpuArch::GFX908;
        if(arch_str.find("gfx940") != std::string::npos)
            return GpuArch::GFX940;
        if(arch_str.find("gfx941") != std::string::npos)
            return GpuArch::GFX941;
        if(arch_str.find("gfx942") != std::string::npos)
            return GpuArch::GFX942;
        if(arch_str.find("gfx950") != std::string::npos)
            return GpuArch::GFX950;
        if(arch_str.find("gfx1100") != std::string::npos)
            return GpuArch::GFX1100;
        if(arch_str.find("gfx12") != std::string::npos)
            return GpuArch::GFX12XX;
        return GpuArch::UNKNOWN;
    }

    // 从环境变量获取架构
    static GpuArch get_arch_from_env()
    {
        // 优先使用 AITER_GPU_ARCH 环境变量
        const char* arch_env = std::getenv("AITER_GPU_ARCH");
        if(arch_env == nullptr)
        {
            // 回退到 GPU_ARCHS（可能包含多个架构，取第一个）
            arch_env = std::getenv("GPU_ARCHS");
        }
        
        if(arch_env != nullptr)
        {
            std::string arch_str(arch_env);
            // 如果有多个架构用分号分隔，取第一个
            size_t semicolon_pos = arch_str.find(';');
            if(semicolon_pos != std::string::npos)
            {
                arch_str = arch_str.substr(0, semicolon_pos);
            }
            return parse_arch(arch_str);
        }
        
        // 默认返回 GFX942 (最常用的 MI300 架构)
        return GpuArch::GFX942;
    }

    // 主要的 dispatch 函数：接收抽象的 coherence 类型（int32_t），自动获取架构，返回映射后的值
    // coherence_type: BufferCoherenceType 的整数表示
    // 返回：对应架构的 ck_tile coherence 值
    static int32_t dispatch(int32_t coherence_type)
    {
        // 从环境变量获取架构
        GpuArch arch = get_arch_from_env();
        
        // 将 int32_t 转换为 BufferCoherenceType
        BufferCoherenceType type = static_cast<BufferCoherenceType>(coherence_type);
        
        // 映射到该架构的具体值
        return BufferCoherenceMapper::map_to_arch_value(type, arch);
    }

    // 重载版本：直接接收枚举类型
    static int32_t dispatch(BufferCoherenceType type)
    {
        GpuArch arch = get_arch_from_env();
        return BufferCoherenceMapper::map_to_arch_value(type, arch);
    }

    // 重载版本：显式指定架构
    static int32_t dispatch(BufferCoherenceType type, GpuArch arch)
    {
        return BufferCoherenceMapper::map_to_arch_value(type, arch);
    }

    // 重载版本：使用架构字符串
    static int32_t dispatch(BufferCoherenceType type, const std::string& arch_str)
    {
        GpuArch arch = parse_arch(arch_str);
        return dispatch(type, arch);
    }

    // 获取 ck_tile 枚举（用于模板参数）
    static ck_tile::amd_buffer_coherence_enum get_ck_tile_enum(
        BufferCoherenceType type, GpuArch arch)
    {
        return BufferCoherenceMapper::to_ck_tile_enum(type, arch);
    }

    // 根据数据访问模式和数据大小智能选择 coherence
    // data_size_bytes: 数据大小（字节）
    // is_write: 是否为写操作
    // reuse_expected: 是否期望重用
    static BufferCoherenceType choose_optimal(
        size_t data_size_bytes, 
        bool is_write = false,
        bool reuse_expected = true)
    {
        if(reuse_expected)
        {
            return BufferCoherenceType::DEFAULT;
        }

        // 根据数据大小选择作用域
        if(data_size_bytes > (1ULL << 30))
        { // > 1GB
            return BufferCoherenceType::SYSTEM_NT;
        }
        else if(data_size_bytes > (1ULL << 26))
        { // > 64MB
            return BufferCoherenceType::DEVICE_NT;
        }
        else if(data_size_bytes > (1ULL << 22))
        { // > 4MB
            return BufferCoherenceType::GROUP_NT;
        }
        else
        {
            return BufferCoherenceType::WAVE_NT;
        }
    }

    // 获取架构描述信息
    static const char* get_arch_name(GpuArch arch)
    {
        switch(arch)
        {
        case GpuArch::GFX90A:  return "gfx90a (MI200)";
        case GpuArch::GFX908:  return "gfx908 (MI100)";
        case GpuArch::GFX940:  return "gfx940 (MI300A early)";
        case GpuArch::GFX941:  return "gfx941 (MI300A)";
        case GpuArch::GFX942:  return "gfx942 (MI300A)";
        case GpuArch::GFX950:  return "gfx950 (MI300X)";
        case GpuArch::GFX1100: return "gfx1100 (RDNA3)";
        case GpuArch::GFX12XX: return "gfx12xx (GFX12)";
        default:               return "Unknown";
        }
    }

    // 获取 coherence 类型描述
    static const char* get_coherence_name(BufferCoherenceType type)
    {
        switch(type)
        {
        case BufferCoherenceType::DEFAULT:       return "DEFAULT";
        case BufferCoherenceType::WAVE_NT:       return "WAVE_NT";
        case BufferCoherenceType::GROUP_NT:      return "GROUP_NT";
        case BufferCoherenceType::DEVICE_NT:     return "DEVICE_NT";
        case BufferCoherenceType::SYSTEM_NT:     return "SYSTEM_NT";
        case BufferCoherenceType::GLC:           return "GLC";
        case BufferCoherenceType::SLC:           return "SLC";
        case BufferCoherenceType::GLC_SLC:       return "GLC_SLC";
        case BufferCoherenceType::HIGH_PRIORITY: return "HIGH_PRIORITY";
        case BufferCoherenceType::LAST_USE:      return "LAST_USE";
        default:                                 return "Unknown";
        }
    }
};

// =============================================================================
// 便捷宏定义
// =============================================================================

// 将 aiter coherence type 和架构 dispatch 到具体值
// 使用示例:
// int32_t value = AITER_MAP_COHERENCE(BufferCoherenceType::DEVICE_NT, GpuArch::GFX942);
#define AITER_MAP_COHERENCE(type, arch) \
    aiter::BufferCoherenceMapper::map_to_arch_value(type, arch)

// 获取 ck_tile 枚举值
// 使用示例:
// auto ck_enum = AITER_TO_CK_TILE_ENUM(BufferCoherenceType::DEVICE_NT, GpuArch::GFX942);
#define AITER_TO_CK_TILE_ENUM(type, arch) \
    aiter::BufferCoherenceMapper::to_ck_tile_enum(type, arch)

// Dispatch 并执行代码块
// 使用示例:
// AITER_DISPATCH_COHERENCE(BufferCoherenceType::DEVICE_NT, arch, {
//     my_kernel<<<grid, block>>>(data, CK_COHERENCE_VALUE);
// })
#define AITER_DISPATCH_COHERENCE(type, arch, ...)                              \
    do                                                                         \
    {                                                                          \
        constexpr auto CK_COHERENCE_ENUM =                                     \
            aiter::BufferCoherenceMapper::to_ck_tile_enum(type, arch);         \
        constexpr int32_t CK_COHERENCE_VALUE =                                 \
            static_cast<int32_t>(CK_COHERENCE_ENUM);                           \
        __VA_ARGS__                                                            \
    } while(0)

// =============================================================================
// 全局便捷函数：最简单的使用方式
// =============================================================================

// 获取 buffer coherence 值（自动从环境变量获取架构）
// 使用示例：
//   int32_t coherence = aiter::get_buffer_coherence(BufferCoherenceType::DEVICE_NT);
// 或：
//   int32_t coherence = aiter::get_buffer_coherence(static_cast<int32_t>(BufferCoherenceType::DEVICE_NT));
inline int32_t get_buffer_coherence(int32_t coherence_type)
{
    return BufferCoherenceDispatcher::dispatch(coherence_type);
}

inline int32_t get_buffer_coherence(BufferCoherenceType type)
{
    return BufferCoherenceDispatcher::dispatch(type);
}

// 获取 ck_tile 枚举值（自动从环境变量获取架构）
inline ck_tile::amd_buffer_coherence_enum get_buffer_coherence_enum(int32_t coherence_type)
{
    GpuArch arch = BufferCoherenceDispatcher::get_arch_from_env();
    BufferCoherenceType type = static_cast<BufferCoherenceType>(coherence_type);
    return BufferCoherenceMapper::to_ck_tile_enum(type, arch);
}

inline ck_tile::amd_buffer_coherence_enum get_buffer_coherence_enum(BufferCoherenceType type)
{
    GpuArch arch = BufferCoherenceDispatcher::get_arch_from_env();
    return BufferCoherenceMapper::to_ck_tile_enum(type, arch);
}

} // namespace aiter

