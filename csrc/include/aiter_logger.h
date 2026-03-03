// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

namespace aiter {

    // Log levels matching Python's logging module
    static constexpr int LOG_DEBUG   = 10;
    static constexpr int LOG_INFO    = 20;
    static constexpr int LOG_WARNING = 30;
    static constexpr int LOG_ERROR   = 40;

    inline int get_log_level()
    {
        const char* level_str = std::getenv("AITER_LOG_LEVEL");
        if(level_str == nullptr)
            return LOG_INFO; // default matches Python
        std::string level(level_str);
        for(auto& c : level)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if(level == "DEBUG")
            return LOG_DEBUG;
        if(level == "INFO")
            return LOG_INFO;
        if(level == "WARNING")
            return LOG_WARNING;
        if(level == "ERROR")
            return LOG_ERROR;
        return LOG_INFO; // unknown level defaults to INFO
    }

    inline int current_log_level()
    {
        static const int level = get_log_level();
        return level;
    }

} // namespace aiter

// clang-format off
#define AITER_LOG_DEBUG(msg)   do { if(aiter::current_log_level() <= aiter::LOG_DEBUG)   { std::cout << "[aiter] " << msg << std::endl; } } while(0)
#define AITER_LOG_INFO(msg)    do { if(aiter::current_log_level() <= aiter::LOG_INFO)    { std::cout << "[aiter] " << msg << std::endl; } } while(0)
#define AITER_LOG_WARNING(msg) do { if(aiter::current_log_level() <= aiter::LOG_WARNING) { std::cerr << "[aiter WARNING] " << msg << std::endl; } } while(0)
#define AITER_LOG_ERROR(msg)   do { if(aiter::current_log_level() <= aiter::LOG_ERROR)   { std::cerr << "[aiter ERROR] " << msg << std::endl; } } while(0)
// clang-format on
