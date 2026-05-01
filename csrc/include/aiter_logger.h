// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

    // Get AITER_LOG_MORE value
    inline int get_log_more()
    {
        const char* more_str = std::getenv("AITER_LOG_MORE");
        if(more_str == nullptr)
            return 0;
        return std::atoi(more_str);
    }

    inline int current_log_more()
    {
        static const int more = get_log_more();
        return more;
    }

    // Parse AITER_LOG_MODULE env var to get list of enabled modules
    inline std::vector<std::string> get_log_modules()
    {
        std::vector<std::string> modules;
        const char* modules_str = std::getenv("AITER_LOG_MODULE");
        if(modules_str == nullptr)
            return modules; // empty means all modules
        std::string modules_env(modules_str);
        std::istringstream iss(modules_env);
        std::string module;
        while(std::getline(iss, module, ',')) {
            // Trim whitespace
            size_t start = module.find_first_not_of(" \t");
            size_t end = module.find_last_not_of(" \t");
            if(start != std::string::npos) {
                modules.push_back(module.substr(start, end - start + 1));
            }
        }
        return modules;
    }

    inline const std::vector<std::string>& current_log_modules()
    {
        static const std::vector<std::string> modules = get_log_modules();
        return modules;
    }

    // Check if a module should log (when AITER_LOG_MORE > 0 and AITER_LOG_MODULE is set)
    inline bool should_log_module(const char* module_name)
    {
        // If AITER_LOG_MORE <= 0, no module filtering
        if(current_log_more() <= 0)
            return true;

        const auto& modules = current_log_modules();
        // If no modules specified, allow all
        if(modules.empty())
            return true;

        // Check if module_name is in the list
        std::string mod(module_name);
        for(const auto& m : modules) {
            if(m == mod)
                return true;
        }
        return false;
    }

} // namespace aiter

// clang-format off
#define AITER_LOG_DEBUG(msg)   do { if(aiter::current_log_level() <= aiter::LOG_DEBUG)   { std::cout << "[aiter] " << msg << std::endl; } } while(0)
#define AITER_LOG_INFO(msg)    do { if(aiter::current_log_level() <= aiter::LOG_INFO)    { std::cout << "[aiter] " << msg << std::endl; } } while(0)
#define AITER_LOG_WARNING(msg) do { if(aiter::current_log_level() <= aiter::LOG_WARNING) { std::cerr << "[aiter WARNING] " << msg << std::endl; } } while(0)
#define AITER_LOG_ERROR(msg)   do { if(aiter::current_log_level() <= aiter::LOG_ERROR)   { std::cerr << "[aiter ERROR] " << msg << std::endl; } } while(0)

// Module-specific logging macros (for use when AITER_LOG_MORE > 0)
#define AITER_LOG_DEBUG_MODULE(module, msg)   do { if(aiter::current_log_level() <= aiter::LOG_DEBUG && aiter::should_log_module(module))   { std::cout << "[aiter:" << module << "] " << msg << std::endl; } } while(0)
#define AITER_LOG_INFO_MODULE(module, msg)    do { if(aiter::current_log_level() <= aiter::LOG_INFO && aiter::should_log_module(module))    { std::cout << "[aiter:" << module << "] " << msg << std::endl; } } while(0)
#define AITER_LOG_WARNING_MODULE(module, msg) do { if(aiter::current_log_level() <= aiter::LOG_WARNING && aiter::should_log_module(module)) { std::cerr << "[aiter:" << module << " WARNING] " << msg << std::endl; } } while(0)
#define AITER_LOG_ERROR_MODULE(module, msg)   do { if(aiter::current_log_level() <= aiter::LOG_ERROR && aiter::should_log_module(module))   { std::cerr << "[aiter:" << module << " ERROR] " << msg << std::endl; } } while(0)
// clang-format on
