# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# CMake configuration for OPUS unit tests

cmake_minimum_required(VERSION 3.16)
project(opus_tests LANGUAGES CXX)

# Find HIP
find_package(HIP QUIET)

if(NOT HIP_FOUND)
    message(WARNING "HIP not found. OPUS tests require ROCm/HIP.")
    message(WARNING "Skipping OPUS test build.")
    return()
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/include)

# Compile tests with hipcc
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

# Main test executable
add_executable(test_opus_basic test_opus_basic.cpp)

# Enable testing
enable_testing()
add_test(NAME opus_basic COMMAND test_opus_basic)

# Print configuration
message(STATUS "OPUS Tests Configuration:")
message(STATUS "  HIP found: ${HIP_FOUND}")
message(STATUS "  HIPCC: ${HIP_HIPCC_EXECUTABLE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Include Dir: ${CMAKE_CURRENT_SOURCE_DIR}/../../include")
