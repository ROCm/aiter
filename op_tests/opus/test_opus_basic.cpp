// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_opus_basic.cpp
 * @brief Basic unit tests for OPUS (AI Operator Micro Std) library
 *
 * This file contains unit tests for core OPUS components:
 * - number/seq: compile-time constants and sequences
 * - array: fixed-size arrays with constexpr support
 * - tuple: heterogeneous tuple containers
 * - static_for: compile-time loop unrolling
 * - gmem: global memory abstractions (GPU tests)
 */

#include <iostream>
#include <cassert>
#include <type_traits>
#include "opus/opus.hpp"

// =============================================================================
// Test Utilities
// =============================================================================
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_EQ(a, b, msg) \
    TEST_ASSERT((a) == (b), msg << " expected=" << (b) << " actual=" << (a))

int g_tests_passed = 0;
int g_tests_failed = 0;

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... "; \
        if (test_func()) { \
            std::cout << "PASSED" << std::endl; \
            g_tests_passed++; \
        } else { \
            std::cout << "FAILED" << std::endl; \
            g_tests_failed++; \
        } \
    } while(0)

// =============================================================================
// Number and Sequence Tests
// =============================================================================
bool test_number_basic() {
    // Test number type and literals
    auto n1 = opus::number<5>{};
    TEST_ASSERT_EQ(n1.value, 5, "number<5> value");

    // Test literal operator - bring in specific operator
    using opus::operator""_I;
    auto n2 = 10_I;
    TEST_ASSERT_EQ(decltype(n2)::value, 10, "10_I literal");

    // Test arithmetic operations on numbers
    auto sum = opus::number<3>{} + opus::number<5>{};
    TEST_ASSERT_EQ(decltype(sum)::value, 8, "3 + 5");

    auto diff = opus::number<10>{} - opus::number<3>{};
    TEST_ASSERT_EQ(decltype(diff)::value, 7, "10 - 3");

    auto prod = opus::number<4>{} * opus::number<5>{};
    TEST_ASSERT_EQ(decltype(prod)::value, 20, "4 * 5");

    auto div = opus::number<20>{} / opus::number<4>{};
    TEST_ASSERT_EQ(decltype(div)::value, 5, "20 / 4");

    // Test comparison operations
    constexpr bool lt = (opus::number<3>{} < opus::number<5>{}).value;
    TEST_ASSERT(lt, "3 < 5");

    constexpr bool gt = (opus::number<10>{} > opus::number<5>{}).value;
    TEST_ASSERT(gt, "10 > 5");

    constexpr bool eq = (opus::number<7>{} == opus::number<7>{}).value;
    TEST_ASSERT(eq, "7 == 7");

    return true;
}

bool test_seq_basic() {
    // Test sequence creation
    using s1 = opus::seq<1, 2, 3, 4, 5>;
    TEST_ASSERT_EQ(s1::size(), 5, "seq size");
    TEST_ASSERT_EQ(s1::at(0), 1, "seq[0]");
    TEST_ASSERT_EQ(s1::at(4), 5, "seq[4]");

    // Test get function
    auto val = opus::get<2>(s1{});
    TEST_ASSERT_EQ(val, 3, "get<2>(seq)");

    // Test make_index_seq
    using s2 = opus::make_index_seq<5>;
    TEST_ASSERT_EQ(s2::size(), 5, "make_index_seq<5> size");
    TEST_ASSERT_EQ(s2::at(0), 0, "make_index_seq[0]");
    TEST_ASSERT_EQ(s2::at(4), 4, "make_index_seq[4]");

    // Test make_index_seq with range
    using s3 = opus::make_index_seq<2, 6>;
    TEST_ASSERT_EQ(s3::size(), 4, "make_index_seq<2,6> size");
    TEST_ASSERT_EQ(s3::at(0), 2, "make_index_seq<2,6>[0]");
    TEST_ASSERT_EQ(s3::at(3), 5, "make_index_seq<2,6>[3]");

    // Test make_repeated_seq
    using s4 = opus::make_repeated_seq<7, 3>;
    TEST_ASSERT_EQ(s4::size(), 3, "make_repeated_seq size");
    TEST_ASSERT_EQ(s4::at(0), 7, "make_repeated_seq[0]");
    TEST_ASSERT_EQ(s4::at(1), 7, "make_repeated_seq[1]");

    return true;
}

bool test_seq_reduction() {
    // Test reduce_seq - returns a single-element seq, access via at(0)
    using s = opus::seq<1, 2, 3, 4>;
    auto sum = opus::reduce_seq_sum(s{});
    TEST_ASSERT_EQ(sum.at(0), 10, "sum of 1+2+3+4");

    auto prod = opus::reduce_seq_mul(s{});
    TEST_ASSERT_EQ(prod.at(0), 24, "product of 1*2*3*4");

    return true;
}

// =============================================================================
// Array Tests
// =============================================================================
bool test_array_basic() {
    // Test array creation and access
    opus::array<int, 5> arr;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;

    TEST_ASSERT_EQ(arr[0], 10, "array[0]");
    TEST_ASSERT_EQ(arr[1], 20, "array[1]");
    TEST_ASSERT_EQ(arr.size(), 5, "array size");

    // Test number indexing
    TEST_ASSERT_EQ(arr[opus::number<2>{}], 30, "array[number<2>]");

    // Test fill and clear
    arr.fill(100);
    TEST_ASSERT_EQ(arr[0], 100, "array after fill");
    TEST_ASSERT_EQ(arr[4], 100, "array after fill (last element)");

    arr.clear();
    TEST_ASSERT_EQ(arr[0], 0, "array after clear");

    return true;
}

bool test_array_make_and_get() {
    // Test make_array
    auto arr = opus::make_array(1, 2, 3, 4, 5);
    TEST_ASSERT_EQ(arr.size(), 5, "make_array size");
    TEST_ASSERT_EQ(opus::get<0>(arr), 1, "get<0>(array)");
    TEST_ASSERT_EQ(opus::get<4>(arr), 5, "get<4>(array)");

    // Test concat_array
    auto arr1 = opus::make_array(1, 2);
    auto arr2 = opus::make_array(3, 4);
    auto arr_concat = opus::concat_array(arr1, arr2);
    TEST_ASSERT_EQ(arr_concat.size(), 4, "concat_array size");
    TEST_ASSERT_EQ(opus::get<0>(arr_concat), 1, "concat_array[0]");
    TEST_ASSERT_EQ(opus::get<3>(arr_concat), 4, "concat_array[3]");

    return true;
}

// =============================================================================
// Tuple Tests
// =============================================================================
template <typename T>
constexpr opus::index_t tuple_size(T&&) {
    return opus::remove_cvref_t<T>::size();
}

bool test_tuple_basic() {
    // Test tuple creation
    auto t = opus::make_tuple(1, 2.5, 'a');
    TEST_ASSERT_EQ(tuple_size(t), 3, "tuple size");
    TEST_ASSERT_EQ(opus::get<0>(t), 1, "get<0>(tuple)");
    TEST_ASSERT_EQ(opus::get<1>(t), 2.5, "get<1>(tuple)");
    TEST_ASSERT_EQ(opus::get<2>(t), 'a', "get<2>(tuple)");

    // Test single element tuple
    auto t1 = opus::make_tuple(42);
    TEST_ASSERT_EQ(opus::get<0>(t1), 42, "single element tuple");

    return true;
}

bool test_tuple_concat() {
    auto t1 = opus::make_tuple(1, 2);
    auto t2 = opus::make_tuple(3.0, 4.0);
    auto t_concat = opus::concat_tuple(t1, t2);

    TEST_ASSERT_EQ(tuple_size(t_concat), 4, "concat_tuple size");
    TEST_ASSERT_EQ(opus::get<0>(t_concat), 1, "concat_tuple[0]");
    TEST_ASSERT_EQ(opus::get<1>(t_concat), 2, "concat_tuple[1]");
    TEST_ASSERT_EQ(opus::get<2>(t_concat), 3.0, "concat_tuple[2]");
    TEST_ASSERT_EQ(opus::get<3>(t_concat), 4.0, "concat_tuple[3]");

    return true;
}

bool test_make_repeated_tuple() {
    auto t = opus::make_repeated_tuple<3>(5);
    TEST_ASSERT_EQ(tuple_size(t), 3, "make_repeated_tuple size");
    TEST_ASSERT_EQ(opus::get<0>(t), 5, "make_repeated_tuple[0]");
    TEST_ASSERT_EQ(opus::get<1>(t), 5, "make_repeated_tuple[1]");
    TEST_ASSERT_EQ(opus::get<2>(t), 5, "make_repeated_tuple[2]");

    return true;
}

bool test_merge_peepholed_tuple() {
    using opus::operator""_I;

    // merge_peepholed_tuple fills underscore (_) slots in the peepholed tuple
    // with values from the income tuple, preserving non-underscore elements.
    // tuple<*, *, _, *, _> + tuple<#, @> -> tuple<*, *, #, *, @>

    // Case 1: Two underscores at positions 2 and 4
    auto pt1 = opus::make_tuple(10_I, 20_I, opus::_, 40_I, opus::_);
    auto it1 = opus::make_tuple(99_I, 77_I);
    auto r1  = opus::merge_peepholed_tuple(pt1, it1);
    TEST_ASSERT_EQ(opus::get<0>(r1).value, 10, "merge[0] = 10 (kept)");
    TEST_ASSERT_EQ(opus::get<1>(r1).value, 20, "merge[1] = 20 (kept)");
    TEST_ASSERT_EQ(opus::get<2>(r1).value, 99, "merge[2] = 99 (from income[0])");
    TEST_ASSERT_EQ(opus::get<3>(r1).value, 40, "merge[3] = 40 (kept)");
    TEST_ASSERT_EQ(opus::get<4>(r1).value, 77, "merge[4] = 77 (from income[1])");

    // Case 2: Single underscore at position 0
    auto pt2 = opus::make_tuple(opus::_, 5_I, 6_I);
    auto it2 = opus::make_tuple(100_I);
    auto r2  = opus::merge_peepholed_tuple(pt2, it2);
    TEST_ASSERT_EQ(opus::get<0>(r2).value, 100, "merge single _[0] = 100");
    TEST_ASSERT_EQ(opus::get<1>(r2).value, 5,   "merge single _[1] = 5");
    TEST_ASSERT_EQ(opus::get<2>(r2).value, 6,   "merge single _[2] = 6");

    // Case 3: No underscores -- returns the peepholed tuple unchanged
    auto pt3 = opus::make_tuple(1_I, 2_I, 3_I);
    auto it3 = opus::make_tuple();  // empty income
    auto r3  = opus::merge_peepholed_tuple(pt3, it3);
    TEST_ASSERT_EQ(opus::get<0>(r3).value, 1, "merge no-underscore[0]");
    TEST_ASSERT_EQ(opus::get<1>(r3).value, 2, "merge no-underscore[1]");
    TEST_ASSERT_EQ(opus::get<2>(r3).value, 3, "merge no-underscore[2]");

    // Case 4: All underscores -- result is the income tuple
    auto pt4 = opus::make_tuple(opus::_, opus::_, opus::_);
    auto it4 = opus::make_tuple(7_I, 8_I, 9_I);
    auto r4  = opus::merge_peepholed_tuple(pt4, it4);
    TEST_ASSERT_EQ(opus::get<0>(r4).value, 7, "merge all-underscore[0]");
    TEST_ASSERT_EQ(opus::get<1>(r4).value, 8, "merge all-underscore[1]");
    TEST_ASSERT_EQ(opus::get<2>(r4).value, 9, "merge all-underscore[2]");

    return true;
}

// =============================================================================
// Static For Tests
// =============================================================================
bool test_static_for() {
    // Test static_for with number
    int sum = 0;
    opus::static_for<5>([&](auto i) {
        sum += i.value;
    });
    TEST_ASSERT_EQ(sum, 0+1+2+3+4, "static_for sum");

    // Test static_for with runtime range
    int prod = 1;
    opus::static_for([&](int i) {
        prod *= (i + 1);
    }, 3);
    TEST_ASSERT_EQ(prod, 1*2*3, "static_for runtime product");

    // Test static_for with start and end
    sum = 0;
    opus::static_for([&](int i) {
        sum += i;
    }, 2, 5);
    TEST_ASSERT_EQ(sum, 2+3+4, "static_for range sum");

    return true;
}

bool test_static_ford() {
    // Test nested static_for (static_ford)
    int counter = 0;
    // static_ford with template parameters for dimensions
    opus::static_ford<2, 3>([&](auto i, auto j) {
        (void)i;
        (void)j;
        counter++;
    });
    TEST_ASSERT_EQ(counter, 6, "static_ford 2x3 iterations");

    return true;
}

// =============================================================================
// Type Traits Tests
// =============================================================================
bool test_type_traits() {
    // Test is_constant (using static_assert for compile-time checks)
    static_assert(opus::is_constant_v<opus::number<5>>, "number is constant");
    static_assert(opus::is_constant_v<opus::bool_constant<true>>, "bool_constant is constant");
    static_assert(!opus::is_constant_v<int>, "int is not constant");

    // Test is_seq
    static_assert(opus::is_seq_v<opus::seq<1, 2, 3>>, "seq is seq");
    static_assert(!opus::is_seq_v<int>, "int is not seq");

    // Test is_array
    static_assert(opus::is_array_v<opus::array<int, 5>>, "array is array");
    static_assert(!opus::is_array_v<int>, "int is not array");

    // Test is_tuple
    static_assert(opus::is_tuple_v<opus::tuple<int, float>>, "tuple is tuple");
    static_assert(!opus::is_tuple_v<int>, "int is not tuple");

    // Test is_any_of
    static_assert(opus::is_any_of_v<int, float, double, int>, "int is in list");
    static_assert(!opus::is_any_of_v<char, float, double, int>, "char is not in list");

    return true;
}

// =============================================================================
// Layout Tests
// =============================================================================
bool test_layout_basic() {
    // Test make_layout with shape only
    using opus::operator""_I;
    auto layout_1d = opus::make_layout(opus::make_tuple(10_I));
    (void)layout_1d;

    // Test make_layout with shape and stride
    auto layout_2d = opus::make_layout(opus::make_tuple(3_I, 4_I), opus::make_tuple(4_I, 1_I));
    (void)layout_2d;

    return true;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "OPUS (AI Operator Micro Std) Unit Tests" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;

    std::cout << "--- Number and Sequence Tests ---" << std::endl;
    RUN_TEST(test_number_basic);
    RUN_TEST(test_seq_basic);
    RUN_TEST(test_seq_reduction);

    std::cout << std::endl << "--- Array Tests ---" << std::endl;
    RUN_TEST(test_array_basic);
    RUN_TEST(test_array_make_and_get);

    std::cout << std::endl << "--- Tuple Tests ---" << std::endl;
    RUN_TEST(test_tuple_basic);
    RUN_TEST(test_tuple_concat);
    RUN_TEST(test_make_repeated_tuple);
    RUN_TEST(test_merge_peepholed_tuple);

    std::cout << std::endl << "--- Static For Tests ---" << std::endl;
    RUN_TEST(test_static_for);
    RUN_TEST(test_static_ford);

    std::cout << std::endl << "--- Type Traits Tests ---" << std::endl;
    RUN_TEST(test_type_traits);

    std::cout << std::endl << "--- Layout Tests ---" << std::endl;
    RUN_TEST(test_layout_basic);

    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Results: " << g_tests_passed << " passed, " << g_tests_failed << " failed" << std::endl;
    std::cout << "======================================" << std::endl;

    return g_tests_failed > 0 ? 1 : 0;
}
