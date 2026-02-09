// OPUS Unit Tests — standalone, no dependencies beyond hipcc + opus.hpp
// Build: hipcc -std=c++20 -O2 -I../ test_opus.cpp -o test_opus && ./test_opus
#include <cassert>
#include <cstdio>
#include <type_traits>
#include "opus/opus.hpp"

namespace o = opus;
using o::number;
using o::index_t;

// =========================================================================
// test_number: number<I>, arithmetic, comparison, _I literal
// =========================================================================
void test_number() {
    constexpr auto a = number<4>{};
    constexpr auto b = number<8>{};
    static_assert(a.value == 4);
    static_assert(b.value == 8);

    // arithmetic
    static_assert((a + b).value == 12);
    static_assert((a - b).value == -4);
    static_assert((a * b).value == 32);
    static_assert((b / a).value == 2);
    static_assert((b % number<3>{}).value == 2);

    // bitwise
    static_assert((a & b).value == 0);
    static_assert((a | b).value == 12);
    static_assert((a ^ b).value == 12);
    static_assert((a << number<1>{}).value == 8);
    static_assert((b >> number<1>{}).value == 4);

    // comparison
    static_assert((a == a).value == true);
    static_assert((a != b).value == true);
    static_assert((a < b).value == true);
    static_assert((b > a).value == true);
    static_assert((a <= a).value == true);
    static_assert((b >= a).value == true);

    // unary
    static_assert((-a).value == -4);
    static_assert((+a).value == 4);
    static_assert((!number<0>{}).value == true);

    // _I literal
    using opus::operator""_I;
    constexpr auto c = 42_I;
    static_assert(c.value == 42);

    printf("  test_number: PASS\n");
}

// =========================================================================
// test_seq: seq construction, access, make_index_seq, reduce, concat, pop_front
// =========================================================================
void test_seq() {
    using o::seq;
    using o::make_index_seq;

    constexpr auto s = seq<3, 7, 11>{};
    static_assert(s.size() == 3);
    static_assert(s[0] == 3);
    static_assert(s[1] == 7);
    static_assert(s[2] == 11);

    // at() with number
    static_assert(s.at(number<0>{}) == 3);
    static_assert(s.at(number<2>{}) == 11);

    // get<I>
    static_assert(o::get<0>(s) == 3);
    static_assert(o::get<1>(s) == 7);

    // make_index_seq
    constexpr auto s0 = make_index_seq<5>{};
    static_assert(s0.size() == 5);
    static_assert(s0[0] == 0);
    static_assert(s0[4] == 4);

    // make_index_seq with start, end
    constexpr auto s1 = make_index_seq<4, 9>{};
    static_assert(s1.size() == 5);
    static_assert(s1[0] == 4);
    static_assert(s1[4] == 8);

    // make_index_seq with start, end, step
    constexpr auto s2 = make_index_seq<4, 8, 2>{};
    static_assert(s2.size() == 2);
    static_assert(s2[0] == 4);
    static_assert(s2[1] == 6);

    // make_repeated_seq
    constexpr auto sr = o::make_repeated_seq<5, 3>{};
    static_assert(sr.size() == 3);
    static_assert(sr[0] == 5);
    static_assert(sr[2] == 5);

    // concat_seq
    constexpr auto sc = o::concat_seq(seq<1, 2>{}, seq<3, 4>{});
    static_assert(sc.size() == 4);
    static_assert(sc[0] == 1);
    static_assert(sc[3] == 4);

    // reduce_seq_sum
    constexpr auto rs = o::reduce_seq_sum(seq<1, 2, 3, 4>{});
    static_assert(rs[0] == 10);

    // reduce_seq_mul
    constexpr auto rm = o::reduce_seq_mul(seq<2, 3, 4>{});
    static_assert(rm[0] == 24);

    // seq_pop_front
    constexpr auto sp = o::seq_pop_front(seq<10, 20, 30>{});
    static_assert(sp.size() == 2);
    static_assert(sp[0] == 20);
    static_assert(sp[1] == 30);

    // is_seq_v
    static_assert(o::is_seq_v<decltype(s)>);
    static_assert(!o::is_seq_v<int>);

    printf("  test_seq: PASS\n");
}

// =========================================================================
// test_array: make_array, operator[], fill, clear, size, concat_array, get<I>
// =========================================================================
void test_array() {
    constexpr auto a = o::make_array(10, 20, 30);
    static_assert(a.size() == 3);
    static_assert(a[0] == 10);
    static_assert(a[1] == 20);
    static_assert(a[2] == 30);

    // get<I>
    static_assert(o::get<0>(a) == 10);
    static_assert(o::get<2>(a) == 30);

    // number index
    static_assert(a[number<1>{}] == 20);

    // fill & clear
    auto b = o::make_array(0, 0, 0);
    b.fill(7);
    assert(b[0] == 7 && b[1] == 7 && b[2] == 7);
    b.clear();
    assert(b[0] == 0 && b[1] == 0 && b[2] == 0);

    // concat_array
    constexpr auto c = o::concat_array(o::make_array(1, 2), o::make_array(3, 4));
    static_assert(c.size() == 4);
    static_assert(c[0] == 1);
    static_assert(c[3] == 4);

    // equality
    constexpr auto d = o::make_array(10, 20, 30);
    static_assert(a == d);

    // is_array_v
    static_assert(o::is_array_v<decltype(a)>);
    static_assert(!o::is_array_v<int>);

    printf("  test_array: PASS\n");
}

// =========================================================================
// test_tuple: make_tuple, get<I>, size, concat, flatten, reduce, transform, repeated
// =========================================================================
void test_tuple() {
    constexpr auto t = o::make_tuple(number<3>{}, number<5>{}, number<7>{});
    static_assert(t.size() == 3);
    static_assert(o::get<0>(t).value == 3);
    static_assert(o::get<1>(t).value == 5);
    static_assert(o::get<2>(t).value == 7);

    // concat_tuple
    constexpr auto t1 = o::make_tuple(number<1>{}, number<2>{});
    constexpr auto t2 = o::make_tuple(number<3>{}, number<4>{});
    constexpr auto tc = o::concat_tuple(t1, t2);
    static_assert(tc.size() == 4);
    static_assert(o::get<0>(tc).value == 1);
    static_assert(o::get<3>(tc).value == 4);

    // flatten_tuple
    constexpr auto nested = o::make_tuple(o::make_tuple(number<1>{}, number<2>{}), number<3>{});
    constexpr auto flat = o::flatten_tuple(nested);
    static_assert(flat.size() == 3);
    static_assert(o::get<0>(flat).value == 1);
    static_assert(o::get<1>(flat).value == 2);
    static_assert(o::get<2>(flat).value == 3);

    // reduce_tuple_sum
    constexpr auto rts = o::reduce_tuple_sum(o::make_tuple(number<1>{}, number<2>{}, number<3>{}));
    static_assert(o::get<0>(rts).value == 6);

    // reduce_tuple_mul
    constexpr auto rtm = o::reduce_tuple_mul(o::make_tuple(number<2>{}, number<3>{}, number<4>{}));
    static_assert(o::get<0>(rtm).value == 24);

    // transform_tuple
    constexpr auto tr = o::transform_tuple([](auto x) { return x + number<10>{}; },
                                            o::make_tuple(number<1>{}, number<2>{}, number<3>{}));
    static_assert(o::get<0>(tr).value == 11);
    static_assert(o::get<2>(tr).value == 13);

    // make_repeated_tuple
    constexpr auto rep = o::make_repeated_tuple<3>(number<42>{});
    static_assert(rep.size() == 3);
    static_assert(o::get<0>(rep).value == 42);
    static_assert(o::get<2>(rep).value == 42);

    // is_tuple_v
    static_assert(o::is_tuple_v<decltype(t)>);
    static_assert(!o::is_tuple_v<int>);

    // seq_to_tuple
    constexpr auto st = o::seq_to_tuple(o::seq<5, 10, 15>{});
    static_assert(st.size() == 3);
    static_assert(o::get<0>(st).value == 5);
    static_assert(o::get<2>(st).value == 15);

    // tuple_count
    constexpr index_t nc = o::tuple_count<number<1>>(o::make_tuple(number<1>{}, number<2>{}, number<1>{}));
    static_assert(nc == 2);

    printf("  test_tuple: PASS\n");
}

// =========================================================================
// test_vector: vector_t, make_vector, make_repeated_vector, fill, clear, to_array, to_vector
// =========================================================================
void test_vector() {
    // make_vector
    auto v = o::make_vector(1.0f, 2.0f, 3.0f, 4.0f);
    assert(v[0] == 1.0f);
    assert(v[3] == 4.0f);

    // get<I>
    assert(o::get<0>(v) == 1.0f);
    assert(o::get<3>(v) == 4.0f);

    // make_repeated_vector
    auto rv = o::make_repeated_vector<4>(5.0f);
    assert(rv[0] == 5.0f && rv[3] == 5.0f);

    // fill & clear
    auto vf = o::make_vector(0.0f, 0.0f, 0.0f);
    o::fill(vf, 9.0f);
    assert(vf[0] == 9.0f && vf[2] == 9.0f);
    o::clear(vf);
    assert(vf[0] == 0.0f && vf[2] == 0.0f);

    // to_array
    auto va = o::make_vector(10.0f, 20.0f);
    auto arr = o::to_array(va);
    assert(arr[0] == 10.0f && arr[1] == 20.0f);
    static_assert(o::is_array_v<decltype(arr)>);

    // to_vector from array
    auto arr2 = o::make_array(1.0f, 2.0f, 3.0f);
    auto vec2 = o::to_vector(arr2);
    assert(vec2[0] == 1.0f && vec2[2] == 3.0f);
    static_assert(o::is_vector_v<decltype(vec2)>);

    // is_vector_v
    static_assert(o::is_vector_v<decltype(v)>);
    static_assert(!o::is_vector_v<int>);

    // vector_traits
    static_assert(o::vector_traits<decltype(v)>::size() == 4);
    static_assert(std::is_same_v<o::vector_traits<decltype(v)>::dtype, float>);

    printf("  test_vector: PASS\n");
}

// =========================================================================
// test_slice: static slicing on vector/array/tuple, set_slice
// =========================================================================
void test_slice() {
    // vector slice [end]
    auto v = o::make_vector(10.0f, 20.0f, 30.0f, 40.0f);
    auto vs1 = o::slice(v, number<3>{});  // first 3 elements
    assert(vs1[0] == 10.0f && vs1[2] == 30.0f);
    static_assert(o::vector_traits<decltype(vs1)>::size() == 3);

    // vector slice [start, end]
    auto vs2 = o::slice(v, number<1>{}, number<3>{});  // elements 1..2
    assert(vs2[0] == 20.0f && vs2[1] == 30.0f);
    static_assert(o::vector_traits<decltype(vs2)>::size() == 2);

    // array slice
    auto a = o::make_array(1, 2, 3, 4, 5);
    auto as1 = o::slice(a, number<3>{});
    assert(as1[0] == 1 && as1[2] == 3);
    assert(as1.size() == 3);

    // tuple slice
    constexpr auto t = o::make_tuple(number<10>{}, number<20>{}, number<30>{}, number<40>{});
    constexpr auto ts = o::slice(t, number<2>{});
    static_assert(ts.size() == 2);
    static_assert(o::get<0>(ts).value == 10);
    static_assert(o::get<1>(ts).value == 20);

    // set_slice on vector
    auto vd = o::make_vector(0.0f, 0.0f, 0.0f, 0.0f);
    auto src = o::make_vector(7.0f, 8.0f);
    o::set_slice(vd, src, number<1>{}, number<3>{});
    assert(vd[0] == 0.0f && vd[1] == 7.0f && vd[2] == 8.0f && vd[3] == 0.0f);

    printf("  test_slice: PASS\n");
}

// =========================================================================
// test_layout: make_layout (packed), layout_linear, multi-dim index, packed_shape_to_stride
// =========================================================================
void test_layout() {
    // packed_shape_to_stride
    constexpr auto shape = o::make_tuple(number<4>{}, number<8>{});
    constexpr auto stride = o::packed_shape_to_stride(shape);
    static_assert(o::get<0>(stride).value == 8);
    static_assert(o::get<1>(stride).value == 1);

    // make_layout from shape tuple (packed) — single-arg overload
    constexpr auto ly = o::make_layout(o::make_tuple(number<4>{}, number<8>{}));
    static_assert(ly(number<0>{}, number<0>{}) == 0);
    static_assert(ly(number<1>{}, number<0>{}) == 8);
    static_assert(ly(number<0>{}, number<3>{}) == 3);
    static_assert(ly(number<2>{}, number<5>{}) == 21);

    // make_layout with explicit shape and stride tuples
    constexpr auto ly2 = o::make_layout(o::make_tuple(number<4>{}, number<8>{}),
                                         o::make_tuple(number<16>{}, number<1>{}));
    static_assert(ly2(number<1>{}, number<0>{}) == 16);
    static_assert(ly2(number<2>{}, number<3>{}) == 35);

    // layout_linear: supports inc() and +=
    auto ll = o::make_layout(o::make_tuple(number<2>{}, number<4>{}));
    assert(ll(number<0>{}, number<0>{}) == 0);
    ll += 100;
    assert(ll(number<0>{}, number<0>{}) == 100);
    assert(ll(number<1>{}, number<2>{}) == 106);

    // is_layout_v
    static_assert(o::is_layout_v<decltype(ly)>);
    static_assert(o::is_layout_v<decltype(ll)>);
    static_assert(!o::is_layout_v<int>);

    // 3D layout
    constexpr auto ly3 = o::make_layout(o::make_tuple(number<2>{}, number<3>{}, number<4>{}));
    static_assert(ly3(number<0>{}, number<0>{}, number<0>{}) == 0);
    static_assert(ly3(number<1>{}, number<0>{}, number<0>{}) == 12);
    static_assert(ly3(number<0>{}, number<1>{}, number<0>{}) == 4);
    static_assert(ly3(number<0>{}, number<0>{}, number<3>{}) == 3);

    printf("  test_layout: PASS\n");
}

// =========================================================================
// test_static_for: static_for<N>, range-based, static_ford
// =========================================================================
void test_static_for() {
    // static_for<N>
    int sum = 0;
    o::static_for<5>([&](auto i) { sum += i.value; });
    assert(sum == 10);  // 0+1+2+3+4

    // static_for with range (number constants)
    int sum2 = 0;
    o::static_for([&](auto i) { sum2 += i.value; }, number<2>{}, number<5>{});
    assert(sum2 == 9);  // 2+3+4

    // static_for with step
    int sum3 = 0;
    o::static_for([&](auto i) { sum3 += i.value; }, number<0>{}, number<6>{}, number<2>{});
    assert(sum3 == 6);  // 0+2+4

    // static_for with runtime integers
    int sum4 = 0;
    o::static_for([&](auto i) { sum4 += i; }, 5);
    assert(sum4 == 10);

    // static_ford<N0, N1> (2D)
    int count = 0;
    o::static_ford<2, 3>([&](auto i, auto j) { count++; });
    assert(count == 6);

    // static_ford with seq
    int count2 = 0;
    o::static_ford(o::seq<2, 3>{}, [&](auto i, auto j) { count2++; });
    assert(count2 == 6);

    printf("  test_static_for: PASS\n");
}

// =========================================================================
// test_type_traits
// =========================================================================
void test_type_traits() {
    static_assert(o::is_constant_v<number<5>>);
    static_assert(o::is_constant_v<o::bool_constant<true>>);
    static_assert(!o::is_constant_v<int>);

    static_assert(o::is_seq_v<o::seq<1, 2, 3>>);
    static_assert(!o::is_seq_v<int>);

    static_assert(o::is_tuple_v<o::tuple<number<1>, number<2>>>);
    static_assert(!o::is_tuple_v<int>);

    static_assert(o::is_array_v<o::array<int, 3>>);
    static_assert(!o::is_array_v<int>);

    static_assert(o::is_vector_v<o::vector_t<float, 4>>);
    static_assert(!o::is_vector_v<float>);

    using layout_type = decltype(o::make_layout(o::make_tuple(number<4>{}, number<8>{})));
    static_assert(o::is_layout_v<layout_type>);
    static_assert(!o::is_layout_v<int>);

    static_assert(o::is_underscore_v<o::underscore>);
    static_assert(!o::is_underscore_v<int>);

    static_assert(o::is_dtype_v<o::fp32_t>);
    static_assert(o::is_dtype_v<o::fp16_t>);
    static_assert(o::is_dtype_v<o::bf16_t>);
    static_assert(o::is_dtype_v<o::fp8_t>);
    static_assert(o::is_dtype_v<o::i32_t>);
    static_assert(o::is_dtype_v<o::u8_t>);
    static_assert(!o::is_dtype_v<double>);

    static_assert(o::is_packs_v<o::fp4_t>);
    static_assert(o::is_packs_v<o::int4_t>);
    static_assert(o::is_packs_v<o::uint4_t>);
    static_assert(o::is_packs_v<o::e8m0_t>);
    static_assert(!o::is_packs_v<int>);

    printf("  test_type_traits: PASS\n");
}

// =========================================================================
// test_underscore: underscore, merge_peepholed_tuple, tuple_count
// =========================================================================
void test_underscore() {
    static_assert(o::is_underscore_v<o::underscore>);
    static_assert(!o::is_underscore_v<number<0>>);

    constexpr auto t = o::make_tuple(number<1>{}, o::underscore{}, number<3>{}, o::underscore{});
    constexpr auto cnt = o::tuple_count<o::underscore>(t);
    static_assert(cnt == 2);

    // merge_peepholed_tuple: replace underscores with income values
    constexpr auto pt = o::make_tuple(number<1>{}, o::underscore{}, number<3>{}, o::underscore{});
    constexpr auto it = o::make_tuple(number<10>{}, number<20>{});
    constexpr auto merged = o::merge_peepholed_tuple(pt, it);
    static_assert(o::get<0>(merged).value == 1);
    static_assert(o::get<1>(merged).value == 10);
    static_assert(o::get<2>(merged).value == 3);
    static_assert(o::get<3>(merged).value == 20);

    // no underscores -> return original
    constexpr auto pt2 = o::make_tuple(number<5>{}, number<6>{});
    constexpr auto merged2 = o::merge_peepholed_tuple(pt2, it);
    static_assert(o::get<0>(merged2).value == 5);
    static_assert(o::get<1>(merged2).value == 6);

    printf("  test_underscore: PASS\n");
}

// =========================================================================
// test_embed: embed(x, y) dot product of tuples
// =========================================================================
void test_embed() {
    constexpr auto x = o::make_tuple(number<2>{}, number<3>{}, number<4>{});
    constexpr auto y = o::make_tuple(number<5>{}, number<6>{}, number<7>{});
    constexpr auto dot = o::embed(x, y);
    // 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
    static_assert(dot.value == 56);

    constexpr auto a = o::make_tuple(number<1>{}, number<2>{});
    constexpr auto b = o::make_tuple(number<3>{}, number<4>{});
    constexpr auto dot2 = o::embed(a, b);
    static_assert(dot2.value == 11);  // 1*3 + 2*4

    printf("  test_embed: PASS\n");
}

// =========================================================================
// test_packed_types: fp4_t, int4_t, uint4_t, e8m0_t
// =========================================================================
void test_packed_types() {
    static_assert(o::sizeof_bits_v<o::fp4_t> == 4);
    static_assert(o::num_packs_v<o::fp4_t> == 2);
    static_assert(sizeof(o::fp4_t) == 1);

    static_assert(o::sizeof_bits_v<o::int4_t> == 4);
    static_assert(o::num_packs_v<o::int4_t> == 2);

    static_assert(o::sizeof_bits_v<o::uint4_t> == 4);
    static_assert(o::num_packs_v<o::uint4_t> == 2);

    static_assert(o::sizeof_bits_v<o::e8m0_t> == 8);
    static_assert(o::num_packs_v<o::e8m0_t> == 1);

    static_assert(o::num_packs_v<float> == 1);

    static_assert(o::sizeof_bits_v<float> == 32);
    static_assert(o::sizeof_bits_v<int32_t> == 32);
    static_assert(o::sizeof_bits_v<int8_t> == 8);
    static_assert(o::sizeof_bits_v<void> == 0);

    static_assert(o::is_packs_v<o::fp4_t> && o::is_dtype_v<o::fp4_t>);
    static_assert(o::is_packs_v<o::int4_t> && o::is_dtype_v<o::int4_t>);
    static_assert(o::is_packs_v<o::uint4_t> && o::is_dtype_v<o::uint4_t>);
    static_assert(o::is_packs_v<o::e8m0_t> && o::is_dtype_v<o::e8m0_t>);

    // pack indexing (extract sub-values from packed byte)
    o::fp4_t pack;
    pack.value = 0x53;  // 0101 0011 -> pack[0]=3(lo nibble), pack[1]=5(hi nibble)
    assert(pack[0] == 3);
    assert(pack[1] == 5);
    assert(pack[number<0>{}] == 3);
    assert(pack[number<1>{}] == 5);

    printf("  test_packed_types: PASS\n");
}

// =========================================================================
// test_adaptor: p_dim/y_dim types, mfma_adaptor static constexpr members
// Note: adaptor functions (pickup_shape, shape_a, etc.) are OPUS_D
// (device-only) when compiled with hipcc, so we test only the constexpr
// member values that are accessible from host.
// =========================================================================
void test_adaptor() {
    using o::p_dim;
    using o::y_dim;

    // p_dim and y_dim are distinct types
    static_assert(!std::is_same_v<p_dim, y_dim>);

    // mfma_adaptor constexpr members for 16x16x16_f16
    using A16 = o::impl::mfma_adaptor<o::mfma_f32_16x16x16_f16>;
    static_assert(A16::grpm_a == 16);
    static_assert(A16::grpn_b == 16);
    static_assert(A16::grpk_a == 4);    // 64 / 16
    static_assert(A16::grpk_b == 4);
    static_assert(A16::grpn_c == 16);
    static_assert(A16::grpm_c == 4);    // 64 / 16
    static_assert(A16::pack_a == 4);
    static_assert(A16::pack_b == 4);
    static_assert(A16::pack_c == 4);
    static_assert(A16::rept_a == 1);    // elem_a(4) / pack_a(4)
    static_assert(A16::rept_b == 1);
    static_assert(A16::rept_c == 1);    // elem_c(4) / pack_c(4)

    // mfma_adaptor constexpr members for 32x32x8_f16
    using A32 = o::impl::mfma_adaptor<o::mfma_f32_32x32x8_f16>;
    static_assert(A32::grpm_a == 32);
    static_assert(A32::grpn_b == 32);
    static_assert(A32::grpk_a == 2);    // 64 / 32
    static_assert(A32::grpn_c == 32);
    static_assert(A32::grpm_c == 2);    // 64 / 32
    static_assert(A32::pack_a == 4);    // min(8, 4) = 4
    static_assert(A32::pack_b == 4);
    static_assert(A32::pack_c == 4);    // min(4, 16) = 4
    static_assert(A32::rept_a == 1);    // 4 / 4
    static_assert(A32::rept_c == 4);    // 16 / 4

    // mfma_adaptor_swap_ab inherits from mfma_adaptor
    using A16S = o::impl::mfma_adaptor_swap_ab<o::mfma_f32_16x16x16_f16>;
    static_assert(A16S::grpm_a == 16);  // inherited
    static_assert(A16S::grpn_b == 16);  // inherited

    printf("  test_adaptor: PASS\n");
}

// =========================================================================
// test_mfma_types: MFMA type alias static properties (compile-time, no GPU)
// =========================================================================
void test_mfma_types() {
    {
        using M = o::mfma_f32_16x16x16_f16;
        static_assert(M::wave_m == 16);
        static_assert(M::wave_n == 16);
        static_assert(M::wave_k == 16);
        static_assert(M::warp_size == 64);
        static_assert(std::is_same_v<M::dtype_a, o::fp16_t>);
        static_assert(std::is_same_v<M::dtype_b, o::fp16_t>);
        static_assert(std::is_same_v<M::dtype_c, o::fp32_t>);
        static_assert(M::elem_a == 4);
        static_assert(M::elem_b == 4);
        static_assert(M::elem_c == 4);
    }
    {
        using M = o::mfma_f32_32x32x8_f16;
        static_assert(M::wave_m == 32 && M::wave_n == 32 && M::wave_k == 8);
        static_assert(M::elem_a == 4);
        static_assert(M::elem_b == 4);
        static_assert(M::elem_c == 16);
    }
    {
        using M = o::mfma_f32_32x32x8_bf16;
        static_assert(std::is_same_v<M::dtype_a, o::bf16_t>);
        static_assert(M::wave_m == 32 && M::wave_n == 32 && M::wave_k == 8);
    }
    {
        using M = o::mfma_f32_32x32x16_fp8_fp8;
        static_assert(M::wave_m == 32 && M::wave_n == 32 && M::wave_k == 16);
        static_assert(std::is_same_v<M::dtype_a, o::fp8_t>);
        static_assert(M::elem_a == 8);
        static_assert(M::elem_c == 16);
    }
    {
        using M = o::mfma_f32_16x16x32_fp8_fp8;
        static_assert(M::wave_m == 16 && M::wave_n == 16 && M::wave_k == 32);
        static_assert(M::elem_a == 8);
        static_assert(M::elem_c == 4);
    }

    printf("  test_mfma_types: PASS\n");
}

// =========================================================================
// test_warp_size: get_warp_size() returns 64 on host
// =========================================================================
void test_warp_size() {
    constexpr auto ws = o::get_warp_size();
    static_assert(ws == 64);
    assert(ws == 64);

    printf("  test_warp_size: PASS\n");
}

// =========================================================================
// test_functional: plus, minus, multiplies, divides functors
// =========================================================================
void test_functional() {
    constexpr auto a = number<6>{};
    constexpr auto b = number<3>{};

    static_assert(o::plus{}(a, b).value == 9);
    static_assert(o::minus{}(a, b).value == 3);
    static_assert(o::multiplies{}(a, b).value == 18);
    static_assert(o::divides{}(a, b).value == 2);

    assert(o::plus{}(10, 20) == 30);
    assert(o::multiplies{}(5, 7) == 35);

    printf("  test_functional: PASS\n");
}

// =========================================================================
// main
// =========================================================================
int main() {
    printf("opus unit tests\n");
    printf("================\n");
    test_number();
    test_seq();
    test_array();
    test_tuple();
    test_vector();
    test_slice();
    test_layout();
    test_static_for();
    test_type_traits();
    test_underscore();
    test_embed();
    test_packed_types();
    test_adaptor();
    test_mfma_types();
    test_warp_size();
    test_functional();
    printf("================\n");
    printf("All tests passed!\n");
    return 0;
}
