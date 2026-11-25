// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Builds the entries of one GPU family's decode, batched decode or prefill
// table with one head dim and dtype, and registers their launch functions.
// Each kernel is a translation unit (rdna_attn_unit.cuh) of a module built
// for one head dim and dtype, so the builds compile in parallel.
#pragma once

#include <iterator>
#include <utility>

#include "rdna_decode_attn.cuh"
#include "rdna_prefill_attn.cuh"

namespace rdna {

template <Arch A, HeadDim kHeadDim, Dtype kDtype, size_t... I>
bool register_decode(std::index_sequence<I...>) {
  (
      [] {
        constexpr DecodeConfig c = decode_configs<A>()[I];
        if constexpr (c.head_dim == kHeadDim && c.dtype == kDtype)
          decode_launchers(A)[I] = &launch_decode<c, false, A>;
      }(),
      ...);
  return true;
}

template <Arch A, HeadDim kHeadDim, Dtype kDtype>
bool register_decode() {
  return register_decode<A, kHeadDim, kDtype>(
      std::make_index_sequence<std::size(decode_configs<A>())>{});
}

template <Arch A, HeadDim kHeadDim, Dtype kDtype, size_t... I>
bool register_decode_batch(std::index_sequence<I...>) {
  (
      [] {
        constexpr DecodeConfig c = decode_batch_configs<A>()[I];
        if constexpr (c.head_dim == kHeadDim && c.dtype == kDtype)
          decode_batch_launchers(A)[I] = &launch_decode<c, true, A>;
      }(),
      ...);
  return true;
}

template <Arch A, HeadDim kHeadDim, Dtype kDtype>
bool register_decode_batch() {
  return register_decode_batch<A, kHeadDim, kDtype>(
      std::make_index_sequence<std::size(decode_batch_configs<A>())>{});
}

template <Arch A, HeadDim kHeadDim, Dtype kDtype, size_t... I>
bool register_prefill(std::index_sequence<I...>) {
  (
      [] {
        constexpr PrefillConfig c = prefill_configs<A>()[I];
        if constexpr (c.head_dim == kHeadDim && c.dtype == kDtype)
          prefill_launchers(A)[I] = &launch_prefill<c>;
      }(),
      ...);
  return true;
}

template <Arch A, HeadDim kHeadDim, Dtype kDtype>
bool register_prefill() {
  return register_prefill<A, kHeadDim, kDtype>(
      std::make_index_sequence<std::size(prefill_configs<A>())>{});
}

}  // namespace rdna
