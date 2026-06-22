// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// pybind for the dedicated CK-free KV-cache write module (module_cache_reshape).
// Exposes the same reshape_and_cache / reshape_and_cache_flash symbols as
// module_cache, but from a TU that builds on any arch (incl. gfx1201 RDNA4).
#include "aiter_stream.h"
#include "cache.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("reshape_and_cache",
          &aiter::reshape_and_cache,
          py::arg("key"),
          py::arg("value"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("slot_mapping"),
          py::arg("kv_cache_dtype"),
          py::arg("k_scale")    = std::nullopt,
          py::arg("v_scale")    = std::nullopt,
          py::arg("asm_layout") = false);
    m.def("reshape_and_cache_flash", &aiter::reshape_and_cache_flash);
}
