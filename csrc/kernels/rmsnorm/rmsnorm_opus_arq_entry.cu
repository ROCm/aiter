// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// module_rmsnorm_quant replacement C entrypoint. Dispatches out_code to the per-out-dtype
// arq launchers (each compiled in its own TU) so int8/fp8/fp4 build in parallel. Also
// holds the no-quant (out dtype == in dtype) launcher, which is small.
#include "rmsnorm_opus_arq.hpp"

#define OPUS_EXPORT extern "C" __attribute__((visibility("default")))

namespace aiter {
// no-quant: out dtype follows in dtype (out=in), so it can't use the fixed-OUT_T macro.
void opus_arq_noquant(OPUS_ARQ_PARAMS)
{
    if(in_code == 1)
        launch_arq_io<bf16_t, bf16_t>(OPUS_ARQ_ARGS);
    else
        launch_arq_io<fp16_t, fp16_t>(OPUS_ARQ_ARGS);
}
} // namespace aiter

// out_code: -1 no-quant, 0 int8, 1 fp8, 2 fp4x2. in_code: 0 fp16, 1 bf16.
// no-quant / per-token / grouped / fp4, fused-add (rin != 0), gemma, smooth (xscale != 0),
// shuffle_scale, strided rows.
OPUS_EXPORT void add_rmsnorm_quant_opus_raw(size_t out,
                                            size_t rout,
                                            size_t scale,
                                            size_t in,
                                            size_t rin,
                                            size_t weight,
                                            size_t xscale,
                                            float epsilon,
                                            int m,
                                            int n,
                                            float qmax,
                                            int in_code,
                                            int out_code,
                                            int in_s,
                                            int rin_s,
                                            int rout_s,
                                            int out_s,
                                            int group_size,
                                            int shuffle,
                                            int gemma,
                                            int cu_num,
                                            size_t stream)
{
    using namespace aiter;
    if(m <= 0 || n <= 0)
        return;
    auto* o    = reinterpret_cast<void*>(out);
    auto* ro   = reinterpret_cast<void*>(rout);
    auto* sc   = reinterpret_cast<void*>(scale);
    auto* i    = reinterpret_cast<const void*>(in);
    auto* ri   = reinterpret_cast<const void*>(rin);
    auto* w    = reinterpret_cast<const void*>(weight);
    auto* xsc  = reinterpret_cast<const void*>(xscale);
    auto s     = reinterpret_cast<hipStream_t>(stream);
    // dispatch on out_code -> per-dtype launcher (each in its own TU)
    if(out_code < 0)
        opus_arq_noquant(in_code, o, ro, sc, i, ri, w, xsc, epsilon, m, n, qmax, in_s, rin_s,
                         rout_s, out_s, group_size, shuffle, gemma, cu_num, s);
    else if(out_code == 0)
        opus_arq_i8(in_code, o, ro, sc, i, ri, w, xsc, epsilon, m, n, qmax, in_s, rin_s, rout_s,
                    out_s, group_size, shuffle, gemma, cu_num, s);
    else if(out_code == 1)
        opus_arq_fp8(in_code, o, ro, sc, i, ri, w, xsc, epsilon, m, n, qmax, in_s, rin_s, rout_s,
                     out_s, group_size, shuffle, gemma, cu_num, s);
    else
        opus_arq_fp4(in_code, o, ro, sc, i, ri, w, xsc, epsilon, m, n, qmax, in_s, rin_s, rout_s,
                     out_s, group_size, shuffle, gemma, cu_num, s);
}
