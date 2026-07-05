// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// arq fp4x2 (MXFP4) out, grouped + shuffle. Own translation unit (heaviest / most
// arch-specific: gfx950 pk_fp4, gfx1250 pk8_fp4 conversions).
#include "rmsnorm_opus_arq.hpp"

namespace aiter {
OPUS_ARQ_DEFINE(opus_arq_fp4, opus::fp4_t)
} // namespace aiter
