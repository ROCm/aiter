// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Every prebuilt variant of the bf16-Q x fp8-K decode index-scoring pass.
//
// Instantiation only, exactly like pa_sparse_block_score_decode.cu: the table
// lives in topk_index_score.hpp and the entry point dispatches into it.
// FOURTEEN specialisations are built here: seven (H, Q) cells --
// {1,4} x {1,2,4,8} minus (4,8), which exceeds the 16 MFMA columns -- each on
// both AUX_K legs, 0 and 3. All seven cells are currently CERTIFIED, so
// opus_idx_score_cell_certified and opus_idx_score_cell_built coincide and the
// N16 bypass is unused.
//
// They are kept as SEPARATE predicates even while they agree, because they
// diverge the moment a cell is built ahead of its evidence -- which is the state
// this path was in for most of its life, and the state any future cell starts
// in.
//
// This comment said "eight cells, only the two (1,4) legs dispatchable" until a
// review caught it (finding D10). It had been stale since the table widened.
//
// RESOLVED STRUCTURALLY (2026-09-23): these sources moved into their OWN
// module, module_topk_index_score, whose flags carry no per-source glob at all.
// The hazard below was real while this file lived in module_msa_sparse_attention
// and was avoided only by the filename not matching; it is now out of that
// module's scope entirely. The P3 verification stays in the job scripts anyway
// -- a structural argument is stronger than a check, but it is not a reason to
// delete the check that would catch a regression in the structure.
//
// HISTORICAL NOTE (the hazard this replaced):
// FILENAME WAS LOAD-BEARING. module_msa_sparse_attention applies
// "-mllvm -amdgpu-mfma-vgpr-form=1" through the per-source glob
// "pa_sparse_block_score_*.cu" (optCompilerConfig.json). This file is named
// topk_index_score.cu and does not match that glob -- but that is
// NAMING LUCK, NOT A STRUCTURAL GUARANTEE (lead ruling,
// team-message-0a4d4bcc): a rename or a widened pattern reinstates the hazard.
// The flag must therefore be excluded by VERIFICATION at P3 -- dump the actual
// compile command for this source and confirm the flag is absent -- never
// concluded from this filename. If it is present: STOP and report. It moves
// register allocation on the bf16-MFMA path, and the accepted config sits at
// VGPR 61, THREE registers from dropping below 8 waves/SIMD, which is the
// occupancy premise the grid lever rests on.
#include "topk_index_score.hpp"

namespace aiter {
namespace sparse_attn {
OPUS_IDX_SCORE_TABLE(OPUS_IDX_SCORE_DEFINE)
} // namespace sparse_attn
} // namespace aiter
