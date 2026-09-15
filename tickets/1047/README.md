# SILOTIGER-1047 notes

Harness pins, rocprof, and pasted tables live here from phase 1 onward.

## Phase 0

- Oracle: `aiter/ops/flydsl/kernels/qsa/oracle.py`
- Shapes: `aiter/ops/flydsl/kernels/qsa/shapes.py`
- Surface: `aiter/ops/flydsl/qsa.py` (no K1/K2 launcher yet)
- Gate: `HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py`

Tie-break: smaller block index on equal finite scores (`top_k_per_row_decode`).

Live AMD bar is vLLM **main** `qwen4_exp/amd/ops/qsa.py` (PR 53896 merged 2026-08-31). Record SHAs when the harness is wired.
