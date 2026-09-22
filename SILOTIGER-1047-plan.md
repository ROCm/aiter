# SILOTIGER-1047 — FlyDSL QSA indexer scorer + sparse GQA

Ship **FlyDSL** kernels for the Qwen Sparse Attention (QSA) block in
Qwen3.8-Flash-Next / Qwen4-preview (`qwen4_exp`): (1) paged indexer **scorer**
plus fused top-k, (2) **sparse GQA** attend on the selected tokens.

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040).
Ticket: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047).
Source dump: `SILOTIGER-1047.md`.

**Primary bar is whatever AMD serving actually launches today**, not the
unmerged AITER PR. That is vLLM-vendored Triton plus HIP top-k in
`qwen4_exp/amd/ops/qsa.py` (vLLM PR 53896). Beat that end-to-end on one QSA
layer (`indexer + select + attend`) before claiming a win.

**Secondary bar** is unmerged AITER Triton/Gluon in
[ROCm/aiter#4882](https://github.com/ROCm/aiter/pull/4882): beat its portable
Triton path, and beat its gfx950 Gluon path **on the shapes Gluon actually
dispatches**. #4882 is a competitor, not production.

QSA is **not** every layer: hybrid is `GGGQ` × 12, so **12 of 48** layers (plus
MTP, which can reuse indices).

**In scope:** FlyDSL K1 (scorer + fused top-k), FlyDSL K2 (sparse GQA), an
`aiter/ops/flydsl/` op surface, a correctness+perf harness that reports family A
and family B separately, and a vLLM `qwen4_exp` opt-in (`auto` / FlyDSL /
Triton).

**Out of scope:** GR (SILOTIGER-1041 / 1042); DSA / `fp8_mqa_logits` retarget;
sparse MLA prefill; SWA; FlashInfer / NVIDIA TRT-LLM QSA as an acceptance bar;
fusing `qsa_pre_indexer` before K1/K2 beat the live AMD bar.

Work the phases **in order**. Later items assume earlier ones have landed.
Leave checkboxes unchecked until that item is done; paste tables and notes under
the relevant phase as evidence.

## Progress

- [x] 0. Repo layout + oracle (no kernel yet)
- [x] 1. Harness: pin live AMD, #4882, fp32 oracle; measure who dominates
- [ ] 2. FlyDSL K1 (scorer + fused top-k) — family A then B
- [ ] 3. FlyDSL K2 (sparse GQA) — family A then B
- [ ] 4. Wire `aiter/ops/flydsl/` + vLLM `qwen4_exp` opt-in
- [ ] 5. Optional fusions (only after the bar)

## Locked decisions

These locks apply to **this ticket** unless a later note explicitly supersedes
them.

- **Skills (read, do not recall).** Before writing or reviewing FlyDSL
  for this ticket, Read
  `.claude/skills/flydsl-kernel-authoring/SKILL.md` and follow it.
  For `op_tests/test_flydsl_qsa.py`, also Read
  `.claude/skills/aiter-op-test/SKILL.md`.
  Do not start kernel code from memory of those skills.
  Cleanup/modernization of existing kernels uses
  `flydsl-kernel-code-cleanup`, not authoring, and is not a phase
  of this plan unless a later lock says otherwise.
- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 6**
  (`HIP_VISIBLE_DEVICES=6`).
- **Primary vs secondary.** The must-beat path is **live vLLM AMD** (Triton MQA
  + HIP top-k + Triton expand+tail + Triton sparse GQA `num_stages=1`). #4882
  Triton/Gluon is a named competitor column. NVIDIA
  `qwen4_exp/nvidia/ops/qsa.py` is not a harness column and is never the AMD
  gate.
- **Do not hide a loss.** Report each named backend separately. A win vs #4882
  does not cover a loss to live vLLM, or the reverse.
- **Family A is the production must-win.** Flash-Next / `Qwen/Qwen3.8-Flash-Next`:
  indexer `H=4`, `kv_heads=1`, `D=128`, `r=4`, budget 2048 → `K_B=512` blocks,
  expand+tail width ≤ 2051; sparse GQA **24×2** (group **12**), `head_dim=256`,
  partial RoPE 64, sigmoid output gate, **BF16** main QSA cache (not FP8 KV).
- **Family B is Gluon-parity only.** Keep #4882 Gluon-validated shapes so FlyDSL
  is measured where Gluon was tuned: indexer `H` is 4 or 8, `D=128`; sparse GQA
  `D=128`, group size **5**, `selection_width=2051`. Released Flash-Next GQA is
  group 12 / `D=256`, so **#4882 Gluon will not auto-dispatch there**. Family A
  vs #4882 is Triton-vs-FlyDSL; family B vs #4882 is Triton-and-Gluon-vs-FlyDSL.
- **Wrong kernels stay unused.** Do not retarget FlyDSL `fp8_mqa_logits`
  (`H % 16 == 0`, dense, weighted ReLU), MLA sparse decode, SWA, or DSA
  (`H=32` FP8 with per-head `w_h` over tokens). QSA is `H=4` BF16, no `w_h`,
  over **mean-pooled blocks**. Pad-to-16 of DSA is a prototype only.
- **K1 stays fused on ``visible <= 512``.** Stream paged compressed index-K,
  ReLU-sum per complete block, and emit ids without a score matrix. Same
  “score-plus-top-k” idea as DSA fused indexer work; different ABI.
- **Family A K1 may go unfused above 512 blocks.** The ticket requires the
  scores, the top-512, and expand+tail (lines 47-52); it does not require one
  kernel. Long rows score into an `[M, n_blocks]` fp32 buffer with BLOCK_N=32
  MFMA workgroups; single-request prefill batches 16 query rows per workgroup.
  Selection is `flydsl_top_k_per_row_decode(stable=True)` below 32768 columns
  and `topk_select(..., tie='low')` streaming radix at or above that width.
  Family B long rows reuse the same scorers; ``H=8`` is a second compile.
- **Family B K1 long-L uses the family A scorer.** Emit remains
  ``visible <= 512`` for ``H`` 4 and 8, plus the #4882 published indexer
  point. Longer rows dispatch into family A's BLOCK_N=32 MFMA scorer plus
  radix (``H=4`` same compile as family A; ``H=8`` a parameterized compile).
  The 2g bitonic tile merge is removed.
- **Score math.** `I_ib = sum_h ReLU(dot(q[h], k_bar[b]))` for complete blocks
  only (`p_b + r - 1 <= i`). Optional serving scale `1/sqrt(128)` is allowed
  **only if it cannot change top-k argmax**. `eps` is unused.
- **Top-k tie-break.** On equal finite scores, keep the **smaller block
  index** (live AMD HIP `top_k_per_row_decode`). Incomplete blocks are `-inf`
  and are not selected. Remaining slots are `-1`.
- **Two kernels, one op surface.** K1 = scorer + fused top-k; K2 = sparse GQA.
  Expand+tail may live in K1’s epilogue or K2’s prologue until phase 5. Public
  wrappers under `aiter/ops/flydsl/`; kernels under
  `aiter/ops/flydsl/kernels/` (QSA-named files, not a reuse of `mqa_logits/`).
- **Arch.** Compile **gfx942 and gfx950** separately if LDS/VGPR models differ.
  gfx950 should use the extra LDS (live AMD and #4882 both left sparse GQA
  `num_stages=1` on the vLLM AMD path). Do not union decode GEMV and prefill
  MFMA in one instantiation if that costs occupancy.
- **Shapes in the harness.** `M` is flattened tokens. Decode `1..8` and prefill
  512 / 2048 / 8192 plus at least one long-context length (**32k or 128k**) so
  indexer scaling is visible. Both archs.
- **Correctness.** Independent fp32 oracle: block-causal ReLU-sum scores, exact
  top-512 (tie-break documented), expand+tail, then standard GQA on those
  positions. Cross-check vs vLLM Triton `qsa.py` and vs #4882 on shared shapes.
  Authoritative math: Qwen3.8-Next tech report §2.1 / QSA. Fusing top-k may
  change fp32 score order vs materializing full logits; gate K1 on the oracle
  with a documented tolerance, and require **set equality** of selected blocks
  (or a fixed tie policy).
- **Two-layer tests in `op_tests/test_flydsl_qsa.py`.** Same split as the
  667 warp-decode file: one module, two runners. Pytest collects `test_*`;
  `@benchmark` sweeps must **not** be named `test_*`.
  - **Correctness (pytest gate):** zero-arg or `@pytest.mark.parametrize`
    unit cases (`test_*`). They may call the oracle / plumbing with tiny
    shapes. Do not put required shape args on a `test_*` without parametrize
    — `@benchmark` hides the inner signature, so pytest will call it with
    no arguments and `log_args` will raise.
  - **Perf sweep (`__main__`):** `@benchmark` + `run_perftest` candidate
    loop, named `bench_*` (e.g. `bench_qsa_family_a_plumbing`). Torch/oracle
    **not** timed into the table. `us` + TFLOPS + TB/s + `err` per candidate.
    One markdown summary table per bench fn. `__main__` guard; `get_gfx()`
    gate in `main()`. No hand-written ratio columns. Family A and family B
    are **separate tables**.
  Both commands below must stay green. `python -m pytest` is the fast
  correctness gate; the script is the sweep. A file that only works as
  `python3 op_tests/test_flydsl_qsa.py` is incomplete.
- **#4882 pin.** Do not wait for merge to start the harness. Until it lands,
  pin a PR head (runtime-tested parent `2462d5b64`; later heads may be
  docs-only). Forced `gluon` on a dispatch miss must error; `auto` falls back
  to Triton. If #4882 merges, retarget the competitor pin to `main`.
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale HSACO
  cannot mask a bad rewrite.
- **Warm-up / claims.** Warm up by duration. Interleave paired rounds. Do not
  claim a GQA win from the group-5 `D=128` Gluon bench, and do not claim an
  indexer win from DSA `H=32` FP8 numbers. End-to-end gate is the **whole
  chain** (launches, bytes, and fused K1/K2), not a single kernel vs its Triton
  twin in isolation.
- **Cache policy (`--rotate`).** Perf rows share one `run_perftest`
  `num_rotate_args` across every named backend. Default `[1]` is hot (reuse
  one buffer set; existing GPU-6 tables). `0` auto-sizes extra copies from
  L2; `N>1` is explicit copies. Timed calls pass paged caches as args so
  deepcopy actually clones them — zero-arg closures cannot rotate. Do not
  compare a cold FlyDSL cell to a hot rival or to #4882's published
  `do_bench` µs. HIP-graph replay stays hot (addresses are captured).
- **AITER fused MoE / tgemm / vision FA** may be on in the same process; they
  are **not** QSA baselines.
- **Gate (after a kernel exists).** Both layers:
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 -m pytest op_tests/test_flydsl_qsa.py -q
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 op_tests/test_flydsl_qsa.py
  ```
  Paths may move; keep this snippet in sync. Until the op_test exists, the
  phase-1 harness command is the gate. Phase 0/1 unit cases must pass
  pytest even before a kernel exists.

## Subtasks

### 0. Repo layout + oracle (no kernel yet)

Stand up files and the math reference so later kernels have a single ABI to
hit. Do not start FlyDSL K1 until the oracle matches the ticket math on a
tiny dense case.

Proposed layout (adjust only if a later lock says so):

- `aiter/ops/flydsl/kernels/qsa/` — K1 / K2 kernels
- `aiter/ops/flydsl/qsa.py` — public wrappers + arch dispatch
- `op_tests/test_flydsl_qsa.py` — pytest `test_*` unit cases + `__main__`
  `bench_*` family A / B tables
- `tickets/1047/` — harness notes, rocprof, competitor pins, pasted tables

- [x] Oracle: block-causal ReLU-sum, documented top-k tie-break, expand+tail,
      GQA on selected positions (fp32, then cast).
- [x] Document family A / B tensor shapes and dtypes in one comment block on
      the wrapper (or a tiny `qsa_shapes.py`) so tests and kernels share them.
- [x] **Done when:** oracle is importable, covered by a small **pytest**
      unit case, and agrees with a hand-checked 1-row / few-block example
      from the tech report formula.

  Layout: `aiter/ops/flydsl/kernels/qsa/{shapes,oracle}.py`,
  `aiter/ops/flydsl/qsa.py`, `op_tests/test_flydsl_qsa.py`, `tickets/1047/`.
  Gate: `HIP_VISIBLE_DEVICES=6 python3 -m pytest op_tests/test_flydsl_qsa.py -q`
  (CPU-safe `test_*`; no kernel compile). Script `__main__` may also run
  those unit cases, but pytest is the correctness gate.

  Layout: `aiter/ops/flydsl/kernels/qsa/{shapes,oracle}.py`,
  `aiter/ops/flydsl/qsa.py`, `op_tests/test_flydsl_qsa.py`, `tickets/1047/`.
  Gate: `HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py` (CPU-safe
  unit cases; no kernel compile).

### 1. Harness: pin live AMD, #4882, fp32 oracle; measure who dominates

This phase answers **which kernel to land first** at 8k / 32k / 128k / 1M
(open question in the ticket). No FlyDSL win is claimed here. Land as
separate steps (plumbing → live AMD → #4882 → rocprof).

- [x] Family A plumbing only: paged indexer-K and GQA K/V, shuffled
      `block_table`, `M`/`L` sweep, oracle on dense vs gathered. No
      competitor kernels. Sweep lives in `bench_*`, not `test_*`. Gate:
      pytest `-q` then `HIP_VISIBLE_DEVICES=6 python3 op_tests/test_flydsl_qsa.py`
- [x] Pin **vLLM AMD live path** (`qwen4_exp/amd/ops/qsa.py` + HIP
      `top_k_per_row_decode`). Record the exact vLLM / AITER SHAs in
      `tickets/1047/`. Vendored Triton subset:
      `aiter/ops/triton/_triton_kernels/attention/qsa_vllm_amd.py`
      (vLLM `836bb3839ffe`). HIP top-k via
      `aiter.ops.topk._hip_top_k_per_row_decode` (not FlyDSL top-k) when
      `module_top_k_per_row.so` is present. Phase 1 tables were recorded
      without that module (oracle tie-break on MQA logits).
- [x] Pin **#4882 Triton** (`qsa_paged_mqa_logits` / expand /
      `qsa_sparse_paged_gqa`) onto family A paged tensors. PR head
      `150c7bc12b45`; Triton-only launchers in
      `aiter/ops/triton/attention/qsa_4882.py`. Separate markdown table vs
      oracle. Gluon not imported.
- [x] Pin **#4882 Gluon** (gfx950, Triton `>= 3.6`, forced `backend="gluon"`
      on family B: indexer H 4/8 D=128, GQA group 5 / D=128 / width 2051).
      Family A GQA is not launched on Gluon. Kernels:
      `aiter/ops/triton/_gluon_kernels/gfx950/attention/qsa_{paged_mqa_logits,sparse_paged_gqa}.py`.
- [x] Family A table and family B table; never merge them. Family A:
      plumbing, live AMD, #4882 Triton. Family B: #4882 Triton and Gluon
      (`bench_qsa_family_b_4882_triton` is a separate table). Pasted in
      `tickets/1047/README.md` phase 1f (`M∈{1,8}`, `L∈{512,8192,32768}`).
- [x] rocprof **one real QSA layer** (indexer through GQA) at short and long
      `L`, including HIP graph replay at decode. Driver:
      `tickets/1047/profile_qsa_layer.py`; notes in `tickets/1047/README.md`
      (phase 1e). Full layer HIP graph captured at decode `M=1`.
- [x] Record whether **indexer or GQA dominates** on this GPU at 8k / 32k /
      128k (and 1M if the machine can hold it). **No swap of phases 2 vs 3**
      was the call at 1e. HIP `module_top_k_per_row.so` was **absent**; decode
      select wall was oracle/`torch.topk` on MQA logits, not
      `_hip_top_k_per_row_decode`. rocprof MQA was ~3 µs. That ranking does
      **not** apply to production HIP select (later select-only 2d times are
      a different measurement, not a new full-layer rocprof). Prefill
      `M=512` under the fallback: GQA slightly ahead at 8k; select ahead at
      32k. 128k decode fits; 1M not run.
- [x] **Done when:** both family tables exist with live AMD + oracle + #4882
      where it dispatches; a short note states which side of QSA dominates at
      the locked lengths on GPU 6.

### 2. FlyDSL K1 (scorer + fused top-k) — family A then B

K1 streams paged compressed index-K tiles, computes the ReLU-sum score, keeps
a local top-k, merges globally. Output: `block_ids [M, 512]` on family A
(or local k on B). Expand+tail can still be a separate launch in this phase.

- [ ] Family A (`H=4`, `D=128`, `k=512`, paged compressed blocks, complete-block
      causal bound).
- [x] **2a.** Family A decode kernel, correctness only: paged stream + local
      top-512, `block_ids [M, 512]`, no global score matrix; oracle **set
      equality** on short `L` (`n_blocks <= 512`); `us` vs live AMD recorded
      without a win claim. Expand still a separate launch.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Set equality `err=0`.
After the FlyDSL authoring pass, a static TV-layout tiled copy stages Q in LDS,
BF16 K rows use 128-bit buffer-copy fragments, and a parallel bitonic merge
keeps the best 512 of 1024 candidates. Times are **not** a win claim:

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 97.7 | 43.5 | 0 | 0 |
| 8 | 512 | 128 | 102.0 | 54.5 | 0 | 0 |
| 1 | 2048 | 512 | 117.1 | 51.0 | 0 | 0 |
| 8 | 2048 | 512 | 121.5 | 67.4 | 0 | 0 |

This env still lacks `module_top_k_per_row.so`; the live AMD column uses the
oracle tie-break on vLLM MQA logits, same as phase 1. These AMD microseconds
are **not** the 2d bar.

- [x] **2b.** Long-`L` merge in the same kernel: 512-slot tiles, running
      top-512 in LDS, no `[M, n_blocks]` score buffer, no call into
      `topk_per_row_*`. Oracle set equality at 8k / 32k / 128k. Times
      recorded, not a win claim.

**Superseded by 2d.** The no-score-buffer / no-`topk_per_row_*` rule stated
above was dropped for family A long rows: the bitonic merge lost to an
unfused scorer plus `flydsl_top_k_per_row_decode`. Kept as the historical
record of the fused merge and its measurements.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. `err=0`. Each tile is
merged with the running top-512 by a 55-stage in-LDS bitonic network:

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 97.7 | 43.5 | 0 | 0 |
| 8 | 512 | 128 | 102.0 | 54.5 | 0 | 0 |
| 1 | 2048 | 512 | 117.1 | 51.0 | 0 | 0 |
| 8 | 2048 | 512 | 121.5 | 67.4 | 0 | 0 |
| 1 | 8192 | 2048 | 459.8 | 56.0 | 0 | 0 |
| 8 | 8192 | 2048 | 479.6 | 63.5 | 0 | 0 |
| 1 | 32768 | 8192 | 1844.8 | 69.4 | 0 | 0 |
| 8 | 32768 | 8192 | 1889.8 | 152.4 | 0 | 0 |
| 1 | 131072 | 32768 | 7415.0 | 96.2 | 0 | 0 |
| 8 | 131072 | 32768 | 7651.1 | 278.7 | 0 | 0 |

This env still lacks `module_top_k_per_row.so`; the live AMD column uses the
oracle tie-break on vLLM MQA logits. These AMD microseconds are **not** the
2d bar.

- [x] **2c.** Prefill `M=512` uses the **same** wave64-per-row instantiation.
      Occupancy did not die vs decode, so there is no second compile. Separate
      prefill table; oracle set equality; times recorded, not a win claim.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. `err=0`. Wall time at
`M=512` matches decode `M=1` at the same `L` (the GPU was idle at decode):

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 512 | 512 | 128 | 97.2 | 72.0 | 0 | 0 |
| 512 | 2048 | 512 | 117.3 | 92.0 | 0 | 0 |
| 512 | 8192 | 2048 | 458.4 | 161.1 | 0 | 0 |
| 512 | 32768 | 8192 | 1826.2 | 509.2 | 0 | 0 |

This env still lacks `module_top_k_per_row.so`; the live AMD column uses the
oracle tie-break on vLLM MQA logits. These AMD microseconds are **not** the
2d bar.

- [x] **2d.** Family A K1 keeps fused emit for ``n_blocks <= 512``. Long rows
      materialize fp32 scores with BLOCK_N=32 BF16 MFMA workgroups, then call
      decode radix below 32768 columns and streaming radix
      (`topk_select`, ``tie='low'``) at or above that width. BLOCK_N=16
      remains buildable but lost on prefill. Both paths return `block_ids [M, 512]`.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Oracle set equality `err=0`
on both columns. `aiter/jit/module_top_k_per_row.so` is loaded
(`import [module_top_k_per_row]`; oracle-fallback warning did not fire).
The AMD column is Triton MQA + `_hip_top_k_per_row_decode` + expand.

The FlyDSL column keeps the **`n_blocks <= 512` fast path**: every complete
block is in the top-512, so it writes ids without scoring. For longer rows,
the old single-workgroup bitonic dispatch is replaced by a global
`[M, n_blocks]` fp32 score buffer. The scorer pads H=4 to an MFMA 16-row
tile, splits D=128 across two waves, and reduces the partials before summing
ReLU across the four real heads. BLOCK_N=32 beat BLOCK_N=16 at every
material prefill point. Single-request prefill instead uses the MFMA M
dimension for 16 query rows, reuses each K tile across those rows, and keeps
the one-row scorer for decode and multi-request inputs. Selection is
`flydsl_top_k_per_row_decode(stable=True)` until 32768 columns, then streaming
radix. Expand remains separate.

A 64-bit MSD binary radix-select on the 1024-candidate tile (score order,
then inverted id, 32 bits each, wave reduce + two barriers per bit) was
measured and **not shipped**. Set equality held. Decode ``M=1`` at 8k /
32k was ~126 / ~497 µs vs the kept bitonic ~106 / ~419 µs in the 2d
HIP-loaded table. Sixty-four digit passes cost more barriers than the
55-stage sort.

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 2.0 | 9.5 | 0 | 0 |
| 1 | 2048 | 512 | 2.0 | 10.3 | 0 | 0 |
| 1 | 8192 | 2048 | 10.5 | 19.1 | 0 | 0 |
| 8 | 8192 | 2048 | 12.7 | 22.0 | 0 | 0 |
| 1 | 32768 | 8192 | 14.1 | 20.3 | 0 | 0 |
| 8 | 32768 | 8192 | 19.4 | 25.2 | 0 | 0 |
| 1 | 131072 | 32768 | 28.3 | 30.0 | 0 | 0 |
| 8 | 131072 | 32768 | 46.3 | 52.9 | 0 | 0 |
| 512 | 512 | 128 | 3.5 | 18.1 | 0 | 0 |
| 512 | 2048 | 512 | 3.6 | 29.1 | 0 | 0 |
| 512 | 8192 | 2048 | 29.3 | 83.1 | 0 | 0 |
| 512 | 32768 | 8192 | 68.2 | 247.7 | 0 | 0 |

Streaming radix at 32768 columns moves 128k decode from 33.7 / 54.4 us to
28.3 / 46.3 us (`M=1` / `M=8`) and crosses live AMD. The 16-row scorer moves
8k / 32k prefill from 83.2 / 288.7 us to 29.3 / 68.2 us and beats live AMD
at both points.

- [ ] Family B (`H` 4 or 8, Gluon-validated indexer shapes).
- [x] **2e.** Family B decode kernel, ``H=4`` only, correctness: paged
      emit when ``n_blocks <= 512``, `block_ids [M, 512]`, no global score
      matrix; oracle **set equality**; `us` vs #4882 Triton (and Gluon on
      gfx950) recorded without a win claim. Expand still a separate launch.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Set equality `err=0`.
HIP `module_top_k_per_row.so` is loaded. Separate table from family A.
Times are **not** a win claim (2h). ``H=8`` is 2f; long-`L` is 2g.

| m | seq_len | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 1 | 512 | 128 | 2.0 | 13.1 | 13.2 | 0 |
| 8 | 512 | 128 | 3.0 | 15.8 | 15.7 | 0 |
| 1 | 2048 | 512 | 2.0 | 10.2 | 10.4 | 0 |
| 8 | 2048 | 512 | 3.0 | 11.4 | 11.5 | 0 |

- [x] **2f.** Family B ``H=8`` is a **second compile**, same emit path as 2e
      (``n_blocks <= 512``), oracle set equality; `us` vs #4882 Triton and
      Gluon recorded without a win claim.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Set equality `err=0`.
``H=4`` and ``H=8`` are separate compiles. Times are **not** a win claim
(2h).

| m | seq_len | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 1 | 512 | 128 | 2.0 | 13.8 | 13.2 | 0 |
| 8 | 512 | 128 | 2.8 | 16.2 | 15.5 | 0 |
| 1 | 2048 | 512 | 2.0 | 11.0 | 10.3 | 0 |
| 8 | 2048 | 512 | 2.9 | 12.1 | 11.4 | 0 |

- [x] **2g.** Long-`L` merge in the same family B kernel: 512-slot tiles,
      running top-512 in LDS, no `[M, n_blocks]` score buffer. Oracle set
      equality at 8k / 32k / 128k for ``H`` 4 and 8. Times vs #4882 recorded;
      **not** a win claim.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. `err=0`. Same bitonic
tile merge as family A 2b. ``n_blocks > 512`` loses to #4882 MQA + HIP
radix. Do not chase a select win. 2h is emit / short-L only.

| m | seq_len | H | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|--:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 1 | 8192 | 4 | 2048 | 106.4 | 16.3 | 16.2 | 0 |
| 8 | 8192 | 4 | 2048 | 107.7 | 18.3 | 18.1 | 0 |
| 1 | 32768 | 4 | 8192 | 418.8 | 18.4 | 18.4 | 0 |
| 8 | 32768 | 4 | 8192 | 422.9 | 23.1 | 21.6 | 0 |
| 1 | 131072 | 4 | 32768 | 1716.1 | 27.7 | 27.2 | 0 |
| 8 | 131072 | 4 | 32768 | 1734.2 | 49.5 | 47.7 | 0 |
| 1 | 8192 | 8 | 2048 | 123.4 | 17.0 | 16.2 | 0 |
| 8 | 8192 | 8 | 2048 | 124.8 | 19.4 | 18.2 | 0 |
| 1 | 32768 | 8 | 8192 | 487.2 | 19.4 | 18.5 | 0 |
| 8 | 32768 | 8 | 8192 | 492.2 | 26.4 | 22.1 | 0 |
| 1 | 131072 | 8 | 32768 | 2022.0 | 29.9 | 27.4 | 0 |
| 8 | 131072 | 8 | 32768 | 2036.2 | 56.2 | 45.3 | 0 |

- [x] **2h.** Family B K1 beats #4882 Triton and Gluon on emit /
      ``visible <= 512`` (``H`` 4 and 8) and on the published indexer point
      (``M=32``, ``H=4``, ``D=128``, ``page_size=8``, ``n_blocks=512``).
      Same `block_ids [M, 512]`; no score matrix. Long-L is 2i (family A
      scorer), not a 2h gate.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Oracle set equality
`err=0`. Emit already beats both #4882 columns in 2e/2f. Published point
maps ``pages=512`` to 512 compressed keys packed at ``page_size=8``
(64 pages, ``L=2048``), so emit applies. Expand still separate.

| m | seq_len | page_size | H | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us | flydsl_k1 err |
|--:|--------:|----------:|--:|---------:|-------------:|----------------------:|---------------------:|--------------:|
| 32 | 2048 | 8 | 4 | 512 | 2.9 | 12.1 | 11.9 | 0 |
- [x] Family A may materialize `[rows, n_blocks]` FP32 scores for long rows;
      short rows retain fused emit. Family B long rows reuse those scorers
      (``H=8`` is a second compile).
- [x] **2i.** Family B long-L drops the 2g bitonic and dispatches into
      family A's MFMA scorer plus radix. ``H=4`` shares family A's compile;
      ``H=8`` is a parameterized second compile. Emit kernels no longer
      contain a bitonic path. GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`.
      Oracle set equality. 8k / 32k beat both #4882 columns. Decode
      ``M=1`` 128k is a small loss vs Gluon (same regime as family A vs
      #4882 Triton).

| m | seq_len | H | n_blocks | flydsl_k1 us | 4882_triton_select us | 4882_gluon_select us |
|--:|--------:|--:|---------:|-------------:|----------------------:|---------------------:|
| 1 | 512 | 4 | 128 | 2.0 | 13.5 | 13.3 |
| 1 | 8192 | 4 | 2048 | 10.5 | 16.3 | 16.2 |
| 1 | 32768 | 4 | 8192 | 14.2 | 18.4 | 18.5 |
| 1 | 131072 | 4 | 32768 | 28.6 | 28.0 | 27.2 |
| 8 | 512 | 4 | 128 | 2.6 | 15.8 | 15.8 |
| 8 | 8192 | 4 | 2048 | 12.8 | 18.2 | 18.1 |
| 8 | 32768 | 4 | 8192 | 19.8 | 23.4 | 21.9 |
| 8 | 131072 | 4 | 32768 | 46.5 | 50.7 | 48.1 |
| 1 | 512 | 8 | 128 | 2.0 | 14.2 | 13.3 |
| 1 | 8192 | 8 | 2048 | 10.7 | 17.0 | 16.3 |
| 1 | 32768 | 8 | 8192 | 14.4 | 19.4 | 18.6 |
| 1 | 131072 | 8 | 32768 | 28.6 | 30.0 | 27.5 |
| 8 | 512 | 8 | 128 | 2.7 | 16.3 | 15.6 |
| 8 | 8192 | 8 | 2048 | 12.8 | 19.4 | 18.1 |
| 8 | 32768 | 8 | 8192 | 19.9 | 26.2 | 21.9 |
| 8 | 131072 | 8 | 32768 | 44.9 | 56.4 | 45.1 |
- [ ] gfx942 and gfx950.
- [ ] Gate vs live vLLM AMD (`MQA Triton + HIP top-k`) and vs #4882 Triton;
      beat Gluon on gfx950 **where Gluon dispatches**.
- [ ] **Done when:** selected-block **set equality** (or documented tie policy)
      vs the oracle; family A beats live AMD on emit / ``visible <= 512`` and
      through 32k decode; family B beats #4882 Triton and Gluon on emit /
      ``visible <= 512``, the published indexer point, and 8k / 32k long-L
      (128k ``M=1`` vs Gluon remains a small recorded loss).

### 3. FlyDSL K2 (sparse GQA) — family A then B

K2 attends uncompressed paged K/V at the expanded token indices. Prefetch K/V
for the selected runs (512 runs of 4 plus a short tail on A; 2051-wide list on
B). Split-K as needed.

- [ ] Family A: 24 Q / 2 KV, group 12, `D=256`, softmax scale, sigmoid gate
      weights if fused into the epilogue. Out: BF16 `o [M, 24, 256]`
      (pre-`o_proj`).
- [x] **3a.** Family A decode kernel: one WG per ``(row, kv_head)``, paged
      gather of selected K/V, online softmax, preallocated ``o [M, 24, 256]``.
      Pytest vs ``qsa_sparse_gqa`` (``checkAllclose``, rtol/atol ``1e-2``).
      ``bench_qsa_family_a_k2`` vs live AMD ``qsa_sparse_paged_attention``
      (``num_stages=1``). Times recorded, **no win claim**. Expand, RoPE, and
      sigmoid stay unfused. Prefill is 3c; split-K is 3b.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. `err=0` vs the oracle on
decode ``M∈{1,8}``. Width is the expanded list (2051), so GQA flops do not
grow with ``L``. The kernel is a correctness-first GEMV (one thread per
``D``, 12 sequential block reductions per token) and is ~1000× the live AMD
column; 3b/3d own occupancy and the win.

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | flydsl_k2 err | vllm_amd_gqa err |
|--:|--------:|---------:|------:|-------------:|----------------:|--------------:|-----------------:|
| 1 | 512 | 128 | 2051 | 10638.0 | 11.0 | 0 | 0 |
| 8 | 512 | 128 | 2051 | 10685.7 | 14.4 | 0 | 0 |
| 1 | 2048 | 512 | 2051 | 10685.4 | 11.2 | 0 | 0 |
| 8 | 2048 | 512 | 2051 | 10839.6 | 15.7 | 0 | 0 |
| 1 | 8192 | 2048 | 2051 | 10684.5 | 11.3 | 0 | 0 |
| 8 | 8192 | 2048 | 2051 | 10968.6 | 16.3 | 0 | 0 |
| 1 | 32768 | 8192 | 2051 | 10706.1 | 11.2 | 0 | 0 |
| 8 | 32768 | 8192 | 2051 | 11013.9 | 16.2 | 0 | 0 |

- [x] **3b.** Split-K plus LSE merge on the same 3a ABI (no second math path).
      Host split count follows live AMD decode occupancy (64 splits when
      ``M * Hk <= 8``). Pytest vs oracle; decode times vs live AMD recorded,
      **no win claim**. Prefill is 3c.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. `err=0`. Decode ``M=1``
drops from 3a's ~10.6 ms to ~103 µs (~100×) but remains ~10× live AMD
(``~10 µs``). ``M=8`` is ~246 µs vs ~16 µs. 3d owns the win (vector K/V
tiles / MFMA), not more splits.

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | flydsl_k2 err | vllm_amd_gqa err |
|--:|--------:|---------:|------:|-------------:|----------------:|--------------:|-----------------:|
| 1 | 512 | 128 | 2051 | 101.3 | 10.0 | 0 | 0 |
| 8 | 512 | 128 | 2051 | 236.8 | 13.3 | 0 | 0 |
| 1 | 2048 | 512 | 2051 | 103.3 | 10.2 | 0 | 0 |
| 8 | 2048 | 512 | 2051 | 247.9 | 16.0 | 0 | 0 |
| 1 | 8192 | 2048 | 2051 | 103.6 | 10.3 | 0 | 0 |
| 8 | 8192 | 2048 | 2051 | 245.7 | 16.2 | 0 | 0 |
| 1 | 32768 | 8192 | 2051 | 103.7 | 10.2 | 0 | 0 |
| 8 | 32768 | 8192 | 2051 | 245.0 | 16.2 | 0 | 0 |

- [x] **3c.** Prefill ``M=512`` uses the **same** split-K instantiation.
      Occupancy of the GEMV (1024 WGs at ``splits=1``) did not require a
      second compile. Do **not** union decode GEMV and prefill MFMA. Separate
      prefill table; oracle ``checkAllclose``; times recorded, not a win claim.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. `err=0`. Host split
count matches live AMD (``M * Hk > 512`` → 1 split), so each WG walks the
full 2051-wide list. Wall is ~11.7 ms vs live AMD ~200–285 µs, and vs
decode 3b ~103 µs at ``M=1`` (64 splits). 3d owns tiles/MFMA.

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | flydsl_k2 err | vllm_amd_gqa err |
|--:|--------:|---------:|------:|-------------:|----------------:|--------------:|-----------------:|
| 512 | 512 | 128 | 2051 | 11690.7 | 200.7 | 0 | 0 |
| 512 | 2048 | 512 | 2051 | 11709.1 | 211.8 | 0 | 0 |
| 512 | 8192 | 2048 | 2051 | 11831.8 | 246.8 | 0 | 0 |
| 512 | 32768 | 8192 | 2051 | 11884.8 | 284.6 | 0 | 0 |

- [ ] **3d.** Family A ``BLOCK_N=16`` QK/PV MFMA (group padded to 16),
      128-bit paged K/V gather (one page translate per column owner),
      same split-K ABI. gfx950 QK uses ``16x16x32``; gfx942 retains
      ``16x16x16``. Softmax runs on wave 0 off the reduced C fragment,
      so the ``s`` LDS tile is gone. A **loop-top barrier** keeps the
      next gather from overwriting ``k``/``v``/``p`` LDS while the
      previous tile's PV is still reading them — without it prefill
      ``M=512`` shows ~0.05% of elements outside ``1e-2``. One wave
      computes split weights in the merge. The next K/V tile is prefetched
      into registers before current-tile QK and carried across the runtime
      loop. Decode beats #4882 Triton GQA but **does not** beat live AMD
      (~1.5–1.8× at ``M=8``, ~1.8–1.9× at ``M=1``). Prefill is ~2×
      AMD/#4882. Not a win claim; gfx950 extra LDS remains 3g.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. `err=0` on decode and
prefill. Host splits keep decode at 64 and cap prefill at 8. Register
prefetch improves the vectorized-gather point by ~3–5% at decode and
~4–5% at prefill; it does not close the gap to live AMD.

Decode:

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | 4882_triton_gqa us | flydsl_k2 err |
|--:|--------:|---------:|------:|-------------:|----------------:|-------------------:|--------------:|
| 1 | 512 | 128 | 2051 | 18.7 | 10.0 | 112.9 | 0 |
| 8 | 512 | 128 | 2051 | 23.0 | 13.3 | 115.5 | 0 |
| 1 | 2048 | 512 | 2051 | 19.1 | 10.3 | 115.9 | 0 |
| 8 | 2048 | 512 | 2051 | 24.5 | 16.0 | 126.3 | 0 |
| 1 | 8192 | 2048 | 2051 | 19.1 | 10.3 | 117.4 | 0 |
| 8 | 8192 | 2048 | 2051 | 24.7 | 15.1 | 121.9 | 0 |
| 1 | 32768 | 8192 | 2051 | 19.0 | 10.2 | 118.7 | 0 |
| 8 | 32768 | 8192 | 2051 | 25.0 | 15.4 | 123.1 | 0 |

Prefill:

| m | seq_len | n_blocks | width | flydsl_k2 us | vllm_amd_gqa us | 4882_triton_gqa us | flydsl_k2 err |
|--:|--------:|---------:|------:|-------------:|----------------:|-------------------:|--------------:|
| 512 | 512 | 128 | 2051 | 516.5 | 203.2 | 222.2 | 0 |
| 512 | 2048 | 512 | 2051 | 530.7 | 214.1 | 236.9 | 0 |
| 512 | 8192 | 2048 | 2051 | 546.5 | 248.3 | 260.3 | 0 |
| 512 | 32768 | 8192 | 2051 | 548.7 | 285.7 | 281.5 | 0 |

A second gfx950 ``BLOCK_N=64`` compile was measured and **not shipped**.
Live AMD's prefill rule (``BLOCK_N=64``, ``splits=1``) is ~1.01–1.17 ms
here; keeping the 8-split cap is ~1.12–1.29 ms. Both lose to the
``BLOCK_N=16`` / 8-split kernel (~0.54–0.57 ms). The 64-wide tile needs
~90 KB LDS (over gfx942's 64 KB) and four N-subtiles of QK/PV. Decode
stays ``BLOCK_N=16``.

Folding the P LDS barrier by running softmax on every wave and
``shuffle_idx``-transposing P into the PV A fragment was measured and
**not shipped**. Decode was flat-to-worse (~19.8–20.1 µs at ``M=1``).
Prefill ~0.66 ms vs the kept ~0.52–0.55 ms, with ~0.002–0.007% of
elements outside ``1e-2`` at ``M=512``. The four-wave ``exp`` plus
per-lane shuffles cost more than the barrier they replaced. Loop-top +
K/V + C + P barriers stay.

128-bit ``partial_out`` loads in the LSE merge (64-thread wave, 8-wide
along ``D``) were measured and **not shipped**. Decode ``M=1`` went to
~24.5 µs from the kept ~18.7–19.1 µs; prefill was only a small win
(~0.51–0.54 ms). Scalar per-``D`` merge stays.

The original ``BLOCK_N=16`` kernel was then replaced with a
live-AMD-shaped FlyDSL port. One builder emits two specializations:
decode follows the live ``BLOCK_N=16`` / 256-thread split policy, while
prefill uses ``BLOCK_N=64`` / 128 threads / ``splits=1`` and writes
``out`` directly without launching merge. Splits divide complete
``BLOCK_N`` tiles, and one column owner translates each logical token
before sharing page/off/live through LDS. The tile loop gathers paged K then V; gfx942 aliases those tiles so
BLOCK_N=64 stays under 64 KiB, while gfx950 keeps separate K and V
buffers (~75 KiB prefill). QK is gfx950 K32 / gfx942 K16 and PV is
K16. Online softmax stays in log2 space. Split partials are FP32 and
merge on two waves. Q MFMA fragments stay in registers. The old split
and merge kernels are no longer callable.
The old split and merge kernels are no longer callable.

GPU 6 / gfx950 / ``FLYDSL_RUNTIME_ENABLE_CACHE=0``: all 18 QSA pytest
cases pass, including decode and prefill oracle checks at ``1e-2``.
The standard ``L=512``, width-2051 bench is correct but slower than both
the replaced kernel and live AMD:

| m | flydsl port µs | live AMD µs | #4882 Triton µs | err |
|--:|----------------:|------------:|-----------------:|----:|
| 1 | 23.3 | 9.9 | 112.8 | 0 |
| 8 | 22.2 | 13.5 | 115.4 | 0 |
| 512 | 722.2 | 202.7 | 222.3 | 0 |

This completes the structural port, not the phase-3 performance gate.
An authoring-alignment pass replaced scalar K/V LDS stores with
``make_tiled_copy`` + ``UniversalCopy128b``, kept V row-major rather than
scalar-transposing it, changed wave/lane mapping to ``idx2crd``, and moved
QK/PV issue to ``fx.gemm``. GPU 6 / gfx950 remains correct (18 pytest
cases, ``err=0``): width-2051 ``L=512`` is 23.8 / 24.1 / 588.1 us at
``M=1/8/512`` versus the first port's 23.3 / 22.2 / 722.2 us. Thus the
blocked LDS stores improve prefill ~19% but do not improve decode.
Occupancy ISA on GPU 6 / gfx950 at width 2051 (the pytest decode case
is ``ns=1`` and is not this dump):

| kernel | ds_write b128/b32/b16 | ds_read b128/b64/b32 | barrier | MFMA | VGPR | LDS B |
|---|---:|---:|---:|---:|---:|---:|
| decode split `bn16_blk256_ns32` | 5 / 15 / 4 | 4 / 1 / 12 | 6 | 6 | 79 | 21376 |
| decode merge `ns32_blk128` | 0 / 2 / 0 | 8 / 0 / 1 | 1 | 0 | 112 | 260 |
| prefill split `bn64_blk128_ns1` | 36 / 17 / 16 | 18 / 0 / 13 | 6 | 48 | 257 | 76736 |

No ``ds_read_b16``. Leftover ``ds_write_b16`` is P/C. Both split
kernels keep 6 barriers. gfx950 dual-KV LDS is 21 KiB decode / 75 KiB
prefill.

QK B now uses wave ``make_tiled_copy_B`` rather than scalar ``k_lds``
gathers. Width-2051 ``L=512`` stays 24.0 / 24.1 / 588.6 us (``err=0``);
ISA is flat on decode and +2 ``ds_read_b32`` on prefill. Matching
``BLOCK_N``/waves/splits still does not mean matching Triton's ISA.

**Do not retry** MMA-native PV A/B (`make_tiled_copy_A/B` or a TV copy
of P) on this row-major ``[BN, D]`` V layout. copy_B's tile is
``(MFMA-N, K) = (D, tokens)``, so K along tokens is stride-D and is
not a contiguous 64-bit B fragment (oracle ``err≈0.98``). A TV 64-bit
P copy that matched the old four-element fragment compiled and
matched the oracle but was ISA-neutral / slightly more VGPR on decode
and was reverted.

**Do not retry** widening leftover P/C LDS stores: 128-bit C fragment
stores plus shuffle-pack 64-bit P stores (`lane_m % 4 == 0` after
``shuffle_xor`` 1/2/3). Oracle ``err=0``, but width-2051 ``L=512``
went to 24.5 / 24.8 / 637.3 us vs 24.0 / 24.1 / 588.6. Scalar P/C
stores stay.

Dropped the barrier between V gather and wave-0 softmax (softmax
reads C/live, not V). Split-kernel occupancy ISA is 6 ``s_barrier``
(was 7). Width-2051 ``L=512`` is 24.3 / 24.1 / 586.7 us (``err=0``)
— flat. Prefill VGPR 165 vs 169. Do **not** retry all-wave softmax
to fold the P barrier.

**Do not retry** replacing cooperative page translate with per-thread
(and softmax) redundant `indices`/`page_table` loads. Oracle ``err=0``,
but width-2051 ``L=512`` went to 24.1 / 23.8 / 608.9 us vs 24.3 /
24.1 / 586.7; prefill paid for dropping the translate barrier. Keep
phys/page/live LDS.

**Do not retry** packing `phys/page_off/live/pad` into one
`[BLOCK_N,4]` Int32 LDS row with one 128-bit vector store per
translated column. GPU 6 / gfx950 focused decode+prefill pytest passed
and ISA changed three metadata `ds_write_b32` to one `ds_write_b128`,
but width-2051 `L=512` HEAD → packed median (three runs) was
20.18 → 20.19 us at `M=1`, 20.86 → 20.69 us at `M=8`, and
515.98 → 518.41 us at `M=512`. The only gain was 0.81%, under the
3% keep gate; keep the three scalar LDS rows.

**Do not retry** hoisting gfx950 V `buffer_load` (`g_copy`) to before
the K-publish `lgkmcnt(0)` / `s_barrier`. Distinct from overlapping
the next tile's K with PV. GPU 6 / gfx950 focused decode+prefill
pytest passed and decode ISA issued V loads before that barrier, but
width-2051 `L=512` HEAD → hoist median (three runs) was
20.18 → 20.13 us at `M=1`, 20.86 → 20.96 us at `M=8`, and
515.98 → 518.30 us at `M=512`. Keep V gather after the K-publish
barrier.

**Kept** K (and gfx942 aliased KV) LDS row pad `_K_STRIDE = D+8`.
Does not change decode token-major V. GPU 6 / gfx950 focused
decode+prefill pytest passed. Width-2051 `L=512` HEAD → pad median
(three runs) was 20.18 → 20.09 us at `M=1`, 20.86 → 20.52 us at
`M=8`, and 515.98 → 424.09 us at `M=512` (−17.8% prefill). PMC
conflict/wave dropped (decode 1032 → 384, prefill 93984 → 51744);
MFMA/wave stayed 24 / 1584. New keep-gate baseline **20.09 / 20.52 /
424.09**. C-LDS is still unpadded.

**Do not retry** padding C-LDS's 64-lane axis to 66 (`_C_LANE_STRIDE=66`,
logical 64). GPU 6 / gfx950 focused decode+prefill pytest passed, but
width-2051 `L=512` K-pad → C-pad median (three runs) was
20.09 → 20.31 us at `M=1`, 20.52 → 19.98 us at `M=8` (−2.63%), and
424.09 → 431.45 us at `M=512`. Keep unpadded
`(n_subtiles, num_waves, 64, 4)`.

**Do not retry** MMA-native QK A (`make_tiled_copy_A` of global Q into
the QK A fragment). The compiler aborted in
`CopyOpUniversalCopyType::emitAtomCallSSA`. Keep the 128-bit `g_copy`
plus `q_off`/`from_elements` extract and pad-head zeroing.

gfx950 now has separate ``[BN, D]`` K and V LDS tiles (gfx942 still
aliases). The tile schedule is unchanged, so the post-QK barrier still
publishes C; it is no longer an alias-overwrite wait. GPU 6 / gfx950:
18 pytest cases, ``err=0``, width-2051 ``L=512`` is 24.4 / 24.1 /
519.3 us vs 24.3 / 24.1 / 586.7. Prefill is ~11% faster. Occupancy
ISA: decode split LDS 21376 (was 13184) VGPR 79 (was 69); prefill LDS
76736 (was 43968) VGPR 257 (was 165); still 6 ``s_barrier``.

Decode split-K is retuned only for ``M * Hk <= 4``: 32 splits replaces
64. Width-2051 ``L=512`` ``M=1`` improves from 24.4 to 20.7 us (~15%)
with ``err=0``; ``M=8`` keeps 32 splits and is flat at 24.0 us, while
prefill stays 519.5 us. The decode merge drops from 34
``ds_read_b128`` at 64 splits to 8 at 32. **Do not retry** 16 splits
for the ``4 < M * Hk < 32`` regime: ``M=8`` regressed to 31.3 us from
24.1 us.

**Do not retry** extra prefill split-K when ``M * Hk > 512``. Raising
that bucket from 1 to 4 splits keeps ``err=0`` and 18 tests, but
width-2051 ``L=512`` prefill went 519.5 → 548.4 us; decode was
unchanged. Direct ``out`` writes with one split stay.

**Do not retry** overlapping the next K gather with PV. A local
``@flyc.jit`` dispatcher, localized page-map LDS views, and explicit
``index``-to-``Int64`` conversion make the runtime ``if is_first`` /
``if has_next`` tiled-copy path compile and match the oracle. It still
regresses width-2051 ``L=512`` prefill to 530.0 us from 519.3 us
(~2.1%); decode is flat at 24.2 / 24.1 us and all 18 tests pass. The
pipeline was reverted. Do not retry the old 90 KiB ping-pong mapping.

On gfx950, this tile's V gather now runs after the K barrier and before
QK (separate ``v_lds``). gfx942 still gathers V after QK so it can
overwrite the aliased KV tile. GPU 6 / gfx950: 18 pytest cases,
``err=0``, width-2051 ``L=512`` is 19.8 / 22.7 / 513.5 us vs 20.7 /
24.0 / 519.5. Occupancy ISA is unchanged (6 barriers, decode VGPR 79,
prefill VGPR 257).

**Do not retry** token-major ``(D, BLOCK_N)`` V LDS written with
per-element stores plus ``make_tiled_copy_B`` PV **on prefill / both
launch paths**. Oracle ``err=0`` and 18 tests pass, and decode improved
to 18.6 / 21.2 us, but prefill regressed to 613.0 us from 513.5.
Decode-only gfx950 ``BLOCK_N=16`` token-major V plus tiled PV-B is the
mapping that landed: width-2051 ``L=512`` is 18.5 / 21.1 / 516.2 us vs
19.8 / 22.7 / 513.5.

**Do not retry** replacing that decode-only token-major mapping with
prefill's row-major ``[BLOCK_N, D]`` V LDS, 128-bit tiled V stores, and
scalar stride-``D`` PV gathers. GPU 6 / gfx950 stayed exact and removed
the decode V scalar-store cluster (``ds_write_b16`` 20 → 4,
``ds_write_b128`` 3 → 5, no ``ds_bpermute``), but width-2051 ``L=512``
regressed 18.65 → 19.60 us at ``M=1`` and 21.15 → 22.82 us at ``M=8``;
``M=512`` was flat at 513.29 → 513.31 us. Keep decode token-major V:
eliminating its shuffle-free scalar stores is not worth the PV-read cost.

**Do not retry** decode-only ``[BLOCK_N, D]`` V LDS plus
``LDSReadTrans16_64b`` ``make_tiled_copy_B`` PV on
``(16, 16):(1, D)``. GPU 6 / gfx950 stayed exact and did emit
``ds_read_b64_tr_b16`` (4) with packed V stores (``ds_write_b16`` 20 → 4,
``ds_write_b128`` 3 → 5, VGPR 71 → 69), but width-2051 ``L=512``
regressed 18.65 → 19.16 us at ``M=1`` and 21.15 → 22.35 us at ``M=8``;
``M=512`` was flat at 513.29 → 512.71 us. This is not the earlier
hand-rolled ``(n0+lane_m%4)*D + (d-lane_m%4)`` miss. Keep token-major
decode V: a correct transpose PV-B does not buy back the layout change.

**Do not retry** prefill-only ``LDSReadTrans16_64b`` ``make_tiled_copy_B``
PV on already row-major ``[BLOCK_N, D]`` with ``(16, 16):(1, D)``.
GPU 6 / gfx950 stayed exact (``err=0``), decode was flat (18.63 / 21.22 us),
and occupancy ISA emitted ``32× ds_read_b64_tr_b16`` at the same 257 VGPR /
36 ``ds_write_b128``, but width-2051 ``L=512`` ``M=512`` only moved
513.29 → 506.16 us (~1.4%). Below the 3% keep gate. Keep scalar
``v_lds[n, d]`` gathers on BN64.

**Do not retry** decode-only ``[BLOCK_N, D]`` packed V stores plus
retiling PV so each lane holds 4 consecutive D (``UniversalCopy64b``)
and reducing n with vector FMA instead of MFMA. GPU 6 / gfx950 stayed
exact (``err=0``) and dropped PV ``v_mfma_f32_16x16x16`` (4 → 0) with
packed V stores (``ds_write_b16`` 20 → 4, ``ds_write_b128`` 3 → 5), but
VGPR 71 → 129 and width-2051 ``L=512`` regressed 18.65 → 23.67 us at
``M=1`` and 21.15 → 29.31 us at ``M=8``; ``M=512`` was flat.
Keep token-major decode V and MFMA PV: D-contiguous register B without
MFMA is not a win.

**Do not retry** ``BLOCK_N=64`` / 8 splits / 128 threads for
``4 < M * Hk < 32``. ``M=8`` stayed correct (``err=0``) but width-2051
``L=512`` went 22.7 → 33.5 us; ``M=1`` and prefill were flat. Keep that
bucket on ``BLOCK_N=16`` / 32 splits / 256 threads.

**Do not retry** four-wave 256-thread ``BLOCK_N=64`` / ``splits=1``
prefill on the dual-KV mapping. ``err=0``, but width-2051 ``L=512``
prefill went 513.5 → 719.9 us; decode was flat. Keep two-wave 128
threads. This is not the old ~90 KiB aliased 256-thread BN64 mapping.

**Do not retry** dropping gfx950 K32 ``q_off`` so Q fragments index
``q_vec[i]`` instead of ``q_vec[q_off + i]``. ``err=0``, but
width-2051 ``L=512`` stayed 19.8 / 23.3 / 514.9 us vs 19.8 / 22.7 /
513.5. Prefill VGPR is not this extract.

**Do not retry** a second token-major V LDS (``v_t``) filled by scalar
scatters from the 128-bit row-major gather, plus ``make_tiled_copy_B``
PV. ``err=0``, but width-2051 ``L=512`` went 19.9 / 23.3 / 935.7 us
vs 19.8 / 22.7 / 513.5. Keep one row-major V tile.

**Do not retry** 64-bit ``UniversalCopy`` PV-A loads from contiguous
``p_lds[h, n:n+4]``. Oracle ``err=0``; width-2051 ``L=512`` was 19.8 /
23.1 / 516.0 us vs 19.8 / 22.7 / 513.5. The 16 ``ds_read_u16`` P reads
are not the split-kernel gap. Keep scalar P gathers.

**Do not retry** ``ds_read_b64_tr_b16`` PV-B from unswizzled row-major
``[BLOCK_N, D]`` V (lane groups of 4, address ``(n0+lane_m%4)*D + (d-lane_m%4)``).
Decode oracle ``err≈0.98``. Same class of miss as MMA-native PV B on this
layout. Keep scalar ``v_lds[n, d]`` gathers.

**Do not retry** unrolling the 32-split LSE merge (``range_constexpr``
instead of ``scf.for``). ``err=0``; width-2051 ``L=512`` was 18.6 /
21.2 / 513.0 us vs the kept 18.5 / 21.1 / 516.2. Decode is flat; prefill
does not launch merge. Keep the runtime split loop.

**Do not retry** a 256-thread LSE merge (one ``D`` lane per thread,
drop the ``di`` loop). ``err=0``; width-2051 ``L=512`` was 18.6 /
21.3 / 513.6 us vs the kept 18.5 / 21.1 / 516.2. Decode is slightly
worse; prefill does not launch merge. Keep two waves and two ``D``
lanes per thread.

**Do not retry** BF16 ``partial_out`` (keep FP32 split partials). GPU 6 /
gfx950 stayed exact (``err=0``); width-2051 ``L=512`` was 18.91 / 21.23 /
515.40 us vs 18.65 / 21.15 / 513.29. Decode is flat-to-worse; prefill
does not use split partials. Keep FP32 ``partial_out``.

**Do not retry** nontemporal (``cache_modifier=2``) 128-bit K/V gathers.
GPU 6 / gfx950 stayed exact (``err=0``); width-2051 ``L=512`` went
18.65 → 19.12 us at ``M=1``, 21.15 → 22.87 us at ``M=8``, and
513.29 → 528.44 us at ``M=512``. Keep cached ``BufferCopy128b`` K/V loads.

**Do not retry** 64-bit ``BufferCopy64b`` ``partial_out`` loads in the
128-thread LSE merge with consecutive ``D`` pairs (``d=2*tid``). GPU 6 /
gfx950 stayed exact (``err=0``); merge VGPR 112 → 63 and ISA emitted
``buffer_load_dwordx2``. Split-kernel occupancy ISA was unchanged. Width-2051
``L=512`` went 18.65 → 18.76 us at ``M=1`` and 21.15 → 21.62 us at ``M=8``;
``M=512`` was flat (no merge). This is not the earlier 8-wide / 64-thread
128-bit merge miss. Keep scalar per-``D`` merge loads.

**Do not retry** a 64-thread LSE merge with 4-wide ``BufferCopy128b``
``partial_out`` loads (``d=4*tid``, one wave, ``block=64``). GPU 6 /
gfx950 stayed exact (``err=0``); width-2051 ``L=512`` went
18.65 → 18.84 us at ``M=1`` and 21.15 → 21.64 us at ``M=8``;
``M=512`` was flat (no merge). Distinct from the 8-wide / 64-thread
and 2-wide / 128-thread merge misses. Keep two-wave scalar merge.

**Do not retry** restoring 64 decode splits for ``M * Hk < 32``.
That count was already replaced by 32 because merge traffic dominated
(``M=1`` 24.4 → 20.7 us). Re-measuring it on GPU 6 / gfx950 stayed
exact (``err=0``) and width-2051 ``L=512`` went 18.65 → 23.26 us at
``M=1`` and 21.15 → 27.06 us at ``M=8``. Keep 32 decode splits.

**Do not retry** nontemporal (``cache_modifier=2``) 32-bit stores of
FP32 ``partial_out``. GPU 6 / gfx950 stayed exact (``err=0``);
width-2051 ``L=512`` went 18.65 → 18.82 us at ``M=1`` and
21.15 → 21.07 us at ``M=8`` (under the 3% keep gate);
``M=512`` was flat (no partials). Keep default cached epilogue stores.

**Do not retry** ``amdgpu-expert-scheduling-mode`` on the K2 compile
hints. GPU 6 / gfx950 stayed exact (``err=0``); width-2051 ``L=512``
went 18.65 → 18.88 us at ``M=1`` and 21.15 → 21.56 us at ``M=8``.
Keep the default scheduler.

**Do not retry** isolated shuffle-pack 64-bit P LDS stores
(``UniversalCopy64b`` from ``lane_m % 4 == 0`` after ``shuffle_xor``
1/2/3). GPU 6 / gfx950 stayed exact (``err=0``); width-2051 ``L=512``
went 18.65 → 19.28 us at ``M=1``, 21.15 → 21.98 us at ``M=8``, and
513.29 → 566.07 us at ``M=512``. This is the P half of the earlier
P/C widening miss. Keep scalar ``p_lds[h, n]`` stores.

- [ ] Family B: group 5, `D=128`, width 2051 — vs #4882 Triton **and** Gluon.
- [ ] gfx942 and gfx950; gfx950 uses extra LDS vs the live `num_stages=1` path.
- [ ] Decode (`M=1..8`) and prefill instantiations are **not** forced into one
      kernel if occupancy suffers.
- [ ] **Done when:** GQA `err` vs oracle is within the documented tol; family A
      beats live AMD sparse GQA and #4882 Triton; family B beats #4882 Triton
      and Gluon on the published points. Do **not** cite the group-5 Gluon
      number as a family A GQA win.

### 4. Wire `aiter/ops/flydsl/` + vLLM `qwen4_exp` opt-in

- [ ] Public wrappers + lazy export from `aiter/ops/flydsl/__init__.py`.
- [ ] End-to-end op_test: indexer through GQA as one QSA layer (launches,
      bytes, fused K1/K2), family A and B tables, HIP graph replay at decode.
- [ ] vLLM `qwen4_exp` opt-in with the same three-way backend idea as #4882
      (`auto` / FlyDSL / Triton). `auto` must not silently pick a backend that
      fails the family A gate.
- [ ] **Done when:** one documented command on GPU 6 shows family A e2e beating
      live AMD; competitor columns still present; vLLM opt-in is callable
      without editing the default AMD path.

### 5. Optional fusions (only after the bar)

Do not start this phase to “make K1/K2 look better.”

- [ ] Fuse expand+tail into K2 (or keep it in K1’s epilogue — pick one and
      lock it here).
- [ ] Fuse `qsa_pre_indexer` (`Gemma RMSNorm + partial MRoPE + compress`) only
      after K1/K2 already beat live AMD. Live AMD today is unfused
      (`GemmaRMSNorm + triton_mrope`; no `qsa_pre_indexer.py` on AMD).
- [ ] MTP IndexShare (`indexer.skip_topk`): GQA-only launch, no scorer.
- [ ] **Done when:** fused path matches the unfused oracle; e2e still beats
      live AMD; skip-topk is a harness row, not a surprise.

## Non-goals (do not pull into this plan)

- Changing GPU from the locked `HIP_VISIBLE_DEVICES=6`.
- Waiting on #4882 to merge before the harness or K1.
- Treating NVIDIA QSA Triton, FlashInfer, or TRT-LLM as the AMD acceptance bar.
- Mixing GR kernels (SILOTIGER-1041 / 1042) into these files.
- Claiming indexer wins from DSA `H=32` FP8 numbers.
- Mass-comment cleanup as its own commit unless a phase’s diff is unreadable
  without it.

## Open questions (resolve into locks; do not guess in code)

Leave these open until the named phase produces evidence. When resolved, move
the answer into **Locked decisions** and check the item.

- [ ] Where indexer vs GQA dominates on this GPU at 8k / 32k / 128k / 1M
      (phase 1). Sets whether to land K1 or K2 first after the harness.
      Phase 1e answered this under **fallback** top-k only. Production HIP
      select is not a new full-layer rocprof; 2d/e2e must not use the
      fallback AMD column.
- [ ] Whether FlyDSL K1 should emit **block ids** or already-expanded **token
      ids** (phase 2/5).
- [ ] Packed vs padded `M` (varlen) from a real vLLM prefill trace (phase 4).
- [ ] MTP IndexShare wiring in vLLM vs aiter-only skip-topk (phase 5).
- [ ] #4882 merge timing: keep competing even if it lands; retarget the pin
      rather than pausing.
