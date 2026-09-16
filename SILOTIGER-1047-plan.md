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
- **K1 must not write a full score matrix.** Stream paged compressed index-K,
  ReLU-sum per complete block, keep a **local** top-512 (or local top-k on
  family B), merge to a global 512 (or k). Same “score-plus-top-k” idea as DSA
  fused indexer work; different ABI.
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

- [ ] **2d.** Family A K1 beats live vLLM AMD select on the harness lengths
      (`M∈{1,8,512}`, `L∈{512,2048,8192,32768,131072}`), still `block_ids [M,512]`,
      no `[M, n_blocks]` score buffer.

GPU 6 / gfx950 / `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Oracle set equality `err=0`
on both columns. **Not checked.** `aiter/jit/module_top_k_per_row.so` is
loaded (`import [module_top_k_per_row]`; oracle-fallback warning did not
fire). The AMD column is Triton MQA + `_hip_top_k_per_row_decode` + expand.

The FlyDSL column is eight-wave K1 plus a **`visible <= 512` emit path**,
a **wave-local tile sort**, a **per-tile pair merge**, a **column split**
on decode rows with more than one 512-slot tile, and a **software-pipelined
score_col** (prefetch next D-chunk of paged K while the current ReLU-sum
runs). The inter-wave bitonic merge fuses each reverse-upper step with its
first cross-half compare, removing three workgroup barriers per tile. Prefill
stays single-WG. Workspace is `[M, 8, 512]`, not `[M, n_blocks]`. Expand still
separate. HIP `module_top_k_per_row.so` is loaded. Oracle set equality `err=0`
on both columns.

**Not checked:** `L<=2048` still beats HIP select. Pipelined scoring
plus the fused first inter-wave compare leave decode 8k/32k/128k at
~35 / ~52 / ~153µs vs HIP ~19 / ~20 / ~30µs. Prefill serial 8k/32k is
~137 / ~537µs, still behind HIP (~84 / ~250µs).

| m | seq_len | n_blocks | flydsl_k1 us | vllm_amd_select us | flydsl_k1 err | vllm_amd_select err |
|--:|--------:|---------:|-------------:|-------------------:|--------------:|--------------------:|
| 1 | 512 | 128 | 2.0 | 9.4 | 0 | 0 |
| 8 | 512 | 128 | 2.9 | 11.4 | 0 | 0 |
| 1 | 2048 | 512 | 2.0 | 10.2 | 0 | 0 |
| 8 | 2048 | 512 | 2.9 | 11.4 | 0 | 0 |
| 1 | 8192 | 2048 | 35.2 | 19.0 | 0 | 0 |
| 8 | 8192 | 2048 | 35.4 | 21.6 | 0 | 0 |
| 1 | 32768 | 8192 | 52.3 | 20.4 | 0 | 0 |
| 8 | 32768 | 8192 | 52.0 | 25.2 | 0 | 0 |
| 1 | 131072 | 32768 | 153.4 | 30.0 | 0 | 0 |
| 8 | 131072 | 32768 | 161.2 | 53.1 | 0 | 0 |
| 512 | 512 | 128 | 3.6 | 18.7 | 0 | 0 |
| 512 | 2048 | 512 | 3.5 | 29.3 | 0 | 0 |
| 512 | 8192 | 2048 | 137.4 | 83.5 | 0 | 0 |
| 512 | 32768 | 8192 | 536.9 | 249.5 | 0 | 0 |

- [ ] Family B (`H` 4 or 8, Gluon-validated indexer shapes).
- [ ] No `[rows, n_blocks]` FP32 score buffer.
- [ ] gfx942 and gfx950.
- [ ] Gate vs live vLLM AMD (`MQA Triton + HIP top-k`) and vs #4882 Triton;
      beat Gluon on gfx950 **where Gluon dispatches**.
- [ ] **Done when:** selected-block **set equality** (or documented tie policy)
      vs the oracle; family A beats live AMD on the harness lengths; family B
      beats #4882 Triton and Gluon on the published indexer bench points.

### 3. FlyDSL K2 (sparse GQA) — family A then B

K2 attends uncompressed paged K/V at the expanded token indices. Prefetch K/V
for the selected runs (512 runs of 4 plus a short tail on A; 2051-wide list on
B). Split-K as needed.

- [ ] Family A: 24 Q / 2 KV, group 12, `D=256`, softmax scale, sigmoid gate
      weights if fused into the epilogue. Out: BF16 `o [M, 24, 256]`
      (pre-`o_proj`).
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
