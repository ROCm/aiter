# FlyDSL HSTU Backward Kernel — Phased TDD Implementation Plan

*Date: 2026-07-07*
*Scope: a concrete, test-driven roadmap for implementing the **HSTU attention backward**
kernel in FlyDSL, mirroring the existing FlyDSL forward and matching the PyTorch reference
gradients.*

Companion documents:
- Math / derivation: [`2026-07-07_HSTU_theory.md`](./2026-07-07_HSTU_theory.md)
- Environment & constraints: [`HSTU_bwd_kernel_dev.md`](./HSTU_bwd_kernel_dev.md)

---

## 0. TL;DR

We build `flydsl_hstu_attention_bwd` incrementally, one **red → green → refactor** loop per
capability, always validating against `torch.autograd.grad` on the reference
`torch_hstu_attention`. We start from the *smallest correct thing* (a single-`(b,h)`,
causal-only, dense, non-sequence-parallel kernel that produces `dV` only) and grow toward the
full, tiled, masked, sequence-parallel kernel that plugs into a `torch.autograd.Function`.

> **Status (living doc, last updated 2026-07-07).**
> - ✅ **Setup** — FlyDSL HSTU forward vendored into `meta/aiter` from the fork
>   (`robin/users/relbers/hstu_attention_fwd` @ `01bfb38`); forward compiles + passes on `gfx950`.
> - ✅ **Phase 0** — scaffold, reference oracle, `_validate_bwd_inputs`, host stub. Green.
> - ✅ **Phase 1** — `dV` only, causal, single-writer kernel. Green (bf16/f16 × 3 shapes).
> - ✅ **Phase 2** — `dK` + `dS` machinery (resident V, `dA=dO·Vᵀ`, SiLU′ gate). Green.
> - ✅ **Phase 3** — `dQ` via a second Q-owned kernel; full dense causal backward. Green.
> - ✅ **Phase 4** — masking variants (`num_targets`, `max_attn_len`, `contextual_seq_len`,
>   combos, asymmetric dims). Green (26 tests).
> - ✅ **Interlude** — perf baseline captured on **MI300X (`gfx942`)**: `flydsl` provider added
>   to the recsys `bench_hstu.py`/`sweep_hstu.py`; ms/TFLOP-s vs Triton recorded over P1–P4 ×
>   {512,1024,2048,16384} × {fwd,bwd}. Results in `docs/HSTU_backward_optimization_log.md`.
>   Headline: FlyDSL bwd ~2.4–3.5× faster at small batch (B=120); ~parity→0.77× at large batch
>   (B=1024), gap closing with seq_len (near parity at N=16384).
> - ✅ **Phase 5** — tiling + config/CSV plumbing: cached `_compile_bwd_launcher`, tuned→default→
>   custom override chain, `_bwd_tuned_config_map`/`_BWD_CSV_COLUMNS`. Green (33 tests: +4
>   block-override, +3 tuned-CSV).
> - ❎ **Phase 6 (N/A)** — sequence-parallel `dQ` sync is **subsumed by the Phase-3 two-kernel
>   design**: `dV`/`dK` (KV-owned) and `dQ` (Q-owned) are each fully tile-parallel *and*
>   single-writer, so there is no `dQ` read-modify-write to guard. The vestigial `sequence_parallel`
>   knob was removed.
> - ✅ **Phase 7 (partial)** — bench + tuning: added `op_tests/op_benchmarks/flydsl/tune_hstu_attn_bwd.py`
>   (sweeps tile configs, emits the tuned CSV). Committed `hstu_attention_bwd_tuned.csv` for P1–P4 ×
>   {512,1024,2048} dense causal on `gfx942`. Finding: `(192,32,4,0)` wins at N=2048 (~6–9% over
>   default), default `(64,32,4,0)` best at N≤1024. End-to-end: B1024/H8/N2048 bwd 71.5→64.0 ms
>   (−10.5%), Triton gap 1.29×→1.16×. Remaining: mask-regime + N=16384 tuning, and `gfx950` (34 tests green).
> - ✅ **Phase 8** — autograd integration: `FlydslHstuAttention(torch.autograd.Function)` + a
>   `flydsl_hstu_attention(...)` drop-in wrapper (fwd + bwd as one differentiable op). Green
>   (`test_flydsl_autograd_end_to_end`, 6 cases). **Full suite: 40 tests.**
> - ✅ **Phase 9** — layout-algebra cleanup & de-duplication: shared `hstu_attention_common.py`
>   (`grouped_loader`, `swz_col`, `decode_lane` via `idx2crd`); both backward kernels import it.
>   Behavior-preserving (40 tests green; B1024/H8/N2048 bwd 63.95 ms, no regression).
>
> **Design note.** The backward is **two lock-free kernels**: a KV-owned kernel
> (`hstu_attention_bwd.py`) for `dV`/`dK` (reduce over the query index) and a Q-owned kernel
> (`hstu_attention_bwd_dq.py`) for `dQ` (reduce over the key index). Splitting by reduction axis
> keeps both single-writer with no atomics; `dS` is recomputed in each (recompute ≫ HBM traffic).
> `flydsl_hstu_attention_bwd` launches both.
>
> **Known caveat.** The vendored PyTorch oracle `torch_hstu_attention` only supports
> `attn_dim == hidden_dim` (its `qkv_to_padded_dense` reshape assumes a uniform `D`). Symmetric-dim
> tests use it; asymmetric-dim tests use a hand-rolled per-sequence dense causal oracle
> (`hstu_bwd_reference_causal_dense`) in the test module.

The forward already exists and pins every convention we must match; the backward's job is
"same conventions, five matmuls + one elementwise stage, reduce over the query index instead of
the key index."

---

## 1. Guiding principles

1. **Test first, always.** Every phase begins by writing a *failing* test that pins the exact
   behavior we want, then we make it pass with the minimum kernel change, then we refactor.
   No kernel code is written before there is a test that would catch it being wrong.

2. **The PyTorch reference is ground truth.** `torch_hstu_attention`
   (`meta/aiter/op_tests/triton_tests/utils/hstu_attention_ref.py`) + `torch.autograd.grad`
   produces `dQ, dK, dV`. Every gradient assertion compares against it. The Triton backward
   (`_hstu_attn_bwd`) is a *structural* blueprint (tiling, recompute, locks), not the numeric
   oracle.

3. **Mirror the forward, don't reinvent.** The backward must reproduce the forward's numerical
   recipe *exactly* (see theory §7.2), reuse its host plumbing shape (theory §7.3), and honor
   the same constraints. Where the forward made a choice (fast-math SiLU, `alpha`-in-score,
   `1/N`-in-epilogue, causal-only, dtype ∈ {f16,bf16}, `to_id` mask order), the backward makes
   the same one.

4. **Correct before fast.** Get a lock-free, single-writer, dense, causal-only kernel numerically
   correct first. Only then add masking variants, tiling knobs, sequence-parallel `dQ` sync, and
   tuning. Never trade correctness for speed in an early phase.

5. **One capability per phase.** Each masking variant (`num_targets`, `max_attn_len`,
   `contextual_seq_len`) and each performance feature lands behind its own test(s) and, where the
   forward does so, its own compile-time flag (`has_targets`, `has_window`, `has_contextual`).

### 1.1 Environment & hard constraints (from `HSTU_bwd_kernel_dev.md`)

- **Device pinning is mandatory:** run everything with `HIP_VISIBLE_DEVICES=7` — shared node.
- Python env: `/workspaces/git/meta/aiter/flydsl_venv`.
- Target arch: **MI350 = `gfx950` (CDNA4)**; keep `gfx942` (CDNA3) paths working since the
  forward supports both (`_arch_dma_params`, LDS swizzle period differ).
- Backward must be **compatible with the forward** and mirror what needs mirroring.

Example invocation for every test/bench run in this plan:

```bash
cd /workspaces/git/meta/robin_aiter
HIP_VISIBLE_DEVICES=7 /workspaces/git/meta/aiter/flydsl_venv/bin/python -m pytest \
  op_tests/flydsl_tests/test_flydsl_hstu_attention_bwd.py -x -q
```

---

## 2. The math we are implementing (one-screen recap)

Forward (per `(b,h)`, sequence length `n`, constant `N = MAX_SEQ_LEN`):

$$
S=\alpha QK^\top,\quad A = M\circ \tfrac1N\,\text{silu}(S),\quad O = AV.
$$

Backward (given `dO`; **recompute** `S, σ` from `Q,K` — nothing is stashed by the forward):

$$
\mathrm dV = A^\top \mathrm dO,\qquad
\mathrm dA = \mathrm dO\,V^\top,\qquad
\mathrm dS = M\circ \tfrac1N\,\sigma(1+S(1-\sigma))\circ \mathrm dA,
$$
$$
\mathrm dQ = \alpha\,\mathrm dS\,K,\qquad
\mathrm dK = \alpha\,\mathrm dS^\top Q.
$$

Five matmuls + one pointwise stage. No softmax, no `logsumexp`, no row-sum correction. See
theory §4–§5 for the full derivation and §5.2–§5.3 for the tiling/`dQ`-hazard discussion.

**Numerical recipe to match the forward bit-closely** (theory §7.2):
- `alpha` folded into `S` before SiLU, and it reappears in `dQ, dK` (not in `dV`).
- `1/N` multiplies `dS` (mirrors the forward hoisting `1/N` to the O epilogue).
- SiLU via fast/unsafe `exp2`/`rcp` intrinsics, under `fast_fp_math=True, unsafe_fp_math=True`.
- fp32 matmul accumulation, cast to `{f16,bf16}` only on store.

---

## 3. Target file layout

We create these files, mirroring the forward's split:

| File | Role | Mirrors |
|---|---|---|
| `meta/robin_aiter/aiter/ops/flydsl/kernels/hstu_attention_bwd.py` | device kernel + `validate_*` / `build_*` | `kernels/hstu_attention_fwd.py` |
| `meta/robin_aiter/aiter/ops/flydsl/hstu_attention_kernels.py` (extend) | add `flydsl_hstu_attention_bwd` host entry point + config plumbing | existing `flydsl_hstu_attention_fwd` |
| `meta/robin_aiter/op_tests/flydsl_tests/test_flydsl_hstu_attention_bwd.py` | pytest suite (the TDD driver) | `test_flydsl_hstu_attention.py` |
| `meta/robin_aiter/op_tests/flydsl_tests/_hstu_bwd_reference.py` (optional) | shared autograd-reference helper | reuses `generate_hstu_attn_inputs` |
| `meta/robin_aiter/aiter/ops/flydsl/hstu_attention_bwd_tuned.csv` (later) | tuned configs | `hstu_attention_tuned.csv` |
| `meta/robin_aiter/op_tests/op_benchmarks/flydsl/bench_hstu_attn.py` (extend) | add backward benchmark vs Triton bwd | existing fwd bench |

Reuse verbatim from the forward test module (`test_flydsl_hstu_attention.py`):
`generate_hstu_attn_inputs`, `_generate_sparse_seq_len`, `_apply_sl`, and the CSV-plumbing test
patterns.

---

## 4. Test harness design (the oracle)

The single most important piece of infrastructure is the reference-gradient helper. Written once
in Phase 0, reused by every later phase.

```python
def hstu_bwd_reference(N, alpha, q, k, v, seq_offsets, causal, num_targets,
                       max_attn_len, contextual_seq_len, dout):
    """Ground-truth dq, dk, dv via autograd on the torch reference."""
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    out = torch_hstu_attention(
        N, alpha, qf, kf, vf, seq_offsets, causal,
        dropout_pr=0.0, training=False, num_targets=num_targets,
        max_attn_len=max_attn_len, contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=0,
    )
    dq, dk, dv = torch.autograd.grad(out, (qf, kf, vf), grad_outputs=dout.float())
    return dq, dk, dv
```

Notes / decisions:
- Do the reference in **fp32** for a clean oracle; the FlyDSL kernel runs in `{f16,bf16}` and is
  compared with **relaxed tolerances** (the forward test uses `atol=1e-3, rtol=0` on
  `out * max_seq_len`; the backward will need per-tensor tolerances, likely looser for `dQ/dK`
  than `dV` because two extra matmuls accumulate error). Establish tolerances empirically in
  Phase 3 and record them as named constants in the test module.
- Mirror the forward test's scaling trick if needed (`* max_seq_len`) to lift small magnitudes
  out of the denormal/bf16-underflow regime before comparing.
- Keep a **fp32-vs-fp32 "closed form" unit test**: with `silu=identity, M=1, N=1, α=1` the
  gradients collapse to plain bilinear attention (`dQ=dA·K`, `dK=dAᵀQ`, `dV=AᵀdO`) — a cheap
  sanity oracle independent of the reference (theory §4.5 sanity check).

---

## 5. Phased plan

Each phase lists **Red** (tests to write first), **Green** (kernel work to make them pass),
**Refactor**, and **Exit criteria**. Phases are ordered so each builds on a green predecessor.

### Phase 0 — Scaffolding & the failing harness  *(no kernel math yet)*  ✅ DONE

**Goal:** stand up the test file, the reference oracle, input generation, and a stub host API,
so that the first real test can fail for the right reason.

- **Red:**
  - `test_flydsl_hstu_attention_bwd.py` imports a (not-yet-existing) `flydsl_hstu_attention_bwd`
    and a `hstu_bwd_reference` helper.
  - One trivial test: `test_reference_oracle_runs` — builds tiny inputs via
    `generate_hstu_attn_inputs`, calls `hstu_bwd_reference`, asserts shapes of `dq,dk,dv`. This
    validates the oracle itself (no FlyDSL yet).
  - One `xfail`/skipped test `test_flydsl_bwd_importable` that will drive Phase 1.
- **Green:**
  - Add `hstu_bwd_reference` (§4). Add a `flydsl_hstu_attention_bwd` **stub** in the host module
    that raises `NotImplementedError` (so imports resolve).
  - Add input-validation scaffolding `_validate_bwd_inputs` mirroring `_validate_inputs`
    (checks `dout` shape == `v` shape, dtype match, device match, rank-3, jagged offsets).
- **Refactor:** factor `generate_hstu_attn_inputs` import (reuse forward test's generator).
- **Exit:** oracle test green; validation-unit tests green; kernel tests present but skipped.

### Phase 1 — `dV` only, single `(b,h)`, dense, causal-only  *(first real GPU output)*  ✅ DONE

**Why `dV` first?** `dV = Aᵀ dO` needs the *forward* recompute (`S, σ, A`) plus **one** matmul
and no `alpha`. It exercises jagged indexing, recompute, masking gate, and an MFMA reduction over
the query index — the scariest new plumbing — while producing only one of three outputs. It is
the cleanest first green.

- **Red:**
  - `test_bwd_dv_causal_small`: single sequence (`batch` chosen so `batch*num_heads % 8 == 0`,
    e.g. `batch=8, heads=1` or `batch=2, heads=4`), small `n` (≤ one/two tiles), `head_dim =
    hidden_dim = 16` or `64`, bf16 and f16. Compare only `dv` to the oracle with relaxed tol.
    Leave `dq, dk` unasserted (kernel may return zeros for them for now).
- **Green:**
  - New `kernels/hstu_attention_bwd.py` with `validate_hstu_attention_bwd` (copy the forward's
    arch/dtype/shape/divisibility checks) and `build_hstu_attention_bwd`.
  - Device kernel: **KV-parallel** grid (each program owns a `BLOCK_N` block of K/V rows), loop
    over query blocks `start_m`; recompute `S = α QKᵀ`, `σ`, `A = M∘silu(S)/N` for the tile
    (reuse the forward's `silu_scale_batch`, `to_id`, `keep_col`, `pack_p`, K-swizzle, V-load
    idioms), then accumulate `dv += Aᵀ dO` via one MFMA. Store `dv`.
  - Non-sequence-parallel: one writer per `(b,h)` — no locks/atomics.
- **Refactor:** extract the shared "recompute S/σ/A tile" routine so later phases reuse it for
  `dS`.
- **Exit:** `dv` matches oracle (causal, no variants) across bf16/f16 and a couple of small shapes.

> **Implemented as (Phase 1, actual).** `kernels/hstu_attention_bwd.py`. The kernel is a faithful
> relabel of the forward: because `dV` reduces over the query index, each program **owns a KV tile
> (`BLOCK_M`) and streams query tiles (`BLOCK_N`)**. Q↔K roles swap (streamed **Q** staged to LDS
> via the swizzled DMA path; resident **K** in registers), **dO** takes V's register-prefetch→LDS
> path, and the causal bound becomes a **lower** bound on the streamed q tiles (`q ≥ kv`). The
> MFMA `GEMM1(A=Q, B=K) → S[q,kv]` fragment is reused as the `GEMM2` A-operand to give `P^T[kv,q]`,
> so `dV += P^T·dO` contracts the streamed query index; `1/N` is applied in the epilogue exactly as
> the forward does. `dV` rows are owned by a single program → lock-free. `dQ/dK` are returned as
> zeros by the host until Phases 2–3. Test: `test_flydsl_bwd_dv_causal` (bf16/f16 × 3 symmetric
> shapes). Phase-1 masking is causal-only; the variant flags are rejected in
> `validate_hstu_attention_bwd`.

### Phase 2 — Add `dK` (and `dS` machinery), still single-writer  ✅ DONE

**Goal:** produce `dS = M∘(1/N)σ(1+S(1-σ))∘(dO Vᵀ)` and reduce it into `dK = α dSᵀ Q`. `dK` is a
reduction over the query index like `dV`, so it fits the same KV-parallel accumulator with no
new synchronization.

- **Red:**
  - `test_bwd_dk_causal_small` (extends the Phase-1 shapes): assert `dv` **and** `dk`.
  - Add the SiLU-derivative unit check: a tiny fp32 path test that `dS` equals
    `dA·σ(1+S(1-σ))/N` on masked entries and `0` elsewhere (can be done at the reference level to
    lock the formula before trusting the kernel).
- **Green:**
  - In the device kernel, after recompute compute `dAᵀ = V dOᵀ` (mirror Triton
    `dqk_trans = dot(v, trans(do))`), apply the pointwise gate `σ(1+S(1-σ))/N` and the **same**
    `keep_col` mask, giving `dSᵀ`. Accumulate `dk += dSᵀ Q`; multiply by `alpha` **once** at store
    (copy the forward-adjacent Triton optimization `dk = dk * alpha`).
- **Refactor:** unify the pointwise gate so forward `silu` and backward `silu'` share the
  `exp2`/`rcp` sigmoid to keep numerics identical.
- **Exit:** `dv, dk` match oracle; `alpha` applied exactly once; mask identical to forward.

> **Implemented as (Phase 2, actual).** Extended `kernels/hstu_attention_bwd.py`. Added **V** as a
> second resident (owned-KV) register operand, a new MFMA `dA = dO·Vᵀ` (dO A-operand read from the
> LDS it was already staged in for `dV`; V as B-operand, contracting the hidden dim), and a
> `silu_and_grad_batch` helper that returns both `silu(αS)` and the gate `silu'(αS) = σ(1+αS(1−σ))`
> from the *same* fast `exp2`/`rcp` sigmoid as the forward. `compute_s_tile` retains `(grad, keep)`
> per fragment element so `dS = keep·(1/N)·silu'·dA` is formed after dO publishes; that `dS`
> fragment is reused as the A-operand of `dK += dSᵀ·Q` (Q's B-operand read from its resident LDS
> tile, over `head_dim` chunks), with **α applied once** in the epilogue. `dK` reduces over the
> streamed query index → still single-writer, no atomics. Tests: `test_flydsl_bwd_dv_dk_causal`
> (asserts both `dv` and `dk`) + `test_silu_derivative_formula`. `dQ` still returned as zeros.

### Phase 3 — Add `dQ` (single-writer loop), full dense causal backward  ✅ DONE

**Goal:** complete the trio. `dQ = α dS K` is a reduction over the **key** index, so in the
KV-parallel layout each program contributes a **partial** `dQ` for the query rows it touched.
Phase 3 uses the **`SEQUENCE_PARALLEL = False`** path: one program per `(b,h)` loops all KV
blocks sequentially, so `dQ` has a single writer and needs **no atomics/locks** (theory §5.3,
staging step 1).

- **Red:**
  - `test_bwd_all_causal_small`: assert `dq, dk, dv` together, small shapes, bf16 + f16.
  - `test_bwd_all_causal_multi_tile`: `n` spanning several `BLOCK_M`/`BLOCK_N` tiles and a jagged
    batch with varied sequence lengths (reuse `generate_hstu_attn_inputs` with `sparsity=0.5`).
  - `test_bwd_closed_form_identity`: the `silu=identity, M=1, N=1, α=1` collapse (theory §4.5).
  - Establish and record per-tensor tolerances (constants `ATOL_DV`, `ATOL_DQK`, …).
- **Green:**
  - Grid orientation: single program per `(b,h)` (not KV-parallel yet) so it owns the whole `dQ`
    for the sequence; inside, loop KV blocks and, for each, add the partial
    `dq += α Kᵀ dSᵀ`-equivalent into a private/loop-carried `dQ` accumulator, storing at the end.
    (Alternatively keep KV-parallel but restrict to one KV program per `(b,h)` — pick whichever
    maps cleaner onto FlyDSL carried-accumulator loops; decide in-code and note it.)
  - Correct **causal KV bounds** (`low`/`high`, theory §5.4): only query blocks with
    `start_m >= start_n` contribute; skip provably-masked tiles.
- **Refactor:** clean up the tile loop into `run_kv_block` / `run_q_block` helpers analogous to
  the forward's `run_kv_tile`.
- **Exit:** `dq, dk, dv` all match oracle on single-tile, multi-tile, and jagged cases;
  closed-form identity test green. **This is the milestone: a numerically-correct HSTU backward.**

> **Implemented as (Phase 3, actual).** Rather than a single program per `(b,h)` with a `dQ` HBM
> read-modify-write, `dQ` is produced by a **separate Q-owned kernel** (`hstu_attention_bwd_dq.py`)
> that mirrors the forward's orientation (owns a query tile, streams K/V) and adds `dO` as a
> resident operand. Per streamed KV tile it recomputes `Sᵀ[kv,q] = K·Qᵀ`, forms the SiLU′ gate +
> causal mask, computes `dA[kv,q] = V·dOᵀ`, gates it to `dS`, then reuses that `dS` fragment as the
> A-operand of `dQ += dS·K` (K re-read from its resident LDS tile as the B-operand), with `α`
> applied once in the epilogue. Because `dQ` rows are owned by exactly one program, it is
> single-writer/lock-free — no `dQ` RMW and no atomics needed. The KV-owned `dV`/`dK` kernel is
> unchanged. This costs one extra `S`/`dS` recompute vs. a fused kernel, which is the intended
> recompute-over-bandwidth trade. Test: `test_flydsl_bwd_all_causal` (asserts `dq,dk,dv`). The
> `silu=identity` closed-form check is deferred (the kernel can't disable SiLU; the autograd oracle
> already covers correctness).

### Phase 4 — Masking variants, one flag at a time  ✅ DONE

Each variant is gated by a compile-time flag exactly as the forward (`has_targets`,
`has_window`, `has_contextual`) and reuses the forward's `to_id` order (**contextual shift, then
target-tail clamp**, diagonal always valid).

- **Phase 4a — `num_targets` (`has_targets`)**
  - **Red:** `test_bwd_targets` with `target_size=20` (mirror forward test param row 2). Build
    `num_targets`, assert all three grads.
  - **Green:** thread `num_targets` + `max_id` clamp into recompute + `keep_col`. Verify the
    backward masks bit-identically to the forward at sequence boundaries (the classic bug site —
    theory §6 gotchas).
- **Phase 4b — `max_attn_len` sliding window (`has_window`)**
  - **Red:** `test_bwd_window` (`max_attn_len=64`, forward test row 3).
  - **Green:** add the window term to `keep_col` (`dist <= max_attn_len`) and the low-tile skip
    bound to the KV loop (mirror forward `kv_lower`/`win_tile_start`). The backward's loop bounds
    must skip the *same* tiles the forward did.
- **Phase 4c — `contextual_seq_len` (`has_contextual`)**
  - **Red:** `test_bwd_contextual` (`contextual_seq_len=64`, forward test row 4).
  - **Green:** add the prefix `id` shift+clamp and the prefix-opener term (`row_id==0` attends
    whole prefix), plus the prefix block's opened KV range. This is the fiddliest mask; give it
    its own multi-tile and jagged cases.
- **Phase 4d — combinations & asymmetric dims**
  - **Red:** asymmetric `head_dim != hidden_dim` and `%64 != 0` dims (forward test rows 5–7:
    `(96,96)`, `(128,64)`, `(96,192)`), plus a targets+window combo if the reference supports it.
  - **Green:** ensure `HEAD_DIM_K` rounding / K-swizzle handling matches the forward for
    non-64-divisible dims.
- **Exit:** full parametrized suite mirroring `test_flydsl_hstu_attention`'s matrix, but asserting
  gradients. A single `@pytest.mark.parametrize` table like the forward's drives it.

> **Implemented as (Phase 4, actual).** Both kernels gained `to_id` (contextual shift → target-tail
> clamp), `max_id`, and the full `keep` predicate (causal/diagonal, `dist ≤ max_attn_len` window,
> `q_id==0 & kv_id<max_id` contextual opener), gated by build-time `has_targets/has_window/
> has_contextual`. The `dQ` (Q-owned) kernel copies the forward's mask + KV-range bounds verbatim
> (window `kv_lower` skip + contextual prefix opener → `seq_len`). The `dV`/`dK` (KV-owned) kernel
> uses the same predicate transposed; its streamed-query lower bound stays causal (`q ≥ kv`) except
> under `has_contextual`, where it drops to 0 so prefix queries (which attend keys above their
> diagonal) are included. **Gotcha found:** FlyDSL branch-ifies Python `if/else` on build
> constants, so a var assigned only inside both branches isn't visible after — use the forward's
> default-then-override idiom. Asymmetric dims (`attn_dim ≠ hidden_dim`, incl. non-64-divisible)
> validated via a hand-rolled dense oracle. Tests: `test_flydsl_bwd_variants` (targets, window,
> contextual, window+targets × bf16/f16) + `test_flydsl_bwd_asymmetric_dims`. Full suite: 26 green.

### Interlude — Performance baseline & tracking  *(after Phase 4, before Phase 5)*  🔜

**Goal.** Now that the backward is numerically correct (Phases 1–4), start **capturing runtime and
throughput** on the production shapes so later optimization (Phases 5–7) has a baseline to measure
against. We do **not** write our own bench script — we reuse the existing recsys harness and add a
`flydsl` provider next to its `aiter_triton` one.

**Reuse the recsys harness** (`meta/mvonstra-amd/recsys-kernels/recsys_harness/`):
- `bench_hstu.py` already has the whole machinery: a `BenchSpec` dataclass, deterministic jagged
  input generation (`_build_inputs`), `bench_one(spec, provider)` → `BenchRow` via
  `triton.testing.do_bench`, a CLI, and a `--correctness-check`. `common.hstu_flops` computes FLOPs
  with the right convention (`fwd = f1+f2`, **`bwd = 3·f1 + 2·f2`**), and it already benches an
  `aiter_triton` provider in both `--mode fwd` and `--mode bwd` — our natural comparison point.
- The input layout matches ours exactly: `q,k,v` are `(L, H, ·)` jagged with `seq_offsets` and
  optional `num_targets`, `alpha = 1/attn_dim`, causal.

**Hook-up (small, ~2 provider branches).** Add a `flydsl` provider to `bench_hstu.py`:
- In `_provider_fn` (forward): call `flydsl_hstu_attention_fwd(N=seq_len, alpha, q,k,v, seq_offsets,
  causal=True, num_targets, max_attn_len, contextual_seq_len)`.
- In `_make_bwd_bench_fn` (backward): `dout = torch.randn_like(v)`, then
  `return lambda: flydsl_hstu_attention_bwd(seq_len, alpha, q,k,v, dout, seq_offsets, True,
  num_targets, max_attn_len, contextual_seq_len)` — no autograd wrapper / pre-allocated grad buffers
  needed (it returns `(dq,dk,dv)` and launches both the KV-owned and Q-owned kernels; `do_bench`
  times the pair end-to-end, which is the number we care about).
- The `flydsl`/`aiter_triton` provider branches import lazily, so a minimal run needs only
  `torch`, `triton`, and our `aiter` — not `generative_recommenders`/`fbgemm`/`gdpa`.

**Metrics captured** (per shape, into a CSV via the harness's `write_rows_csv`): `ms_mean`,
`tflops_achieved`, `pct_of_bf16_roof`, plus shape metadata. Primary tracking metric is `ms_mean`
(and TFLOP/s); treat `pct_of_bf16_roof` as approximate — see caveat.

> **Caveat (roofline on MI350).** `common.py` only knows `MI300X` for a HIP device
> (`BF16_TFLOPS_PEAK = 1307`). We run on **MI350 / gfx950 (CDNA4)**, whose BF16 peak differs, so
> `pct_of_bf16_roof` is against the *wrong* peak until we add an MI350 branch to `common.py`'s
> device detection. Until then, compare **ms / TFLOP/s** across providers (apples-to-apples) rather
> than the roofline percentage.

**Production shapes to track** (all satisfy our `(batch·heads) % 8 == 0` contract):

| Shape | batch (B) | heads (H) | attn_dim | hidden_dim | notes |
|---|---|---|---|---|---|
| P1 | 1024 | 4 | 128 | 128 | large batch |
| P2 | 1024 | 8 | 128 | 128 | large batch, more heads |
| P3 | 120  | 4 | 128 | 128 | small batch |
| P4 | 120  | 8 | 128 | 128 | small batch, more heads |

Sweep each over a couple of `seq_len` (e.g. 512 / 1024 / 2048), `dtype=bf16`, and the mask regimes
we support (dense causal; then `target_size`, `max_attn_len`, `contextual_seq_len` singly), in both
`--mode fwd` and `--mode bwd`. `seq_len`, mask params, and dtype are all CLI/`BenchSpec` knobs.

**Workflow.**
1. Gate on correctness first (our pytest suite is green; the harness `--correctness-check` compares
   the FlyDSL *forward* against `pytorch_ref`). A backward correctness cross-check vs `aiter_triton`
   bwd can be added to the harness later if useful.
2. Run e.g.:

```bash
cd /workspaces/git/meta/mvonstra-amd/recsys-kernels/recsys_harness
HIP_VISIBLE_DEVICES=7 /workspaces/git/meta/aiter/flydsl_venv/bin/python bench_hstu.py \
  --providers flydsl aiter_triton --mode bwd \
  --batch-size 1024 --heads 4 --attn-dim 128 --hidden-dim 128 \
  --seq-len 1024 --target-size 0 --max-attn-len 0 --contextual-seq-len 0 \
  --out-csv runs/hstu_bwd_flydsl_baseline.csv
```

3. Commit the baseline CSV so Phases 5–7 (tiling, tuning) can be compared against it.

**Exit:** a `flydsl` provider merged into `bench_hstu.py`, a committed baseline CSV over P1–P4 ×
{fwd,bwd} × a small seq_len set, and FlyDSL-vs-Triton `ms`/TFLOP-s numbers recorded. *(This
interlude is measurement-only; it changes no kernel code and is a prerequisite for the Phase 7
tuning loop, which will reuse the same harness + CSV.)*

### Phase 5 — Tiling, block sizes & config plumbing  ✅ DONE

**Goal:** make the kernel tunable and give it the same host-side config machinery as the forward.

- **Red:**
  - `test_bwd_block_size_overrides`: run correctness across a few `(block_m, block_n, num_waves,
    waves_per_eu)` combos (explicit overrides) and assert grads still match — proves tiling
    independence.
  - CSV-plumbing tests mirroring the forward's `test_tuned_csv_*` (picked up, missing→empty,
    best-duration-wins) but pointed at a `_bwd` CSV and `_CSV_COLUMNS` (add
    `sequence_parallel` column).
- **Green:**
  - Add to `hstu_attention_kernels.py`: `_get_bwd_default_config`, `_get_bwd_tuned_config`,
    `_compile_bwd_launcher` (with `@functools.lru_cache`), `_validate_bwd_inputs`, and the public
    `flydsl_hstu_attention_bwd(N, alpha, q, k, v, seq_offsets, causal, num_targets, max_attn_len,
    contextual_seq_len, dout, *, block_m=…, …, stream=…)` returning `(dq, dk, dv)` — shaped as a
    sibling of `flydsl_hstu_attention_fwd`, reusing `_run_compiled`, `get_dtype_str`, the
    tuned→default→custom override chain.
  - Pre-allocate/zero `dq` (the accumulated output) as the Triton wrapper does.
- **Refactor:** deduplicate the CSV parsing/`_problem_key` logic shared with the forward (extract
  a small helper module if it pays off).
- **Exit:** grads correct across multiple tile configs; config selection unit-tested.

> **Implemented as (Phase 5, actual).** Added to `hstu_attention_kernels.py`, mirroring the
> forward: `_BWD_CSV_COLUMNS` (forward schema + a `sequence_parallel` column), a cached
> `_bwd_tuned_config_map` (best-duration-wins, reuses the forward `_problem_key`), `_get_bwd_tuned_config`,
> a conservative `_get_bwd_default_config` (`64/32/4/0`, valid across all supported/asymmetric dims),
> and a `@lru_cache`d `_compile_bwd_launcher` that resolves **tuned → default → custom-override** and
> builds the `(dV/dK, dQ)` launcher pair. `flydsl_hstu_attention_bwd` now routes through it (public
> signature unchanged). Default tuned CSV path: `hstu_attention_bwd_tuned.csv` (not committed yet —
> populated by the tuning phase). Tests: `test_flydsl_bwd_block_size_overrides` (4 configs) +
> `test_bwd_tuned_csv_*` (3). Full suite: 33 green. *(No `sequence_parallel` column — see Phase 6.)*

### Phase 6 — Sequence-parallel `dQ` synchronization  ❎ N/A (subsumed by Phase 3)

> **Resolution.** This phase existed to guard a `dQ` read-modify-write hazard that arises **only in a
> fused, KV-parallel backward** (theory §5.2–§5.3), where many KV programs add partial `dQ` into the
> same query rows. The Phase-3 implementation instead uses a **separate Q-owned `dQ` kernel**
> (`hstu_attention_bwd_dq.py`, grid `num_q_tiles·batch·num_heads`, register `dq_acc`, single store),
> alongside the KV-owned `dV`/`dK` kernel (grid `num_kv_tiles·batch·num_heads`). **Both kernels are
> already fully tile-parallel *and* single-writer**, so there is no cross-program `dQ` accumulation to
> synchronize — the goal of `SEQUENCE_PARALLEL=True` (full parallelism) and the safety of
> `SEQUENCE_PARALLEL=False` (no races) are achieved simultaneously by construction. Adding
> scratch/atomics/locks would only add a slower path. The vestigial `sequence_parallel` kwarg and CSV
> column were removed. The trade we accept is one extra `S`/`dS` recompute in the `dQ` kernel
> (recompute-over-bandwidth), which the interlude benchmarks show is competitive. The
> `test_bwd_sequence_parallel_matches_serial` / race-stress tests below are therefore not applicable
> (no serial-vs-parallel duality exists). Perf tuning of the tile configs moves to Phase 7.

<details><summary>Original (fused-kernel) plan, kept for context</summary>

**Goal:** allow KV blocks to run as **separate programs**, which requires guarding the `dQ`
read-modify-write (theory §5.3). This is the trickiest, most GPU-specific part and is deliberately
last.

- **Red:**
  - `test_bwd_sequence_parallel_matches_serial`: same inputs, assert
    `bwd(sequence_parallel=True) == bwd(sequence_parallel=False)` bit-for-bit-ish (tight tol,
    same dtype) — cross-checks the sync path against the already-trusted serial path.
  - A **stress/race test**: large `n`, many heads, repeated runs, to surface nondeterministic
    races (run several times; grads must be stable).
- **Green (choose one strategy, benchmark-driven — theory §5.3/§7.3):**
  1. **Scratch + reduce (recommended first):** each KV program writes its partial `dQ` to a
     scratch buffer keyed by `(kv_block, q_rows)`, then a second lightweight pass reduces. No
     atomics; easiest to get correct in FlyDSL.
  2. **Float `atomic_add`** into global `dQ` if `gfx950`/ROCm exposes it efficiently.
  3. **Spin-lock per query block** (port of Triton `atomic_cas`/`atomic_xchg`) — only if 1/2 lose
     on perf; reframe in FlyDSL primitives, don't port Triton literally.
- **Refactor:** hide the chosen strategy behind a `SEQUENCE_PARALLEL` compile-time flag and a
  `sequence_parallel` config/CSV column.
- **Exit:** sequence-parallel path matches serial path deterministically; race stress test stable.

</details>

### Phase 7 — Benchmarking & tuning  ✅ DONE (gfx942, dense causal)

**Goal:** know how fast we are and produce a tuned CSV, matching the forward's tuning workflow.

- **Red / gate:** a `bench` sanity assert (grads still correct at bench shapes) so tuning can't
  silently regress correctness.
- **Green:**
  - Extend `op_tests/op_benchmarks/flydsl/bench_hstu_attn.py` with a backward path comparing
    against `triton_hstu_attention_bwd` (`_hstu_attn_bwd`) on the target arch; report TFLOP/s and
    GB/s (reuse `get_flops`/`get_bytes` from the triton test util, doubling for bwd matmul count).
  - Sweep `(block_m, block_n, num_waves, waves_per_eu, sequence_parallel)` and emit
    `hstu_attention_bwd_tuned.csv`.
- **Exit:** benchmark runs on `gfx950` (and `gfx942`), tuned CSV committed, perf recorded relative
  to Triton bwd.

> **Implemented as (Phase 7, actual).** Rather than extend the fork's single-shape forward bench, a
> dedicated tuner `op_tests/op_benchmarks/flydsl/tune_hstu_attn_bwd.py` sweeps a curated
> `(block_m, block_n, num_waves, waves_per_eu)` shortlist per problem, times the full backward (both
> kernels) via `triton.testing.do_bench` on realistic jagged inputs (mirrors the recsys harness's
> length distribution, not the test generator's power-law-clamped one), and writes the fastest config
> per problem to `aiter/ops/flydsl/hstu_attention_bwd_tuned.csv` (best-duration-wins, invalid configs
> auto-skipped). No `sequence_parallel` axis (Phase 6 N/A). Ran the `prod` grid (P1–P4 ×
> {512,1024,2048}, bf16, `gfx942`): **`(192,32,4,0)` is fastest at N=2048** (~6–9% over the
> `(64,32,4,0)` default), default wins at N≤1024. End-to-end via the recsys harness, B1024/H8/N2048
> bwd improved **71.5 → 64.0 ms (−10.5%)**, closing the Triton gap from 1.29× to 1.16×. Correctness of
> the shipped tuned config is locked by adding `(192,32,4,0)` to `test_flydsl_bwd_block_size_overrides`.
> Follow-ups: tune the mask regimes and `N=16384`, and re-tune on `gfx950`. Full suite: 34 green.

### Phase 8 — Autograd integration & end-to-end  ✅ DONE

**Goal:** make the FlyDSL forward + backward usable as a single differentiable op, mirroring
Triton's `_AttentionFunction`.

- **Red:**
  - `test_flydsl_autograd_end_to_end`: wrap fwd+bwd in a `torch.autograd.Function`, run
    `torch.autograd.gradcheck`-style or compare `.grad` after `.backward()` on a scalar loss
    against `torch_hstu_attention` autograd, across the parametrized matrix.
- **Green:**
  - Add a `FlydslHstuAttention(torch.autograd.Function)` whose `forward` calls
    `flydsl_hstu_attention_fwd` and saves `(q,k,v,seq_offsets,num_targets)` + scalars, and whose
    `backward` calls `flydsl_hstu_attention_bwd` and returns `(None, None, dq, dk, dv, None, …)`
    with the arg positions matching the public signature (copy the Triton `_AttentionFunction`
    None-padding pattern).
- **Exit:** end-to-end gradient test green; op is drop-in differentiable.

> **Implemented as (Phase 8, actual).** Added `FlydslHstuAttention(torch.autograd.Function)` to
> `hstu_attention_kernels.py`, mirroring the Triton `_AttentionFunction`: `forward(ctx, N, alpha, q,
> k, v, seq_offsets, causal, num_targets, max_attn_len, contextual_seq_len)` saves `(q,k,v,
> seq_offsets[,num_targets])` + scalars and calls `flydsl_hstu_attention_fwd`; `backward` runs under
> `torch.inference_mode()`, calls `flydsl_hstu_attention_bwd`, and returns `(None, None, dq, dk, dv,
> None, None, None, None, None)` (grads for q/k/v only). A `flydsl_hstu_attention(...)` convenience
> wrapper exposes `.apply`, and both are added to `__all__`. Test `test_flydsl_autograd_end_to_end`
> compares `.grad` after `out.backward(dout)` against the autograd oracle for dense/targets/window ×
> bf16/f16 (6 cases). Full suite: **40 green**.

### Phase 9 — Layout-algebra cleanup & de-duplication  ✅ DONE

**Goal:** reduce hand-rolled integer index arithmetic and remove the ~3× copy-pasted indexing
idioms across the HSTU backward kernels by (a) centralizing shared helpers in one module and
(b) expressing index maps with FlyDSL **layout algebra** (`make_layout` / `idx2crd` / `crd2idx` /
`make_view` / composed swizzle) wherever it is a clean, behavior-preserving win.

**Finding that scopes this phase (from a survey of all three kernels).** There is **no
forward↔backward "layout-algebra gap"**: the forward (`hstu_attention_fwd.py`) and both backward
kernels (`hstu_attention_bwd.py`, `hstu_attention_bwd_dq.py`) share the *same* hybrid style. They
already use layout algebra for the coalesced vector loads (`grouped_loader` = `make_layout` +
`make_view`), but do manual arithmetic for: lane/wave decomposition (`tid // WARP_SIZE`,
`lane % 16`, …), the XOR LDS swizzle (`k_swz_col`/`q_swz_col`), the DMA pass-map loops
(`pair = tid + d*BLOCK_THREADS; row = pair // pairs_per_row; …`), MFMA operand gather, and the
epilogue scatter. Crucially these idioms are **duplicated verbatim** (roles relabelled) across the
three files. Some manual indexing is **inherent and stays**: the `raw_ptr_buffer_load_lds` DMA path,
the bf16 `pack_p`/fragment bitcast, and the jagged `seq_offsets`/`to_id` masking (business logic,
not addressing).

**Non-goals (explicitly out of scope for this cleanup).** A full CuTe-style rewrite to
`TiledMma.partition_A/B/C` and `make_tiled_copy` — no aiter FlyDSL kernel uses those yet, it would be
a large re-architecture of hot, *tuned* kernels, and the forward blueprint doesn't use them either.
No change to the numeric recipe, tile configs, grid, or public API. This phase must be
**behavior-preserving** (same MLIR-level addressing, same results, no perf regression).

**Default scope decision (revisit if perf/risk dictates).** Refactor the **backward kernels only**
(our domain, Phases 1–8), leaving the vendored forward untouched but designing the shared module so
the forward *could* adopt it later. Staged:
- **9a — Shared helper module.** Extract the duplicated idioms into
  `kernels/hstu_attention_common.py`: `grouped_loader`, a `swz_col(tile_row, col, rows, shift)`
  swizzle helper, a lane/wave decomposition helper, and a DMA pass-map builder. Both backward kernels
  import them (removes the copy-paste; single source of truth).
- **9b — Layout-algebra idioms.** Replace manual lane/wave arithmetic with
  `idx2crd(tid, make_layout(...))` + `fx.get(...)` (the "Pattern B" idiom other aiter kernels already
  use), i.e. `(wave, lane)` and `(lane_div_16, lane_mod_16)` come from layouts, not `//`/`%`.
- **9c — (optional) swizzle as a layout.** Evaluate composing the XOR swizzle into the LDS layout
  (`make_composed_layout` + `Swizzle`) instead of applying `col ^ …` at each access; adopt only if it
  stays behavior-identical and doesn't regress perf.

**Red / gate:** the existing **40-test** suite is the correctness guard (this is a pure refactor, so
no new numeric tests are required); add a tiny standalone unit test only for any extracted helper with
self-contained logic (e.g. `swz_col`). **Green:** kernels build and the full suite passes. **Refactor:**
keep each step small and independently test-green. **Exit:** suite green **and** no regression on the
tuned `N=2048` shape (spot-check via the tuner/bench, e.g. B1024/H8/N2048 ≈ 64 ms).

> **Implemented as (Phase 9, actual).** Added `kernels/hstu_attention_common.py` with `grouped_loader`
> (unchanged; `make_layout` + `make_view`), `swz_col(tile_row, col, rows, shift)` (the shared XOR
> swizzle), and `decode_lane(tid, num_waves, warp_size, mfma_n)` which derives
> `(wave_id, lane, lane_div_16, lane_mod_16)` from two `idx2crd(tid, make_layout(...))` maps instead of
> `//`/`%`. Both backward kernels (`hstu_attention_bwd.py`, `hstu_attention_bwd_dq.py`) now import
> these; their local `grouped_loader`/`*_swz_col`/lane-math were removed (`*_swz_col` kept as one-line
> aliases so call sites are untouched). **Gotcha:** `idx2crd`/`fx.get` yield MLIR `index`-typed values;
> the kernels' address math is `i32`, so `decode_lane` casts each coordinate back with `fx.Int32(...)`
> (otherwise MLIR fails to verify `arith.muli(index, i32)`). Kept behavior-preserving and
> perf-neutral (B1024/H8/N2048 bwd 63.95 ms vs 64.04 ms pre-refactor). Scope held to the **backward**
> kernels; the vendored forward is untouched (the shared module is forward-adoptable later). No
> `TiledMma`/`make_tiled_copy` rewrite (deliberate non-goal). Full suite: **40 green**.

---

## 6. Cross-cutting correctness gotchas (guard each with a test)

From theory §6 and §7.2 — each deserves at least one targeted assertion:

- **Identical mask fwd/bwd**, including diagonal-always-valid and target/contextual position
  clamps. Mismatches show up only near sequence boundaries → add a test whose failure would be
  boundary-local (small `n`, `num_targets>0`, `contextual_seq_len>0`).
- **`1/N` is the constant `MAX_SEQ_LEN`, not `1/n`.** Easy to conflate in jagged code → test with
  `n < N` sequences.
- **`alpha` appears in `S`, `dQ`, `dK` but NOT `dV`.** → the closed-form identity test plus a test
  with `alpha != 1` catches double/missing application.
- **SiLU derivative** uses the exact `σ(1+S(1-σ))` from recomputed `S,σ` (fast-math `exp2`/`rcp`),
  not a library sigmoid → keep fwd/bwd sigmoid shared.
- **fp32 accumulation, cast on store** → without it, `dQ/dK` allclose will fail tolerance.
- **Jagged offset-then-mask** for every load/store; row coords cast to i64 to avoid address
  overflow on packed tensors (copy the forward's `fx.Int64(...)` discipline).

---

## 7. Definition of done

1. `flydsl_hstu_attention_bwd` returns `dq, dk, dv` matching `torch.autograd.grad` on
   `torch_hstu_attention` within recorded tolerances, for the full parametrized matrix
   (causal; `num_targets`, `max_attn_len`, `contextual_seq_len`, and asymmetric/`%64≠0` dims),
   in f16 and bf16.
2. Serial and sequence-parallel paths agree deterministically.
3. Host plumbing (validate/build/compile/config/CSV) mirrors the forward.
4. A `torch.autograd.Function` exposes fwd+bwd end-to-end and passes a gradient test.
5. Benchmarked against Triton bwd on `gfx950` (and `gfx942`), tuned CSV committed.
6. Every phase's tests remain green (full suite runs with `HIP_VISIBLE_DEVICES=7`).

---

## 8. Phase → test map (quick reference)

| Phase | Capability | Primary tests (new) | Status |
|---|---|---|---|
| 0 | Scaffold + oracle | `test_reference_oracle_runs`, `_validate_bwd_inputs` units | ✅ done |
| 1 | `dV`, causal, single-writer | `test_flydsl_bwd_dv_causal` (bf16/f16 × 3 shapes) | ✅ done |
| 2 | `dK` + `dS` | `test_flydsl_bwd_dv_dk_causal`, `test_silu_derivative_formula` | ✅ done |
| 3 | `dQ`, full dense causal | `test_flydsl_bwd_all_causal` (dq/dk/dv, bf16/f16 × 3 shapes) | ✅ done |
| 4 | Mask variants | `test_flydsl_bwd_variants`, `test_flydsl_bwd_asymmetric_dims` | ✅ done |
| — | Perf baseline (interlude) | `flydsl` provider in recsys `bench_hstu.py`; baseline CSVs over P1–P4 | ✅ done |
| 5 | Tiling + config | `test_flydsl_bwd_block_size_overrides`, `test_bwd_tuned_csv_*` | ✅ done |
| 6 | Sequence-parallel `dQ` | — (subsumed by Phase-3 two-kernel design; knob removed) | ❎ N/A |
| 7 | Bench + tune | `tune_hstu_attn_bwd.py` + `hstu_attention_bwd_tuned.csv` | ✅ done (gfx942, dense causal) |
| 8 | Autograd integration | `test_flydsl_autograd_end_to_end` | ✅ done |
| 9 | Layout-algebra cleanup / de-dup | existing 40-suite as guard | ✅ done |

---

### File references (for implementers)

- Theory / derivation: `meta/aiter/docs/2026-07-07_HSTU_theory.md`
- Reference oracle: `meta/aiter/op_tests/triton_tests/utils/hstu_attention_ref.py` (`torch_hstu_attention`)
- Triton bwd blueprint: `meta/aiter/aiter/ops/triton/_triton_kernels/attention/hstu_attention.py` (`_hstu_attn_bwd`)
- Triton host wrapper + autograd: `meta/aiter/aiter/ops/triton/attention/hstu_attention.py` (`triton_hstu_attention_bwd`, `_AttentionFunction`)
- FlyDSL forward host API: `meta/robin_aiter/aiter/ops/flydsl/hstu_attention_kernels.py`
- FlyDSL forward device kernel: `meta/robin_aiter/aiter/ops/flydsl/kernels/hstu_attention_fwd.py`
- FlyDSL forward tests (harness to mirror): `meta/robin_aiter/op_tests/flydsl_tests/test_flydsl_hstu_attention.py`
- FlyDSL forward bench: `meta/robin_aiter/op_tests/op_benchmarks/flydsl/bench_hstu_attn.py`
