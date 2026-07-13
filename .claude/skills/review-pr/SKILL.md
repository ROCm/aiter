---
name: review-pr
description: AI code review for aiter PRs. Catches perf regressions, silent correctness bugs, dispatch gate holes, and AI-generated code patterns. Invoke with a PR number; works through fetch → semantic understanding → rule checklist → verdict. Add new rules here as patterns emerge from real reviews.
argument-hint: <PR number>
---

# aiter PR Review

---

## Step 1 — Fetch

```bash
PR=$1  # PR number from skill argument
REPO="ROCm/aiter"

# Full metadata
gh pr view $PR --repo $REPO --json title,body,number,labels,files,author,reviews,comments > /tmp/pr_meta.json

# Diff
gh pr diff $PR --repo $REPO > /tmp/pr.diff

# Linked issue (extract from body "fix: #NNN" or "close #NNN")
ISSUE=$(cat /tmp/pr_meta.json | python3 -c "
import json,re,sys
body = json.load(sys.stdin).get('body','') or ''
m = re.search(r'(?:fix|close|resolve)[s]?[: ]*#(\d+)', body, re.I)
print(m.group(1) if m else '')
")
[ -n "$ISSUE" ] && gh issue view $ISSUE --repo $REPO --json title,body > /tmp/pr_issue.json

# Prior reviewer comments (top-level)
cat /tmp/pr_meta.json | python3 -c "
import json,sys
d = json.load(sys.stdin)
for r in d.get('reviews',[]):
    b = (r.get('body','') or '').strip()
    if b: print(f'[REVIEW {r[\"author\"][\"login\"]}] {b[:200]}')
for c in d.get('comments',[]):
    b = (c.get('body','') or '').strip()
    if b: print(f'[COMMENT {c[\"author\"][\"login\"]}] {b[:200]}')
"

# Inline review comments (line-level code comments — often more specific than top-level)
gh api "repos/$REPO/pulls/$PR/comments" | python3 -c "
import json,sys
comments = json.load(sys.stdin)
for c in comments:
    author = c.get('user',{}).get('login','')
    body = (c.get('body','') or '').strip()
    path = c.get('path','')
    line = c.get('line') or c.get('original_line','')
    if body and 'copilot' not in author.lower() and 'bot' not in author.lower():
        print(f'[INLINE {author}] {path}:{line}')
        print(f'  {body[:250]}')
" 2>/dev/null
```

Read the diff and PR body before proceeding.

---

## Step 2 — Semantic Understanding (answer all 5 before rules)

Work through these by reading the diff, not the description alone.

**Q1 — What specifically changed computationally?**
Not "improves perf" — what algorithm/formula/data flow changed?
_Answer:_

**Q2 — Hardware scope: which arch(es), precision(s), execution phase(s)?**
gfx942 / gfx950 / gfx1250? fp16/bf16/fp8? decode / prefill / both?
_Answer:_

**Q3 — Does this change any public aiter API?**
New symbol in `aiter/ops/*.py`, new kwarg on existing op, change to `aiter/__init__.py`?
_Answer:_

**Q4 — Performance claim: what is the mechanism?**
Not "faster" — WHY is it faster? (fewer memory round-trips, fewer kernel launches, better tiling?)
_Answer:_

**Q5 — Does the description explain WHY or only WHAT?**
"Fuses kernels for speedup" = surface. "Eliminates intermediate HBM write between rmsnorm and quant" = understanding.
If surface-level only → treat as elevated AI-code risk.
_Answer:_

---

## Step 3 — PR Type Classification

Check which type(s) apply; these determine which Step 5 categories are mandatory.

- [ ] **New kernel / new Triton op** → B1 (dispatch gate), B2 (tl.load mask), B4 (new routing value unhandled?), A1 (sibling variants), D1 (atomic zero-init), HK6 (UT)
- [ ] **New constexpr / routing flag / new dtype or arch value added** → B4 (do ALL dispatch branches handle the new value, or assert on it?), C4 (new arch string literal?)
- [ ] **Tuning config (CSV / YAML)** → D3 (hipblaslt), HK4 (kpack:1)
- [ ] **Dispatch logic change** → B1 (silent bypass), B3 (string normalization), B4 (new value unhandled?), A3 (scope too broad)
- [ ] **Replaces existing kernel as default** → D2 (rollback env-var)
- [ ] **Core file change** (see Tier table below) → full Step 4 risk assessment
- [ ] **Refactor / rename** → HK2 (unrelated files), variable name mismatch check
- [ ] **FP8 / quantization path** → C1 (fnuz by dtype), C2 (fp8_max hardcoded), D1 (atomic zero-init)
- [ ] **Perf / benchmark PR** → P1 (numbers with units), P5 (setup cost excluded?), P2 (production shapes), P3 (reproducible)
- [ ] **Test / benchmark only** → P2 (production shapes), HK6 (aiter-op-test format)
- [ ] **Async / multi-stream** → G1 (stream sync missing), G1b (blocking queue.get without timeout in serving code)
- [ ] **FlyDSL kernel** → D10 (compile result called?), D10b (arith.unwrap() before arith.bitcast?)
- [ ] **New if/elif dispatch with variable assignment** → D1b (UnboundLocalError on uninitialized path)

---

## Step 4 — Core File Risk Assessment

**What makes a file "backbone"?** Apply these three questions to any file in the diff — including new files not in the table below.

```
Q1 — Tier 1 test: If this file has a Python syntax error or fails to import,
     does `import aiter` still succeed?
     → NO  → Tier 1 (system-critical: aiter itself breaks)

Q2 — Tier 2 test: Does this file contain the Python dispatch logic that
     selects which kernel to run for an op class,
     AND is that op used by >1 production model family (DSv3, Kimi, MiniMax…)?
     → YES → Tier 2 (op-class critical: wrong result for ALL users of that op)

Q3 — Tier 2 alt: Is this file the public aiter API for an op
     (`from aiter import X` imports from here)?
     → YES → Tier 2 (signature change silently breaks all consumers)

Otherwise → Tier 3 (individual kernel or model-specific code).
```

The table below is the current snapshot — use it to confirm, but Q1/Q2/Q3 to classify new files.

Backbone files ranked by git commit frequency (2025–2026) and blast radius:

| Tier | File | Git commits | Blast radius | Failure mode |
|------|------|-------------|-------------|--------------|
| **1** | `aiter/jit/core.py` | 182 | **ALL ops** — JIT compilation engine | Any import of aiter fails; zero ops load |
| **1** | `aiter/__init__.py` | 52 | **ALL** vLLM/SGLang/ATOM users | `ImportError` or silent namespace truncation below broken import |
| **1** | `aiter/ops/*.py` (any) | varies | All consumers of that op | `AttributeError` at call time in downstream |
| **2** | `aiter/fused_moe.py` | 119 | All MoE models (DeepSeek, Kimi, MiniMax) | Wrong expert routing, silent accuracy drop |
| **2** | `aiter/ops/mha.py` | 89 | All MHA attention paths | Wrong attention output, crash |
| **2** | `aiter/ops/attention.py` | 66 | MLA/paged attention dispatch | Wrong KV, accuracy drop |
| **2** | `aiter/ops/gemm_op_a8w8.py` | 59 | All FP8 quantized GEMM | Wrong matmul result, silent accuracy drop |
| **2** | `aiter/mla.py` | 57 | All MLA decode/prefill (DSv3/Kimi) | Wrong KV, accuracy drop, crash |
| **2** | `aiter/tuned_gemm.py` | 52 | All GEMM-backed ops | `assert False` crash or silent fallback to slow path |
| **2** | `aiter/ops/moe_op.py` | 51 | MoE op dispatch table | Wrong dispatch, wrong expert weights |
| **2** | `aiter/ops/quant.py` | 49 | All quantization paths | Wrong scale, silent accuracy drop |
| **3** | Individual kernel `.py`/`.cu` | — | Ops using that kernel | Depends on kernel type |

**`aiter/__init__.py` special rule**: The import block must NOT be wrapped in try/except.
Any new import added here → check the imported module for bare `ImportError` paths that
could silently truncate the namespace.

**`aiter/jit/core.py` special rule**: This file bootstraps the entire JIT compilation pipeline.
A syntax error, wrong default, or broken env-var handling here means zero aiter ops load.
Changes here require e2e smoke test across all GPU arch targets.

**Mandatory backbone checks — must be answered before writing the verdict:**

For **Tier 1** files (jit/core.py, __init__.py, ops/*.py):
- [ ] List every public symbol changed. Grep for all callers across aiter itself: `grep -rn '<symbol>' aiter/`. If a caller is not covered by the PR's test, flag it.
- [ ] For `__init__.py`: does the new import have a bare `ImportError` path that could silently truncate the namespace?
- [ ] For `jit/core.py`: is there an e2e smoke test that loads all kernels on gfx942 AND gfx950 after this change?
- [ ] State explicitly: if this change is wrong, what breaks and how would it be detected? (all ops fail / one op family fails / silent wrong value)

For **Tier 2** files (fused_moe, mha, attention, gemm, mla, tuned_gemm, quant):
- [ ] Which model families (DSv3, Kimi, MiniMax, GLM…) use this op? Is at least one from each family in the test?
- [ ] Are production shapes tested? At minimum: decode (M=1, TP=4/TP=8) AND prefill (ISL=4096, TP=4/TP=8).
- [ ] Does the change affect gfx942 only, gfx950 only, or both? If both, are both arch paths tested?

**AI code red flag — verbatim duplication across backbone files:** Same algorithm copy-pasted into 2+ backbone files with only variable names changed. See D5.

---

## Step 5 — Rule Checklist

Six failure categories — work all six in order. Severity per finding: 🔴 block / ⚠️ should fix / 📝 note.

| Category | Core question | Key triggers |
|---|---|---|
| **A. Coverage gaps** | Same bug elsewhere? Same code other configs? | `_opt`, `_prefill_opt`, `_v2`; shared path; broad `if` condition |
| **B. Silent bypass** | Does every input reach the right branch? | gated-off param; string alias; non-aligned dim; proxy metric |
| **C. Hardcoded arch/dtype** | Does the constant break on another GPU or fp8 flavor? | `240.0`, `448.0`; arch name for fnuz; `bf16` fixed |
| **D. Uninitialized state** | Is the buffer clean before atomic/kernel launch? | `::empty()`+`atomic_fmax`; `fill_(0)` missing |
| **E. Cross-repo sync** | Does the consumer know about this change? | new aiter symbol; default-preserving new param; plugin bridge |
| **F. Resource duplication** | Does the change double GPU memory silently? | new `_preshuffled`/`_quantized` weight alongside original |

---

### A — Coverage Gaps
_"Fixed one path; the same bug lives in a sibling."_

**A1 — Sibling kernel not fixed** ⚠️ (🔴 if in Tier-1/2 backbone)
Fix changes address calc, bounds check, type widening, or data layout in a CUDA/HIP kernel:
scan the same file for variants named `_opt`, `_prefill`, `_decode`, `_prefill_opt`, `_v2`, `_fast`.
Real example (PR#3841): strided q_nope OOB fix applied to decode kernel; `_prefill_opt` in the same file had the same bug unfixed.
→ `⚠️ A1: same bug may exist in [variant] — check kernel family in this file`

**A2 — Shared path, no cross-model validation** ⚠️
Changed code shared across model families (not model-specific): validated on all?
Real example (PR#3891): valarLip: "please make sure e2e CI passes before changes to common part."
→ `⚠️ A2: change touches shared path — e2e or cross-model validation needed`

**A3 — Activation condition broader than validated scope** ⚠️
New dispatch condition (e.g., `if is_deepseek():`) enables a kernel for more archs/models than tested.
Real example (vLLM#16435): FusedMoE activated for wrong model families → follow-up restrict PR needed.
→ `⚠️ A3: activation condition [X] enables more than validated scope [Y]`

---

### B — Silent Bypass
_"The code looks complete but certain inputs silently take the wrong path."_

**B1 — Dispatch gate with unchecked parameter** 🔴
New `if/elif/else` branch: for each parameter gated off — is it **asserted** (None/zero) or **forwarded**?
If neither: wrong results, no crash, no error.
Trigger: `dropout_p`, `window_size`, `block_table`, `logits_soft_cap`, `alibi_slopes`, `is_causal`.
Real example (PR#3576): `block_table is not None` False-branch computed dense attention silently.
Real example (PR#3390): `is_causal=True` not forwarded → "fake causal" fmha passed all CI.
→ `🔴 B1: [param] silently ignored in [branch] — assert or forward`

**B2 — Triton tl.load / tl.store without mask** 🔴
Unmasked load when dim is not a multiple of BLOCK_SIZE → silent garbage read, no segfault.
Common non-aligned dims: `seqlen`, `vocab_size`, `hidden_dim`, `num_heads`, `head_dim`, `kv_lora_rank`.
→ `🔴 B2: tl.load at [line] missing mask= — silent OOB on non-aligned inputs`

**B3 — String dispatch without normalization** ⚠️
`quant_type == "per_token"` before normalizing: aliases `"fp8_per_token"`, `"per-token"`, `QuantType.per_Token` silently miss the branch.
Real example (PR#3981): raw string compare in `parallel_state.py` — alias callers missed torch-compile fast path.
→ `⚠️ B3: string dispatch [cond] without normalization — aliases fall through to slow path`

**B4 — New dispatch value not handled by all paths, no warning** ⚠️/🔴
When a PR introduces a new routing value to a multi-way dispatch — a new dtype string (`'fp4'`), a new arch string (`'gfx1201'`), a new layout flag (`SWIGLU_INTERLEAVED`), a new constexpr enum value — every reachable dispatch branch must either (a) handle it explicitly, (b) fall through to a documented safe default, or (c) assert/warn before the wrong branch is reached. If any reachable branch silently falls through to behavior that is wrong for the new value, flag it.
Severity: 🔴 if the wrong path produces incorrect output silently (wrong layout, wrong kernel, wrong scale). ⚠️ if the wrong path is a safe-but-suboptimal default (e.g., generic tile depths instead of tuned fp4 depths).
Exception: an upstream assert/raise/isinstance check that prevents the bad value from entering the branch → not B4. A runtime assert that fires for the dangerous combo → not B4.
FP self-check: Is the uncovered branch actually reachable with the new value? Is there a caller contract (documented or asserted) guaranteeing the bad combo never occurs?
Real examples: GGUU flag not wired into gfx950 Triton path — runtime assert guards the explicitly dangerous combo but the remaining gap is silent (aiter#4169); fp4 silently falls through to `in_dtype in ('fp8','int8')` tile table, uses generic preload depths (aiter#3941); cross-attention + mt=1 on gfx1250 falls through to get_heuristic_kernel with no gfx1250 kernel compiled for that combo (aiter#3939).
→ `🔴/⚠️ B4: [new value] reaches [branch] which assumes [old value] — [what wrong thing happens] — add assert or explicit handling`

**B5 — Triton `tl.constexpr` safety check disabled without invariant proof** ⚠️
A `tl.constexpr` bool that gates a validity check (e.g., `CHECK_NEG_ONE_SENTINEL`, `CHECK_BOUNDS`) can be set `False` by a caller to skip the check. If the invariant the check enforces is not independently guaranteed on that path, illegal memory access or silent wrong values result.
Trigger: new `tl.constexpr` bool in a Triton kernel that disables a bounds/sentinel/validity check; caller comment says "X path can disable this" without documenting what guarantees the invariant holds on that path.
Real example (ATOM#1498): `CHECK_NEG_ONE_SENTINEL=False` disables the -1 slot filter in the paged prefill kernel; illegal access if any -1 slot appears without the check.
→ `⚠️ B5: [constexpr] disables [check] — document which caller invariant guarantees no [invalid value] on that path`

**B6 — API propagation incompleteness** 🔴/⚠️
When an API surface changes in dimension X, all downstream receivers (Y) must be updated. Unhandled propagation silently falls through to wrong behavior (Z).

| Sub-type | X (what changed) | Y (downstream not updated) | Z (failure) | Sev |
|----------|-----------------|---------------------------|-------------|-----|
| param-discard | new param in signature | function body | value accepted but never used | ⚠️/🔴 |
| param-removed | param removed from call | same-file call sites | TypeError at call time | 🔴 |
| repr-key | new Gluon constexpr | kernel repr key list | stale JIT binary served | 🔴 |
| arch-discard | arch-specific kwarg | non-target-arch path | kwarg silently discarded | ⚠️ |
| dispatch-silent | multi-backend fallback | caller logging | backend switch with no diagnostic | ⚠️ |
| rename | public symbol renamed | all importers | AttributeError at import time | 🔴 |

Severity (param-discard): 🔴 if param controls output correctness (`expert_mask`, `q_scale`, `kv_scale`); ⚠️ for performance knobs or optional features with working defaults.
Exception: method override where base class forces the signature but subclass legitimately ignores the param — flag as 📝 (structural discard, not a bug).
Real examples (param-discard): `expert_mask` accepted but `# return None` commented out → TP expert-parallel callers silently routed wrong; `v_scale` strides never computed — `sc_off` indexes v_scale_ptr using k_scale strides, wrong scale on non-contiguous tensors (aiter#3959); `gate_up` discarded when `is_guinterleave=False` (aiter#4167).
→ `🔴/⚠️ B6-[sub-type]: [what changed] — [downstream not updated] — [failure]`

**B7 — Over-conservative assert blocks valid shapes** ⚠️
`assert M % tileM == 0` when the kernel pads internally and handles non-aligned M.
Real example (PR#3998): wrapper asserted alignment; asm kernel padded — valid small-M shapes rejected at the Python layer.
FP self-check: Does the kernel actually handle non-aligned inputs, or does the assert reflect a real hardware requirement?
→ `⚠️ B7: assert [constraint] may be unnecessary — verify kernel handles non-aligned inputs before removing`

---

### C — Hardcoded Arch / Dtype Assumptions
_"The constant is correct for gfx942/fnuz; it silently breaks on gfx950 or OCP e4m3."_

**C1 — FP8 fnuz check uses arch name** ⚠️
`if "gfx942" in arch: treat_as_fnuz()` — wrong. Same arch can have both fn and fnuz in flight.
Check IS fnuz: `tensor.dtype == fp8_fnuz`. Gate CONVERSION by arch is OK; inspection must use dtype.
Real example (PR#4073): valarLip: "check _is_fnuz by tensor's DType instead of arch."
→ `⚠️ C1: fnuz check uses arch name — use tensor.dtype comparison`

**C2 — FP8 scale bound hardcoded** ⚠️
`fp8_max = 240.0` → correct for fnuz (e4m3fnuz max=240), wrong for OCP e4m3 (max=448).
Use `get_dtype_max(dtype)` to derive; add a runtime guard if gfx942-only.
Real example (PR#4015): yzhou103: "would break for OCP e4m3 (max=448)."
→ `⚠️ C2: fp8_max hardcoded to [value] — use get_dtype_max(dtype)`

**C3 — Dtype hardcoded without checking actual tensor** ⚠️
Fixed `bf16`, `fp8_e8m0`, or similar in a forward path that handles multiple configs.
Real examples: ATOM#1423 "not always bf16"; ATOM#1458 "hard code to fp8_e8m0?"
→ `⚠️ C3: dtype hardcoded to [type] — should derive from actual tensor/config`

**C4 — New GPU arch string literal in dispatch condition** ⚠️
**FP self-check first (do this before deciding to fire):** Search the unchanged lines of this file for the same arch string (e.g., `'gfx1250'`). If that string already appears on an unchanged line → **do not fire** (pre-existing style, not a new violation). Only proceed if the arch string is genuinely new to this file.
Trigger (only after self-check passes): a new `+` line introduces an arch string literal in a dispatch condition (`if arch == 'gfx1250':`, `if 'gfx950' in arch_name:`), rather than routing through the central kernel registry or a named constant.
Also exempt: arch strings used only in comments, docstrings, or directory path strings; arch strings imported from a central registry module; arch strings used as **capability guards inside a kernel-specific wrapper function** (not in the centralized dispatch layer) — e.g., `get_gfx() == 'gfx1250'` inside `flydsl_flash_attn_batch_func` determines whether the FlyDSL variant is available; that check belongs in the wrapper, not in the central registry, and does not trigger C4 (aiter#3870).
Real examples: `'gfx1250'` new to `fused_mxfp4_quant.py` dispatch logic where no prior arch literals existed (aiter#3937 → fire C4); `'gfx1201'` added to `unified_attention.py` where `'gfx1250'` was already on line 79 (aiter#3956 → skip, pre-existing style); `get_gfx() == "gfx1250"` inside FlyDSL wrapper `flydsl_flash_attn_batch_func` (aiter#3870 → skip, capability guard not centralized dispatch).
→ `⚠️ C4: new arch string '[gfxNNNN]' hardcoded in dispatch — route through arch registry or named constant`

---

### D — Uninitialized / Boundary State
_"The code writes or reads memory that was never properly initialized."_

**D1 — Atomic reduction on uninitialized buffer** 🔴
`atomic_fmax(*ptr, val)` = `*ptr = max(*ptr, val)`. If `*ptr` is uninitialized (from `::empty()`),
garbage dominates the max → corrupted amax → corrupted FP8 descale → silent wrong quantization.
Trigger: `atomic_fmax` / `atomic_max` + `::empty()` or non-zeroed allocation near it.
Severity: 🔴 for atomic accumulation (atomic_fmax, atomicAdd) — garbage propagates into every output element. ⚠️ for partial-sum buffers where a zero-weight coefficient mathematically cancels the contribution (e.g., online softmax with empty batch: `exp(-inf) × garbage = 0`); still flag because `0.0 × NaN = NaN` on IEEE hardware if the allocator returns dirty pages.
Real example (PR#4015): yzhou103: "AiterTensor::empty does not zero-initialize... garbage in v_amax silently corrupts descale."
→ `🔴 D1: [buffer] passed to atomic_fmax not zero-initialized — use ::zero() not ::empty()`

**D1b — Python-side UnboundLocalError from conditional assignment** 🔴
A variable is assigned inside an `if/elif` branch but referenced unconditionally after the block. Python does not detect this statically — `UnboundLocalError` or `NameError` fires only at runtime when the skipped branch is exercised. Silent in test environments that never hit the uninitialized path.
Trigger: new `if/elif` gate assigns a variable (`result = ...`) on some branches; a later line references it without a pre-block default. Check: is there a `var = None` or `var = default_val` before the if-block?
Exception: if there is a definitive `else` branch that also assigns the variable, or if the variable is only ever used inside the branch that assigns it.
Real example (ATOM#860): `needs_independent_noise` returned from `prepare_model()` tuple but assigned only in one branch of `prefill_forward` — other branch paths raised `NameError` when the sampler tried to use it.
→ `🔴 D1b: [var] assigned only inside [branch] but referenced unconditionally — add [var = default] before the if-block`

**D2 — New default path without rollback env-var** ⚠️
New implementation replaces existing default before wide validation: is there an env var to revert?
Real example (PR#3266): flydsl sort replaced opus sort; reviewer: "gate flydsl behind env var until validated on broader workloads."
→ `⚠️ D2: new default path needs rollback env-var for safe rollout`

**D3 — hipblaslt in CSV/YAML tuning config** 🔴
Any `+` line with `hipblaslt` in a tuning file. Not persistent across Docker; causes hangs.
→ `🔴 D3: hipblaslt config must not be committed`

**D4 — Invariant reversal without citation** 🔴
A documented safety invariant is reversed: old comment says "must X because Y" → new code removes X claiming "X not needed" but no spec/asm/test is cited to prove Y no longer holds.
Trigger: `::zeros() → ::empty()` / `torch.zeros → torch.empty` where old comment mentions "must" / "required" / "read back as zero"; assert deletion without explanation; `.contiguous()` removal; zero-init removal with contradicting justification.
Real example (aiter#4043): old: "trailing pad must read back as zero for the asm reader, so zero-initialise it here" → new: "trailing pad is never read by the asm reader, so no zero-init is needed" — two comments directly contradict; PR cites no spec. Human reviewers missed this, only saw the profiling screenshot.
→ `🔴 D4: [operation] reverses a documented safety invariant — cite the spec/asm/test proving new assumption is safe`

**D5 — Verbatim duplication across backbone files** ⚠️
The same fix is copy-pasted into 2+ Tier 1/2 backbone files with trivial name substitution (different variable names, identical algorithm and comments). AI code signature: changes look symmetric but each file's invariants may differ and were not independently verified.
Trigger: nearly identical `+` blocks appearing in two backbone files in the same PR diff; same formula / same comment structure / same magic constants, only variable names differ.
Real example (ATOM#1493): chunked indexer loop copy-pasted verbatim between `deepseek_v2.py` and `deepseek_v4.py` — same `(budget_rows // 128) * 128` formula, same `bit_length() - 1` fallback, same comment block, only variable names changed.
→ `⚠️ D5: identical algorithm in [file_a] and [file_b] — was correctness verified independently in each context, or copy-pasted?`

**D6 — Fake / meta function dtype or shape mismatch** 🔴
When a `gen_fake` / `_fake` / `abstract_impl` function is added or modified, its return tensor dtypes and shapes must match the real op exactly. torch.compile uses the fake to infer output types; a wrong dtype compiles cleanly but causes a dtype assertion or silent wrong values at runtime.
Trigger (1): diff contains a `_fake` / `gen_fake` function alongside the real op; compare each return tensor's dtype and shape against the real op's actual output.
Trigger (2): real op's return dtype or arity changes in the diff but no corresponding `_fake` / `gen_fake` change appears — the existing fake is now stale and will produce wrong types.
Real example (aiter#4110): `fused_allreduce_rmsnorm_quant_fake` returned `torch.empty_like(res_inp)` (bf16) as first element, but real op returns fp8 — wrong dtype for torch.compile's dtype checks. Human reviewers missed this entirely.
→ `🔴 [fake_fn] return [N] dtype is [X] but real op returns [Y] — torch.compile will assert or silently miscompute`

**D7 — New compile_op without fake function** 🔴
A new `@compile_ops` / `torch.library.custom_op` is added but has no corresponding `_fake` / `gen_fake` / `abstract_impl`. torch.compile traces the graph using fake tensors; without a fake, the op is a black box → runtime crash or silent fallback to eager inside a compiled region.
Trigger: diff adds a new function decorated with `@compile_ops` or `torch.library.custom_op`; grep for a `_fake` or `gen_fake` function with the same op name — if absent, flag.
→ `🔴 D7: [op_name] has no fake/abstract implementation — torch.compile will crash or silently fall back to eager`

**D8 — Kernel wrapper missing contiguous check** ⚠️
Python wrapper passes tensor to C++ / HIP kernel but doesn't assert `.is_contiguous()` or call `.contiguous()`. If the caller passes a strided tensor (slice, `.T`, output of non-contiguous `view()`), the kernel reads from wrong addresses — completely silent wrong result.
Trigger: new Python wrapper that calls a `@compile_ops` or C-extension kernel; check that non-trivially-shaped inputs (anything other than a freshly allocated `torch.empty`) are either asserted contiguous or explicitly made contiguous before the call.
→ `⚠️ D8: [tensor] passed to [kernel] without contiguous check — add .contiguous() or assert .is_contiguous()`

**D9 — INT32 overflow in GPU pointer arithmetic** 🔴
C++ kernel launcher or Python wrapper computes a buffer offset, record count, or index in `int32` (or Python `torch.int32`) when the product of dimensions can exceed 2^31 (~2 billion) at production scale.
Common patterns: `token_id * (num_heads * head_dim)` overflows at token_id > 16M with H=32, D=128; `seq_start * K` overflows for long-context at seq_start > 256K with K=8192; gfx1250 TDM block descriptor count fields computed as Python int default to int64 — a missing `.to(torch.int32)` cast silently produces wrong offsets.
Trigger: any arithmetic involving `token_id`, `seq_start`, `batch_offset`, or `total_tokens` that produces a buffer address or array index without an explicit widening to int64 before the multiply; or a TDM descriptor field that feeds into block offset computation without an explicit int32 cast.
Real examples: `out_base = token_id * num_heads * head_dim` in int32 overflows at scale (PR#3844); forward kernel uses `Int32(seq_start) * Int32(K)` while the backward kernel correctly uses int64 (PR#4113).
→ `🔴 D9: [expr] in int32 — widen [token_id / seq_start / total_tokens] to int64 before multiplying by [stride]`

**D10 — FlyDSL compile result stored but never called** 🔴
`flyc.compile(exe, *args)` on a cache-miss path compiles and stores the `CompiledFunction` object (`exe._cf = cf`) but does NOT call it — `cf(*args)` is absent. Every first-invocation of a new (shape, arch, dtype) combination silently no-ops the entire kernel launch and returns the uninitialized `torch.empty` output to the caller with no error.
Trigger: a cache-miss branch in a `_run_compiled`-style function that calls `flyc.compile(...)` and then returns without executing the compiled result.
Note: `flyc.compile()` ONLY compiles; it does NOT execute. The compiled result must be explicitly called with `cf(*args)` on the same branch. Do not confuse this with Triton's `@triton.jit` which auto-executes on first call.
Real example (aiter#3987): `tensor_shim.py` — cold-start on any new shape returns garbage output; all `_launch()` call sites through `fused_moe_gfx942.py` inherit this behavior.
→ `🔴 D10: [fn] compiles on cache-miss but does not call the result — add cf(*args) on the same branch`

**D10b — FlyDSL arith.bitcast requires arith.unwrap() on operand** 🔴
Inside a FlyDSL kernel, passing a raw DSL value directly to `arith.bitcast(val, target_type)` causes a type error at JIT-compile time — DSL values must be unwrapped with `arith.unwrap(val)` first. This fails silently in Python (no static type error) and only crashes at kernel JIT time when the shape/dtype combo is first encountered.
Trigger: any `arith.bitcast(...)` call in a FlyDSL kernel where the first argument is a DSL expression (result of an arithmetic op, a load, or a `const_expr`) rather than a plain Python literal. Check: is `arith.unwrap(...)` wrapping the value?
Real example (aiter#3944): `arith.bitcast(val, ...)` inside a bf16/f16 output path without `arith.unwrap()` — JIT type error on first invocation of that dtype branch.
→ `🔴 D10b: [expr] passed to arith.bitcast without arith.unwrap() — wrap as arith.unwrap([expr]) first`

---

### E — Cross-Repo Sync
_"The change is incomplete without a matching update in another repo."_

**E1 — New aiter symbol or kwarg without linked aiter PR** ⚠️
New `from aiter import X`, new kwargs on aiter calls, new aiter usage: PR description links an aiter PR?
New kwargs may require an aiter version not yet released.
Real example (ATOM#1494): `emit_bf16=True` kwarg added → needed aiter PR first.
→ `⚠️ E1: new aiter usage — corresponding aiter PR not mentioned`

**E2 — New param with backward-compatible default is dead code** 📝
New param added with default that preserves old behavior: the fix only activates when a consumer passes non-default. Who updates the consumer?
Real example (PR#3773): `max_seqlen=-1` added in aiter; fix never activated until ATOM passed actual value.
→ `📝 E2: new API param needs consumer-side update to activate — follow-up tracked?`

**E3 — Plugin bridge not updated** ⚠️
PR changes KV layout, function signature, or data structure that `deepseek_v4_bridge.py` / `sglang_bridge.py` read directly.
Real example (ATOM#1423): paged-SWA layout changed; bridge still used old layout.
→ `⚠️ E3: [structure] changed — plugin bridge sync needed`

---

### F — Resource Duplication
_"The change pins the same data twice on GPU without freeing the original."_

**F1 — New weight variant alongside original** ⚠️
New `w13_weight_preshuffled` / `w_quantized` stored as a new attribute alongside `w13_weight`: both pinned simultaneously → double HBM for that weight.
Real example (ATOM#1469): valarLip: "this will make us pin double weight."
Check: is the original freed after the new variant is created?
→ `⚠️ F1: [new_attr] stored alongside [original] — doubles HBM; is original freed?`

---

### G — Multi-Stream Synchronization
_"Written on stream A, consumed on stream B — no sync between them."_

**G1 — Missing HIP/CUDA stream synchronization** 🔴
HIP/CUDA streams execute concurrently by default. A tensor produced on stream A and consumed by a kernel on stream B without an explicit sync between them causes the consumer to read garbage — no crash, no error, silent wrong output.
Trigger: diff introduces a non-default `torch.cuda.Stream`, passes an explicit `stream=` argument to a kernel, or prepares buffers/weights on a side stream that are later consumed during forward pass on the compute stream. Check: is there `stream.synchronize()`, `stream.wait_stream(other)`, `hipEventRecord` + `hipStreamWaitEvent`, or `torch.cuda.current_stream().wait_stream(other)` between the last write on stream A and the first read on stream B?
→ `🔴 G1: [tensor] written on [stream A] consumed on [stream B] without sync — add stream.wait_stream() or hipStreamWaitEvent`

**G1b — Blocking queue.get() without timeout in production serving code** ⚠️
`queue.get()` without `timeout=` in a worker or service thread that depends on an external producer (decode loop, stream consumer, request handler). If the producer exits abnormally, the worker blocks forever — no crash, no log, hung process.
Trigger: `queue.get()` or `asyncio.Queue.get()` inside a `while True:` worker loop in production serving paths (entrypoints, engine loop, scheduler) without `timeout=` and without a corresponding `except queue.Empty` / `asyncio.TimeoutError` handler or a `done` flag.
Exception: test code, CLI tools, or one-shot scripts where a hang is detectable (CI timeout, interactive TTY).
→ `⚠️ G1b: [worker] blocks on queue.get() without timeout — add timeout= and handle Empty/TimeoutError to survive producer failure`

---

### Performance Evidence (always check)

**P1 — Perf PR without benchmark numbers** ⚠️
Trigger words: perf, optimize, fuse, faster, improve, +X%, replace kernel, OOM fix that changes algo.
Description must have numbers with units (ms, tokens/s, TFLOPS, %, speedup). Screenshots ≠ numbers.
Exception: PRs adding benchmarks/tests for existing ops without claiming improvement.
→ `⚠️ P1: perf claimed — no benchmark numbers with units`

**P2 — Benchmark covers only toy shapes** ⚠️
Numbers exist but only for M≤256, only 1 token, or one model.
Production: DSv4 E=385/topk=7, GPT-OSS 120B, Kimi-K2.5; token range 1→16384.
→ `⚠️ P2: benchmark missing production shapes — [what's absent]`

**P3 — Perf claim not reproducible** ⚠️
Missing: test script, ROCm version, GPU model, TP config, model checkpoint.
→ `⚠️ P3: perf claim missing reproduction info — [what's absent]`

**P4 — TP split shapes not covered** ⚠️
New attention / norm kernel tested only at full head count (TP=1 equivalent). At TP=4/8, `num_heads_q` / `num_heads_k` per device is divided by TP. A kernel that passes at H=128 may OOB at H=32 (TP=4) if shape math doesn't account for the split.
Trigger: new kernel taking `num_heads_q` / `num_heads_k`; PR test shows only one head count without a TP=4 or TP=8 variant.
→ `⚠️ P4: test covers only TP=1 head count — verify at num_heads÷TP=4 (e.g., [128→32])`

**P5 — Benchmark timing excludes one-time setup cost** ⚠️
Perf numbers exist but the timing window starts AFTER a one-time setup step whose cost is borne by real users on every cold start: weight shuffle/preshuffle, first-call JIT compile, hipMemsetAsync on large buffers, or precompile of variant kernels. Omitting this makes a net-regression look like a speedup.
Trigger: PR description shows a speedup but the benchmark script (or description) shows timing begins after `shuffle_weight()`, after `warmup_iters`, or inside an already-warm JIT cache. Check: would a user deploying this op from scratch see the same number?
Real examples: aiter#4166 — `shuffle_weight` excluded from timing, claimed 1.14x win is actually ~0.83x regression when included; aiter#3944 — FlyDSL precompile launches 7 dummy kernels on live stream at cold-start, benchmark captures only warm-cache latency.
→ `⚠️ P5: timing window excludes [setup step] — re-run benchmark including [shuffle_weight / first-call JIT / precompile] to confirm net improvement`

---

### Housekeeping (quick scan)

| Check | Trigger | Flag |
|---|---|---|
| Temp script committed | `.sh`, `runperf*.py`, `test_local_*.py` in diff | `⚠️ HK1: [file] looks temporary — remove before merge` |
| Unrelated files | Files with no connection to PR purpose | `⚠️ HK2: [file] appears unrelated` |
| `sys.path` at module level | `sys.path.insert(` / `sys.path.append(` in non-test `.py` | `⚠️ HK3: sys.path mutation — use relative imports` |
| kpack:1 in gfx950 config | `kpack: 1` in added YAML/CSV for gfx950 | `📝 HK4: kpack:1 on gfx950 is anti-pattern` |
| N-th op variant | 3rd+ variant of same op family | `📝 HK5: consider unified API — [N]th variant of [op]` |
| No UT for new op | New Triton/HIP op, no `op_tests/test_*.py` | `📝 HK6: new op needs UT following aiter-op-test format` |
| TODO/stub in new path | `# TODO`, `# FIXME`, `raise NotImplementedError`, lone `pass` on a `+` line inside a new branch | `⚠️ HK7: [location] — incomplete implementation in new code path` |
| `develop=True` on new op | `@compile_ops(..., develop=True)` in added code | `⚠️ HK8: develop=True bypasses JIT cache — remove before op leaves experimental` |
| Undocumented new env var | `os.environ.get("AITER_...` on a `+` line | `📝 HK9: new env var [NAME] not documented — add to README or known knobs list` |
| Test reference dtype promotion | New test reference impl uses Python float literal (`1.0 + weight`, `0.5 * x.float()`) or explicit upcast (`.to(torch.float32)`, `.double()`) promoting to fp32 while kernel runs in bf16/fp8 — comparison calibrated against wrong-precision baseline | `⚠️ HK10: reference [fn] promotes to fp32 — cast back to [kernel dtype] before comparison` |
| New third-party dependency | New package in `requirements*.txt`, `setup.py`, `pyproject.toml`; or new top-level `import [pkg]` not already a project dep. Exception: ROCm system packages (`amdsmi`, `hip`, `rccl`) are intentionally not on PyPI — flag only if there is no `try/except ImportError` guard AND no comment explaining the ROCm-only dependency | `📝 HK11: new dependency [pkg] — add to requirements, or add try/except ImportError with a comment for ROCm system packages` |

---

## Step 6 — AI Code Diagnostic

For each question below, note if the answer is a warning sign:

| Question | Warning sign |
|----------|-------------|
| Does description explain mechanism (WHY) or just action (WHAT)? | Only WHAT → elevated risk |
| Are perf numbers suspiciously clean? (exact 2.0x, 1.5x, 3.0x) | Could be cherry-picked or fabricated |
| Are perf claims only trace screenshots with no numeric values? | Screenshots ≠ numbers; reviewer will ask |
| Does the test only cover M=1 or M=16? | AI defaults to toy shapes |
| Are gated-off parameters asserted or silently ignored? | Silent → B1 violation |
| Does code introduce `sys.path`, `os.environ` mutations at module level? | Global state leak → HK3 |
| Were unrelated files committed alongside the actual change? | AI commit artifact → HK2 |
| Is the new default path revertible? | No env-var gate → D2 violation |
| Is "Test Plan" / "Test Result" section left as template comment? | Empty = untested, AI-generated description |
| PR description footer says "🤖 Generated with Claude Code" or similar AI attribution? | Author may not understand the change — elevated manual review priority |

If 3+ warning signs: note "elevated AI code risk — recommend thorough manual verification of the dispatch logic and test coverage."

---

## Step 7 — Free-Form Review

After the rule checklist, read the diff as a domain expert:
- Does the approach make sense given the hardware constraints?
- Are there correctness concerns not caught by the rules above?
- LDS limits: gfx942 = 64KB, gfx950 = 64KB per CU. gfx1250 (RDNA4) has 320KB LDS per CU but `ds_read`/`ds_write` immediate offset is only 16-bit (max 65535 = 64KB). If LDS allocation exceeds 64KB on gfx1250, the compiler uses VGPRs for the LDS address → VGPR spill → perf regression or compile failure. Real example (PR#4031): reviewer caught OPUS kernel on gfx1250 would hit this.
- For new Triton kernels: BLOCK_SIZE choices, num_warps, num_stages — are they reasonable for MI300X? Large BLOCK_SIZE can push LDS over limit causing test failures (Real example: PR#3808, 10 LDS-exhaustion failures in Triton batched GEMM configs).
- `.contiguous()` before kernel calls when tensor may have non-standard strides?
- For mixed FP8 dtype paths (fn vs fnuz): gfx942 KV cache is fnuz by default, but Q quantization may emit fn (e.g., DSv4 Flash fused indexer). A kernel handling mixed fn/fnuz inputs needs explicit dtype dispatch — silent dtype mismatch compiles but produces wrong values. Real example (PR#3913): reviewer asked "why is there a mixed FN/FNUZ path?" and asked for `if arch == "gfx942":` guard on the fnuz *conversion* path.
- For FlyDSL/assembly kernels: hardware tile size constants (MFMA M=16, N=16, K=32 for MI300X FP8) should be named constants, not raw magic numbers (16, 32) scattered across the kernel. Real example (PR#3913): vpietila asked "add named constants MFMA_M=16, MFMA_N=16, MFMA_K=32 and use them throughout."

---

## Step 7.5 — Blind-Spot Check

Before writing the verdict, answer this one question in full:

**"Is there any correctness risk, resource hazard, or behavioral edge case in this diff that none of Steps 1–7 above caught?"**

If the answer is yes, add it to the findings. If the answer is no, proceed.

---

## Step 8 — Verdict

**Output rules (strictly enforced):**
- Run Steps 1–7 internally. Do NOT narrate steps, do NOT show checklists, do NOT show which rules fired.
- Output ONLY the card below. Nothing before it, nothing after it.
- If there are no findings, the findings section is omitted entirely.
- "What it does" must be one sentence, written for a reviewer who hasn't read the diff.

```
## aiter PR #NNN — [title]

**[One sentence: what this PR does, in plain terms.]**

[✅ LGTM | ⚠️ NEEDS WORK | 🔴 BLOCK]

🔴 [specific finding — what, where, why it matters]
⚠️ [specific finding]
📝 [note]
```

Each finding must have **three parts**:
1. **Problem** — what exactly is wrong, with file/line if relevant
2. **Impact** — what goes wrong at runtime if this is not fixed (wrong output / crash / perf regression)
3. **Action** — end with a verb phrase: "**Author must** [do X]" or "**Reviewer should ask** [Y]" — no verb = incomplete finding, do not include

Do NOT use rule codes (P1, D4, A1…) in output — they are internal labels only.

Examples of good findings:
- `🔴 fused_qk_norm_rope_cache_quant.py:463 changes torch.zeros → torch.empty, but the old comment says "trailing pad must be zero for asm reader" and the new comment claims "never read" — if padding IS read, every quantized output is corrupted. **Author must** cite the asm spec or a test proving padding is not read.`
- `⚠️ PR claims fp8 latency is now 1.3–1.5x better, but the benchmark starts timing after shuffle_weight() completes — users pay that cost on every cold start. **Author must** re-run with shuffle_weight included in the timing window and confirm the result is still positive.`
- `⚠️ Chunked indexer logic is copy-pasted verbatim into deepseek_v2.py and deepseek_v4.py. If v4's variable semantics differ, the formula silently produces wrong KV offsets for v4 callers. **Author must** confirm correctness was verified independently under v4's variable layout.`
- `📝 No corresponding ATOM consumer PR mentioned. **Reviewer should ask** who will pass emit_bf16=True to activate this path.`

Examples of bad findings (too vague, no action verb):
- `⚠️ Missing perf numbers` — no impact stated, no action
- `🔴 D4 violation` — rule code means nothing to a reviewer
- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion

---

## Adding New Rules

When a human reviewer catches something real that this skill missed:
1. Add it to Step 5 with a real PR example as evidence
2. Increment the rule number
3. Commit with message: `review-pr: add R[N] from PR#[NNN] — [one line description]`

The skill grows from real review history, not hypothetical patterns.
