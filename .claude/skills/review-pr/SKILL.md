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

- [ ] **New kernel / new Triton op** → B1 (dispatch gate), B2 (tl.load mask), A1 (sibling variants), D1 (atomic zero-init), HK6 (UT)
- [ ] **Tuning config (CSV / YAML)** → D3 (hipblaslt), HK4 (kpack:1)
- [ ] **Dispatch logic change** → B1 (silent bypass), B3 (string normalization), A3 (scope too broad)
- [ ] **Replaces existing kernel as default** → D2 (rollback env-var)
- [ ] **Core file change** (see Tier table below) → full Step 4 risk assessment
- [ ] **Refactor / rename** → HK2 (unrelated files), variable name mismatch check
- [ ] **FP8 / quantization path** → C1 (fnuz by dtype), C2 (fp8_max hardcoded), D1 (atomic zero-init)
- [ ] **Test / benchmark only** → P2 (production shapes), HK6 (aiter-op-test format)
- [ ] **Async / multi-stream** → G1 (stream sync missing)

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

**B4 — Over-conservative assert blocks valid shapes** ⚠️
`assert M % tileM == 0` when the kernel pads internally and handles non-aligned M.
Real example (PR#3998): wrapper asserted alignment; asm kernel padded — valid small-M shapes rejected.
→ `⚠️ B4: assert [constraint] may be unnecessary — verify kernel handles non-aligned inputs`

**B5 — Triton `tl.constexpr` safety check disabled without invariant proof** ⚠️
A `tl.constexpr` bool that gates a validity check (e.g., `CHECK_NEG_ONE_SENTINEL`, `CHECK_BOUNDS`) can be set `False` by a caller to skip the check. If the invariant the check enforces is not independently guaranteed on that path, illegal memory access or silent wrong values result.
Trigger: new `tl.constexpr` bool in a Triton kernel that disables a bounds/sentinel/validity check; caller comment says "X path can disable this" without documenting what guarantees the invariant holds on that path.
Real example (ATOM#1498): `CHECK_NEG_ONE_SENTINEL=False` disables the -1 slot filter in the paged prefill kernel; illegal access if any -1 slot appears without the check.
→ `⚠️ B5: [constexpr] disables [check] — document which caller invariant guarantees no [invalid value] on that path`

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

---

### D — Uninitialized / Boundary State
_"The code writes or reads memory that was never properly initialized."_

**D1 — Atomic reduction on uninitialized buffer** 🔴
`atomic_fmax(*ptr, val)` = `*ptr = max(*ptr, val)`. If `*ptr` is uninitialized (from `::empty()`),
garbage dominates the max → corrupted amax → corrupted FP8 descale → silent wrong quantization.
Trigger: `atomic_fmax` / `atomic_max` + `::empty()` or non-zeroed allocation near it.
Real example (PR#4015): yzhou103: "AiterTensor::empty does not zero-initialize... garbage in v_amax silently corrupts descale."
→ `🔴 D1: [buffer] passed to atomic_fmax not zero-initialized — use ::zero() not ::empty()`

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
| New third-party dependency | New package in `requirements*.txt`, `setup.py`, `pyproject.toml`; or new top-level `import [pkg]` not already a project dep | `📝 HK11: new dependency [pkg] — justify why it's needed and add to requirements` |

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

Each finding must have two parts:
1. **Problem** — what exactly is wrong, with file/line if relevant
2. **Decision needed** — what the human reviewer needs to verify or ask for

Do NOT use rule codes (P1, D4, A1…) in output — they are internal labels only.

Examples of good findings:
- `🔴 fused_qk_norm_rope_cache_quant.py:463 changes torch.zeros → torch.empty, but the old comment says "trailing pad must be zero for asm reader" and the new comment claims "never read" — direct contradiction. Author must cite the asm spec or a test proving padding is not read.`
- `⚠️ PR claims fp8 latency is now 1.3–1.5x better, but description has only screenshots, no token/s numbers. Author must provide concrete benchmark data with units.`
- `⚠️ Chunked indexer logic is copy-pasted verbatim into two Tier-2 files. Author must confirm correctness was verified independently in each file's context.`
- `📝 No corresponding ATOM PR mentioned — who will call emit_bf16=True?`

Examples of bad findings (too vague, no action):
- `⚠️ Missing perf numbers` — no decision, no action
- `🔴 D4 violation` — rule code means nothing to a reviewer
- `📝 Check backbone files` — reviewer does not know what to do

---

## Adding New Rules

When a human reviewer catches something real that this skill missed:
1. Add it to Step 5 with a real PR example as evidence
2. Increment the rule number
3. Commit with message: `review-pr: add R[N] from PR#[NNN] — [one line description]`

The skill grows from real review history, not hypothetical patterns.
