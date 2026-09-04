# Rule bodies

Not loaded with the skill. `triage.py expand` emits only the blocks the diff derives,
into $WORK/rules_expanded.txt. Rule ids and text are unchanged from when these lived in
SKILL.md; edit them here.

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
FP self-check (do this before firing): confirm the loaded dim is NOT guaranteed a multiple of BLOCK_SIZE — i.e. it is not padded/rounded up at allocation, not `tl.cdiv`-tiled with a masked tail elsewhere, and not already guarded by an enclosing mask or a caller-side pad. An unmasked load on a provably-aligned dim is safe; do not fire. Name the concrete non-aligned dim value that triggers the OOB.
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
| param-removed | param removed from signature | all call sites (cross-repo if public) | TypeError at call time | 🔴 |
| repr-key | new Gluon constexpr | kernel repr key list | stale JIT binary served | 🔴 |
| arch-discard | arch-specific kwarg | non-target-arch path | kwarg silently discarded | ⚠️ |
| dispatch-silent | multi-backend fallback | caller logging | backend switch with no diagnostic | ⚠️ |
| rename | public symbol renamed | all importers (cross-repo if public) | AttributeError at import/call time | 🔴 |

Severity (param-discard): 🔴 if param controls output correctness (`expert_mask`, `q_scale`, `kv_scale`); ⚠️ for performance knobs or optional features with working defaults.
**Public-API scope:** if the changed symbol is a public op (`from aiter import X`, or lives in `aiter/ops/*.py` / `aiter/__init__.py`), param-removed and rename break cross-repo consumers (ATOM / SGLang / vLLM), not just same-file call sites — the downstream to check is every repo that imports it. Also apply E1 (is a linked consumer PR mentioned?) and E5 (owner sign-off for a stable core-API contract).
Exception: method override where base class forces the signature but subclass legitimately ignores the param — flag as 📝 (structural discard, not a bug).
FP self-check (rename / param-removed): before firing, confirm the old symbol is NOT preserved by a compatibility shim added in the same PR — a new same-named `def` wrapper (keeps `from aiter import old_name` resolving), an alias, or a binding pin (`@compile_ops(..., fc_name='old_name')` keeps the C++ symbol even when the Python fn is renamed). A rename/removal behind such a shim is backward-compatible — do not fire. Real non-example (aiter#4227): `get_mla_metadata_v1` renamed to `_impl`, but a same-named wrapper + `fc_name='get_mla_metadata_v1'` preserved both the Python and C++ symbols → not B6.
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
FP self-check: if the constant sits on a path already runtime-guarded to a single dtype/arch (e.g. inside an `if arch == 'gfx942':` block), the hardcode is safe there — do not fire; fire only when the path handles multiple fp8 flavors.
Real example (PR#4015): yzhou103: "would break for OCP e4m3 (max=448)."
→ `⚠️ C2: fp8_max hardcoded to [value] — use get_dtype_max(dtype)`

**C3 — Dtype hardcoded without checking actual tensor** ⚠️
Fixed `bf16`, `fp8_e8m0`, or similar in a forward path that handles multiple configs.
FP self-check first: search the unchanged lines of this file for the same hardcoded dtype — if it already appears pre-existing on the same path, this is not a new violation (do not fire as new). Fire only when the hardcode is newly introduced, or the path newly handles more than one dtype/config.
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
Scope: D2 is about a **temporary rollback kill-switch** for a risky default swap (meant to be removed once validated) — NOT a permanent feature-flag knob. aiter maintainers generally reject new *permanent* env vars (see HK9): a MoE activation knob added in #3593 was reverted in #4225. If the safe path can be auto-derived from dtype/arch/shape instead of an env var, prefer that; reserve the env var for a genuine short-lived rollback.
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

**D9 — INT32 overflow in GPU pointer arithmetic** 🔴 *(scanner-backed — read the scan output, not this prose)*

`scan_index_width.py` decides this family structurally and its candidate list is the input to
the finding; `triage.py` therefore does not put D9 on the read list. The text below
documents what the scanner looks for and what still needs a human — naming the production
scale at which the product exceeds 2^31. A rule that has a scanner should be consumed through
the scanner: re-reading the prose adds nothing and costs attention that the families without
one need.

C++ kernel launcher or Python wrapper computes a buffer offset, record count, or index in `int32` (or Python `torch.int32`) when the product of dimensions can exceed 2^31 (~2 billion) at production scale.
Common patterns: `token_id * (num_heads * head_dim)` overflows at token_id > 16M with H=32, D=128; `seq_start * K` overflows for long-context at seq_start > 256K with K=8192; gfx1250 TDM block descriptor count fields computed as Python int default to int64 — a missing `.to(torch.int32)` cast silently produces wrong offsets.
Trigger (structural, NOT a name list): a multiplication that feeds pointer or index arithmetic, where at least one operand derives from a **non-`constexpr` parameter of the enclosing kernel** — a value supplied at runtime, which is the only kind that can grow past 2^31 — and no operand is widened to 64 bits, counting a widening applied on an earlier line and carried in through a local name. `constexpr` tile constants bound the product at compile time and are excluded. Also fires on a TDM descriptor field feeding block offset computation without an explicit int32 cast.
**Why the trigger is structural, and why saying so was not enough:** an earlier version of this rule listed the names `token_id`, `seq_start`, `batch_offset`, `total_tokens`. Three real defects used none of them — `stride_out_batch`, `block_id`, `physical_block`, `context_kv_idx` — and the rule stayed silent on all three (aiter#1674 ×2, aiter#3541). The rule text was then rewritten to say "structural" while still defining index-shaped and stride-shaped by name, and the scanner behind it matched two name lists against operand text. Measured on aiter#4978, the PR that introduced the `moe_wgrad` overflow later fixed by #5132: **0 of the real defect lines were reported**, the one `moe_wgrad` candidate emitted was an already-`int64` line, and 390 candidates were produced overall. The scanner is now an AST pass with no name lists at all; on the same diff it reports 4 of 4, and on #5132 it reports none. Do not narrow it back to a name list.
**Production scale.** Step 1 printed `validate-kernel-pr/production_scale.md` directly beneath the candidate list: pool sizes, batch limits and stride semantics that the diff does not contain. Use those numbers to name the triggering case the 🔴 gate requires; if none of them puts the product past 2^31, clear the candidate and say so.

**The candidate list is already in context.** Step 1 ran `scan_index_width.py` over the diff and printed, per file, every distinct index×stride expression reaching pointer arithmetic with no 64-bit widening. Work that list: clear each candidate, and fire D9 only where you can name the production scale at which the product exceeds 2^31. If the list is empty, say so rather than skipping the category silently. **If the scan printed a `NOT SCANNED` section, D9 cannot be cleared** — those files were never examined, and an empty candidate list that excluded them is not evidence of absence. Report the unscanned files instead of reporting no candidates.
Real examples: `out_base = token_id * num_heads * head_dim` in int32 overflows at scale (PR#3844); forward kernel uses `Int32(seq_start) * Int32(K)` while the backward kernel correctly uses int64 (PR#4113).
→ `🔴 D9: [index expr] in int32 — widen [index operand] to int64 before multiplying by [stride], overflows at [concrete production scale]`

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

**D11 — Struct field added or removed with a pinned ABI** 🔴

_"The layout is a contract with something that was compiled separately."_

Trigger: a `struct`/`class` in `.h`/`.cuh`/`.cu` gains or loses a field, and the diff touches
no `offsetof(` / `static_assert(sizeof(...))` line. aiter pins every kargs struct a
hand-written code object reads — `csrc/py_itfs_cu/` carries 40 such assertions across 37
files, and 6% of open PRs touch one.

Inserting a field anywhere but the end shifts every offset after it. The assertions exist to
turn that into a compile error rather than a wrong-address launch; the failure to look for is
a PR that changes the layout and leaves them alone.

Real example (aiter#5220): two `int`s added between `stride_qo_h` and `stride_kv_page` of
`pa_sparse_prefill_kargs`. `sizeof` goes 112 → 120 and the offsets of `stride_kv_page` and
`softmax_scale` each shift by 8, so all three assertions in the *same translation unit the PR
edits* fail. The assertion text says what is actually required: rebuild the gfx1250 code
objects, then update the table.

FP self-check: appending at the end of the struct shifts nothing, and a PR that updates the
assertions alongside the field is the correct shape — do not fire on either.
→ `🔴 D11: [field] added to [struct] before [next field] — sizeof and the offsets after it shift; update the PA_GFX1250_CO_ABI table and rebuild the code objects, or append at the end`

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

**E4 — Downstream CI skipped on a change downstream consumes** 🔴
aiter's downstream tests (ATOM, SGLang, vLLM) are SKIPPED BY DEFAULT and only run when a label is added — `ci:atom` (DeepSeek-R1-0528, GPT-OSS-120B), `ci:sglang` (DeepSeek-R1-MXFP4, Qwen 3.5), `ci:vllm` (GPT-OSS-120B, DeepSeek-R1-0528, Kimi-K2.5), `ci:all` (all of the above), or `ci:atom_full` (ATOM accuracy suite; only for FlyDSL/Triton upgrades). A PR that changes an op a downstream consumes can pass every *aiter* check with the downstream job skipped, merge green, and break the downstream silently — visible only after merge.
**Staleness guard:** the label→model mapping here is a snapshot. Before quoting a specific model for a label, confirm the current `ci:*` definitions in `.github/workflows/*.yaml` — the model roster rotates, and a stale mapping produces a confidently-wrong label recommendation.
Which label (be precise, do not reflexively pick `ci:all`):
1. **Dispatch reachability** — is the changed/new kernel wired into a default dispatch path? A pure-additive, arch-gated kernel not in any default path is unreachable by downstream → exempt, no label needed.
2. **Map activation → model** — if reachable, read the branch's activation condition (arch × dtype × shape × model gate) and map it to a model (e.g. 128-head fp8 MLA decode on gfx950 → DeepSeek-V4).
3. **Minimal label set** — DeepSeek is exercised by atom+sglang+vllm (≈ `ci:all`); Qwen 3.5 → `ci:sglang` only; Kimi-K2.5 → `ci:vllm` only; GPT-OSS-120B → `ci:atom`+`ci:vllm`.
Fallback: if you cannot trace the activation to a specific model but the diff changes the behavior of an mla/fused_moe/attention/quant/gemm/jit-core path, default to `ci:all` — a wasted CI run is far cheaper than a broken downstream.
Check: in the PR's statusCheckRollup, if `Atom Test` / `Kimi Downstream` / `Sglang Downstream` is `skipped` AND the diff touches a downstream-consumed op, coverage is missing.
Real example (aiter#3459): a DeepSeek-V4 128-head MLA decode kernel passed Aiter Test (success) with Atom Test SKIPPED and no `ci:*` label; after merge the Atom Test went red — the MLA change broke ATOM, invisible pre-merge.
→ `🔴 E4: [op] is consumed by [ATOM/SGLang/vLLM] but its downstream CI is skipped — add ci:all (or the minimal ci:atom/sglang/vllm) and require it green before merge`

**E5 — Stable core-API change needs owner sign-off, not just CI + one approve** 🔴
Modifying the BEHAVIOR / SIGNATURE / DEFAULT DISPATCH / NUMERIC SEMANTICS of a long-lived, widely-consumed API — `fused_moe.py`, `mla.py`, `ops/attention.py`, `ops/mha.py`, `ops/quant.py`, `gemm_op_a8w8.py`, `moe_op.py`, `jit/core.py` — must not be self-merged by a contributor or landed on a single reviewer's approval. These are downstream contracts; green CI (even `ci:all`) only covers the models/shapes it knows, not every downstream version or call path — necessary but not sufficient.
Trigger: diff changes the behavior/signature/default-path of a Step-4 Tier-1/Tier-2 file — NOT a pure-additive, arch-gated, behavior-preserving change (those are exempt; see E4 step 1).
Who signs off: aiter has **no CODEOWNERS file**, so ownership is de-facto — the top committer of the path is the effective gatekeeper:
`git log --format='%an' -- <file> | sort | uniq -c | sort -rn | head`
For `fused_moe.py` / core MoE dispatch this is currently @valarLip (top committer, and the maintainer who reverted #3593 and gates MoE PRs). Re-derive per file — MLA / attention / quant may have a different top committer.
**The reviewer must proactively notify the owner — do not wait for them to notice the PR.** Post a PR comment that @-mentions them (e.g. `@valarLip`) with a one-line summary of the contract change and an explicit request to approve before merge. Passive "someone should sign off" is not enough; the finding is not resolved until the owner has been actively pinged and has responded. Do not settle for a revert after merge.
Real example (aiter#3593): a `fused_moe.py` env knob merged on CI + one approval, then reverted by a maintainer within the hour — it should have had owner sign-off before merge.
→ `🔴 E5: [file] is a stable downstream-facing contract — do NOT self-merge. **Reviewer must @-mention the de-facto owner (git top-committer; @valarLip for fused_moe) in a PR comment requesting explicit sign-off** before merge, on top of ci:all`

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
Staleness guard: the production config list is a snapshot — verify current E/topk/hidden and the model roster from the model registry or a recent benchmark before asserting what counts as "production".
→ `⚠️ P2: benchmark missing production shapes — [what's absent]`

**P3 — Perf claim not reproducible** ⚠️
Missing: test script, ROCm version, GPU model, TP config, model checkpoint.
→ `⚠️ P3: perf claim missing reproduction info — [what's absent]`

**P4 — TP split shapes not covered** ⚠️
New attention / norm kernel tested only at full head count (TP=1 equivalent). At TP=4/8, `num_heads_q` / `num_heads_k` per device is divided by TP. A kernel that passes at H=128 may OOB at H=32 (TP=4) if shape math doesn't account for the split.
Trigger: new kernel taking `num_heads_q` / `num_heads_k`; PR test shows only one head count without a TP=4 or TP=8 variant.
→ `⚠️ P4: test covers only TP=1 head count — verify at num_heads÷TP=4 (e.g., [128→32])`

**P5 — Benchmark hides a cost real users pay on every call or cold start** ⚠️
The perf claim is measured with the timing window drawn so a *recurring* production cost is excluded: a first-call JIT compile on a path that is NOT cached across calls, or a setup step that runs on the live stream inside the timed region on every cold start. If that cost is real and recurring, omitting it can turn a net regression into an apparent speedup.
Do NOT fire on a genuinely one-time, amortizable setup that production pays once at model init — excluding weight shuffle/preshuffle, model weight loading, or a first-call JIT whose result is cached forever from steady-state per-call latency is CORRECT methodology, not deception. `warmup_iters` before a steady-state loop is standard and by itself is not P5.
FP self-check (do this before firing): is the excluded cost paid **once per deployment** (amortizable → do NOT fire) or **again on every call / every cold start / inside the timed stream** (→ fire)? If you cannot show it recurs, do not fire.
Counter-example (does NOT trigger P5): aiter#4166 preshuffles the static weight once outside the timing loop and honestly reports a geomean 0.69x result — a correct steady-state benchmark, not a hidden cost. Charging that one-time shuffle against a single call to manufacture a "regression" is itself the false positive this rule must avoid.
→ `⚠️ P5: timing window excludes [cost] that recurs per call / per cold start — re-run including it, or confirm it is one-time amortizable`

**P6 — Kernel change whose cost nobody measured** ⚠️
P1–P5 all grade the numbers *the PR supplies*. None of them produces a number, so a kernel PR can clear the whole Performance block on the strength of a table nobody re-ran. Correctness evidence does not cover this: `correctness_repo_tests: pass` means the kernel computes the right values and says nothing about how long it takes.

**Where the measurement now comes from.** The validator has a `perf` stage. When it runs, it times the target on base and on head back to back on the same locked GPU — base with the patch reversed out on the same worktree — and reduces the pair to `median_ratio`, the head speedup over base, oriented so `<1` is always a regression. That is a deterministic result, not an advisory one: below `threshold` (default 0.95, over ≥3 matched rows, both sides exiting cleanly) it writes a `should-fix` finding and the report's verdict becomes `NEEDS_WORK`. Read it out of `stages.perf`; do not re-derive it.

Trigger: Step 1's triage printed `perf triage: REQUIRED` — i.e. the PR changes runtime surface — **and** the head-matched report carries no usable `stages.perf` (absent, or `status: skip`). That is the gap P6 exists to name. If `stages.perf` is `pass` or `fail`, the measurement happened; report the number and do not fire.

Why the stage skips, and what each case means for the card:
- **no benchmark entry point in the target** — the honest common case (26 of aiter's 123 `op_tests/` targets have no timing harness at all). Fire P6, and say the target cannot be timed as written.
- **the PR adds the target** — base has nothing to compare against. Fire P6 with that reason; a head-only number is not a comparison.
- **nonzero exit on either side** — deliberately never a regression, because a truncated log yields a meaningless ratio. Fire P6 and treat the crash as the more interesting finding.
- **fewer than 3 matched rows / no shared timing column** — the two sides measured different things. Fire P6.

**The measurement that counts is base vs head, on this box, back to back.** Running only head against whatever baseline the PR chose reproduces the PR's own comparison; it cannot show a regression, and it silently inherits any staleness in that baseline. When the stage could not run, Step 1 prints the exact two-command manual form for the triaged target.
What to report: the shapes measured, both sides' numbers with units, and the delta. If nothing ran, say why — no idle GPU, no benchmark entry point, arch not available here — and mark every perf statement in the card `[inferred]`.
FP self-check: do NOT fire when the triage says NOT REQUIRED (no runtime surface), when `stages.perf` is `pass`/`fail`, when a manual measurement was taken and reported, or when the PR's own harness is the thing under test and has no steady state to measure. A single sample on a shared box is weak evidence — report the spread or the sample count rather than one number.
Real example (aiter#4538): a FlyDSL kernel whose entire justification was perf was reviewed to `Validation: PASS` with zero timing data; the perf finding stopped at "reviewer should ask", and a maintainer had already posted on the PR that a competing kernel beat it. Measuring afterwards took two `--scenario bench` runs and showed the fused path is where the PR's gain comes from — which the review should have carried in the first place. That gap is what the `perf` stage now closes automatically.
→ `⚠️ P6: kernel changed and no base-vs-head timing exists — [why the perf stage could not run]; measure [target] on both sides, or mark the perf findings [inferred]`

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
| New permanent env-var knob | `os.environ.get("AITER_...` on a `+` line that adds a lasting behavioral flag | `⚠️ HK9: new env-var knob [NAME] — aiter generally rejects permanent env flags (AITER_MOE_FORCE_BF16_ACT in #3593 was reverted by #4225); prefer auto-deriving from dtype/arch/shape. Acceptable only as a temporary rollback kill-switch (see D2), and then must be documented.` |
| Test reference dtype promotion | New test reference impl uses Python float literal (`1.0 + weight`, `0.5 * x.float()`) or explicit upcast (`.to(torch.float32)`, `.double()`) promoting to fp32 while kernel runs in bf16/fp8 — comparison calibrated against wrong-precision baseline | `⚠️ HK10: reference [fn] promotes to fp32 — cast back to [kernel dtype] before comparison` |
| New `test_*.py` pytest collects nothing from | New `op_tests/**/test_*.py` written as a `main()` script -- no `def test_`, no `class Test` -- and the PR adds no other collectable test | `⚠️ HK12: [file] is named like a test but pytest collects nothing from it; HK6 reads as satisfied while the new code has no CI coverage. Add a `def test_` entry point, or say plainly in the PR that the code is manually tested only. (aiter tolerates the style -- 3.8% of open PRs, and the tree already has such files -- so this is about the coverage claim, not the file.)` |
| New third-party dependency | New package in `requirements*.txt`, `setup.py`, `pyproject.toml`; or new top-level `import [pkg]` not already a project dep. Exception: ROCm system packages (`amdsmi`, `hip`, `rccl`) are intentionally not on PyPI — flag only if there is no `try/except ImportError` guard AND no comment explaining the ROCm-only dependency | `📝 HK11: new dependency [pkg] — add to requirements, or add try/except ImportError with a comment for ROCm system packages` |

---

### T — Triton / Gluon Kernel

_"The kernel is written in Python, so the C++ rules never look at it."_

These fire only on `aiter/ops/triton/`, `_triton_kernels/`, `_gluon_kernels/`, `/gluon/`.
Triton is already 195 of 600 open aiter PRs and rising, and the generic families say almost
nothing about them — `ops-wrapper` fires on 88% of Triton PRs and `modified-kernel` on 79%,
which is a label, not a triage. Each family below fires on at most 52% of Triton PRs.

**T1 — `tl.load` / `tl.store` with no `mask`** 🔴 (⚠️ if the bound is provably a multiple of the block)

Trigger: a `tl.load(...)` or `tl.store(...)` whose args contain no `mask=`. Present in 23% /
19% of Triton PRs respectively. Triton does not bounds-check; an unmasked access at the tail
tile reads or writes past the tensor. Reading garbage is the quiet version, writing is the
one that corrupts a neighbouring allocation.
FP self-check: masking is unnecessary when the axis length is a `tl.constexpr` multiple of
the block, or when the pointer arithmetic is already clamped (`tl.minimum`, `% n`). Confirm
which, and say so — do not fire on "no mask" alone.
→ `🔴 T1: tl.load at [file:line] has no mask; [dim] is not a multiple of [BLOCK] so the last program reads [n] elements past the end`

**T2 — `mask=` with no `other=`** ⚠️

Trigger: `tl.load(..., mask=...)` and no `other=`. 10% of Triton PRs. The masked-off lanes
default to zero — usually fine for a sum, wrong for a max/min reduction (zero beats a
negative running max) and wrong for any accumulator seeded from the load.
Check what consumes the loaded value before clearing this.
→ `⚠️ T2: masked load at [file:line] feeds a max-reduction with no other=-inf; padded lanes win the max`

**T3 — `num_warps` / `num_stages` carried over unre-tuned** ⚠️

Trigger: `num_warps` / `num_stages` / `num_ctas` in added lines. 50% of Triton PRs — the
single most common Triton edit. These are per-shape and per-arch: a config tuned on gfx942
is not tuned for gfx950, and one copied from a sibling kernel with a different BLOCK is not
tuned at all. `num_stages` above what LDS can hold silently drops occupancy rather than
failing.
This is a perf finding, so it needs perf evidence: without a base-vs-head number it is
`[inferred]` and phrased as a question, per Step 8.
→ `⚠️ T3: num_stages raised to [n] for [kernel] with no measurement on [arch]; is this tuned here or inherited from [sibling]?`

**T4 — AMD launch knobs hardcoded across archs** ⚠️

Trigger: `waves_per_eu`, `matrix_instr_nonkdim`, `kpack`. 34% of Triton PRs. These are
AMD-specific and MI300/MI355 differ; `matrix_instr_nonkdim` in particular selects an MFMA
instruction that must exist on the target arch.
→ `⚠️ T4: matrix_instr_nonkdim=[n] set unconditionally; gfx950 [does/does not] have that MFMA shape`

**T5 — accumulator precision** 🔴

Trigger: `tl.dot(`, `.to(tl.float32)`, `allow_tf32`, `input_precision`. 25% of Triton PRs.
Two failure shapes: an accumulator left in fp16/bf16 over a long K loop loses the tail of the
sum, and `tl.dot` inputs downcast to save registers change the result silently. On fp8 paths
check the scale is applied in fp32, not after a downcast.
→ `🔴 T5: acc declared [dtype] over K=[n]; at [scale] the running sum loses [x] — accumulate in float32 and cast once at the store`

**T6 — grid and `program_id` disagree** 🔴

Trigger: `grid=lambda`, `tl.program_id`, `tl.cdiv(`. 27% of Triton PRs. The launch grid is
computed on the host and the tile mapping inside the kernel; nothing checks that they agree.
A `cdiv` on the host with a kernel that assumes exact division leaves the tail tile
unprocessed — output is correct everywhere the tests look and wrong in the last block.
Check the host grid expression against every `program_id` axis, including which axis is which
after a swizzle.
→ `🔴 T6: grid is cdiv(M,BLOCK_M) x cdiv(N,BLOCK_N) but the kernel derives pid_n from pid // grid_m using the pre-swizzle grid_m; tiles [range] are never written`
