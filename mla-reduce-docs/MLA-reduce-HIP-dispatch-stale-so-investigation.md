# MLA reduce HIP dispatch `TypeError` — investigation & run results

**Date:** 2026-06-23 · **Branch:** `flydsl-mla-decode-reduce` @ `ce20aec1f` · **Reporter:** Matti Eskelinen
**Status:** Root-caused. Two independent findings; the default HIP path is sound.

---

## 1. Symptoms

Matti ran, with the FlyDSL fallback **disabled**:

```bash
AITER_MLA_REDUCE_FLYDSL=0 python op_tests/test_mla_sparse.py -n16,2 -b 1 -c 21 -k 512 -d fp8 -kvd fp8
```

and reported two things: (a) `TypeError: mla_reduce_v1() missing 1 required positional argument: 'final_output'`, and (b) a consistent GPU illegal memory access he suspected was "something wrong in the original kernel as well."

Both are explained below. Neither is a defect in the default HIP reduce.

---

## 2. Finding 1 — `TypeError` is a stale JIT module

It is a **stale compiled `module_mla_reduce.so`** on the reporter's machine, predating the current 8-arg signature.

- **The executed HIP path is byte-identical to pre-FlyDSL `main`.** With `AITER_MLA_REDUCE_FLYDSL=0`, `_flydsl_mla_reduce_enabled()` returns `False` on the env check (`aiter/mla.py:29`) *before flydsl is imported*; the branch taken is `aiter.mla_reduce_v1(...)` (`aiter/mla.py:67`), passing all 8 args — identical to `b92a6bab1^`.
- **The arg-checker validates against the compiled `.so`, not the stub.** `aiter/jit/core.py` parses the signature from the module's pybind docstring (`op.__doc__`, ~L1520–1558), overrides `func.__signature__` (L1560), then runs `inspect.getcallargs` (L1575). The "missing `final_output`" error reflects the signature baked into the reporter's `.so`, which predates `max_seqlen_q`/`final_output` (landed at `af113a3f8`/`d17e8239c`, before this branch).

Current source is internally consistent at 8 args: Python stub `aiter/ops/attention.py:1127`, pybind `csrc/include/rocm_ops.hpp:1977`, C++ `csrc/kernels/mla/reduce.cu:1024`.

**Why only HIP trips, never FlyDSL.** `flydsl_mla_reduce_v1` is a pure-Python launcher that calls the compiled kernel directly, never via `torch.ops.aiter` arg-checking. The HIP path is the sole consumer of `module_mla_reduce.so`, so it alone can hit a stale-`.so` mismatch — exactly Matti's asymmetry.

**Reproduction attempt:** on container `mla_reduce_bench` (MI300X/gfx942, same commit `ce20aec1f`), Matti's exact command **runs clean** — the `TypeError` does not reproduce. Our `/tmp/aiter_jit/module_mla_reduce.so` (built Jun 18) carries the 8-arg signature; his is older. That is the entire delta.

---

## 3. Finding 2 — FlyDSL reduce faults on multi-token decode (`decode_qlen > 1`)

The IMA is in the **FlyDSL reduce port**, not the HIP kernel, and only on multi-token decode.

**Faulting kernel pinned.** Under `AMD_LOG_LEVEL=3` the last launch before the fault is `mla_reduce_kernel_0` (the FlyDSL reduce), dispatched right after the stage-1 asm kernel `aiter::mla_a16w16_...`. So the OOB access is inside the FlyDSL kernel.

**Trigger isolated by A/B (only `decode_qlen` changes):**

| Reduce path | env | dtype | `decode_qlen` | Result |
|---|---|---|---|---|
| HIP `kn_mla_reduce_v1` | FLYDSL=0 | fp8/fp8 | 2 | clean, `decode:err = 0.339172` |
| FlyDSL port | FLYDSL=1 | bf16 | **1** | clean, `decode:err = 0` |
| FlyDSL port | FLYDSL=1 | fp8 | **1** | clean, `decode:err = 0.347290` |
| FlyDSL port | FLYDSL=1 | bf16 | **2** | **was** GPU memory access fault (exit 134) → **FIXED: clean, `decode:err = 0`** |
| FlyDSL port | FLYDSL=1 | fp8 | **2** | **was** GPU memory access fault (exit 134) → **FIXED: clean, `decode:err ≈ 0.36`** (fp8 quant delta, matches qlen=1) |

- **Original HIP kernel exonerated:** it runs the exact same shape + metadata clean (answers Matti's worry directly).
- **dtype-independent:** the qlen=1→clean / qlen=2→fault crossover held for *both* bf16 and fp8. dtype moves only the accuracy delta; only `decode_qlen` moved clean-vs-fault. So if Matti ran his original **fp8** command with the FlyDSL flag on, the IMA he saw was *this* bug — distinct from the §2 `TypeError` and the §3.1 tolerance delta.

**Fix verified (2026-06-23, GPU restored).** After adding the empty-tile guard (below), bf16 qlen=2 runs clean 3/3 with `decode:err = 0`; fp8 qlen=2 clean with `decode:err ≈ 0.36` (its own qlen=1 baseline is 0.388 — same fp8 quantization delta, not a reduce error); qlen=1 bf16/fp8 unchanged; `FLYDSL=0` default path unaffected. No fault under any repeat.

**Mechanism (true root cause — corrected after on-GPU dump).** The original "static" hypothesis (NTG / `*max_seqlen_q` row layout) was *wrong for this path*: a runtime dump of the real reduce args shows `max_seqlen_q = 1` even at `decode_qlen = 2` — multi-token is encoded as **more reduce tiles** (`num_reduce_tile = 2`), not as a wider q-range. The actual trigger: the real decode metadata hands the reduce a **degenerate `reduce_indptr = [0, 0, 0]`** (every tile has `n_splits = 0` — no partials for this invocation) with an **uninitialized `reduce_final_map`** (garbage q-ranges). HIP skips such tiles via `reduce_tile_start == last_reduce_tile → return false` and `else if(num_splits > 1)` (`csrc/kernels/mla/reduce.cu:691,743`), never dereferencing the garbage. The FlyDSL kernel had **neither guard**: it unconditionally read `reduce_final_map[tile]` → garbage `q_start/q_end` → the seq-loop stored out of bounds → *intermittent* fault (passed when the garbage happened to give `q_start ≥ q_end`, faulted otherwise; serializing with `AMD_SERIALIZE_KERNEL=3` masked it). **Fix:** clamp the seq-loop's upper bound to its lower bound when `n_splits ≤ 1` so a no-work tile runs zero iterations and never touches the garbage range (`aiter/ops/flydsl/kernels/mla_reduce.py`, `has_work = cmpi sgt n_splits, 1; ub_seq = has_work.select(q_end, seq0)`). The grid-y = `max_seqlen_q` / NTG-strided-loop changes from the prior plan are retained — harmless here (grid-y = 1) and correct if a future metadata revision does widen `max_seqlen_q`.

**Why the standalone matrix missed it.** The standalone `op_tests/test_flydsl_mla_reduce.py` always builds **well-formed** metadata: every tile has `n_splits ∈ {2,3,4,...}` and a valid `reduce_final_map`. It never constructs the **degenerate `n_splits = 0` tile with uninitialized `reduce_final_map`** that the real `get_mla_metadata_v1` emits for this shape — and that degenerate tile is the sole trigger. (The matrix was also extended to `M = max_seqlen_q ∈ {1,2,4}` while chasing the original NTG hypothesis; those 144 cases pass, but they exercise a layout the production path doesn't actually use here.) The real gap was **empty-tile handling**, not multi-token row mapping. Opt-in only, so the default HIP path was unaffected throughout. *Follow-up worth adding:* a standalone case with `reduce_indptr = [0, 0, 0]` + garbage `reduce_final_map` to lock in the guard.

### 3.1 Separate observation — fp8 numerical tolerance (NOT a reduce bug)

The runs surface a stage-1 fp8 decode-attention accuracy delta vs the bf16 golden (e.g. `mla_decode-absorb_fp8 [golden vs aiter_asm]` FAILED, ~38.6% elements; `[golden fp8 vs aiter_asm]` only a *warning* at ~4.1%). **Confirmation:** re-running the same shape with bf16 inputs (`-d bf16 -kvd bf16`), through the identical reduce path, gives `decode:err = 0` exactly. So the delta is **fp8 input quantization upstream of the reduce** — the reduce is exonerated. Track separately (§5.5).

---

## 4. Environment note (container drift, fixed before §3 re-runs)

After a container restart, bare `python3` resolved `aiter` to a second editable install at `/opt/aiter` (via its `.pth`/egg-link on `sys.path`) instead of workspace `/aiter`. `/opt/aiter` hard-pins `flydsl == 0.1.2` (`/opt/aiter/aiter/ops/flydsl/__init__.py:15`, exact equality), but the container's flydsl was upgraded to `0.2.0`, so every run aborted at import ("CK and HIP ops are disabled") before any kernel.

**Fix:** `pip uninstall -y amd-aiter` (drops the `/opt/aiter` editable), then `pip install -e .` from `/aiter` (its tree accepts `flydsl >= 0.1.8` — `aiter/ops/flydsl/__init__.py:16`, a `>=` check, so `0.2.0` is fine). Run with `AITER_JIT_DIR=/tmp/aiter_jit` so the loader uses the container-built `.so` files, not the in-tree prebuilt ones (which need `GLIBCXX_3.4.31`, absent on this host). This restores the original fault-run environment; all §3 results were re-verified under it.

Logs in `mla-reduce-docs/`: `_flydsl_run.log` (bf16 qlen=2 fault), `_flydsl_amdlog.log` (dispatch trace), `_flydsl_qlen1.log` (clean bf16 qlen=1), `_flydsl_fp8.log` (fp8 qlen=2 fault), `_flydsl_fp8_qlen1.log` (clean fp8 qlen=1), `_hip_fp8_qlen2.log` (clean HIP fp8 qlen=2).

---

## 5. Fix & recommendations

**Reporter unblock (immediate).** Force a rebuild of the one stale module, then re-run:
```bash
rm -f /home/AMD/meskelin/aiter/aiter/jit/module_mla_reduce*.so
# re-run his command; module_mla_reduce recompiles (~3 min)
```
Or point `AITER_JIT_DIR` at a fresh dir. The recompiled module carries the 8-arg signature and the HIP dispatch succeeds. *Caveat:* rebuild confirmed on our box, not his — worth a one-line DM confirming his `AITER_JIT_DIR` cleared, ruling out a second JIT dir shadowing the build.

1. **Make stale-`.so` failures self-explanatory.** Wrap the check at `aiter/jit/core.py:1575` so a signature/arity mismatch against `op.__doc__` raises with a hint ("compiled module likely stale — delete it / clear `AITER_JIT_DIR` to rebuild") instead of a bare `getcallargs` `TypeError` pointing at the call site.
2. **Signature-drift guard (optional).** `check_args()` already computes drift and only logs a warning (`core.py:1663`); consider surfacing a stale-module mismatch loudly or invalidating the cached `.so`.
3. **Fix the FlyDSL multi-token fault (§3). ✅ DONE (2026-06-23).** Root cause was a **missing empty-tile guard**, not NTG: the real metadata has `max_seqlen_q = 1` and feeds `reduce_indptr = [0,0,0]` (`n_splits = 0`) with garbage `reduce_final_map`. Fixed by clamping the seq-loop to zero iterations when `n_splits ≤ 1` (mirrors HIP `reduce.cu:691,743`). bf16 qlen=2 now clean (`decode:err = 0`); fp8 qlen=2 clean (fp8 quant delta only); qlen=1 + default path unchanged. Logs: `_flydsl_qlen2_FIXED.log`, `_flydsl_fp8_qlen2_FIXED.log`, `_flydsl_matrix_FIXED.log`. *Remaining:* add a standalone regression case with a degenerate (`n_splits=0`, garbage fmap) tile.
4. **Track the fp8 tolerance separately (§3.1).** Distinct item for the `mla_decode-absorb_fp8` delta; unrelated to dispatch, should not block closing Matti's report.

---

## 6. One-line summary

Two independent findings. (1) The `TypeError: ... missing ... 'final_output'` is a **stale `module_mla_reduce.so`** predating the 8-arg signature — Matti's exact command runs clean on our container; rebuilding that module is the fix. (2) The **illegal memory access** is a **FlyDSL reduce port bug on multi-token decode** (`decode_qlen > 1`) — now **fixed**. The real trigger (confirmed by an on-GPU arg dump) is **not** the `*max_seqlen_q` layout the static analysis first guessed (real `max_seqlen_q = 1`), but a **missing empty-tile guard**: the metadata feeds a degenerate `reduce_indptr = [0,0,0]` with uninitialized `reduce_final_map`, and the kernel read that garbage q-range and stored OOB (intermittently). Fixed by skipping tiles with `n_splits ≤ 1`, as HIP already does (`reduce.cu:691,743`). The **original HIP kernel runs the same shape clean** (exonerated); the FlyDSL path is opt-in, so the default was unaffected throughout.
