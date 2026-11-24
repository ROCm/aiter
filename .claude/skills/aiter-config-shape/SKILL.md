---
name: aiter-config-shape
description: How to add/upload tuned config CSVs under aiter/configs (incl. model_configs/) without introducing duplicate shapes, and how to find & resolve duplicate-shape collisions. Use whenever adding a model's tuned config, merging/uploading config CSVs, editing anything under aiter/configs/**, or when a run hits "duplicate shape entries during merge".
argument-hint: [model name or aiter/configs/**/*.csv file]
---

# aiter config shape-collision standard

Tuned config CSVs in `aiter/configs/` are **not read one file at a time**. At
runtime `aiter/jit/core.py::AITER_CONFIGS.get_config_file` merges, per family,
the canonical `aiter/configs/<name>.csv` **plus every**
`aiter/configs/model_configs/*<name>*.csv`, and `update_config_files`
de-duplicates that merge on a key derived from the matching **untuned** file's
columns (+ `cu_num`/`gfx`/`_tag`). If two rows across the merged files share that
key, the merge **raises** `RuntimeError: Found N duplicate shape entries during
merge of '<family>'`.

Consequence: a per-model config you add can be perfectly fine alone yet collide
with a *different* model's config once both are on `main`. Single-PR CI only
merges your file with current `main`, so two PRs that each add the same shape to
different model files both pass, then break `main` after both land
(**cross-PR / merge-skew hazard**). Reviewers routinely forget this.

Follow this whenever you add "tuned configs for model Y", upload/merge config
CSVs, or touch anything under `aiter/configs/**`.

## The hard rules

1. **A key is (untuned columns) + `cu_num`, plus `gfx`/`_tag` when those columns
   are present.** Never hand-pick key columns — they come from the family's
   `*_untuned_*.csv` header, exactly as runtime derives them (`update_config_files`
   appends `cu_num` if absent, `gfx` only when a `gfx` column exists in the merge,
   and `_tag` only when a `_tag` column exists). Do not assume `M,N,K` alone: e.g.
   `a8w8` also keys on
   `q_dtype_w`, `bf16` on `bias,dtype,outdtype,scaleAB,bpreshuffle`, `fmoe` on the
   full `token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_*,q_type,
   use_g1u1,doweight_stage1` (+ `_tag`).
2. **Your file is merged with the canonical file AND every other model file of
   the same family.** Before uploading, check the merge, not just your file in
   isolation. The family is decided by the substring in the filename
   (`*<tuned_file_name>*`), e.g. `dsv4_bf16_tuned_gemm.csv` joins the
   `bf16_tuned_gemm` family.
3. **One shape, one row across the whole family.** If the shape already exists
   (in the canonical file or another model's file), do **not** re-add it. If yours
   is genuinely faster, replace the existing row rather than adding a second.
4. **Run the collision guard and make it pass before you push.** (below)
5. **Resolve, never suppress.** Fix by removing the redundant row (keeping the
   lowest-`us` winner) — do not widen keys, rename shapes, or delete the guard.
6. **`gfx` matters.** Archs that share a `cu_num` (e.g. gfx950 vs gfx1250 both
   report 256) are only distinguishable by `gfx`. Keep the real `gfx` the tuner
   wrote; don't drop the column. Legacy files without `gfx` are backfilled from
   `cu_num` (256→gfx950, 80/304→gfx942) — do not rely on that for new archs.

## Detect: run the guard

The authoritative check drives the **real runtime merge** (no re-implemented
logic, so it can't drift) against a temp copy of `aiter/configs/`:

```bash
python3 -m unittest op_tests.tuning_tests.test_config_shape_collision -v
```

- Requires torch (importing `aiter` pulls it in); it skips cleanly where torch is
  absent, so run it in an env/container that has torch.
- A failing family surfaces the runtime `RuntimeError`: a `Found N duplicate shape
  entries during merge of '<family>'` message, the colliding rows as a
  `Duplicate rows:` table (the merged config columns — no `_src`/key line is
  printed), and an `Updated files:` list of the source CSVs that were rewritten.

To see it the way production does (runtime prints the colliding rows), trigger
the merge directly on a scratch copy:

```python
import shutil, tempfile, aiter.jit.core as core
tmp = tempfile.mkdtemp(); shutil.copytree("aiter/configs", f"{tmp}/aiter/configs")
core.AITER_ROOT_DIR = tmp
type(core.AITER_CONFIGS).get_config_file.cache_clear()
core.AITER_CONFIGS.get_config_file(
    "AITER_CONFIG_GEMM_BF16", f"{tmp}/aiter/configs/bf16_tuned_gemm.csv", "bf16_tuned_gemm"
)  # raises RuntimeError listing duplicate rows + which files
```

## Find which rows/files collide

The `RuntimeError` (and the test failure) already prints every duplicate row and
its source file. To locate them yourself for a family, merge its files and group
by the runtime key (read the key from the untuned header — don't invent it):

```python
import glob, os, pandas as pd
name = "bf16_tuned_gemm"                      # the family
untuned = f"aiter/configs/{name.replace('tuned','untuned')}.csv"
key = pd.read_csv(untuned, nrows=0).columns.str.strip().tolist() + ["cu_num", "gfx"]
files = [f"aiter/configs/{name}.csv"] + [
    p for p in glob.glob(f"aiter/configs/model_configs/*{name}*.csv")
    if "untuned" not in os.path.basename(p)
]
df = pd.concat([pd.read_csv(f).assign(_src=os.path.basename(f)) for f in files if os.path.exists(f)])
key = [k for k in key if k in df.columns]
print(df[df.duplicated(key, keep=False)].sort_values(key)[key + ["us", "_src"]].to_string(index=False))
```

## Resolve: the auto-dedup already exists — just trigger it on the real tree

**Do not write a dedup script.** `update_config_files` (`aiter/jit/core.py`)
already resolves collisions: for each duplicate key it keeps the lowest-`us` row,
**writes the pruned CSVs back to the source files**, and raises asking you to
re-run. Your job is only to trigger that write-back against the *real* tree and
commit the result.

- **Trigger it (one command):** run the guard's `--fix` mode. Unlike the default
  detection (which runs on a temp copy to stay read-only), `--fix` runs on the
  **real** checkout so `update_config_files`' write-back lands where you can
  commit it:

  ```bash
  python3 op_tests/tuning_tests/test_config_shape_collision.py --fix
  ```

  It resolves every family, keeps the lowest-`us` row per shape, rewrites the
  source CSVs, and prints which files/rows changed. (Equivalent to running the
  model/op/tuner once, or calling `core.AITER_CONFIGS.get_config_file(...)` on the
  real tree — same built-in auto-dedup, no extra logic.)
- **Commit** the rewritten config CSVs (`git diff` shows the pruned rows), then
  re-run without `--fix` to confirm clean.
- **Manual alternative:** if you'd rather edit by hand, from the duplicate list
  delete the slower-`us` copy — usually the shape already exists in the canonical
  or an older model file, so remove it from the file you are adding. Keep exactly
  one row per key (the lowest `us`).

After resolving, re-run the guard until it passes.

## Cross-PR awareness (for authors and reviewers)

- Your green CI does **not** prove `main` stays green — it never saw the other
  in-flight config PRs. When adding shapes shared across models (common decode
  shapes, MoE token grids), assume another PR may add the same shape.
- Backstop (once wired): if this guard is added to a **push-to-`main`** job, two
  PRs that race and both land a colliding shape turn `main` red immediately — fix
  forward by pruning the duplicate (lowest-`us` wins). The guard is not wired into
  CI yet, so today this is a manual/local check, not an automatic backstop.
- Reviewing a config PR: skim other open PRs touching the same family, and trust
  the guard rather than eyeballing shapes.

## References (single source of truth — keep in sync)

- Guard test: `op_tests/tuning_tests/test_config_shape_collision.py` (drives the
  real merge; add a family here if you register a new `AITER_CONFIG_*` file).
- Runtime merge + dedup: `aiter/jit/core.py::get_config_file` /
  `update_config_files` (key derivation, gfx backfill, `_tag`, write-back).
- GPU end-to-end counterpart: `op_tests/tuning_tests/test_run_config.py` (runs
  every merged shape).

## Anti-patterns

- ❌ Checking only your new file, not the family-wide merge.
- ❌ Hand-picking key columns (missing `gfx`, adding non-key `libtype`) instead of
  reading the untuned header.
- ❌ Adding a second row for a shape that already exists somewhere in the family.
- ❌ "Fixing" a collision by widening the key, renaming a shape, or disabling the
  guard.
- ❌ Assuming green PR CI means `main` is safe (ignores concurrent config PRs).
- ❌ Dropping/zeroing the `gfx` column so shapes look distinct.
- ❌ Re-implementing the merge/dedup in a new script — call the runtime or the
  guard test instead.
