---
name: flydsl-aot-export
description: >
  Route an aiter FlyDSL kernel launch through the export_to_c AOT cache
  (aot_factory / bind_aot / FlyDSLAotCache) so a new process loads a prebuilt
  object instead of tracing. Use when converting a wrapper's _run_compiled
  launch to the AOT cache, when reviewing such a conversion (is every
  constexpr in the key or specialization_key?), when an entry falls back to
  the JIT path or raises a baked-argument error, and when validating a
  conversion without the target GPU.
argument-hint: [wrapper module, op name or cache name]
allowed-tools: Read Edit Bash Grep Glob
---

# FlyDSL AOT cache: converting a launch to export_to_c

How to move a `_run_compiled(launcher, ...)` launch onto
`aiter/ops/flydsl/aot_cache.py`, and how to check the result. The design, the
invalidation rules and the validated cases are in `docs/flydsl_aot_cache.md`;
read its §4.1 (keys), §4.2 (invalidation) and §6 (adopting) before a
non-trivial conversion. This file is the procedure and the checks.

Converted wrappers to copy from:

| Shape of the launcher | Example |
|---|---|
| Factory takes the constexprs, launcher takes runtime arguments only | `gemm_kernels.py` (`_preshuffle_launcher`), `gemm_a8w8_blockscale.py` (`_compile_gemm`) |
| Module-level `@flyc.jit` with constexpr launch arguments | `kernels/gemm_a16w16_gfx950.py` (`bind_aot` under key `()`, the config in `specialization_key`) |
| Constexprs bound by a cached `@flyc.jit` wrapper | `mxscale_preshuffle_kernels.py` (`_bound_gemm_exe`), `batched_gemm_a8w8_gfx950.py` |
| Launcher built inside a wrapper and kept in its own dict | `kernels/mxmoe_dispatcher.py` (`bind_aot` with `make_exe`) |
| Factory argument the cache cannot key (a tuple of dataclasses) | `kernels/mega_moe/mega_moe_stage1.py` (`_stage1_bundle_launcher`: the configs as tuples of their fields) |
| Caller keeps the function `flyc.compile` returned, or preloads it | `kernels/flydsl_dispatch_combine_intranode_op.py` (`_compile_and_keep`, `_preload_compiled` of a `lazy` handle) |

## The one rule

**Everything that changes the compiled code must be in the key.** The key is
the factory's arguments plus `_run_compiled`'s `specialization_key`. Arch, CU
count, FlyDSL itself, the kernel's source files and the environment variables
those files read are tracked for you. Tensor dtype, rank and unit-stride axis
are in the disk digest, but not in the in-memory key.

So, for each launch, list what the trace depends on and tick each item off:

- every `Constexpr[T]` parameter of the launcher, including those left to
  their default;
- every value the factory closes over;
- a tensor dtype that can vary for one launcher (`q.dtype`, an output dtype);
- a device, when the factory keeps one launcher per device (pass the
  `torch.device`, not its index: on disk a device is only its type, so one
  entry serves every GPU);
- a module global or a config file the kernel reads at trace time. The cache
  does not see these: pass the value in the key.

## Steps

1. **Find the launcher and what selects it.** Read the `_run_compiled` call
   and the launcher's signature. Note which parameters are `Constexpr`, which
   object caches the launcher today (`functools.cache`, a module dict, a
   module-level function) and who else launches the same launcher.
2. **Declare what the cache is built from** in `aiter/aot/flydsl/sources.py`:
   add `"my_kernel": {"sources": [...], "env": [...]}` to `AOT_CACHES`. List
   every file the kernel's code comes from besides the file that creates the
   cache and `COMMON_SOURCES` (paths relative to `aiter/ops/flydsl`; a
   directory stands for its `.py` files), and every variable those files read.
   Reuse a shared list there if the kernel family has one.
3. **Pick the pattern.**

   Constexprs go to the factory:

   ```python
   @aot_factory("my_kernel")            # the cache name; above functools.cache
   @functools.cache
   def _my_launcher(N, K, tile_m, out_dtype):
       from .kernels.my_kernel import compile_my_kernel   # import inside
       return compile_my_kernel(N, K, tile_m, out_dtype)

   _run_compiled(_my_launcher.lazy(N, K, tile_m, out_dtype), *args, stream)
   ```

   A module-level launcher that takes its constexprs at launch: a factory with
   no argument, and the constexprs in `specialization_key`:

   ```python
   @aot_factory("my_kernel")
   @functools.cache
   def _my_launcher():
       from .kernels.my_kernel import launch_my_kernel
       return launch_my_kernel

   spec = (tile_m, tile_n, out_is_f16, split_k)       # every Constexpr argument
   _run_compiled(_my_launcher.lazy(), *runtime_args, *spec, specialization_key=spec)
   ```

   A launcher built where a factory cannot return it (a dict of launchers, a
   launcher shared by several argument sets): bind it where it is built, keyed
   by the values that determine it, and give `make_exe` a fresh build:

   ```python
   bind_aot(launch, _MY_AOT, dataclasses.astuple(traits),
            make_exe=lambda: _build(traits)["launch"])
   ```

4. **Launch `factory.lazy(...)`, not `factory(...)`.** A hit then never calls
   the factory. Move the kernel-module imports into the factory and out of the
   wrapper's `_lazy_import`, or the hit still pays the `@flyc.kernel`
   definitions (a few hundred ms per module in a fresh process).
5. **One launcher, one cache.** When several wrappers launch the same
   launcher, define its factory once and import it. Two factories
   returning one launcher object bind it to two keys, and it falls back to
   the JIT path.
6. **Update what else compiles the kernel.** A preload or warm-up helper must
   go through the same launch path under compile-only
   (`_run_compiled(factory.lazy(), ...)` with `COMPILE_ONLY=1` set, or
   `_preload_compiled(factory.lazy(), ...)`, which builds the entry in a
   prebuild and loads it elsewhere), or it warms FlyDSL's cache and not the
   entry the launch loads. A call site that keeps what `flyc.compile`
   returned and calls it directly uses `_compile_and_keep(handle, ...)`. An
   AOT replay under `aiter/aot/flydsl/` needs no change if it enters at the
   public op.
7. **Document it.** Add the cache name, wrapper and key to
   `docs/flydsl_aot_cache.md` §5, and remove the op from the "not converted"
   list in §7.

## Checks before calling it done

Key coverage:

- [ ] Every `Constexpr` launch argument is in `specialization_key`, by value
      and in a stable order. Compare the tuple with the launcher signature
      line by line; defaults count.
- [ ] `specialization_key` tuples that reach one launcher from different call
      sites cannot be equal for different constexprs (different lengths, or a
      leading tag).
- [ ] Key elements are ints, floats, bools, strings, `None`, torch dtypes and
      devices, and tuples/lists/dicts of those. A numpy scalar, an Enum or a
      tensor-derived value leaves the launch on the JIT path. Convert at the
      call site (`int(x)`, `bool(x)`, `.value`).
- [ ] Every flag has one type, whatever the caller passed: coerce with
      `bool()` / `int()` before building the key and the launch arguments. A
      constexpr launch argument is compared by value with the one its entry
      was built with, and `False` and `0` differ there, so a key first
      launched with `False` raises when launched with `0`. As factory
      arguments they hash alike but give two disk digests.

Launch arguments:

- [ ] Runtime arguments are tensors, pointer arguments (`ptr_arg`,
      `flyc.from_c_void_p`), `fx` numerics or a stream. A torch stream is
      converted only as the last argument; pass `fx.Stream(...)` elsewhere.
- [ ] One key always launches with the same argument list. A later launch
      of a key with more arguments than its first, or too few to reach its
      constexprs, raises: do not pass an optional trailing constexpr on some
      calls of a key and not on others.

Behaviour kept:

- [ ] Nothing else refers to the launcher global, dict or helper you removed
      (`grep` the repo, including `op_tests/`, `csrc/` tuners and
      `aiter/aot/`).
- [ ] The wrapper has no AOT branch of its own: `_run_compiled` handles mode
      `0`, the fallback and compile-only.
- [ ] A kernel file shared with another arch keeps that arch's path as it
      was, unless you validate it too.

## What leaves a stale entry

An entry is looked up by four things, and by nothing else: the **key**
(factory arguments, `specialization_key`, `Constexpr` launch arguments), the
**contents of the declared source files** and the **declared variables**
(both in `aiter/aot/flydsl/sources.py`, plus the creating file), and the
**fingerprint** (FlyDSL and its own compile variables, arch, CU count).
Whatever else the generated code depends on is invisible: change it, and a
later process loads the old kernel without an error. When you convert or
review a launcher, look for each of these.

| What the kernel depends on | Example | Detected? | Do this |
|---|---|---|---|
| A value the factory or kernel closes over that is not a factory argument | `tile = _pick_tile(device)` inside the factory, then used in the kernel | No | Make it a factory argument, or pass it in `specialization_key` |
| A module global changed at run time | `kernels_common.FAST_MATH = True` set by a caller or a test | No | Pass the value through the key; do not patch module state |
| A constant in a file that is not declared | a helper the wrapper (outside `kernels/`) imports; `importlib.import_module(name)`; another installed package | Only an import between two files under `kernels/` (prebuild and unit test raise) | Add the file to the cache's `"sources"` |
| A non-Python input read while building | a tuning CSV or JSON the factory reads its tiles from | No | Read it in the wrapper and pass the values as factory arguments, or declare the file in the cache's `"sources"` |
| An environment variable a declared file reads | `os.environ.get("AITER_FLYDSL_MY_SWITCH")` in a kernel file | Yes, by a literal name: a prebuild and the unit test raise if it is not in the cache's `"env"` | Add it to the cache's `"env"`; if it cannot change generated code, to `ENV_NOT_CODE` in `aiter/aot/flydsl/sources.py` |
| An environment variable read in a file that is not declared, or passed in from a caller | the wrapper reads `MY_SWITCH` and branches in the factory | No | Read it in the wrapper and pass its value as a factory argument |
| A variable read under a computed name | `os.environ.get(f"AITER_X_{stage}")` | No | Do not do this in kernel code; read it by a literal name |
| A device property other than arch and CU count | memory size, device name, a driver version | No | Pass it in the key |
| A `Constexpr` launch argument missing from `specialization_key` | `launch(..., tile: Constexpr[int])` | Yes: stored per entry and compared; two values under one key raise | Still put it in `specialization_key` |
| Tensor dtype, rank or unit-stride axis that varies for one launcher | the same launcher called with fp8 and bf16 | On disk yes (the digest holds them); in memory no | Put it in `specialization_key` |
| Linked bitcode | `link_extern(..., bitcode_path=...)` | Yes: a prebuild raises if it is not declared | Add its path to the cache's `"sources"` |
| `True`, `1` and `1.0` as one key | a flag passed as `1` by one caller and `True` by another | No (they hash alike in memory) | Coerce to one type at the call site |

Two more rules that are not about the key:

- **Only a prebuild writes the cache.** A launch with no entry logs an error
  and traces on the JIT path. So a launcher no replay under
  `aiter/aot/flydsl/` covers never hits: add its shapes to a replay.
- **A variable is an input of the caches that declare it, and of no other.**
  Setting a variable a kernel's files read (an `AITER_FLYDSL_*` one too) to
  a value the prebuild did not have makes that kernel's entries miss. The
  variables the common sources read (`COMMON_ENV` in that file) and FlyDSL's
  own compile variables apply to every cache.

`AITER_FLYDSL_AOT_CACHE=verify` re-traces every entry a process loads and
compares it byte for byte, so it finds a stale entry of any kind above, but
only when the process loads that entry with a value that differs from the
build's. Run the op's test under it after a prebuild, with another value of
anything the kernel may depend on.

## Validating

Use a fresh `AITER_FLYDSL_AOT_CACHE_DIR` and `FLYDSL_RUNTIME_CACHE_DIR` for
each round. The log line `[FlyDSL AOT] built <path>.o` marks a build. A
`[FlyDSL AOT]` error line that mentions `the JIT path` marks a launch the
cache did not serve: after a prebuild of the same shapes it is a failure of
the conversion.

1. **Build.** Prebuild: run the op's replay (`python -m aiter.aot.flydsl.<op>`)
   or the op once under `COMPILE_ONLY=1`. Expect one `built` line per
   specialization and no fallback. A launch never builds: without an entry it
   logs an error (`no prebuilt entry`) and traces on the JIT path.
2. **Hit.** Run the op on the same directory in a new process. Expect no
   `built` line and no `no prebuilt entry` error.
3. **Verify.** Run it again with `AITER_FLYDSL_AOT_CACHE=verify`: every entry
   loaded from disk is re-traced and byte-compared. A mismatch means the key
   misses an input.
4. **Mode 0.** Run with `AITER_FLYDSL_AOT_CACHE=0`: the JIT path must still
   work, since it is the fallback.
5. **Sidecar.** Open a `<digest>-<variant>.json` of the new cache: `baked`
   lists the constexpr argument indexes and values, `sources` the files the
   cache declares. Compare them with what the kernel is built from: a file
   that is not declared invalidates nothing (§4.2 of the doc). Declare
   variables the kernel code reads under `"env"` in `aiter/aot/flydsl/sources.py`.
6. **Result.** On the target GPU, compare the op's output bitwise against
   mode `0`: prebuild the entries (the op's replay, or its test once under
   `COMPILE_ONLY=1`), then run its op test on that cache directory.

Without the target GPU, steps 1-5 run compile-only on any host:

```bash
# tuned GEMM / MoE rows of one arch, through the public op
GPU_ARCHS=gfx1250 python -m aiter.aot.flydsl.gemm --csv <tuned.csv>
```

For an op with no replay module, call the wrapper on FakeTensors:

```python
import os
os.environ["GPU_ARCHS"] = "gfx1250"       # before importing aiter
os.environ["AITER_AOT_REPLAY"] = "1"
import torch
from aiter.aot.flydsl.dry_run import dry_run

with dry_run("gfx1250", 256, python_ops=True):
    my_op(torch.empty(4, 128, 576, dtype=torch.bfloat16, device="cuda"), ...)
```

Fake storage addresses are made up, so a host check that compares pointers
(an overlap or alignment test) can fail there for no real reason; patch that
check in the script, not in the wrapper. Say in the PR that the validation
was compile-only and that step 6 is still owed.

Run `python -m pytest -q tests/test_flydsl_aot_launch.py
tests/test_flydsl_aot_cache.py` after touching `aot_cache.py`, `aot_launch.py`
or `tensor_shim.py`, and after a FlyDSL pin bump.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `was compiled with constexpr ...; pass every constexpr launch argument in specialization_key` | One key launched with two values of a constexpr, or with one value in two types (`False` and `0`) | Add the argument to `specialization_key`; coerce flags to one type |
| `was compiled for N launch arguments and is launched with ...` | One key launched with different argument lists | Pass the same arguments at every launch of the key |
| `no exported entry for ...; using the JIT path for it` | The build, export or load of the entry failed; the reason is in the message | Fix the reason (an argument type `export_to_c` does not support, a stream that is not last) |
| `one launcher bound to keys ... and ...` | Two factories or keys return one launcher object | One factory per launcher; for shared launchers use `bind_aot` at the build |
| `cannot bind a dict` / `tuple` | The factory returns several launchers | `bind_aot` each launcher where it is built |
| `... launching through the JIT path` at bind time | A key element the cache cannot store | Convert it to a plain value at the call site |
| `no prebuilt entry in ...; tracing it on the JIT path` | A launch found no entry: only a prebuild (`COMPILE_ONLY`) builds one | Run the op's replay in `aiter/aot/flydsl`, or add the shape to its list |
| `changed after this process hashed it` | A declared source was edited while a prebuild ran | Rerun the prebuild |
| `does not declare [...], which its sources import` | A declared kernel file imports a kernel file missing from the cache's `"sources"` in `aiter/aot/flydsl/sources.py` | Add it to the cache's `"sources"` |
| `links [...], which it does not declare` | The kernel links bitcode missing from the cache's `"sources"` | Add its path there |
| A second prebuild builds again | The key, a declared source or a declared variable differs between runs | Diff the two sidecars |
| `verify` mismatch | The trace reads something the key does not hold | Find it (a global, a config, a device property) and put it in the key |
| The preload/tuner precompile does not speed up the first launch | It compiles the raw launcher | Step 6 above |

## Limits to state, not to hide

- A hit skips the trace. With `bind_aot` on a launcher built before the bind,
  the kernel definition still runs once per process.
- `factory.lazy()` handles and loaded entries are not evicted, whatever
  `maxsize` the factory's `lru_cache` has.
- `COMPILE_ONLY` is read for the whole process, so a launch on another thread
  during a preload compiles and does not run. Preload in tuner and prebuild
  workers, not beside inference.
- Entries are built only by a prebuild: a replay under `aiter/aot/flydsl/`,
  or the op run once under `COMPILE_ONLY=1`. A launcher no replay covers
  never hits; add its shapes to a replay when converting it.
