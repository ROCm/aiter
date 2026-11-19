# FlyDSL AOT cache: skipping the per-process trace

This page covers three things:

- Why FlyDSL 0.5.0 kernels are slow on first call in every new process.
- The general ways to deal with it, and why aiter picks an exported-object
  cache.
- The design of aiter's `aiter/ops/flydsl/aot_cache.py`, with the preshuffle
  GEMM as the first user.

## 1. The problem

FlyDSL 0.5.0 identifies a compiled artifact by the traced MLIR IR. To look up
its disk cache, it first runs the Python kernel to produce that IR. The
artifact cache saves the MLIR→ISA lowering, but not the trace. The comment at
`flydsl/compiler/jit_function.py:1199` says so: *"Every ordinary JIT/compile
request traces before ANY artifact hit"*.

The result is that every new process pays the full trace once for every
specialization it uses: each GEMM tile config, each MoE stage and each
`(N, K)`. Measured costs:

| kernel | trace on a warm FlyDSL disk cache |
|---|---|
| preshuffle GEMM (`kernels/preshuffle_gemm.py`) | 70–320 ms per config |
| MoE / attention kernels | 1–4 s per config |

A serving process that loads a tuned CSV with hundreds of configs therefore
spends minutes tracing at startup, or stalls on the first request that hits
each config.

## 2. Design options

### 2.1 Warm up at startup

Enumerate the configs a process will use (for a GEMM, one per unique tile
config of the tuned table: `M` is a runtime `Int32`, so it costs no compile),
compile them in parallel worker processes into FlyDSL's disk cache, then
trace each config once more in the main process to get a handle.

This moves the cost to startup and parallelizes the lowering, but it **does
not remove the trace**: every process still traces every config, because
FlyDSL 0.5.0 has no other way to produce a handle.

### 2.2 aiter today: build-time replay

`aiter/aot/flydsl/` already replays the tuned CSVs through the public op
entry points at build time. `dry_run.py` runs them under `FakeTensorMode`
with `COMPILE_ONLY=1` in parallel workers, which fills FlyDSL's disk cache.
This is the warm-up of §2.1 done at `setup.py` time, so it has the same
limit: each process still traces every config on first use.

### 2.3 Exported-object cache

Export each compiled specialization as a host object (`export_to_c`) under
a key the wrapper builds from its constexprs, and load the object on a later
hit (`flyc.load_module`) without tracing. The open questions are how to key
an entry and how to invalidate it when a kernel source changes:

- hashing a whole kernel directory into the cache directory is safe but
  invalidates every kernel on any edit;
- a per-caller list of source files is finer, but a helper missing from the
  list is a silent stale hit;
- recording the files the trace actually runs (§4.2) is per kernel and needs
  no list.

### 2.4 Comparison

| | removes trace on reload | needs FlyDSL internals | invalidation |
|---|---|---|---|
| startup warm-up | no (moves it to startup) | no | FlyDSL's own IR key |
| aiter AOT `dry_run` | no | no | FlyDSL's own IR key |
| this design | **yes** | no | toolchain/env dir + **per-entry recorded sources** + key |
| a cache of FlyDSL handles keyed by private state | yes | **yes** (rejected) | fragile |

## 3. FlyDSL building blocks

| API | use |
|---|---|
| `flyc.compile(exe, *args, compile_only=True)` | Compiles without launching and returns an exportable `CompiledFunction`, also under `COMPILE_ONLY=1` (where the call without `compile_only` returns `None`). The cache passes CPU placeholders, so FakeTensors work too. |
| `CompiledFunction.export_to_c(dir, file_name, function_prefix="")` | Writes `<file_name>.o` and `<file_name>.h`. `dir` must exist. Byte-for-byte deterministic for the same input. |
| `flydsl.compiler.protocol.cache_signature(jit_arg)` | Documented as "for keying user-managed caches of `flyc.compile` handles". For a torch tensor it gives `(TorchTensorJitArg, dtype, align, use_32bit_stride, shape=(-1,…), stride with the unit axis)`. |
| `flydsl.compiler.JitArgumentRegistry.get(type(v))` | Returns the JIT-argument constructor `flyc.compile` itself uses, for building `cache_signature`. |
| `flyc.load_module(path)` | Loads an `export_to_c` object in memory (no system linker), checks its schema and host triple, loads the GPU module on the current device (on another device at its first call there) and returns a `CompiledFunction` whose argument marshalling comes from the ABI in the object metadata. In FlyDSL since 0.5.1.dev1058, which aiter pins; with an older FlyDSL every key takes the JIT path. |

The object carries `SYM__metadata`, a JSON with the ABI slots, `arch`,
`backend`, `flydsl_version`, `host_triple` and `schema_version` (3). Each
tensor contributes two slots, a data pointer and a packed layout struct with
its dynamic shape and stride values; a scalar slot holds its value; the stream
slot holds a `hipStream_t`.

The APIs above are stable under FlyDSL's `docs/api_stability.md`
(including `flyc.load_module`). The cache also uses three FlyDSL
internals, which a FlyDSL release may move or rename without notice:

| internal | use |
|---|---|
| `flydsl.compiler.jit_argument.PointerJitArg` | Constructed with a null pointer as the placeholder of a pointer argument, so the compile-only export does not launch. |
| `flydsl.utils.env` (`env.compile.compile_only`) | Detects `COMPILE_ONLY`. |
| `fx.Stream._is_stream_param` | Recognizes a stream argument, whose placeholder is a null stream. |

Re-run `op_tests/test_flydsl_aot_cache.py` on every FlyDSL pin bump.

## 4. aiter design: `aiter/ops/flydsl/aot_cache.py`

The design is the exported-object cache of §2.3, built on the FlyDSL APIs
in §3.

```
op wrapper ──► FlyDSLAotCache.launch(key, make_exe, *launcher_args)
                 │
                 ├─ memory hit (key) ───────────────────────► CompiledFunction(*args)
                 │
                 └─ digest = sha256(name, key, cache_signature of each runtime arg)
                    dir    = $AITER_FLYDSL_AOT_CACHE_DIR/<fingerprint>/<name>/
                    │
                    ├─ some <digest>-<srchash>.json whose recorded sources all
                    │  still hash the same, and its .o exists
                    │    ──► flyc.load_module(.o)   ~2-3 ms, no trace, make_exe not called
                    └─ miss (flock <digest>.lock, re-check)
                         with sys.setprofile recording every aiter .py that runs:
                           exe = make_exe()
                           flyc.compile(exe, *CPU placeholders, compile_only=True)
                         export_to_c into a temp dir
                         os.replace(.json), then os.replace(.o) last
                         ──► flyc.load_module(.o)
```

The process that builds an entry loads it like every other process, so the
exported object is what runs everywhere.

### 4.1 Keys

- **`key`** is chosen by the caller. Within one cache `name` it must
  determine the specialization: every constexpr given to the factory, plus
  every tensor dtype that can vary. It is the in-memory lookup key, so the
  hit path does no hashing.
- **The disk digest** also hashes `cache_signature` of every runtime
  argument (dtype, rank, unit-stride axis) and the Python type of every
  scalar. A key that forgets something the signature covers can then at
  worst miss on disk; it cannot load another process's mismatched `.o`.
- **Key elements** are ints, floats, bools, strings, `None`, torch dtypes
  and devices, and tuples, lists and dicts of those. The disk digest uses a
  normalized form (`_normalize_key`): dtypes become strings, a device becomes
  only its type, since one `.o` serves every GPU. A device in `key` thus
  separates entries in memory only, which the JIT path needs (§4.3). Anything
  else raises, because its `repr` may hold a memory address and would give
  every process a different digest.
- **Scalar types** come from the ABI in the object metadata, so a hit never
  builds `exe`. That matters: running the `compile_preshuffle_gemm(...)`
  factory alone costs about 230 ms of AST rewriting, and importing a kernel
  module about 30 ms, so wrappers import the compiler inside `make_exe`.

### 4.2 Invalidation (how `.o` files stay current)

There are two levels.

**Level 1: the directory.** `fingerprint()` selects the top-level directory.
It is computed on each miss; only the FlyDSL hash is cached per process, since
an AOT build switches arch and environment within one process. It covers everything that is not aiter
source:

- the cache schema, Python `major.minor` and `platform.machine()`;
- `flydsl.__version__`, **the contents of every FlyDSL `.py`**, and the name,
  size and mtime of FlyDSL's native `.so`/`.a` files;
- `torch.version.hip` and the target arch. That is
  `torch.cuda.get_device_properties().gcnArchName`, or FlyDSL's
  `get_rocm_arch()` when there is no device or `FLYDSL_GPU_ARCH` is set;
- every `AITER_FLYDSL_*` / `FLYDSL_*` env var, plus
  `HSA_OVERRIDE_GFX_VERSION` and `COMPILE_OPT_LEVEL`. Variables that cannot
  change code are excluded: dump dirs, the cache's own variables, and
  `FLYDSL_GPU_ARCH`, which is hashed as the arch it selects. Otherwise an AOT
  build, which sets it, and the runtime, which reads the device, would use
  different directories.

**Level 2: per-entry sources.** aiter sources are not hashed as a directory.
During a miss, `make_exe()` and the trace run under `sys.setprofile`, which
records the `co_filename` of every Python function called. Every recorded
`.py` inside the aiter package (except `aot_cache.py` itself) is stored with
its sha256 in the entry's sidecar `.json`, together with any
`extra_sources`. A lookup accepts an entry only if all of those files still
hash the same. The per-process digest cache makes this one `sha256` per file
per process.

For the preshuffle GEMM the recorded set is `gemm_kernels.py`,
`kernels/preshuffle_gemm.py`, `kernels/kernels_common.py` and
`ops/triton/__init__.py` (an import side effect, harmless). So:

- editing `kernels/act.py`, or any MoE kernel, still hits every GEMM entry;
- editing `kernels/preshuffle_gemm.py` rebuilds only the GEMM entries, and
  only when each is next used.

Entries are stored per source version, `<digest>-<sources hash>.{o,json}`.
Switching a file back (e.g. `git checkout` of another branch and back) finds
the earlier build again instead of rebuilding it. Old variants are never
read once their sources change; delete them, or old fingerprint directories,
to reclaim space.

Recording costs about 80 ms on a miss (≈110 ms → ≈190 ms), once per entry
per machine. A hit pays nothing for it.

Known holes, and how they are handled:

| hole | example | mitigation |
|---|---|---|
| A module-level value read from a file none of whose functions run | `TILE = 64` in `consts.py`, imported by name | `setprofile` sees calls, not attribute reads. Pass the file in `extra_sources`, or run `verify` mode |
| Code outside the aiter package affects the kernel | a helper in another installed package | Pass it in `extra_sources`, or run `verify` mode |
| The trace reads state that is neither env nor source | CU count, a global set at runtime, a config file | Put it in `key`. `verify` mode detects it |
| An incomplete hand-written `key` | a forgotten constexpr | The disk digest still covers the argument types. `verify` mode re-exports and compares |
| A partially written or corrupt entry | crash during a build | `.o` is published last with `os.replace`. A load failure rebuilds the entry once |

Recording the files the trace actually ran gets per-kernel granularity
without a list to maintain (§2.3). The remaining hole (constants read
without a call) is the reason `verify` mode exists.

Modes, set with `AITER_FLYDSL_AOT_CACHE`:

- `1` (default): on. It is safe on by default because per-entry source tracking makes edits during kernel development
  rebuild only what they touch.
- `0`: off. The op uses the existing `_run_compiled` path.
- `verify`: like `1`, but on every disk hit it re-traces, re-exports, and
  byte-compares the `.o`. A mismatch raises and names the stale entry.
  `export_to_c` is deterministic, so a mismatch always means the key or the
  fingerprint missed an input. Use it in CI and after FlyDSL bumps.

### 4.3 Launching

A hit is the `CompiledFunction` `flyc.load_module` returns. It fills the ABI
slots from the arguments with FlyDSL's own `CallState` and calls the entry;
a nonzero status raises. Pointer arguments (`ptr_arg`, a `PointerJitArg`) are
compiled with a null pointer of the same element type, address space and
alignment: a non-null pointer counts as a live device argument and
`flyc.compile` would launch on it.

**Every mode goes through `launch`.** Mode `0` and keys that cannot be
exported or loaded (an argument type `export_to_c` does not support) use
`_JitKernel`, which calls `_run_compiled(exe, *args,
specialization_key=key)`. A fallback logs one warning per key. Caller bugs
(`_UsageError`, e.g. an unsupported key element) and `verify` failures
raise instead. On this path a handle only launches on the device it was
compiled on (FlyDSL issue 4 in §7), so a wrapper whose factory is cached per
device puts the device in `key`, as `gemm_a8w8_blockscale` does. A loaded
entry works on every device.

**The stream argument.** `torch.cuda.current_stream()` builds a Python
stream object and costs about 5 us, more than half of a launch. Wrappers
pass `current_stream_arg(device)` from `kernels/tensor_shim.py`, which reads
the raw handle with `torch._C._cuda_getCurrentRawStream` and wraps it in
`fx.Stream`: about 0.6 us. `current_raw_stream` returns the int for code that
keys buffers by stream.

### 4.4 Build-time prebuild (compile-only)

Under `COMPILE_ONLY=1`, including aiter's `aot/flydsl/dry_run.py` with
FakeTensors, `launch()` still builds the `.o` but neither loads nor launches
it. The placeholders are made with `unset_fake_temporarily`. This holds even
for a key the process already launched: like `_run_compiled`, `launch` never
dispatches under `COMPILE_ONLY`. The hot path tests `"COMPILE_ONLY" in
os.environ` (0.1 us) and reads the FlyDSL option (0.8 us) only when it is
set.

Under `COMPILE_ONLY`, `launch()` also runs FlyDSL's own compile-only launch
(`_run_compiled`), as the JIT path does, after the export. The replay in
`aot/flydsl/gemm.py` needs it: `requested_kernels()` learns which kernel an M
requests from that launch, and bisects M between two rows when their ends
request different kernels. Without it, a key whose `.o` is already on disk
(from an earlier run, or another M in the same run) traces nothing; the
bisection then stops early and skips kernels. On the mxfp8 CSV below this
lost 7 of 49 kernels. The export runs first because FlyDSL's in-process
trace cache would otherwise hide the sources the entry records. When the
export just built the entry, its own compile already showed the replay the
request and filled FlyDSL's cache (the same artifact), so the JIT launch is
skipped. Otherwise it runs and also fills FlyDSL's own cache for mode `0`
and the fallback, like before this cache.

Cost, gfx950 mxfp8 and a4w4 CSVs above, run one after the other on fresh
directories: 155 s at HEAD, 164 s with the cache (62 GPU binaries, the same
set). A replay whose entries all exist still traces every request.

Parallel replays share both caches safely. FlyDSL's own disk cache takes a
per-artifact file lock, so one process runs LLVM and the others load its
result; the export takes `flock <digest>.lock`. Measured on the mxfp8 CSV,
two replays at once on one directory (40 workers): 58 LLVM compiles and 58
exports for 58 kernels, none twice. Each process still traces the kernels it
requests, which the replay needs. With `FLYDSL_RUNTIME_ENABLE_CACHE=0` or an
IR dump, FlyDSL skips its cache and lock, and processes compile the same
kernel in parallel; exports stay single. `flock` assumes a local file system.

Files under `aiter/aot/` are not recorded as sources: the replay's hooks run
inside the trace, and editing the replay must not invalidate every entry.
Measured on the bmm rows: editing another kernel's file rebuilds nothing,
editing `kernels/bmm_a8w8_mxscale_gfx950.py` rebuilds exactly its 8 entries,
and reverting the edit reuses the earlier ones.

Running the AOT dry run with `AITER_FLYDSL_AOT_CACHE=1` and a fixed
`AITER_FLYDSL_AOT_CACHE_DIR` therefore prebuilds `.o` files for a target arch
on any machine. The dry run's fingerprint equals the one a process on that
arch computes, so the runtime hits them without tracing.
Ops routed through `torch.ops.aiter_routed` (`aiter/plugin`) are the
exception: under the dry run's `FakeTensorMode` their registered fake runs
instead of the op, so the replay never reaches FlyDSL for them and they are
not prebuilt. This predates the cache and is left to a separate change.

## 5. Validated cases

### 5.1 gfx942: `flydsl_preshuffle_gemm_a8`

```python
key = (tuple(sorted(compile_kwargs.items())), x_scale.dtype, w_scale.dtype)
_PRESHUFFLE_AOT_CACHE.launch(
    key, lambda: _get_compile_fn()(**compile_kwargs), *args
)
```

- All launcher tensors are flat contiguous views. `compile_kwargs`
  (`N`, `K`, tiles, `in_dtype`, `out_dtype`, `split_k`, …) together with the
  two scale dtypes determine the signature.
- `M` and `N` are runtime `Int32`s, so one entry serves every `M`. For a
  tuned CSV that means one entry per unique `(N, K, tile config)`.

MI308X (gfx942), FlyDSL 0.5.0.dev1039 (with `flyc.load_module`),
`op_tests/test_flydsl_aot_cache.py`, each mode in a fresh process. "first ms" is the wall time of the first call of that config in the
process.

| m | n | k | qdtype | jit first ms | aot miss first ms | aot hit first ms | aot verify first ms | jit us | aot hit us | bitwise equal |
|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|
| 1 | 1024 | 512 | fp8 | 463.6 | 864.3 | 17.0\* | 357.9 | 4.14 | 4.48 | True |
| 1 | 1024 | 512 | int8 | 174.4 | 171.8 | 3.7 | 93.2 | 4.15 | 4.48 | True |
| 200 | 2048 | 1024 | fp8 | 219.5 | 176.7 | 3.4 | 90.7 | 10.27 | 10.25 | True |
| 200 | 2048 | 1024 | int8 | 280.3 | 174.2 | 3.3 | 93.8 | 9.86 | 9.99 | True |
| 64 | 1024 | 2048 | fp8 | 241.4 | 209.5 | 26.1\*\* | 122.7 | 9.47 | 9.73 | True |
| 64 | 1024 | 2048 | int8 | 218.9 | 186.9 | 3.4 | 102.2 | 9.41 | 9.91 | True |
| 128 | 4096 | 1024 | fp8 | 316.2 | 192.3 | 3.4 | 102.8 | 11.19 | 11.30 | True |
| 128 | 4096 | 1024 | int8 | 343.7 | 194.6 | 3.5 | 102.5 | 11.29 | 11.43 | True |

\* one-time process setup, mostly the fingerprint (about 15 ms). The first
`load_module` in a process takes about 9 ms, later ones about 2 ms.
\*\* includes the first allocation of the split-K semaphore.

- The JIT column is measured with FlyDSL's own disk cache already warm, so
  it is the trace alone.
- The miss pays the trace, the export and about 80 ms for source recording,
  once per entry per machine. A hit costs about 2-3 ms.
- Kernel time is unchanged. Host time of the whole wrapper is 36.4 µs on a
  hit against 37.7 µs through JIT (m=1, n=1024, k=512, 2000 calls).

Also verified (`AITER_FLYDSL_AOT_CACHE_DIR` fresh each time):

- Two processes missing the same entries concurrently build each entry once,
  leave no `.tmp-*`, and a third process only hits.
- CUDA graph capture and replay match eager, also when the first call of
  an entry (and so its `load_module`) happens inside the capture, and when
  the capture is on a second GPU of the process.
- One entry launched on all 8 GPUs of one process gives identical results.
  `load_module` creates a HIP context only on the current device; another
  device loads the module on its first call there.
- Alternating `cuda:0` and `cuda:1` in one process is correct. (Through the
  JIT path, mode `0`, this fails at HEAD as well; see FlyDSL issue 4.)
- `verify` mode passes on good entries and raises on a tampered `.o`.
- Editing an unrelated kernel file still hits; editing
  `kernels/preshuffle_gemm.py` or the wrapper rebuilds only these entries,
  and restoring it hits the earlier variants with no rebuild.

### 5.2 gfx950: `run_gemm_a8w8_blockscale`

```python
key = (N, K, kernel_name, Out.dtype, XQ.device)
_AOT_CACHE.launch(
    key,
    lambda: _compile_gemm(N, K, kernel_name, XQ.device.index, Out.dtype),
    ...
)
```

`_compile_gemm` keeps one launcher per device, so the device is in `key`
(memory only; the `.o` is shared). Validated compile-only on gfx942 under
`dry_run("gfx950", 256)` for the tuned kernel
`flydsl_blockscale_8w_256x256x128_F8_F8_B16_ps0`, M=8192 N=768 K=7168, bf16
and fp16 output:

- the first process builds two `.o` (metadata `arch: gfx950`), a second
  process builds nothing, `verify` passes;
- the final ISA of each exported kernel is byte-identical to the JIT compile
  of the same call (`FLYDSL_DUMP_IR`, 1487 and 1360 lines);
- the dry run's fingerprint equals the one a gfx950 process computes.

`python -m aiter.aot.flydsl.gemm --csv` on the tuned CSV builds the same
entries (same fingerprint) through the public op. Runtime on gfx950 hardware
is still to be checked.

### 5.3 gfx950: mxfp8 / a4w4 GEMMs

Three launcher families, five caches:

| Cache | Wrapper | Key |
|---|---|---|
| `mxscale_preshuffle_gemm`, `_fused`, `mxscale_preshuffle_splitk_reduce` | `mxscale_preshuffle_kernels.flydsl_mxscale_preshuffle_gemm` (also a4w4) | the launcher's constexprs; `(split_k, out_dtype)` for the reduce |
| `gemm_a8w8_mxfp8_8wave` | `gemm_a8w8_mxfp8_8wave` | `sorted(compile_kwargs.items())` |
| `bmm_a8w8_mxscale_gfx950` | `batched_gemm_a8w8_gfx950` | the launcher's constexprs |

Every key also holds the device of the output or first tensor, for the JIT
path (FlyDSL issue 4 in §7); the disk digest keeps only the device type.

The constexprs are bound by a cached `@flyc.jit` wrapper per key, which copies
the kernel launcher's `compile_hints` (the bmm one sets an LLVM option).

Validated compile-only on gfx942 with `GPU_ARCHS=gfx950 python -m
aiter.aot.flydsl.gemm` on two tuned CSVs: 20 a8w8 mxfp8 rows (bmm split-K
1/2/4/8, 8-wave, preshuffle split-K 1/2/4 and fused 3/7) and 3 a4w4 rows
(split-K 7/8 with the separate reduce):

- the replay requests the same 49 kernels as without the cache, and every
  final ISA is byte-identical (`FLYDSL_DUMP_IR`, md5 of `*final_isa*`);
- it builds 62 `.o`; a second run with the same directory builds nothing and
  requests the same 49 kernels.
- Rerun with FlyDSL 0.5.0.dev1039: mode `0` and two runs of mode `1` give the
  same 62 `gpu.binary` blobs (compared in FlyDSL's cache `.pkl` files, not the
  per-name ISA dumps, which keep only the last of several compiles), identical
  to the earlier FlyDSL as well.

Runtime on gfx950 hardware is still to be checked. gfx1250 mxfp8 kernels are
not converted.

## 6. Adopting it for another op

0. **Import the kernel compiler inside `make_exe`.** A hit then never pays
   the import (about 30 ms of AST rewriting per kernel module).
1. **Launch arguments are runtime only.** Every launcher parameter that
   varies per call is a tensor, an `fx` numeric (`Int32`, …) or an
   `fx.Stream`. Constexprs go to the factory (`make_exe`) and into `key`.
2. **`key` covers the specialization.** Include every constexpr, every
   tensor dtype that can vary, and any trace-time state that is not in the
   environment or the sources. Examples: CU count, a mode global. Tensor
   rank and unit-stride axis are covered by the disk digest. Include them in
   `key` too if they can vary for the same `key`, as the FlyDSL
   compiled-handle rules in `aiter/ops/flydsl/AGENTS.md` require.
3. **Pass extra sources.** If the kernel reads module-level constants from
   a file none of whose functions run during the trace, or uses code outside
   the aiter package, pass those paths as `extra_sources=`.
4. **Call `launch` unconditionally.** It handles mode `0`, the fallback
   and compile-only; the wrapper needs no branch.
5. **Test it.** Run the op's test under `AITER_FLYDSL_AOT_CACHE=verify` once,
   and add a bitwise check against the JIT path.

## 7. Limitations and next steps

FlyDSL-side issues found with an earlier `flyc.load_module`. Issues 1-3 are
fixed by the `load_module` FlyDSL ships (0.5.0.dev1039 here: first load about
9 ms with no subprocess, one HIP context, a hit's host time below the JIT
path's). Issue 4 is in the JIT path and remains.

1. **The first load runs `rocm_agent_enumerator`** (~145 ms). It resolves the
   runtime libraries through `get_backend()`, which detects the target arch;
   linking needs no arch. Resolving them from `_get_backend_class(metadata
   ["backend"])` (with `jit_runtime_lib_basenames` a classmethod) makes the
   first load 3.4 ms.
2. **It loads the module on every visible device**, which creates a HIP
   context on each: about 186 MiB per extra GPU in a process that uses one,
   multiplied by every TP rank. Loading on the current device and, when the
   entry returns -2 (not loaded on this device), loading there and retrying
   costs nothing on the hot path and leaves other GPUs untouched.
3. **Slot fills are generic**: about 1.0 µs per tensor layout slot and 0.3 µs
   per data slot, against 0.2-0.3 µs for the JIT path's generated fills. One
   generated fill per slot (as `TorchTensorJitArg` does) takes a call from
   16.2 to 11.1 µs.
4. **A JIT `CompiledFunction` launches only on its first device.**
   `JitFunction._mem_cache` is keyed without the device, so `flyc.compile` on
   another device returns a handle whose module is not loaded there:
   `hipErrorInvalidDevice`, printed, not raised. A loaded `.o` is not
   affected.

- Nothing garbage-collects old fingerprint directories or source variants
  yet. Build directories (`.tmp-*`) left by a crashed process are removed
  after an hour.
- Wire `AITER_FLYDSL_AOT_CACHE=1` into `aiter/aot/flydsl` so a wheel or
  container image can ship prebuilt `.o` files for its tuned CSVs.
- `.lock` files, one per digest, are not removed.
