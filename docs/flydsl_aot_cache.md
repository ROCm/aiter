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
- recording the files a trace runs needs no list, but what a trace runs
  depends on what the process ran before (a memoized factory, a module
  already imported), so the recorded set is not reliable;
- a list of source files declared per cache (§4.2), as FlashInfer's
  `gen_jit_spec(name, sources)` has, is per kernel family and explicit. A
  file missing from the list is checked for instead of detected at run time.

### 2.4 Comparison

| | removes trace on reload | needs FlyDSL internals | invalidation |
|---|---|---|---|
| startup warm-up | no (moves it to startup) | no | FlyDSL's own IR key |
| aiter AOT `dry_run` | no | no | FlyDSL's own IR key |
| this design | **yes** | no | toolchain/env dir + **sources declared per cache** + key |
| a cache of FlyDSL handles keyed by private state | yes | **yes** (rejected) | fragile |

## 3. FlyDSL building blocks

| API | use |
|---|---|
| `flyc.compile(exe, *args, compile_only=True)` | Compiles without launching and returns an exportable `CompiledFunction`, also under `COMPILE_ONLY=1` (where the call without `compile_only` returns `None`). The cache passes CPU placeholders, so FakeTensors work too. |
| `CompiledFunction.export_to_c(dir, file_name, function_prefix="")` | Writes `<file_name>.o` and `<file_name>.h`. `dir` must exist. Byte-for-byte deterministic for the same input. |
| `flydsl.compiler.protocol.cache_signature(jit_arg)` | Documented as "for keying user-managed caches of `flyc.compile` handles". For a torch tensor it gives `(TorchTensorJitArg, dtype, align, use_32bit_stride, shape=(-1,…), stride with the unit axis)`. |
| `flydsl.compiler.JitArgumentRegistry.get(type(v))` | Returns the JIT-argument constructor `flyc.compile` itself uses, for building `cache_signature`. |
| `flyc.load_module(path)` | Loads an `export_to_c` object in memory (no system linker), checks its schema and host triple, loads the GPU module on the current device (on another device at its first call there) and returns a `CompiledFunction` whose argument marshalling comes from the ABI in the object metadata. In FlyDSL since 0.5.1.dev1058 (aiter pins 0.5.1); with an older FlyDSL every key takes the JIT path. |

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

Re-run `python3 -m pytest -q tests/test_flydsl_aot_cache.py` on every FlyDSL pin bump.

## 4. aiter design: `aiter/ops/flydsl/aot_cache.py`

The design is the exported-object cache of §2.3, built on the FlyDSL APIs
in §3.

Kernels opt in at their launcher factory (`aiter/ops/flydsl/aot_launch.py`,
§6); the op wrapper keeps calling `_run_compiled`, which hands a bound
launcher to the cache:

```
op wrapper ──► _run_compiled(launcher, *launcher_args)
                 │  launcher bound by aot_factory / bind_aot
                 ▼
               FlyDSLAotCache.launch(key, make_exe, *launcher_args)
                 │
                 ├─ memory hit (key) ───────────────────────► CompiledFunction(*args)
                 │
                 └─ digest = sha256(name, key, cache_signature of each runtime arg,
                    │                contents of the declared sources,
                    │                values of the declared variables)
                    dir    = $AITER_FLYDSL_AOT_CACHE_DIR/<fingerprint>/<name>/
                    │
                    ├─ some <digest>-<variant>.json that bakes the constexpr
                    │  values of this call, and its .o exists
                    │    ──► flyc.load_module(.o)   ~2-3 ms, no trace, make_exe not called
                    ├─ miss, launching ──► error log, trace on the JIT path;
                    │                      nothing is built or written
                    └─ miss under COMPILE_ONLY, i.e. a prebuild
                       (flock <digest>.lock, re-check)
                         check the declared sources for a forgotten kernel file
                         exe = make_exe()
                         flyc.compile(exe, *CPU placeholders, compile_only=True)
                         check that linked bitcode is declared
                         check that the sources still have the hashed contents
                         export_to_c into a temp dir
                         os.replace(.json), then os.replace(.o) last
```

**Who writes the cache.** Only a prebuild: a launch under `COMPILE_ONLY`,
which is what the replays in `aiter/aot/flydsl` run in their worker
processes (§4.4). A process that launches kernels loads entries and never
builds one; with no entry on disk it logs an error and traces on the JIT
path, as it did before this cache, and FlyDSL's own cache still saves the
LLVM compile. The reason is that Python code is imported once: a process
that has run for a while traces what it imported, which may no longer be
what its source files say, and nothing reliable tells the two apart. A
prebuild worker imports its sources, hashes them and builds within seconds.

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
  and devices, and tuples of those (hashable: the key also looks the kernel up in memory). The disk digest uses a
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
- `flydsl.__version__`, **the contents of every FlyDSL `.py`**, and FlyDSL's
  native libraries: their sha256 from the wheel's `RECORD`, so reinstalling
  the same wheel (e.g. in a later Docker stage) keeps the fingerprint, or, for
  a source build without a `RECORD`, the name, size and mtime of its
  `.so`/`.a` files;
- the ROCm device bitcode (`ocml.bc`, `ockl.bc`, …) FlyDSL links into a
  kernel that calls it: FlyDSL's own digest of the files under the toolkit it
  resolves (bundled, or `ROCM_PATH` / `/opt/rocm` without in-process LLD);
- `torch.version.hip` and the target arch. That is
  `torch.cuda.get_device_properties().gcnArchName`, or FlyDSL's
  `get_rocm_arch()` when there is no device or `FLYDSL_GPU_ARCH` is set;
- the CU count, `chip_info.get_cu_num()`. It honours `CU_NUM`, which the AOT
  dry run sets from each tuned row's `cu_num`, so a prebuild for a CU count
  and a process on such a GPU use the same directory;
- the variables that change what the toolchain makes of any kernel:
  `HSA_OVERRIDE_GFX_VERSION`, `COMPILE_OPT_LEVEL` and the ones FlyDSL keys
  its own artifact cache on (`FLYDSL_COMPILE_OPT_LEVEL`,
  `FLYDSL_DEBUG_ENABLE_DEBUG_INFO`, `LLVM_GFX1250_ANY_XNACK_CODEGEN`, …; the
  list is read from FlyDSL). A variable that kernel code reads, an
  `AITER_FLYDSL_*` one included, is not here: the cache of that kernel
  declares it (level 2), so setting it makes that cache miss and no other.

**Level 2: the sources a cache declares.** aiter sources are not hashed as a
directory, and nothing is detected at run time. Each cache declares the files
its launchers are built from, as FlashInfer's `gen_jit_spec(name, sources)`
does. The declarations of every cache are in one table,
`aiter/aot/flydsl/sources.py`, by cache name; a launcher only names its cache
(`@aot_factory("gemm_a8w8_blockscale")`):

```python
AOT_CACHES = {
    "gemm_a8w8_blockscale": {
        "sources": [
            "gemm_tune/flydsl_gemm_a8w8_blockscale_common.py",
            "kernels/gemm_a8w8_blockscale_8wave.py",
        ],
    },
    "moe_gfx942_gemm": {
        "sources": ["kernels/moe_gemm_2stage/gfx9_fp8"],
        "env": ["AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED", ...],
    },
}
```

So a list is not part of a kernel file: changing one does not touch the
kernel, and the lists of all caches read as one page, with the lists that
several caches share written once. (`sources=` and `env=` of `aot_factory`
and `FlyDSLAotCache` remain for a cache outside that table, a test's say; a
name that is in the table must not pass them.)

- A path is relative to `aiter/ops/flydsl`, or absolute. A directory stands
  for every `.py` below it.
- The file that creates the cache is always one of its sources.
- So are the common sources, `COMMON_SOURCES` of that module, which no
  cache lists: the helpers nearly every kernel is built with
  (`kernels/kernels_common.py`, `kernels/tensor_shim.py`,
  `kernels/buffer_ops.py`) and the per-arch hardware constants kernels size
  themselves by (`aiter/jit/utils/chip_info.py`, `lds_capacity.py`), which are
  outside `aiter/ops/flydsl/kernels`, the only place the import check looks. An edit
  to one of them makes new entries in every cache.
- `"env"` names the environment variables the declared files read,
  `AITER_FLYDSL_*` ones included. The variables the common sources read
  (`COMMON_ENV` there, the kernel-argument preload switches) are
  inputs of every cache. A prebuild, and the unit test, raise for a variable
  a declared file reads by a literal name that is neither declared, nor in
  the fingerprint, nor listed in `ENV_NOT_CODE` there (the target
  selectors, `COMPILE_ONLY`, the cache's own switches). A name that is
  computed (`os.environ.get(f"X_{stage}")`) is not seen. `"get_gfx()"` stands for
  `chip_info.get_gfx()`, for code that picks instructions by it.
- Bitcode a kernel links with `link_extern(..., bitcode_path=...)` is a
  source too, by its absolute path.

The sha256 of each file and the value of each variable go into the digest
that names an entry. So an edit to a declared file, or another value of a
declared variable, makes new entries for that cache only, when each is next
used; every other cache still hits. A file inside the aiter package is named
relative to it and one outside by its file name, so entries built in a source
tree still match once the package is installed elsewhere (a prebuilt wheel).

A file that is not declared invalidates nothing. Two checks find a forgotten
one, both when an entry is built, so a hit pays nothing:

- **A kernel file.** The declared files under `kernels/` are parsed, and
  every file under `kernels/` they import (anywhere in the file, including
  inside a function) must be declared, `tensor_shim.py` included (its
  `ptr_buf_tensor` and friends emit code). The cache's own modules
  (`aot_cache.py`, `aot_launch.py`) are exempt.
- **Bitcode.** The files the compiled kernel links must be declared.

Either one raises `_UndeclaredSourceError`, in every mode; the launch does
not fall back to the JIT path. `tests/test_flydsl_aot_launch.py` runs the
first check over every cache in the tree, so CI reports a forgotten file
without building anything. The checks do not follow imports of a wrapper
outside `kernels/` (a wrapper imports other families' wrappers), a dynamic
import, or code outside the aiter package: declare those by hand.

Entries are stored per version of sources and variables,
`<digest>-<variant hash>.{o,json}`. Switching a file or a variable back (e.g.
`git checkout` of another branch and back) finds the earlier build again
instead of rebuilding it. Old variants are never
read once their sources change; delete them, or old fingerprint directories,
to reclaim space.

The declared sources are hashed once per process, on a cache's first use.
A launch only looks entries up by that digest. A prebuild compares the
contents again before and after each build, since a factory may import its
kernel module only when it builds: a source that changed since it was hashed
is found then, and that build is not stored. A declared source that cannot
be read keeps that cache on the JIT path.

A cache name is unique: creating a cache of one name from two files raises,
since the two would share a directory.

Known holes, and how they are handled:

| hole | example | mitigation |
|---|---|---|
| A source the checks do not see | a helper the wrapper (outside `kernels/`) imports, `importlib.import_module(name)`, a helper in another installed package | Declare it in `sources`. `verify` mode detects a stale entry |
| A variable the kernel code reads that is not declared | `os.environ.get("MY_SWITCH")` in a kernel file | Declare it in `env`, or pass its value in `key`. `verify` mode detects it |
| The trace reads state that is neither env nor source | a global set at runtime, a config file, a device property other than arch and CU count | Put it in `key`. `verify` mode detects it |
| An incomplete hand-written `key` | a value the factory closes over but does not take as an argument | The disk digest still covers the argument types. `verify` mode re-exports and compares |
| A `Constexpr` launch argument missing from `specialization_key` | `launch(..., tile: Constexpr[int])` | Stored per entry and compared on lookup (§6); two values under one key raise |
| Compile hints set by the caller | `with CompilationContext.compile_hints(...)` around the launch | Hashed into the disk digest. In memory a key keeps the function of its first launch in the process, as on the JIT path, so one key must launch under one set of hints |
| A partially written or corrupt entry | crash during a build | `.o` is published last with `os.replace`. A launch that cannot load an entry logs an error and takes the JIT path; it never deletes or rebuilds one. A prebuild rebuilds an object that does not start like an ELF file |

Declared sources give per-family granularity with an explicit list; the
checks above and `verify` mode cover what a list can miss.

Modes, set with `AITER_FLYDSL_AOT_CACHE`:

- `1` (default): on. An edit to a kernel file makes the entries of the caches
  that declare it miss, until a prebuild adds them.
- `0`: off. The op uses the existing `_run_compiled` path.
- `verify`: like `1`, but on every disk hit it re-traces, re-exports, and
  byte-compares the `.o`. A mismatch raises and names the stale entry.
  `export_to_c` is deterministic, so a mismatch always means the key or the
  fingerprint missed an input. Use it in CI and after FlyDSL bumps.

The cache root is `AITER_FLYDSL_AOT_CACHE_DIR`, or by default `flydsl_aot`
in aiter's JIT directory (`get_user_jit_dir()`: `AITER_JIT_DIR`, else
`aiter/jit/` when writable, else `~/.aiter/jit/`). Entries a
`PREBUILD_KERNELS` build writes therefore stay with the package tree, as its
JIT modules do, and a container image that copies the tree carries them.

### 4.3 Launching

A hit is the `CompiledFunction` `flyc.load_module` returns. It fills the ABI
slots from the arguments with FlyDSL's own `CallState` and calls the entry;
a nonzero status raises. Pointer arguments (`ptr_arg`, a `PointerJitArg`) are
compiled with a null pointer of the same element type, address space and
alignment: a non-null pointer counts as a live device argument and
`flyc.compile` would launch on it.

**Every bound launch goes through `launch`.** Mode `0` binds nothing, so
`_run_compiled` takes its JIT path unchanged. Keys that cannot be exported
or loaded (an argument type `export_to_c` does not support) use
`_JitKernel`, which calls `_run_compiled(exe, *args,
specialization_key=key)`. Every launch sent to the JIT path, whatever the reason (no entry, a failed load or export, a key the cache cannot store, a launcher it cannot bind, a factory that was reset), logs one error per key with `logger.error`; nothing raises and the launch runs. Caller bugs
(`_UsageError`: an unsupported element in a key passed to `launch` directly;
a bound launcher with such a `specialization_key` takes the JIT path, §6) and
`verify` failures
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
lost 7 of 49 kernels. The export runs first. When the
export just built the entry, its own compile already showed the replay the
request and filled FlyDSL's cache (the same artifact), so the JIT launch is
skipped. Otherwise it would run and also fill FlyDSL's own cache for mode
`0` and the fallback, like before this cache.

That launch traces the kernel, because FlyDSL derives its key from the
trace, only to find it compiled, and a prebuild requests each kernel many
times: the full `gfx942;gfx950` build makes about 90,000 requests for 17,000
entries, and tracing was 60% of its CPU time. So a process makes the launch
once for an entry it did not build, and answers the entry's later requests
without it:

- the launch derives FlyDSL's key of the kernel the request traces to, and
  the entry's sidecar records the key of the kernel it was built from
  (`"ir"`). Where the two are the same (`_check_launch`), the entry is that
  kernel and FlyDSL's own cache holds it, and the process has confirmed the
  entry. An entry the process built itself is confirmed by its build;
- a later request of a confirmed entry makes no launch. The cache hands the
  recorded key to whoever collects requests
  (`aot_cache.entry_requests(report)`, which `requested_kernels()` enters),
  so the bisection sees the same requests. Collect a replay's requests
  through `requested_kernels()`: watching FlyDSL's launches alone misses
  these, which is the loss described above.

Where the two keys differ, the entry's key does not decide its kernel: an
error says so once, and every request of the entry is launched as before.
So is every request of an entry without a recorded key, with a FlyDSL that
keeps no last compile to read (`JitFunction._last_compiled`, private to it),
inside another launcher's trace, which FlyDSL inlines, while FlyDSL dumps
what it compiles, and under the `COMPILE_ONLY` of a preload (below). A
replay that only identifies kernels confirms nothing, since its launches end
at their key. Under `AITER_FLYDSL_AOT_CACHE=verify` a prebuild leaves no
launch out and raises on a difference.

This adds no key: a request is answered from an entry by the key a serving
process loads it by, after a launch showed that the entry is what the request
traces to. Since each process launches each entry it uses once, `--check`
still looks every kernel up in FlyDSL's cache.

Cost, gfx950 mxfp8 and a4w4 CSVs above, run one after the other on fresh
directories: 155 s at HEAD, 164 s with the cache (62 GPU binaries, the same
set), measured before any launch was left out.

The full `gfx942;gfx950` build (`run_aot`'s stages, 96 workers, fresh
directories, MI308X host): FlyDSL traces 90,500 launches before and 23,500
after, for the same 16,103 kernels in its cache and the same entries in this
one. The MoE stage, whose workers also take the rows of one model together
(`aot/flydsl/moe.py`), goes from 677 s and 737 CPU-min to 180 s and 200
CPU-min. All stages took 35.2 min before; after, 19.3 min on a host that
other builds were loading at the time (MegaMoE, which is LLVM time for 7170
kernels, one set per rank, and gains little, took 614 s there against 486 s
before). No process of that build found an entry built from other IR than
its request traces to. `--check` of the MoE tables on the built caches
passes, and fails with one of FlyDSL's artifacts emptied.

A preload (`tensor_shim._preload_compiled`) also turns `COMPILE_ONLY` on for
the process, to warm one launcher up in a serving or tuning process. That is
not a prebuild: launches of other threads meanwhile build nothing. (They do
not run either, as on the JIT path: `COMPILE_ONLY` is read for the whole
process. Preload before serving, not beside it.)

A preload of a bound launcher (a `factory.lazy(...)` handle or a `bind_aot`
launcher) goes through the cache instead. In a prebuild it builds the entry,
as a launch does. In another process it loads the entry without launching
(`aot_launch.preload_bound`, `FlyDSLAotCache.preload`) and returns the loaded
function, which takes the launcher's arguments; later `_run_compiled`
launches of the key use it too. A key with no entry logs the usual error and
is preloaded on the JIT path, as before. A caller that keeps the function
`flyc.compile` returned and calls it directly (the MegaMoE combine) gets the
same from `tensor_shim._compile_and_keep`, which launches it.

Parallel replays share both caches safely. FlyDSL's own disk cache takes a
per-artifact file lock, so one process runs LLVM and the others load its
result; the export takes `flock <digest>.lock`. Measured on the mxfp8 CSV,
two replays at once on one directory (40 workers): 58 LLVM compiles and 58
exports for 58 kernels, none twice. A process traces a kernel once, to build
its entry or to confirm it, and again where the launch above is still made.
With `FLYDSL_RUNTIME_ENABLE_CACHE=0` or an
IR dump, FlyDSL skips its cache and lock, and processes compile the same
kernel in parallel; exports stay single. `flock` assumes a local file system.

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
@aot_factory("preshuffle_gemm_a8")
@functools.cache
def _preshuffle_launcher(**compile_kwargs):
    return _get_compile_fn()(**compile_kwargs)

_run_compiled(
    _preshuffle_launcher.lazy(**compile_kwargs),
    *args,
    specialization_key=(x_scale.dtype, w_scale.dtype, XQ.device),
)
```

- All launcher tensors are flat contiguous views. `compile_kwargs`
  (`N`, `K`, tiles, `in_dtype`, `out_dtype`, `split_k`, …) together with the
  two scale dtypes determine the signature.
- `M` and `N` are runtime `Int32`s, so one entry serves every `M`. For a
  tuned CSV that means one entry per unique `(N, K, tile config)`.

MI308X (gfx942), FlyDSL 0.5.0.dev1039 (with `flyc.load_module`),
each mode in a fresh process (an earlier, larger version of
`tests/test_flydsl_aot_cache.py`; the test now runs two fp8 shapes and no
timing loop). "first ms" is the wall time of the first call of that config in
the process.

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
- These numbers are from when a launch built a missing entry: the "aot miss"
  column is the trace plus the export, which a prebuild pays now; a launch
  that misses costs what the JIT column shows. A hit costs about 2-3 ms.
- Kernel time is unchanged. Host time of the whole wrapper is 36.4 µs on a
  hit against 37.7 µs through JIT (m=1, n=1024, k=512, 2000 calls).

Also verified (`AITER_FLYDSL_AOT_CACHE_DIR` fresh each time):

- Two prebuilds started together on one empty cache build each entry once,
  leave no `.tmp-*`, and a third only hits.
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
_run_compiled(
    _compile_gemm.lazy(N, K, kernel_name, XQ.device, Out.dtype),
    ...
)
```

`_compile_gemm` (an `aot_factory`) keeps one launcher per device, so the
device is one of its arguments and so in `key`
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
| `mxscale_preshuffle_gemm`, `_fused`, `mxscale_preshuffle_splitk_reduce` | `mxscale_preshuffle_kernels.flydsl_mxscale_preshuffle_gemm` (also a4w4) | the launcher's constexprs; `(split_k, out_dtype, strided)` for the reduce |
| `gemm_a8w8_mxfp8_8wave` | `gemm_a8w8_mxfp8_8wave` | `compile_mxfp8_8wave_gemm`'s arguments |
| `bmm_a8w8_mxscale_gfx950` | `batched_gemm_a8w8_gfx950` | the launcher's constexprs |

The key is each `aot_factory`'s arguments. Every launch also passes the device
of the output or first tensor as `specialization_key`, for the JIT path
(FlyDSL issue 4 in §7); the disk digest keeps only the device type.

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

### 5.4 gfx942 / gfx950: every GEMM and 2-stage MoE launcher of the AOT replays

The replays in `aiter/aot/flydsl` (`gemm.py`, `moe.py`) name the FlyDSL
launchers a build precompiles. On gfx942 and gfx950 every one of them now
launches through this cache:

| Cache | Launcher | How |
|---|---|---|
| `gemm_a16w16_gfx950` | the default HGEMM, one module-level launcher for every config | `bind_aot`; the `specialization_key` is the config items its constexpr param struct is made from, and the device |
| `gemm_a16w16_band_gfx950`, `gemm_a16w16_skinny_gfx950` | band and skinny HGEMM | `aot_factory` |
| `gemm_a8w8_fp8_8wave`, `gemm_a8w8_int8_4wave` | per-token-per-channel a8w8 | `aot_factory` |
| `mixed_moe_gemm1`, `mixed_moe_gemm2`, `mixed_fhmoe_gemm1`, `mixed_fhmoe_gemm2` | MX 2-stage MoE and FHMoE | `aot_factory` |
| `mxfp4_moe_gemm1` | a4w4 stage 1 | `aot_factory` |
| `mxmoe_gemm2` | MX stage 2 (`mxmoe_dispatcher.get_g2`) | `bind_aot` under `get_g2`'s own key |
| `moe_reduction` | stage-2 reduce | `aot_factory` |

Their call sites are unchanged, except the HGEMM, which launched through its
own `flyc.compile` cache (`run_cached`, removed) and now calls
`_run_compiled`. It passes its tensors as
`flyc.from_dlpack(t).mark_layout_dynamic(...)` arguments; a function loaded
with `flyc.load_module` makes a DLPack view of such an argument in every
fill, unless the argument carries its tensor as `torch_tensor`, which the
HGEMM now sets.

Validated compile-only on gfx942 (FlyDSL 0.5.3) with the full replays, each
run twice on one cache directory, recording every `flyc.compile`:

| Replay | Entries | Second run builds | Launchers still compiled by the JIT path |
|---|--:|--:|--:|
| `GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.gemm` | 5240 | 0 | 0 |
| `GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.moe` | 2882 | 0 | 0 |
| `GPU_ARCHS=gfx942 python -m aiter.aot.flydsl.gemm` | 80 | 0 | 0 |
| `GPU_ARCHS=gfx942 python -m aiter.aot.flydsl.moe` | 132 | 0 | 0 |

No launch fell back to the JIT path. With `AITER_FLYDSL_AOT_CACHE=0` the
same replays leave the same FlyDSL artifacts, byte for byte (4874, 2747, 80
and 100 `.pkl` files), so the cache changes no compiled kernel. Runtime on
gfx950 hardware is still to be checked.

Host time of one launch, MI308X, an empty kernel, the launcher's arguments
built on every call:

| Launch arguments | as passed by | JIT path | this cache |
|---|---|--:|--:|
| 6 typed pointers, 2 ints | mxscale, bmm, the gfx942 MoE | 20.5 us | 20.3 us |
| 6 raw addresses, 2 ints, 2 floats | the a16wmix and MX MoE GEMMs | 10.3 us | 11.3 us |
| 5 flat tensors, 2 ints | preshuffle, blockscale, 8-wave | 12.1 us | 14.2 us |
| 3 dynamic-layout tensors with `torch_tensor`, a constexpr struct (arguments prebuilt) | the gfx950 HGEMM | 11.9 us | 12.5 us |
| the same without `torch_tensor` | | 11.8 us | 30.8 us |

The HGEMM row is a launcher of the HGEMM's argument shape run on gfx942; it
gives the same results on both paths for row- and column-major tensors.

### 5.5 gfx942 / gfx950: the planner, GDN, conv and FP8 attention replays

The other replays a build runs (`aiter/aot/flydsl/common.py`, `AOT_MODULES`)
precompile these launchers, which now launch through this cache on gfx942
and gfx950:

| Replay | Arch | Cache | Launcher | How |
|---|---|---|---|---|
| `pa_decode_plan` | gfx942, gfx950 | `pa_decode_plan` | `kernels/pa_decode_plan.py:compile_pa_decode_plan` | `aot_factory`, `plan_pa_decode` launches `.lazy(...)` |
| `chunk_gdn_h` | gfx942 | `chunk_gated_delta_h` | `linear_attention_prefill_kernels._get_or_compile_opt` | `aot_factory` + `.lazy`; the flags the caller passes are coerced with `bool()` |
| `gdn_fused_prefill` | gfx950 | `gdn_fused_prefill` | `kernels/gdr_prefill/gdn_fused_prefill.py:compile_gdn_fused_prefill` | `aot_factory` + `.lazy` |
| `conv` | gfx950 | `conv3d_implicit_gfx950` | `kernels/conv/conv3d_implicit_gfx950.py:compile_conv3d_implicit` | `bind_aot`, keyed by the fields of its `Conv3dImplicitParam`; a variable-resolution kernel leaves `d`/`h`/`w` out and keys the constants it closes over (`_dyn_hw_closure_key`); rerouted |
| `conv` | gfx950 | `conv3d_transpose_ncdhw_ndhwc` | `kernels/conv/conv3d_transpose.py:compile_transpose_ncdhw_ndhwc` | `bind_aot` under `(n, c, BIG)`: `s` is a launch argument, and the kernel bakes only whether `n*c*s` needs 64-bit offsets; rerouted |
| `fmha_fp8` | gfx950 | `flash_attn_fp8_gfx950` | `kernels/flash_attn_func_fp8_gfx950.py:_build_fp8` | `bind_aot` of the builder's `@flyc.jit` launcher (now `_launch.launcher`) under `_build_fp8`'s arguments |

Each cache declares the files its builder runs as sources (the conv plans in
`conv3d_im2col.py`, the FMHA traits in `fmha_gfx950/`; §4.2). The conv wrapper
launched through its own cache (`exe.compile`, then the `CompiledFunction`
kept on the builder's result); `conv_kernels._dispatch` now launches the
`@flyc.jit` launcher through the cache the first time
(`tensor_shim._compile_and_keep`, under the same, empty, `CONV_COMPILE_HINTS`)
and calls the function it got directly afterwards.
`op_tests/test_flydsl_pa_decode.py` spies on `compile_pa_decode_plan`; its
spy now wraps `.lazy`.

No key needed a value moved to the caller: every value these factories read
is one of their arguments (tuned BV, tile, split count, the gfx942 schedule
flag), the arch, or a variable under `aiter/ops/flydsl` read by name.
`AITER_CONV3D_KW_REUSE` and `AITER_CONV3D_DYN_HW` are read by the conv
wrapper and reach the kernel as the `kw_reuse` and `dyn_hw` fields of its
key, so no cache declares them in `env`.

Validated compile-only on gfx942 (MI308X, FlyDSL 0.5.3), each replay run
twice on one cache directory, then under `verify`, recording every
`flyc.compile`; then again under `AITER_FLYDSL_AOT_CACHE=0` with its own
FlyDSL cache:

| Replay | Entries | Second run builds | JIT-path compiles | `verify` | FlyDSL `.pkl`, mode 0 = mode 1 |
|---|--:|--:|--:|---|--:|
| `GPU_ARCHS=gfx942 python -m aiter.aot.flydsl.pa_decode_plan` | 96 | 0 | 0 | passes | 96/96 |
| `GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.pa_decode_plan` | 96 | 0 | 0 | passes | 96/96 |
| `GPU_ARCHS=gfx942 python -m aiter.aot.flydsl.chunk_gdn_h` | 198 | 0 | 0 | passes | 198/198 |
| `GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.gdn_fused_prefill` | 88 | 0 | 0 | passes | 88/88 |
| `GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.conv` | 140 + 9 | 0 | 0 | passes | 139/139 |
| `GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.fmha_fp8` | 92 | 0 | 0 | passes | 92/92 |

The conv replay launches 304 convs and 304 transposes. Resolutions of one
variable-resolution kernel share an entry (140 entries for 130 FlyDSL
artifacts),
and so do the spatial sizes of one transpose (9 entries, 9 artifacts). Key
completeness was checked by re-exporting the launcher on every launch of
each replay above and comparing objects per disk digest: no digest of any
of the six caches produced two different objects.

Also checked on these caches: an edited comment in a kernel file rebuilds
that cache's entries only, and reverting it finds the first variants again;
two replays started together on an empty directory build each entry once,
leave no `.tmp-*` directory, and a third builds nothing. On gfx942,
`op_tests/test_flydsl_pa_decode.py`, `op_tests/test_flydsl_linear_attention_prefill.py`
and `op_tests/test_gated_delta_rule.py` give the same results under modes
`0`, `1` (twice) and `verify` (except `test_chunk_opt_vk_indice[...-hip]`,
the HIP backend, which fails intermittently in every mode), and
`tests/test_flydsl_aot_launch.py` compares the outputs of the planner and
the GDN hidden-state kernel bitwise across mode `0`, a miss and a hit. The
gfx950-only kernels are validated compile-only; their runtime on gfx950 is
still to be checked.

Host time of one launch on MI308X, minimum of five fresh-process runs, the
arguments built on every call (the gfx950 rows are empty kernels of the
same argument shape run on gfx942):

| Launch arguments | Launcher | JIT path | this cache |
|---|---|--:|--:|
| 3 typed pointers, 4 ints, torch stream | `pa_decode_plan` (real) | 16.9 us | 17.2 us |
| 12 tensors (1-D to 5-D), 5 ints, torch stream | `chunk_gated_delta_h` (real) | 22.9 us | 25.1 us |
| 12 flat tensors, 7 ints, torch stream | `gdn_fused_prefill` | 14.3 us | 17.9 us |
| 11 flat tensors, 6 ints, a float, `fx.Stream` | `flash_attn_fp8_gfx950` | 14.1 us | 17.0 us |
| 4 tensors (5-D, 5-D, 2-D, 1-D), 14 `Int64`, through `_dispatch` | variable-resolution conv3d | 13.9 us | 14.5 us |
| 4 tensors, through `_dispatch` | fixed-resolution conv3d | 12.1 us | 12.7 us |
| 2 5-D tensors, an `Int32`, through `_dispatch` | conv transpose | 10.6 us | 10.9 us |

A loaded object fills each tensor slot with FlyDSL's generic fill (§7, issue
3), about 0.25 us per tensor more than the JIT path's generated fills, so
the launchers with 11 or 12 tensors pay 2 to 3.5 us; they run for tens of
microseconds or more. The conv `_dispatch` keeps the function its first
launch got (`tensor_shim._compile_and_keep`: the entry the cache loaded, or
the JIT's when there is none) and calls it directly, as it did with its own
`CompiledFunction` before; the conv rows above predate that. With an empty
four-tensor kernel a steady-state `_dispatch` takes about 11 us with or
without an entry.

### 5.6 gfx950: intranode MegaMoE

`python -m aiter.aot.flydsl.mega_moe` builds a `MegaMoEV2` per rank of each
profile under `dry_run` and calls its `preload_aot_bundles()` and
`quantize()`. Every launcher it compiles now launches and preloads through
this cache:

| Cache | Launcher | How |
|---|---|---|
| `mega_moe_prepare` | `mega_moe_prepare.compile_mega_moe_prepare` (also the fast EP8 prepare it returns) | `aot_factory` |
| `mega_moe_stage1_bundle` | `mega_moe_stage1.compile_mega_moe_stage1_bundle` | `aot_factory` on `_stage1_bundle_launcher`, which takes the `Stage1Config` variants as tuples of their fields: a dataclass is no key element |
| `mega_moe_small_m_stage1`, `mega_moe_large_m_stage1` | `compile_mega_moe_stage1` of `mega_moe_small_m.py` and `mega_moe_large_m.py` | `aot_factory` |
| `mega_moe_stage2`, `mega_moe_stage2_aligned_pair` | `compile_mega_moe_stage2`, `compile_mega_moe_stage2_aligned_pair` (were cached in module dicts) | `aot_factory` |
| `mega_moe_mx_quant` | `quant.build_per_1x32_mx_quant_module` (`per_1x32_mx_quant`) | `aot_factory` |
| `mega_moe_combine` | `make_combine_jit` of a combine after the fused GEMM2 (`combine_no_stage1`) | `aot_factory`; the op keeps the function it gets (`_compile_and_keep`, `_preload_compiled`) |

The runtime keeps the `lazy` handles where it kept launchers, and
`preload_aot_bundles()` preloads through the cache (§4.4): in a serving
process it loads every entry without launching. The dispatch and the
regular `combine()` of the intranode op, which `aiter.dist` uses beyond
MegaMoE and no replay builds, keep their JIT path; so do the fixed-slot
Stage1 (`mega_moe_stage1.compile_mega_moe_stage1`) and `compile_gemm1`,
which the replay does not compile. The combine's launch geometry comes from a
tuning CSV and the device name; both reach the kernel only as `block_num`
and `warp_num_per_block`, which are in its key. `per_1x32_mx_quant` is also
the internode MegaMoE's quant, which so launches through `mega_moe_mx_quant`
too; an `n` the intranode replay does not build logs one error and traces.

Host cost of the lookups that changed, per launch (CPU only, measured
before/after): Stage2 2.7 to 1.6 us, Stage1 bundle 1.7 to 2.5 us (the
variants are turned into tuples, cached), prepare 0.32 to 0.36 us, quant
0.11 to 0.18 us; `_run_compiled` of a kept handle adds 0.2 us to the JIT
path's, before the slot fills of §5.4. The combine keeps calling the
function it kept.

Validated compile-only on gfx942 (FlyDSL 0.5.3, no MORI) with `GPU_ARCHS=gfx950
python -m aiter.aot.flydsl.mega_moe --experts-per-rank 48 --mtpr 8192` (16
jobs), recording every `flyc.compile`:

| | Result |
|---|---|
| Entries after the first run | 674: prepare 352, small-input Stage1 112, Stage2 96, combine 80, Stage1 bundle 16, large-input Stage1 8, aligned-pair Stage2 8, quant 2 |
| Second run on the same directory | 0 builds, no `JIT path` error; 58 s against 167 s |
| `flyc.compile` calls outside the cache | 0 of 1176 |
| `verify` replay on that directory | passes: every entry re-traced to the IR it was built from |
| Key probe (each of the 1176 launches re-exported) | 674 digests, none with two objects |

The default build (`python -m aiter.aot.flydsl.mega_moe`, every production
profile, 234 jobs on 179 workers) writes 7655 entries, 634 MB, in 444 s; a
second run on the same directory builds none in 148 s, and no launch of
either takes the JIT path.

With `AITER_FLYDSL_AOT_CACHE=0` and `=1` the replay leaves the same 594 +
80 FlyDSL artifacts (the same traced-IR names; the 80 combine ones byte for
byte). The GPU binary in 34 to 43 of the 594 differs between any two runs,
two `=0` runs as much as `=0` and `=1`, with identical input IR: LLVM does not
make the same bytes of some prepare kernels twice, so compare these
artifacts by name and IR, not bytes. Runtime on gfx950 with MORI is still to
be checked.

## 6. Adopting it for another op

Opt in at the launcher factory, not at the call site:

```python
@aot_factory("my_kernel")   # declared in aiter/aot/flydsl/sources.py
@functools.cache
def my_launcher(N, K, tile_m, ...):  # its arguments are the key
    return compile_my_kernel(N, K, tile_m, ...)

_run_compiled(my_launcher.lazy(N, K, tile_m, ...), *args, stream)
```

A launcher built inside a wrapper (no factory returning it) binds where it is
built: `bind_aot(launch, FlyDSLAotCache("my_kernel"), (TOPK, N))`.

A factory argument the cache cannot key (e.g. a numpy scalar) leaves that
launcher on the JIT path, and so does such a `specialization_key` (an Enum),
for that key. `factory.cache_clear()` drops the launchers the cache loaded and
keeps the factory on the JIT path for the rest of the process: the caller
wants a new trace, possibly of state the cache does not track (a patched
module global), which an entry on disk would not reflect.

0. **Launch `factory.lazy(...)` on hot paths.** It does not call the factory,
   so a hit pays neither the compiler import nor the `@flyc.kernel`
   definitions (a few hundred ms of AST rewriting per kernel module in a fresh
   process). `_run_compiled(factory(...), ...)` works too, and is what an
   unchanged call site does; it calls the factory once per process and key.
1. **Launch arguments are runtime only.** Every launcher parameter that
   varies per call is a tensor, an `fx` numeric (`Int32`, …) or an
   `fx.Stream` (a torch stream as the last argument is converted). Constexprs
   go to the factory and so into the key. A launcher that keeps a
   `Constexpr[T]` launch argument must pass it in `specialization_key`, as on
   the JIT path. The object ignores such an argument, so each entry stores
   the values it was built with in its sidecar (`baked`) and serves only calls
   that pass them: a prebuild with another value builds its own variant.
   Within a process, one key launched with two values raises, where the JIT
   path would silently run the first. This holds for ints, floats, bools,
   strings, `None` and tuples of those, and so does a launch with more
   arguments (it may pass a constexpr the first launch left to its default)
   or too few for the constexprs the entry was built with.
   A constexpr value the cache cannot compare (a lambda, a `Type[T]`) leaves
   the key on the JIT path, which keeps one compiled function per key and
   does not check; under `verify` it raises, as a failed export or load does.
   (A factory argument or `specialization_key` the cache cannot key takes
   the JIT path under `verify` too.)
2. **The key covers the specialization.** The factory's arguments must
   determine its launcher, as `functools.cache` on it already requires. Pass
   anything else the compiled function depends on, e.g. a tensor dtype that
   can vary for one launcher, as `_run_compiled`'s `specialization_key`, as
   on the JIT path. Arch and CU count are in the fingerprint; declare any other
   variable the kernel's files read in `env`. Tensor rank and unit-stride axis are covered by the
   disk digest.
3. **Declare the sources.** Add the cache to `AOT_CACHES` in
   `aiter/aot/flydsl/sources.py` and list under `"sources"` every file the launcher's
   code is generated from, besides the common sources (§ level 2): the
   kernel module, the helpers it imports, a
   tuning table the wrapper reads its tiles from, linked bitcode (§4.2). The
   first build, and `tests/test_flydsl_aot_launch.py`, raise for a kernel
   file or bitcode that is missing.
4. **Keep the wrapper branch-free.** `_run_compiled` handles mode `0`, the
   fallback and compile-only.
5. **Test it.** CI runs every op test under `AITER_FLYDSL_AOT_CACHE=verify`
   (`.github/scripts/aiter_test.sh`): an entry a test loads from disk, built by
   the wheel's prebuild, is re-traced and compared. `verify` checks
   disk hits only, and a launch never builds, so locally prebuild first (run
   the op's replay, or its test once under `COMPILE_ONLY=1`, on a cache
   directory), then run the test under `verify` on that directory, and add a
   bitwise check against the JIT path.

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
- The prebuild replays tuned CSV rows only. A shape with no tuned row, such
  as a batched mxfp8 GEMM that falls back to its heuristic config, has no
  entry: it traces on the JIT path in every process, with one error per key;
  tuning the shape adds it to the prebuild.
- No replay prebuilds the gfx942 whole-MoE launchers (`moe_gfx942_gemm`,
  `moe_gfx942_sorted_sum`, `moe_gfx942_absmax`, `moe_gfx942_quant_per_tensor`):
  `aiter.aot.flydsl.moe` builds only the 2-stage ones for gfx942. Until one
  does, they take the JIT path.
- Only the launchers in §5 are converted: those of the AOT replays on gfx942
  and gfx950 (GEMMs, 2-stage MoE, the paged-attention planner, the GDN
  prefill kernels, conv3d, the gfx950 FP8 attention and the gfx950
  intranode MegaMoE). Other FlyDSL kernels, e.g. the gfx1250 GEMMs and MoE,
  the gfx1250, TP and internode MegaMoE, the other attention kernels and
  the a8w4 `comm_fused_moe`, still trace and compile on their first call in
  each process. (`moe_reduction` is not
  specific to an arch, so a gfx1250 MoE that calls it launches it through the
  cache, and so does `plan_pa_decode` on gfx1250; neither is validated.)
- A full gfx950 prebuild (`GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.gemm`,
  24136 rows) writes about 5240 entries, 180 MB, in 15 to 20 minutes. Rows
  tuned for another CU count (e.g. the MiniMax-M3 rows for 128 CUs) go to
  that CU count's fingerprint directory, which a GPU with another count
  does not read.
- `.lock` files, one per digest, are not removed.
