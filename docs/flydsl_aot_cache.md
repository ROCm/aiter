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
- `flydsl.__version__`, **the contents of every FlyDSL `.py`**, and FlyDSL's
  native libraries: their sha256 from the wheel's `RECORD`, so reinstalling
  the same wheel (e.g. in a later Docker stage) keeps the fingerprint, or, for
  a source build without a `RECORD`, the name, size and mtime of its
  `.so`/`.a` files;
- `torch.version.hip` and the target arch. That is
  `torch.cuda.get_device_properties().gcnArchName`, or FlyDSL's
  `get_rocm_arch()` when there is no device or `FLYDSL_GPU_ARCH` is set;
- the CU count, `chip_info.get_cu_num()`. It honours `CU_NUM`, which the AOT
  dry run sets from each tuned row's `cu_num`, so a prebuild for a CU count
  and a process on such a GPU use the same directory;
- every `AITER_FLYDSL_*` / `FLYDSL_*` env var, plus
  `HSA_OVERRIDE_GFX_VERSION` and `COMPILE_OPT_LEVEL`. Variables that cannot
  change code are excluded: dump dirs, the cache's own variables, the
  FlyDSL runtime-cache switches (`FLYDSL_RUNTIME_ENABLE_CACHE`, and
  `FLYDSL_RUNTIME_RUN_ONLY`, which the replay's `--check` sets, so it checks
  the entries the build made), the replay's `AITER_FLYDSL_AOT_TIMEOUT` /
  `_MAX_RETRIES` / `_TRACEBACK`, and `FLYDSL_GPU_ARCH`, which is hashed as the arch it selects. Otherwise an AOT
  build, which sets it, and the runtime, which reads the device, would use
  different directories.

**Level 2: per-entry sources.** aiter sources are not hashed as a directory.
During a miss, `make_exe()` and the trace run under `sys.setprofile`, which
records the `co_filename` of every Python function called. Every recorded
`.py` inside the aiter package (except `aot_cache.py` itself) is stored with
its sha256 in the entry's sidecar `.json`, together with any
`extra_sources`. Files inside the aiter package are stored relative to it, so
entries prebuilt in a source tree (`PREBUILD_KERNELS=1 setup.py bdist_wheel`)
still match once the wheel is installed into `site-packages`. A lookup accepts an entry only if all of those files still
hash the same. The per-process digest cache makes this one `sha256` per file
per process.

The profile hook sees calls, not reads of a module-level value: a kernel that
does `from .common import TILE_M` and reads `TILE_M` never calls into
`common.py`. So after the build, the imports in the code of each recorded file
that can feed it (the same spans as the variable scan below) are resolved
statically, and the module a non-function value comes from is recorded too.
The resolution follows re-exports, star imports and namespace packages, and
the names a value is computed from: its statement and the conditions around
it (`BIG = TILE * 2` or `if FLAG: BIG = 2` records where `TILE` or `FLAG` comes
from). A function called at import to compute a value is followed the same
way. `import x` records `x`. A function the kernel imports adds nothing, since
the build records it when it calls it, unless it is memoized; then its body is
followed too. A logger (`x = logging.getLogger(...)`) adds nothing, or
`from aiter import logger` would tie every entry to `aiter/__init__.py`. This
over-records rather than under-records: an imported constant the kernel does
not use costs a rebuild when its file changes. It is static, so a value built
by code it cannot read (`globals()[name] = ...`, `importlib`) is not followed;
an unresolved name records the module it was looked up in.

The sidecar also stores the environment variables the entry depends on, with
their values (`null` if unset), and a lookup requires the same values. They
are found two ways, so no list has to be kept:

- names read by a literal in a recorded kernel file, one under
  `aiter/ops/flydsl/` (`os.environ.get("X")`, `os.getenv("X")`,
  `os.environ["X"]`, `"X" in os.environ`). This covers a module-level read at
  import and a read inside an `lru_cache`d function, which run before or
  outside the build. Only code that can feed the build is scanned: module and
  class-body statements, functions the build called and memoized functions.
  A host-only function the build never called, such as allocator or IPC
  setup reading `PYTORCH_HIP_ALLOC_CONF`, is skipped, so its variable does not
  make prebuilt entries miss. Shared modules are not scanned: `aiter/jit/core.py` alone
  names about 45 variables (`AITER_CONFIG_*`, `AITER_JIT_DIR`, …) and
  `aiter/__init__.py` names `AITER_LOG_LEVEL`, none of which changes a kernel;
- names aiter code reads during the build, seen by the same profile hook
  (`os._Environ.__getitem__` called from a tracked file). This covers a name
  computed at runtime, e.g. `os.environ.get(name)`. Only reads of one name
  count (`[]`, `get`, `getenv`, `in`, `setdefault`). `os.environ.copy()`,
  `dict(os.environ)`, `items()` and the like read every variable; recording
  them would write the whole environment, secrets included, into a sidecar
  that ships with the package, and the entry would hit only on that machine.

Variables the fingerprint owns (`AITER_FLYDSL_*`, `FLYDSL_*`, the extras) are
left to it. So are those that describe the build rather than the code:
`CU_NUM`, `GPU_ARCHS`, `AITER_GPU_ARCHS`, `COMPILE_ONLY`, `AITER_AOT_*`
(`setup.py` sets `AITER_AOT_IMPORT`, the replay `AITER_AOT_REPLAY` and
`AITER_AOT_FEATURE_GUARD`) and `AITER_CONFIG_*`, which the replay points at
the CSVs it replays. The build sets them and the runtime does not; the
fingerprint already hashes the arch and CU count, and the key holds the
config a CSV selects. Only the variables an entry's own
files name are compared, so a switch for another kernel (`AITER_FLY_AR`, an
MoE tuning variable) does not invalidate GEMM entries.

`GPU_ARCHS` is left to the fingerprint, but `chip_info.get_gfx()` returns its
last arch, while the fingerprint hashes the device's. An image setting
`GPU_ARCHS="gfx942;gfx950"` answers `gfx950` on a gfx942, so a kernel choosing
code by `get_gfx()` would build a gfx950 variant into the gfx942 directory.
An entry whose build calls `get_gfx()` therefore stores its value under the
key `"get_gfx()"`, and a lookup compares it like a variable. Its memo is
cleared before each build, so the profile hook sees the call; a value computed
from it before the build, at module level or in a memoized function of a
recorded file, is found by scanning that code. Hashing it into the
fingerprint instead would make every entry miss in such an image, including
those of kernels that never call it. The replay sets `GPU_ARCHS` to the arch
it builds for, so its entries store that arch and a process on that device
hits them. None of the converted
launchers calls `get_gfx()` today.

For the preshuffle GEMM the recorded set is `gemm_kernels.py`, the kernel
modules it builds from (`kernels/preshuffle_gemm.py`, `splitk_epilogue.py`,
`gemm_a16w16_gfx950_utils.py`, `kernels_common.py`, …), `jit/utils/chip_info.py`,
and in the first build of a process the modules it imports (`ops/triton/…`,
an import side effect, harmless). So:

- editing `kernels/act.py`, or any MoE kernel, still hits every GEMM entry;
- editing `kernels/preshuffle_gemm.py` rebuilds only the GEMM entries, and
  only when each is next used.

Entries are stored per version of sources and variables,
`<digest>-<variant hash>.{o,json}`. Switching a file or a variable back (e.g.
`git checkout` of another branch and back) finds the earlier build again
instead of rebuilding it. Old variants are never
read once their sources change; delete them, or old fingerprint directories,
to reclaim space.

A process runs the code it imported, so a source edited after it started
(`git checkout` during a test run) no longer matches what it would trace. A
build that records such a file is not written: that key takes the JIT path
in this process, with a warning, and the next process builds the entry from
the new code.

Recording costs about 80 ms on a miss (≈110 ms → ≈190 ms), once per entry
per machine. A hit pays nothing for it.

Known holes, and how they are handled:

| hole | example | mitigation |
|---|---|---|
| A module-level value reached without an import in build code | `getattr(consts, name)`, `globals()[name] = ...`, or an import inside a function the build did not call | Imports in build code are followed statically (above). Pass the file in `extra_sources`, or run `verify` mode |
| Code outside the aiter package affects the kernel | a helper in another installed package | Pass it in `extra_sources`, or run `verify` mode |
| The trace reads state that is neither env nor source | a global set at runtime, a config file, a device property other than arch and CU count | Put it in `key`. `verify` mode detects it |
| A variable read under a computed name, outside the build | `os.environ.get(name)` at import | Read it in the wrapper and pass it in `key`, or run `verify` mode |
| A variable read outside the build in a shared module | a module-level `os.getenv("X")` in `aiter/jit/` | Name it in kernel code under `aiter/ops/flydsl/`, or pass it in `key`. Today's shared reads are arch, CU count and config paths, all covered |
| A variable read only by copying the environment | `env = dict(os.environ)` then `env["X"]` | Read it by name |
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

## 6. Adopting it for another op

Opt in at the launcher factory, not at the call site:

```python
@aot_factory("my_kernel")            # the cache name
@functools.cache
def my_launcher(N, K, tile_m, ...):  # its arguments are the key
    return compile_my_kernel(N, K, tile_m, ...)

_run_compiled(my_launcher.lazy(N, K, tile_m, ...), *args, stream)
```

A launcher built inside a wrapper (no factory returning it) binds where it is
built: `bind_aot(launch, FlyDSLAotCache("my_kernel"), (TOPK, N))`.

A factory argument the cache cannot key (e.g. a numpy scalar) leaves that
launcher on the JIT path. A module that imports `aot_launch` records it as a
source (§4.2), so editing `aot_launch.py` rebuilds those entries once.

0. **Launch `factory.lazy(...)` on hot paths.** It does not call the factory,
   so a hit pays neither the compiler import nor the `@flyc.kernel`
   definitions (a few hundred ms of AST rewriting per kernel module in a fresh
   process). `_run_compiled(factory(...), ...)` works too, and is what an
   unchanged call site does; it calls the factory once per process and key.
1. **Launch arguments are runtime only.** Every launcher parameter that
   varies per call is a tensor, an `fx` numeric (`Int32`, …) or an
   `fx.Stream` (a torch stream as the last argument is converted). Constexprs
   go to the factory and so into the key.
2. **The key covers the specialization.** The factory's arguments must
   determine its launcher, as `functools.cache` on it already requires. Pass
   anything else the compiled function depends on, e.g. a tensor dtype that
   can vary for one launcher, as `_run_compiled`'s `specialization_key`, as
   on the JIT path. Arch, CU count and the variables the kernel's files read
   are tracked already. Tensor rank and unit-stride axis are covered by the
   disk digest.
3. **Pass extra sources.** If the kernel uses code outside the aiter
   package, pass those paths as `aot_factory(..., extra_sources=)`. Constants
   imported from an aiter module are recorded already.
4. **Keep the wrapper branch-free.** `_run_compiled` handles mode `0`, the
   fallback and compile-only.
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
- The prebuild replays tuned CSV rows only. A shape with no tuned row, such
  as a batched mxfp8 GEMM that falls back to its heuristic config, is built
  at its first call; tuning the shape adds it to the prebuild.
- Only the launchers in §5 are converted. Other FlyDSL kernels, e.g.
  `gemm_a16w16` (hgemm), the a8w4 `comm_fused_moe` and the gfx1250 mxfp8
  GEMM, still trace and compile on their first call in each process.
- A full gfx950 prebuild (`GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.gemm`,
  24136 rows) writes about 3790 entries, 1 GB, in about 15 minutes. Rows
  tuned for another CU count (e.g. the MiniMax-M3 rows for 128 CUs) go to
  that CU count's fingerprint directory, which a GPU with another count
  does not read.
- `.lock` files, one per digest, are not removed.
