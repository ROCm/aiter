# Transactional JIT cache

Code generation uses a fixed `{bd_dir}/{module}/blob.staging` tree so Ninja
input paths, unchanged-file mtimes and debug source paths survive a build.
The compiler writes objects under `{module}/build`. After the binary is
atomically installed, JIT snapshots generated inputs into `{module}/blob`.
Source-cache publication is best-effort: a failed copy or a peer winning the
directory swap logs a warning and leaves the installed binary usable.

`build_module` calls `_jit_compile(..., use_versioner=False)` because it installs
a fixed target name such as `module_deepgemm_opus.so`. Each invocation reaches
Ninja's incremental dependency checks instead of the Python extension versioner.
This avoids compiling a `_v1.so` but installing an older unversioned `.so`, and
prevents the versioner's unchanged-input shortcut from suppressing a retry after
compiler failure, a header-only change, or rebuilding a missing output. Ninja
can still skip compilation when its dependency graph is up to date.

The build carries the token returned by its own codegen through compilation.
It checks the token before and after compilation, and again after copying the
binary into its installation tempfile, before replacing the installed artifact.
A changed/incomplete generation at these checkpoints fails the build without
installing that artifact. Source-cache publication also requires the original
token; it cannot publish a later generation on behalf of the earlier build.
These checks detect overlapping codegen; they do not replace the module lock
or protect against external edits that bypass the generation markers.

## Opus compiled-kid metadata

The tuner reads `{bd_dir}/compiled_kids_opus.json`. Normal runtime dispatch
reads CSV/C++ lookup tables, not this sidecar. A rebuild copies the canonical
sidecar into staging; codegen unions it with CSV kids, heuristic defaults and
the tuner's `--extra_kids` request, then applies validity/architecture filters.
Requests are passed on the command line, never written over the successful
sidecar before compiling. An explicit extra kid must remain in the final compile
set: unknown, off-target-architecture or family-filtered requests fail codegen
before compilation rather than silently returning a binary without that kid.
Filtering historical sidecar or CSV entries for the current target remains
supported. A direct generator invocation without `--compiled_kids_sidecar` still
defaults to `{working_path}/compiled_kids.json`; JIT supplies the staged Opus path.

Before compiling, JIT snapshots the generated sidecar in memory and checks its
generation token. After installing `module_deepgemm_opus.so`, JIT atomically
writes that snapshot back to `{bd_dir}` and publishes an adjacent `.receipt`.
This happens independently of source-cache publication. The receipt contains a SHA-256 of
the sidecar and the installed binary's device, inode, size and nanosecond
mtime/ctime. The tuner skips a rebuild only when the required kids are present,
both fingerprints match, and no unfulfilled explicit rebuild is requested.
Missing/legacy metadata, a replaced or copied
binary, and an interrupted metadata publication conservatively cause a
rebuild. This receipt is a local freshness check, not a portable wheel manifest.
The binary fingerprint comes from an open descriptor for the exact inode this
invocation installed, not a later path lookup that could identify a peer's
replacement. Metadata publication refuses an already replaced binary and never
re-reads a potentially changed staging sidecar to certify the earlier compile.

The tuner's request uses `build_after_wait=True`: if a normal runtime builder
holds the module lock, the tuner waits, acquires the lock, and runs its own
`--extra_kids` invocation. A peer finishing an unrelated build is not treated
as completion of that request. This does not loop on metadata-write failures;
after its own successful compile the binary is usable even if publishing the
receipt fails. Other callers retain the existing skip-after-peer-success policy.

The tuner honors an explicit `AITER_REBUILD` once in the parent even if a current
receipt already covers its candidates. After a successful synchronous build or
reuse of a current binary, it sets the environment variable to `0` so spawned
workers do not rebuild again. It restores the parent's in-process flag; its
rebuilt-module list records the completed request. A failed build restores the
original environment value so the request can be retried, including cancellation
by `KeyboardInterrupt` or `SystemExit`.

`AITER_REBUILD=1` removes the module's build directory and installed `.so`;
the canonical sidecar and receipt live one level above the module directory.
The old kids therefore survive `clear_build`, while the missing/replaced
binary makes an old receipt invalid. A failed compile does not advance the
canonical sidecar. A metadata publication failure logs a warning after binary
installation; a later tuner invocation cannot trust the old receipt.
`AITER_REBUILD=2` removes only the installed `.so`, preserving the module's
incremental build tree. These explicit rebuild modes retain the existing
removal contract: they do not promise that the old binary remains available
to unrelated readers during a rebuild. The tuner waits for its own build
before spawning workers; it does not lock out every other process that could
explicitly rebuild the same module later.

## Storage and reclamation

Each module retains one working source tree and one published source snapshot.
Compilation killed by OOM/SIGKILL leaves the same `blob.staging` path for retry,
not another random staging directory on each attempt. For generated source
size S, budget approximately 2S at rest and up to 3S while copying a changed
snapshot, before objects and binaries. Unchanged files can be hardlinked from
the previous published snapshot into its replacement; staging is kept separate
because generators may overwrite files in place.

New publish/backup/reset directory names include hostname and PID. On the next
codegen for that module, artifacts of a dead local owner are immediately
reclaimed. Live owners and remote owners are retained. Old-format artifacts
without owner information use a 24-hour age grace. Cleanup is invoked by a
build, not by a timer; an unused module's artifacts remain until it is built
again or its build tree is explicitly cleared.

If both publication and rollback fail, the last published backup is retained.
The next locked codegen can restore a dead owner's backup or a backup retained
by the same still-running process. If `blob` is missing, legacy `.blob-backup-*`
and `blob.backup.*` snapshots without owner metadata can also be recovered
immediately under that lock; the 24-hour grace applies to deletion, not recovery.
Staging recovery copies the last published
snapshot into a temporary candidate and atomically installs it only after the
copy completes; an interrupted recovery does not expose a partial staging tree
as a completed generation. If restoration remains denied and `blob`
is absent, cleanup keeps backups regardless of age or dead-owner status.
It still leaves live peers' and remote owners' backups alone. This exceptional
retention prioritizes recoverability over reclaiming the last good snapshot.

CPU-only measurements using the PR's CK submodule commit
`af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`, `--receipt 600`, on macOS:

| Generator configuration | Files | File bytes (MiB) | Allocated blocks (MiB) |
| --- | ---: | ---: | ---: |
| `module_mha_fwd`: fwd | 14,465 | 69.83 | 70.86 |
| `module_mha_varlen_fwd`: fwd + fwd_splitkv | 16,506 | 84.88 | 90.02 |
| `module_mha_batch_prefill`: batch_prefill, ndropout filter | 3,073 | 18.98 | 28.66 |
| bwd, CK generator only | 15,337 | 184.56 | 221.08 |
| `libmha_fwd`: fwd + fwd_splitkv + unfiltered batch_prefill | 22,651 | 122.82 | 147.34 |

The two retained source trees across these three forward modules, two Python
backward modules, and two standalone MHA libraries total approximately
1.95 GiB using the measured allocation. The two Python backward modules also
generate HSA headers, which are not included. Neither object files, binaries,
save-temps output, other modules nor abandoned transactions are included.
Allocation depends on the filesystem; use `du` on the target runner for its
actual total. A CI runner needs this additional source budget on top of its
compiler-output budget; the measurement alone does not establish that its
disk quota is sufficient. `AITER_REBUILD=1` clears the module tree but also
discards incremental objects, so it is not a free cache eviction mechanism.

## Permissions

Staging and the publish candidate root inherit the existing `op_dir` mode.
The candidate is chmod'ed immediately after `mkdtemp`, before copying files;
it does not retain `mkdtemp`'s 0700 mode on publication or a failed copy.
Nested directories and files preserve their source modes. With readable
ancestors and normal umask 022, this preserves 0755 directories and 0644
source files for a root-prebuilt tree later copied/read by a non-root user.
Deliberately restrictive parent/source modes are not widened. CPU tests check
0755 and 0750 roots, nested modes, publication failure and retry paths; actual
ROCm builds and root-to-non-root container execution remain integration checks.
