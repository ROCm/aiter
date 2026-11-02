# Persisting FlyDSL kernel caches across containers

AITER native-extension artifacts, PyTorch/Triton caches and FlyDSL artifacts are
separate caches. Persisting `/root/.cache` or setting `AITER_JIT_DIR` does **not**
necessarily persist FlyDSL's cache.

In FlyDSL 0.4.1, `flydsl.utils.env.runtime.cache_dir` defaults to
`~/.flydsl/cache`; `FLYDSL_RUNTIME_CACHE_DIR` overrides it. For a root container,
the default is `/root/.flydsl/cache`, not `/root/.cache/flydsl`. Other versions
or images can differ: inspect the installed version and resolved path first.

## Give FlyDSL a dedicated persistent directory

Set the variable before importing FlyDSL, including in spawned workers.
For example, on a host with a `render` group (adapt GPU permissions to your host):

```bash
mkdir -p "$PWD/kernel-cache/flydsl"

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v "$PWD/kernel-cache/flydsl:/var/cache/flydsl" \
  -e FLYDSL_RUNTIME_CACHE_DIR=/var/cache/flydsl \
  rocm-aiter:test /bin/bash
```

If the host has no `render` group, supply the correct device-access group rather
than an empty ID. See [non-root container setup](aiter_container_nonroot_setup.md)
for render/video permissions. The runtime user must be able to create
directories, lock files and atomically replace cache files in the mounted
directory. For a non-root image, provision it for that image's UID/GID; do not
make a shared executable cache world-writable.

Alternatively, retain the default path and mount the runtime user's
`$HOME/.flydsl/cache` explicitly. Container `$HOME` can differ from host `$HOME`.
Mounting only AITER's native JIT directory still leaves FlyDSL artifacts in the
container's disposable filesystem.

## Verify from the running container

With FlyDSL 0.4.1, this check does not launch a kernel:

```bash
python3 - <<'PYCODE'
from importlib.metadata import version
from pathlib import Path
import flydsl
from flydsl.utils import env

print("FlyDSL version:", version("flydsl"))
print("FlyDSL source:", flydsl.__file__)
print("cache directory:", Path(env.runtime.cache_dir).expanduser())
print("disk cache enabled:", env.runtime.enable_cache)
PYCODE
```

Also inspect mounts with
`docker inspect --format '{{json .Mounts}}' CONTAINER_NAME`.

An absolute symlink to a host checkout is not a substitute for mounting its
target inside the container. When testing source overrides, verify `__file__`
for `aiter`, `flydsl` and the model package in the actual worker environment.
Keep `AITER_META_DIR`, if set, consistent with the selected AITER checkout.

## Reuse only compatible, trusted artifacts

FlyDSL caches contain serialized compiled artifacts and executable GPU code.
Use only caches produced by trusted builds/users. Separate deployment caches by
pinned software stack and GPU target/stepping; cache keys do not replace
validation of a new deployment. Source, compiler, compile-option or kernel
specialization changes may legitimately require new artifacts.

Retain the normal file-locking and atomic-write behavior. Do not delete or
replace a cache while workers are using it, and do not copy partially written
artifacts from a live writer. A persistent cache does not guarantee coverage
of every future request shape.

For deployments deliberately using `FLYDSL_RUNTIME_RUN_ONLY=1`, populate and
validate required specializations first. Run-only mode rejects cache misses;
it does not warm missing kernels or make incompatible artifacts usable.

## Separate cold-shape latency from steady-state performance

After starting the model:

1. Measure the first request of each relevant prefill/decode shape.
2. Repeat those shapes in the same process.
3. Restart server processes with identical code/options and retain the cache.
4. Repeat in a **new container** with the cache directory mounted explicitly.
5. Check compilation/cache diagnostics and request timings separately.

A health endpoint returning success does not imply all request-dependent GEMM,
attention-planning and MoE specializations have been compiled. A new shape can
select a different tuned kernel even if another request warmed the model.

Preserving kernel artifacts can remove seconds of first-shape compilation
without improving hot request latency or GPU kernel execution. A persistent
kernel cache also does not retain a server's KV/prefix cache across restarts.
When comparing runs, distinguish kernel compilation from other warmed caches
(e.g. filesystem page cache) and from model loading.

## Troubleshooting

| Symptom | Check |
|---|---|
| Fresh containers rebuild exercised kernels | Resolved FlyDSL cache path and mount target; `/root/.cache` alone need not cover it. |
| Same server compiles at a new shape | Selected kernel/specialization, not only whether the model was warmed once. |
| Same shape recompiles with a mounted cache | Software/source paths, options, GPU target, load warnings and cache permissions. |
| Cache exists but run-only fails | Coverage and compatibility; some externally linked artifacts may not be reusable from disk. |
| Long gaps remain after compilation stops | Host dispatch, rank arrival skew and GPU dependencies; not every gap is JIT. |
