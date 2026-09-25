# Reproducing the kernel measurements

A pinned Triton build, the kernel benchmarks, and the checks that the dense path
is unchanged.

The end-to-end speedup figures quoted in the PR description were measured on a
real workload: Qwen3-8B on RULER dataset and the calibration set λ=0.02878.

Requires an MI300X or MI355X, Docker, and this branch checked out.

**Run every command below from the aiter root**, the directory containing
`aiter/` and `op_tests/`. Step 2 mounts `$PWD` into the container, so starting
anywhere else mounts the wrong tree and every later step fails.

---

# Steps

## 1. Build the image

```bash
bash blasst_repro/docker/build.sh             # ~10 min, Triton built from source
```

## 2. Start a container

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --network=host --ipc=host --shm-size 16G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=1 \
  -v "$PWD":/workspace -w /workspace \
  aiter-blasst/triton-chaindot:rocm7.2.4-py3.12-torch2.10.0 bash
```

`TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=1` activates the patch this image exists
for. Without it the image behaves as stock Triton, the step 7 numbers will not
match, and the P@V dot gets a different warp layout from the dense path (see
Background). Step 8 overrides it per arm on purpose.

Inside, confirm the Triton build:

```bash
cat /opt/triton-src-commit.txt      # pinned commit + "patched: 0001-force-chain-dot..."
python3 -c 'import triton; print(triton.__version__)'   # 3.8.0
```

## 3. Install aiter

```bash
AITER_USE_SYSTEM_TRITON=1 pip install -e . --no-build-isolation
python3 -c 'import triton; print(triton.__version__)'   # must still be 3.8.0
```

If Triton changed, the install replaced the patched build; re-run with
`AITER_USE_SYSTEM_TRITON=1`.

## 4. Dense regression suite

```bash
pytest -q op_tests/triton_tests/attention/test_unified_attention.py -k "not gluon"
```

Expect: **1088 passed, 1056 skipped, 0 failed** (~5 min).

## 5. Block-skip suite

```bash
pytest -v op_tests/triton_tests/attention/test_unified_attention_blasst.py
```

Expect: **16 passed, 3 skipped, 0 failed** (~10 s). The 3 skips are gfx1250-only.

## 6. Codegen unchanged when the feature is off (optional)

Optional. Extract the pre-change files, then diff the generated assembly:

```bash
mkdir -p blasst_repro/upstream_ref
git show 52ffe895e:aiter/ops/triton/_triton_kernels/attention/unified_attention.py \
  > blasst_repro/upstream_ref/kernel.py
git show 52ffe895e:aiter/ops/triton/attention/unified_attention.py \
  > blasst_repro/upstream_ref/wrapper.py
echo 52ffe895e > blasst_repro/upstream_ref/REV

bash blasst_repro/codegen_diff.sh
```

Expect: **`RESULT: emitted CODE is IDENTICAL.`** The only listed differences are
`.amdhsa_kernarg_size 184 → 192` and the argument offsets after it — the two
runtime scalars `log2_threshold` and `num_q_blocks`.

## 7. Random-input sweep

```bash
python3 blasst_repro/bench_blasst_ua2d.py \
  --shapes 32x8 --seqlens 16384 \
  --thresholds 1e-9,0.02878,0.1,0.3,1.0,1.5,2.0,4.0
```

Expect, on MI355X:

| λ | speedup | elide | rel_err |
|---|---|---|---|
| 1e-9 | ~0.90× | 0.0% | 0 |
| 0.02878 | ~0.90× | 0.0% | 0 |
| 0.3 | ~0.91× | 0.0% | 0 |
| 1.0 | ~1.03× | 21.4% | 0.245 |
| 2.0 | ~1.43× | 70.9% | 1.025 |
| 4.0 | ~1.68× | 90.2% | 2.214 |

`rel_err` must be exactly **0** on every row where `elide` is 0.0%.

## 8. Chain-dot patch A/B

```bash
bash blasst_repro/run_ab.sh --shapes 32x8 --seqlens 16384
```

Expect the patch worth **~1.07–1.13×** once elision is active, and dense
unaffected (~1.00×).

---

# Background

## Why a custom Triton build is involved

Block skipping wraps the second dot in a conditional, so a skipped tile never
loads V:

```python
if not (ENABLE_BLOCK_SKIP and all_skip):
    acc = tl.dot(P.to(V.dtype), V, acc=acc)
```

That is the whole optimisation — the bandwidth saving comes from not loading V,
not from skipping the multiply — but it puts `P@V` one `scf.if` deeper than the
`QK` dot. The AMD backend picks warp layout using `isChainDotHead` /
`isChainDotTail`, which require both dots in the **same MLIR region**:

```cpp
auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
};
```

Detection fails, and the flash-attention-tuned `warpsPerCTA=[num_warps, 1]`
degrades to a generic `[2, 2]`.

`patch/0001-force-chain-dot-across-scf-if.patch` adds
`isSameOrAcrossOneIfLevel()`, which also accepts two regions related by exactly
one level of `scf.if` nesting. It is gated behind
`TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF` and added to
`CACHE_INVALIDATING_ENV_VARS`, so the compile cache keys on it.

Nothing about this is specific to block skipping — any Triton kernel with a
conditionally-executed second dot hits it.

Measured worth (step 8): roughly **8%** once skipping is active, and **0.999×**
on dense. The gap is near-constant regardless of how much actually elides,
which is the expected shape: warp layout is a compile-time decision, so the cost
is incurred because the conditional *exists*, not because it is taken.


## The caveat on the "stock" arm

`run_ab.sh` runs both arms in one container, because the patch is inert unless
the env var is set. So the `FORCE=0` arm is the **patched build with the patch
disabled**, not a stock Triton build. `isSameOrAcrossOneIfLevel()` returns false
immediately when the variable is unset, so codegen matches stock — but if you
need a literal stock comparison, build the base image without applying the patch
and run `bench_blasst_ua2d.py` there.

The two arms use separate `TRITON_CACHE_DIR`s. The patch does add the variable
to `CACHE_INVALIDATING_ENV_VARS`, but a shared cache across arms has produced
wrong A/B results before and separating them costs nothing.

## Pinning

The Dockerfile pins triton-lang/triton at
`71d121b0690ad2615f19c85a1c29b08b55a9f80e` and applies the patch on top.
Deliberate: the patch is tied to the commit it applies to, and building against
a moving `main` would make these numbers unreproducible and let the patch
silently fail to apply. `/opt/triton-src-commit.txt` inside the image records
both the commit and the fact that the patch was applied.

## Notes

- `block_skip_threshold=0.0` (the default) leaves the kernel unchanged; step 6
  is the proof.
- λ is model-, layer- and length-dependent and must be calibrated; there is no
  safe default, which is why the feature is off unless asked for.
- `AITER_UA_BLASST_SCHED=0` disables the scheduling while keeping skipping.
- `AITER_UA_BLASST_PRELOAD_V=1` keeps V hoisted instead of deferring it.

## Files

| | |
|---|---|
| `docker/Dockerfile` | pinned Triton + patch on the ROCm PyTorch base |
| `docker/build.sh` | build it |
| `patch/*.patch` | the chain-dot patch, self-contained |
| `bench_blasst_ua2d.py` | the benchmark, random inputs |
| `run_ab.sh` | both chain-dot arms, one container, collated |
| `codegen_diff.sh` | proves the dense path is unchanged |
