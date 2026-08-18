# gfx950 Gluon unified attention — softmax rewrite decides whether triton 3.8 compiles

A single algebraic rewrite inside the online-softmax update is the difference between a
kernel that compiles and one that aborts the AMDGPU backend on triton 3.8.

`QK_scale` is a positive scalar, so scaling the `[BLOCK_M, TILE_SIZE]` score tile `S` and
scaling the reduced `[BLOCK_M]` max vector are mathematically equivalent:

```python
# --broken : scale the tile up front
S = S * self.QK_scale
m_ij = gl.maximum(M, gl.max(S, axis=1))
m_ij = gl.where(m_ij > float("-inf"), m_ij, 0.0)
p = gl.exp2(S - m_ij[:, None])

# --fixed  : scale the reduced vector instead
m_ij = gl.maximum(M, gl.max(S, axis=1) * self.QK_scale)
m_ij = gl.where(m_ij > float("-inf"), m_ij, 0.0)
p = gl.exp2(S * self.QK_scale - m_ij[:, None])
```

`ua_kernel.py:softmax_part0_cdna4` picks between them on the `UA_SOFTMAX_BROKEN` env var.

## Run

```bash
./run.sh                # both paths, one process each
python repro.py --fixed
python repro.py --broken
```

Requires a gfx950 (MI355X) device and a triton 3.8. An LLVM abort kills the process, so the
two paths must run in separate processes.

## Result

| triton | LLVM | `--fixed` | `--broken` |
|---|---|---|---|
| 3.8.0 `1f0a8cfc` (main, 2026-08-18) | `b010a18d` | compiles and runs | **SIGABRT, rc 134** |
| 3.7.1 `0263a6a6` (ROCm 7.14) | `1f126a6d` | compiles and runs | compiles and runs |

The abort:

```
llvm/lib/Target/AMDGPU/GCNRewritePartialRegUses.cpp:384:
GCNRewritePartialRegUsesImpl::updateLiveIntervals(...):
  Assertion `NewLI.verify(MRI)' failed.
```

**Still present on today's main.** It was first seen on 3.8.0 `71d3f5cf` (2026-07-29, LLVM
`850a2b1b`); triton has since bumped its LLVM pin to `b010a18d` (triton `640190e`, "Pin LLVM
at b010a18d", #11163) and the abort is unchanged, so this is a retest against newer LLVM
rather than the same build twice.

Assertions are enabled in triton's prebuilt LLVM, so the failure is loud. Whether the same
IR miscompiles silently in a no-assertions build has not been checked.

## Config

`repro.py` uses the smallest config known to trigger it. All three are required:

* `sinks` passed — `sinks=None` selects the non-CDNA4 softmax and compiles either way
* a prefill sequence — pure decode compiles
* `head_size` 64 or 128 — 256 compiles

bf16, causal, no sliding window, no descales.

## Files

| file | |
|---|---|
| `ua_kernel.py` | the Gluon kernel, copied from `aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py` with the `UA_SOFTMAX_BROKEN` switch added and the aiter imports inlined |
| `wrapper.py` | minimal launcher, trimmed from aiter's `_gfx950_unified_attention` to the single-split prefill config |
| `repro.py` | driver |
| `run.sh` | runs both paths |
