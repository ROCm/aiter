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

Requires a gfx950 (MI355X) device. An LLVM abort kills the process (rc 134), so the two
paths must run in separate processes.

## Expected

| triton | `--fixed` | `--broken` |
|---|---|---|
| 3.8 | compiles and runs | **SIGABRT (rc 134)** in codegen |
| 3.7 | compiles and runs | compiles and runs |

On 3.8 (assertions on) the abort is:

```
GCNRewritePartialRegUses.cpp:384: Assertion `NewLI.verify(MRI)' failed
```

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
