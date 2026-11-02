# Direct current-token KV consumption

Based on the preserved ../attention-aiter implementation. This is a C1 Flash
QKV-through-attention region experiment, not an ATOM serving integration.

Keep projection WMMA, all tiled weight TDM, early QB TDM, non-speculative L2
prefetch, fixed-order QA split reduction, and split-KV last-arrival reduction.

## Ownership and readiness

1. QA/KV producers and reducers are unchanged; barrier A publishes the QA latent.
2. QB and per-head norm/RoPE are unchanged; barrier B publishes Q and KVOut.
3. CTA 0 writes KVOut to the persistent cache. There is no barrier C for direct
   mode when pages > 1. Other CTAs do not read the cache row being written.
4. Attention uses TDM gathers for history. Any gather index equal to WRITE_SLOT
   (including invalid-index clamping aliases) is redirected to a different,
   valid history row: (WRITE_SLOT+1)%PAGES. The original slot IDs are retained.
5. Only a tile whose original indices contain WRITE_SLOT loads the 512-element
   KVOut vector, scatters only the matching 1 KiB rows into the LDS KV tile, and
   executes a CTA-local barrier. Both QK and PV WMMA consume this updated tile.
   Duplicate current-token indices work; absent current-token indices need no
   overlay. Partial-tail and invalid masks remain unchanged.
6. Split-KV CTAs publish their partial maximum/denominator/output and increment
   per-head-group arrival counts. Last arrival reduces and writes final output.
   This reduction remains inside the measured launch.
7. Kernel completion ensures the sole persistent cache write completes before
   a subsequent same-stream invocation. Overlapping invocations sharing state
   are unsupported, as before.

For a one-row cache, retain barrier C and the original attention path. The vector
control also retains its barrier. `direct=False` selects the original barrier
path for matched A/B; `direct=True` is the default.

At T=128, BH=16, splits=4: 16 attention CTAs, four consume the new token for an
ordinary unique-index window. There are four last-arrival split reducers. With
splits=1, four attention CTAs consume it, and no split reduction is needed.

The local LDS overlay is an implementation cost; removing a grid barrier does
not automatically ensure a net speedup. Report measured second-round A/B values.
