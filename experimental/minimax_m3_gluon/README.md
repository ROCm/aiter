# MiniMax M3 source-only Gluon port

This experimental port starts at the **already reduced attention output** of a
MiniMax M3 sparse MoE layer. The current region has four intended device stages:

1. One Gluon launch adds the residual, computes Gemma RMSNorm, computes BF16
   router logits, selects top-4 of 128 routed experts plus the shared expert,
   writes the selected AITER BM16 metadata, and zeros MoE2's accumulation buffer.
2. AITER's selected BF16×MXFP4 MoE1 computes the SwiGLU-OAI gate/up.
3. AITER's selected atomic MoE2 writes the local BF16 result.
4. One Gluon launch publishes local results to TP4 peer memory, sums ranks in
   rank order, and emits `(ar_out, h_mid)` for the next layer. It reads the tensor
   returned by MoE2 directly.

This is the MoE boundary of a larger port. ATOM's source kernel also includes
sparse index scoring, top-k page selection, sparse attention, output projection,
and its first TP reduction. Those remain upstream in this snapshot; the three
initial dense layers also remain on the established path. There is no serving
dispatch change yet. The 4-launch count is an intended contract, not a measured
device trace: AITER internals could add launches and must be checked later.

## Source and shape contract

Ported from ROCm/ATOM commit `6e9b3e1c5b50b9e6a3b2f8bcf651a9b0229d3ac0`,
`atom/models/minimax_m3/mono`, especially `post_attn.py`, `stages/router.py`,
`stages/moe_wide.py`, `stages/ffn_reduce.py`, `layout.py`, `weights.py`, and
`runner.py`. This snapshot targets MI355X/gfx950 and TP4 with 1–16 physical
decode tokens, with every row live (graph padding is not yet handled). It assumes hidden 6144, 128 routed experts, top-4, one fused
shared expert, 129 packed A16W4 experts, per-group-32 scales, Gemma norm, and
SwiGLU-OAI with clamp limit 7. The shared route weight is read from the model's
AITER top-k metadata rather than assumed to be one.

| Writer → reader | Buffer | Ordering and reuse |
|---|---|---|
| Score CTA → routing CTA | score key, sigmoid; eight ready words/token | GPU-scope release and side-effecting `sc1` epoch poll; coherent score reads; ready allocated once per width |
| Routing CTA → AITER MoE1 | IDs, weights, BM16 metadata | Same stream launch boundary; each route has one sorted destination |
| Zero CTAs → AITER MoE2 | `SortOut`, BF16 `[B,6144]` | Same stream boundary; zeroing finishes before atomic accumulation |
| AITER MoE2 → Gluon tail | Returned output tensor | Same stream boundary; keep tensor alive through capture/replay |
| Rank → all ranks | Two BF16 values + epoch per aligned 64-bit IPC word | System-scope peer publication and side-effecting `sc1` polling; two slots alternate by epoch |
| Tail → next layer | `ar_out` and `h_mid` | Same stream; layer parity/buffer ownership must be established in serving integration |

Front CTA mapping: CTAs `0..8B-1` compute 16 router experts each; CTA 128
waits for all score tiles, chooses routes, and builds metadata; CTAs 129–255
zero 1024 BF16 elements each. The 256-CTA grid must be resident on 256 CUs
for the intended latency and wait schedule; `pipeline.require_residency` rejects
spills or insufficient compiled occupancy. Compilation resource use and actual
occupancy have not been checked. A16W4 MoE weights remain in AITER's shuffled
layout. The metadata buffer allocates `16 * B * 5` rows, the worst case of one
BM16 block per route; top-k uniqueness makes each expert occur at most B times.

`pipeline.use_hybrid` marks B>=5 as an **unmeasured candidate** because ATOM
switches from per-token to expert-grouped resident MoE there. That source
threshold does not establish where a Gluon/AITER hybrid wins. The port must
benchmark each B and retain the established implementation until it passes
numerical and latency checks. ATOM uses direct peer reduction for small B and
reduce-scatter/all-gather from B=12; this tail presently uses direct exchange
through B=16, another performance decision to measure.

## Source-only checks

```bash
python -m unittest discover -s experimental/minimax_m3_gluon/tests -v
python -m compileall -q experimental/minimax_m3_gluon
```

The tests verify source orchestration, routing selection, shared weight handling,
and metadata capacity/padding on B1–16 without importing a GPU runtime. They do
not compile Gluon or validate numerical output, cache ordering, IPC, occupancy,
graph replay, actual launch count, TP4 serving, or model quality.

## Next device-validation gates

Once GPU execution is authorized: compile the front and tail on the exact
Triton/Gluon build; inspect atomic and `sc1` poll/cache instructions; verify zero spills
and at least one resident CTA/CU; compare route IDs/weights and every metadata
array with ATOM/AITER across 1–16 tokens and replayed changed routes. Check
AITER's selected MoE1/MoE2 names and a trace for hidden launches. Then compare
full local and TP-reduced outputs, replay after allocator churn, all sparse
layers, TTFT/TPOT/throughput and lm-eval. The serving adapter must account for ATOM's `batch_id_per_q_token` pad rows,
TP-uniform width decisions, attention metadata and cached sparse selections. Only
after the MoE region is correct should the preceding output projection and sparse
attention stages be fused into the Gluon front.
