# Per-regime fused-MoE tables

Tables here are **not** merged into the default fused-MoE table. `AITER_CONFIGS`
only globs the top level of `model_configs/`, and so does the AOT prebuild list.
A table here takes effect only when an env var points at it:

| Env var | Used for |
|---|---|
| `AITER_CONFIG_FMOE_PREFILL` | MoE calls made while the host has set the regime to `"prefill"` |
| `AITER_CONFIG_FMOE_DECODE` | MoE calls made while the host has set the regime to `"decode"` |

The host selects the regime with `aiter.fused_moe.set_moe_regime()` or the
`moe_regime()` context manager. See the per-regime block in `aiter/fused_moe.py`.

- **No env var set for a regime:** that regime uses the default
  `AITER_CONFIG_FMOE` table.
- **Shape missing from a regime's table:** the call falls back to the heuristic,
  exactly as it would on the default table.

Prefill and decode need separate tables because they route tokens very
differently, so the same shape wants different kernels. The tuned key carries
only the padded token count, and both regimes land in the same buckets.

## Tables

- `dsv4_fp8fp4_ep8_tuned_fmoe_decode.csv`: DeepSeek-V4-Pro FP4 on gfx950 with
  EP8 and shared-expert fusion. This gives expert=49 (48 routed + 1 fused
  shared), model_dim=7168, inter_dim=3072, topk=6, token buckets 512..16384.
  - Stage1 is tuned against decode's real routing.
  - Stage2 is left on the heuristic's geometry.
  - Intended for `AITER_CONFIG_FMOE_DECODE`, with prefill left on the default
    table.
