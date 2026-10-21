Results for the FP8 QK optimization described in [the implementation report](../../../docs/fused_dsv41_fp8_qk_optimization.md).

GPU 6 only; all-FP8 KV. Timings include Q preparation and output epilogue except the explicitly labeled PR attention-only column. Helpers depend on the pinned external fixture and PR checkouts recorded in the report. Raw profiler databases are retained remotely; their SHA256 hashes are in profile_summary.json.
