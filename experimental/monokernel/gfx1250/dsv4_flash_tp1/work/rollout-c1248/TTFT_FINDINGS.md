# TTFT investigation

The prior C8 TTFT regression did not reproduce in this run. Native versus updated median TTFT was 160.62/155.23 ms at C1, 298.67/292.89 ms at C2, 295.54/283.71 ms at C4, and 290.43/282.58 ms at C8. These are the second client runs, with one server startup per configuration.

Prefill implementation evidence:

- All four servers loaded the same compiled backbone cache: `530db60e73/rank_0/backbone`.
- The fixed-input prefill profiles contain exactly the same ordered sequence of 1,888 kernels for native, previous monokernel, updated monokernel, and the independent updated replica, at both one-sequence and eight-sequence prefill shapes. No monokernel kernel appears in either prefill trace.
- The prefill call audit recorded zero monokernel calls. Source hashes stayed unchanged through serving.

TTFT includes more than one prefill forward. ATOM's `TokenIDProcessor.is_deferred_out` is enabled in this TP1 configuration. `prepare_sampled_ids` returns the previous step's sampled tokens. A token sampled by one prefill can therefore be delivered after another queued prefill, rather than immediately after its own forward. Request arrival and completion timing also determines how prompts are batched.

Direct observations:

- In the updated C2 second client, two single-sequence prefills produced paired TTFTs around 290–312 ms. Two prompts combined into one 2048-token prefill produced paired TTFTs around 147–153 ms. The prefill forwards themselves were around 135–141 ms.
- Native C8 ran eight repetitions of `1 + 7` prefill partitioning and measured 290.43 ms TTFT. The independent updated replica also used `1 + 7` and measured 290.14 ms. The primary updated instance used a mixture of `1 + 7`, `3 + 5`, `2 + 6`, and `4 + 4`, and measured 282.58 ms.
- The controlled bulk client submits eight sibling completions together, with 1024 input tokens and one output token each, using `n=8`, positive temperature and `top_k=1` (the API collapses `n>1` for temperature zero). Two separate client invocations were run; second-client median complete-response times were native 148.12 ms, previous 147.19 ms, updated 139.33 ms, and updated replica 142.10 ms. These are one-token nonstreaming response times, not the 1k/1k serving TPOT benchmark.
- The one-sequence profile has about 25–26 ms of GPU kernel execution within a 141–146 ms launch span; the intervening gaps are substantial. The eight-sequence profile has about 91–96 ms of GPU kernel execution within a 131–135 ms span. CPU enqueue timing tracks the forward span. Per-process dispatch/scheduling variation therefore matters to TTFT even with identical GPU kernel sequences.

Limits of attribution:

The historical native and updated C8 second-run logs both predominantly used `1 + 7`. Those historical runs lack the per-forward GPU timing and full request-level evidence added here. Thus batching alone cannot be claimed to explain that exact old 12.22 ms difference. What is established is that prefill code did not change, the old regression is not reproducible here, and measured TTFT is sensitive to deferred output, batching and runtime timing variation. The identical updated configuration on another GPU differed by 7.6–17.5 ms in TTFT across these concurrency tests while decode throughput stayed close.

Raw evidence: `TTFT_COMPONENTS.json`, `PREFILL_TRACE_COMPARISON.json`, `SERVING_EVIDENCE.json`, `prefill-*.jsonl`, per-request benchmark JSONs, `bulk-*-round2.json`, and the Chrome traces.
