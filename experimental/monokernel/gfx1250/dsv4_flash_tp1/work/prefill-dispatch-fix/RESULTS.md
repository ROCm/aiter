# Decode-only dispatch / native prefill — 2026-10-21

The mode branch now runs outside ATOM's compiled backbone. The backbone uses native layers for all prefill sizes and no longer enters the monokernel custom-op fallback or its four output clones/layer. Eligible decode calls the unchanged full monokernel through eager orchestration recorded in ATOM's full GPU graphs.

## Second-client C8 performance

1k input / 1k output,64requests, one server start per configuration. Median TTFT/TPOT; total throughput includes input+output.

|Configuration|TTFT ms|TPOT ms|Total tok/s|Output tok/s|
|---|---:|---:|---:|---:|
|native|270.845|13.731|1144.406|572.203|
|wrapped|309.871|12.250|1272.717|636.358|
|fused|278.336|12.251|1276.894|638.447|

## Graph evidence

Native and fixed use the same compiled-backbone cache key and byte-identical computation_graph.py: SHA256 81c8e27ebacb2f4a5616cab05edff50874e98f4923f2d8138713d35014d775dd. Both have zero monokernel custom-op call sites. Wrapped control has43 calls; its clones execute inside those opaque callbacks and therefore are not graph-level clone nodes. Full graph snapshots and audit stored here.

## Validation

Serving logs confirm all43 monokernel layers captured for each batch1/2/4/8. Tiny-prefill comparisons: [{"input_tokens": 1, "text_equal": true}, {"input_tokens": 2, "text_equal": false}, {"input_tokens": 4, "text_equal": true}, {"input_tokens": 8, "text_equal": true}, {"input_tokens": 1024, "text_equal": true}].

Single-token decoded-text checks: [{"input_tokens": 1, "first_token_equal": true}, {"input_tokens": 2, "first_token_equal": true}, {"input_tokens": 4, "first_token_equal": true}, {"input_tokens": 8, "first_token_equal": true}, {"input_tokens": 1024, "first_token_equal": true}]. Server did not return token IDs/logprobs; matching text (including replacement characters) does not prove token-ID equality. One of five longer continuations differs; decode numerical differences remain.

GSM8K32examples,3-shot,max512 generated tokens: {"gsm8k": {"name": "gsm8k", "alias": "gsm8k", "sample_len": 32, "exact_match,strict-match": 0.96875, "exact_match_stderr,strict-match": 0.03125, "exact_match,flexible-extract": 0.96875, "exact_match_stderr,flexible-extract": 0.03125}}. Limited smoke test.

## Run

Existing integrated-c8/run.sh now defaults ATOM_DSV4_DECODE_ONLY_DISPATCH=1. Set it to0 only for the old wrapped control. prefill-dispatch-fix/run.sh explicitly selects1. install.py applies the outer dispatch and native-backbone branch to /app/ATOM/atom/models/deepseek_v4.py; decode_forward.py must be available in /work/integrated-c8. Model and matching ROCm/Triton/AITER environment required.

ASCII designs are in ATTENTION_DESIGNS.txt. The monokernel implementations and their prior numerical qualifications are in ../integrated-c8. This patch changes orchestration, not kernel math.
