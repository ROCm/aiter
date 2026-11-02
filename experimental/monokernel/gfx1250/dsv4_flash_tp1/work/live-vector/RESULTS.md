# Flash C1 ATOM live vector result

**Experimental: numerical acceptance failed. These are diagnostic timings.**

Single GPU; 1024 input / 1024 output; four measured requests; seed 0; ignore EOS; no prefix caching. Each configuration used one server startup and two complete clients. Only client 2 is reported; server PID/start-time matched before and after both clients. Total tok/s includes prompt and generated tokens.

| Configuration | Total tok/s | Output tok/s | Mean TPOT (ms) | Mean TTFT (ms) |
|---|---:|---:|---:|---:|
| Original image rerun | 151.919 | 75.959 | 13.033 | 147.696 |
| Native ATOM, hash dispatch corrected | 150.662 | 75.331 | 13.138 | 153.102 |
| Vector layers 0–1, hash dispatch corrected (diagnostic) | 147.832 | 73.916 | 13.389 | 155.663 |

The candidate is 1.88% slower than the corrected native baseline and 0.18% below the historical 148.1 total tok/s. It does not establish a speedup.

Scope: the unfused vector control replaces complete uncompressed hash layers 0 and 1 during C1 decode, including MoE and attention. All other 41 layers remain native. Prefill uses native math with residual materialization at adapter boundaries. This is not a whole-model vector baseline, nor the all-WMMA kernel. Both layers printed VECTOR_ACTIVE during the C1 graph capture.

Correctness: first live layer-0 output relative RMS was 1.984% versus corrected native and 1.009% versus independent PyTorch equations; acceptance gate is 0.1%. Layer 1 has not passed corresponding numerical validation. No GSM8K score is claimed. The post-benchmark France smoke prompt produced different continuations in native and candidate; logs are retained.

The image-specific native Triton decode path ignored the hash-routing callback. The isolated source correction is used in both paired configurations; see ROUTING_FINDING.md. Original image files and published branches/images were not modified.

Source, launchers, patch, raw results, intermediate dumps and logs are in this directory. README.md contains reproduction and resume instructions. The experiment server was stopped after the runs.
