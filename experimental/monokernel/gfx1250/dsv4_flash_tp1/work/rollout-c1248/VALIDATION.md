# Validation and limits

Final serving recipe: C1/C2 use final-output layout fusion (TILE128, OVERLAP0). C4/C8 use input/output layout fusion plus early OA/OB TDM (TILE192, OVERLAP2). Existing bounded QB L2 prefetch, TDM tiled loads, local input quantization and the prior AITER-derived implementations remain enabled. Experimental fine-grained semaphore paths and broad extra prefetch remain off.

The 12 final synthetic full-layer fixtures (C1/2/4/8 × SWA/CSA/HCA) passed exact comparison against the previous monokernel at 13 saved checkpoints, repeated graph execution, and barrier health checks, with zero register spills. The fixtures use position 1023, exercising compressor update boundaries. They do not reproduce every weight/value/context seen in a 1024-token decode trajectory. Individual region bodies are isolated serving code; timings are GPU microseconds, not native ATOM unfused timings.

Not every region got faster. Final FFN/MoE/sum/post/layout fusion improved at all four batch sizes. Standalone early OA/OB TDM is mildly slower at C4/C8. C1 CSA full-layer synthetic timing regressed by 4.1%, despite an end-to-end improvement. All results are retained in REPORT.md; no claim is made that every isolated group improved.

Longer replay testing exposed C2 CSA barrier timeouts with input-layout fusion, including repeated selected-kernel controls. Classic graph mode did not eliminate every timeout. Root cause is not established. That C2 combination is excluded from the final recipe; the less aggressive C2 recipe passed the final checks. Failed candidates remain in the logs and must not be promoted from shorter tests.

Serving validation:

- Every second-client benchmark completed all requested outputs: 8/16/32/64 requests for C1/2/4/8, each 1024 input + 1024 output tokens.
- All 43 decode layers were recorded using the full attention + shared/routed MoE monokernel at every captured batch size. No register spills. Native prefill kernel sequence verified unchanged.
- Controlled fixed-batch, top-k=1, 128-token response comparisons matched previous-versus-updated at all four batch sizes (15/15 sibling outputs); updated replica also matched.
- Diverse concurrent request tests were not fully text-identical at C4/C8. The updated and identical replica differed in two C4 responses, indicating that batching/runtime variation is a confounder; they matched each other at C8 but differed from the previous monokernel in six responses. These mixed-batch differences are unresolved and must not be presented as native numerical qualification.
- The previous and updated monokernels also differ from native ATOM on some text tests. No new GSM8K run was performed. Keep this recipe experimental and opt-in.

Two diagnostic harness issues were fixed without changing the serving benchmark: the first QKV ablation wrapper duplicated baseline coefficient loads (discard those timings; final measurements use the corrected wrapper), and the one-output-token benchmark exporter needed TPOT/ITL metrics retained to avoid a post-run export KeyError. The API does not support nested prompt-token arrays; controlled multi-completion diagnostics use the supported n parameter.
