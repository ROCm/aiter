# MXFP4 MoE Tuning

This context names the concepts used to reduce the MXFP4 A4W4 MoE tuning search space while retaining end-to-end validation.

## Language

**Tuning run**:
One tuner invocation together with the work it starts. Separate invocations are separate tuning runs even when they target the same physical GPU.
_Avoid_: Tuner process, when referring to the whole run

**Per-GPU measurement exclusivity**:
Within one tuning run, independent screening or validation work never overlaps on the same logical GPU; work on different GPUs may overlap. Exclusivity between separate tuning runs is the caller's responsibility.
_Avoid_: Global GPU exclusivity, GPU lock

**Stage candidate**:
A single GEMM1 or GEMM2 kernel configuration that can participate in a compatible two-stage MoE pipeline.
_Avoid_: Candidate, when the stage is ambiguous

**Candidate universe**:
The complete set of pipeline pairs admitted by a specified tuning policy for one shape and execution environment. An unpruned universe and a pruned universe are distinct even when they share the same kernel families.
_Avoid_: Full search, when referring to a set rather than its evaluation

**Candidate pruning**:
The exclusion of stage candidates or pipeline pairs before measured screening or validation. Excluded pairs cannot be recovered by fallback within the remaining candidate universe.

**Search strategy**:
The policy that chooses which pipeline pairs from a candidate universe receive validation runs.
_Avoid_: Search mode, when the candidate universe and evaluation policy need to be distinguished

**Pipeline pair**:
A compatible GEMM1 and GEMM2 stage-candidate pairing that can be executed and validated end to end.
_Avoid_: Kernel pair

**Pipeline compatibility group**:
A set of pipeline pairs that share a compatible GEMM1 intermediate contract and the same final-output setup and accumulation contract. Different kernel families may share a group when those contracts match.
_Avoid_: Equivalence class, subset, bucket

**Accumulation mode**:
The final-output contract of GEMM2: atomic accumulation into a preinitialized output, or non-atomic production that overwrites the output through reduction or scatter. Different accumulation modes belong to different pipeline compatibility groups.

**Shortlist**:
The stage candidates retained by screening for final end-to-end pipeline-pair evaluation.
_Avoid_: Winners

**Screening top-K**:
The maximum number of independently ranked stage candidates retained per screening context. It is distinct from the model's routing top-k.
_Avoid_: Top-k, when the routing or screening meaning is ambiguous

**Routing top-k**:
The number of experts selected for each token by the MoE router.
_Avoid_: Screening top-K

**Reference producer**:
A deterministic, correctness-checked GEMM1 configuration used only to create a layout-faithful intermediate for independent GEMM2 screening. It is not assumed to be performance-optimal.
_Avoid_: Best GEMM1, baseline GEMM1

**Reference consumer**:
A deterministic, correctness-checked GEMM2 configuration used to validate a reference producer or another GEMM1 candidate. It is not assumed to be performance-optimal.
_Avoid_: Best GEMM2, baseline GEMM2

**Screening score**:
The independently measured performance of a stage candidate. It ranks candidates for the shortlist but provides no correctness guarantee and is never written directly as the final tuned result.
_Avoid_: Final latency

**Stage screening context**:
A set of stage candidates that share the same caller-visible input, output, and execution contract, while differing only in performance-tuning parameters. Fixed or derived contract properties remain part of the context even when they do not distinguish separate groups.
_Avoid_: Block-M group

**Validation run**:
An end-to-end execution of a pipeline pair that supplies both its final latency and its correctness result.
_Avoid_: Screening run

**Evaluated-pair cache**:
The per-shape record of every pipeline-pair validation that returned control, including successful, incorrect, catchable-failure, and soft-timeout outcomes. Search expansion reuses these outcomes and never reruns the same pair. A fatal worker death is a shape-level failure, not a cacheable pair outcome.
_Avoid_: Kernel cache

**Ground truth**:
The valid pipeline-pair latency results from an exhaustive search, used to evaluate a reduced search strategy.
_Avoid_: Baseline, reference kernel
