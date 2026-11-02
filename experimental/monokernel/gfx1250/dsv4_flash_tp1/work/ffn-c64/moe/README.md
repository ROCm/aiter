# Flash C64 routed MoE: actual ATOM versus prior fused prototype

Second-round flatgraph GPU3 medians:

| Region | Actual ATOM us | Prototype us |
|---|---:|---:|
| Input MXFP8 quant → routed gate/up GEMM + SwiGLU + MXFP8 requant → down GEMM → expert combine |144.968815|362.835839|
| HERD routing only, unchanged on both sides |9.815043|—|
| BF16 shared-expert add, unchanged |1.275508|—|

Native region invokes installed `_fused_experts_silu_gugu`, the same ATOM helper used by the live C64 decode path, with `ATOM_USE_TRITON_MOE=1`, `AITER_TRITON_USE_HERD=1`, and our serving monokernel disabled. Native four launches are input downcast, two `_moe_gemm_a8w4_decode` launches, and `reduce_grouped_gluon`. Native GEMM configuration is BM16 BN256 BK512,4warps,1buffer, splitK1, no persistent iterations; both grids4224CTAs. Parent live server C64 capture independently confirms these exact names/configurations. Native already fuses gate/up activation and intermediate requantization.

Candidate is the existing `flash/moe_multitoken_prototype.py`, now actually tested at M64. Inputquant remains a separate launch, followed by one fused MoE launch with256residentCTAs, BN128 BK256,3buffers, task loops for384token/expert routes and two global barriers. This is an ablation of the existing prototype, not a newly tuned C64 kernel. It is2.50times slower than native and is not a deployment recommendation.112VGPR,65,824bytesLDS,0spills.

Inputs: all256expert FP4weights/scales from actual checkpoint layer0, native GUGU gate/up interleave and WMMAshuffle. BF16[64,4096] input is synthetic normal(seed6764), not captured serving hidden states. Router logits use actual checkpoint layer0router weights applied to that input. Routing uses actual HERD sqrtsoftplus,top6,renormalization,routedscale1.5,no bias. Although layer0modelconfig has hash routing, current ATOM Triton branch ignores the custom hash callback and invokes this HERD path; live capture confirms no hash_topk. Do not describe these as real-request captured routing statistics. `inputs.pt` preserves synthetichiddenstates and checkpoint-derivedlogits; weights are reloadable from `/data/DeepSeek-V4-Flash` via the script.

Candidate output is bitwise identical to native on initialinput and16additional randominputs with the same fixedrouting (4,194,304additionaloutputvalues), healthzero. These tests do not validate arbitrary changingroute distributions or fullmodelquality. The old prototype was not assumed to supportC64: it was compiled and executed here and passed these checks.

Primary timing uses `attention/graphbatch.py:FlatGraphBatch` from the prior actualmicrobenchmark artifact: capture once, clone unchanged native kernelparameters into a serialflatgraph with identical buffers.16calls/graph,3warmups,20alternating samples perround, second-roundmedian; both rounds and rawsamples saved. Routing/add use128calls/graph,5warmups,40samples. Profiler traces identify kernels; their individual duration samples are not the reported timingmethod.

`bench.py` reproduces checkpointloading, validation and native/candidatetiming. `validate_extra.py` reproduces16inputchecks and routing/addtiming. Both run on oneGPU selected via HIP_VISIBLE_DEVICES. Dependency copies and provenance are included alongside this file.
