# Full attention-layer fusion, Flash C1

Scope: layer0 mHC pre+RMSNorm -> inputquant -> QA/KV -> latentnorm/quant -> QB -> Q/KVnorm/RoPE -> KVcachewrite -> sparseattention+splitreduce -> inverseRoPE/quant -> groupedOA -> quant -> OB -> mHCpost. 1024input/1024output, one decode row; layer0 SWA128. NoMoE, compressor orindexer.

Keep prior artifacts unchanged. prefix/ copies corrected mHC FP32mixture RMS candidate; known QA/attentionnative differences remain. Expected fullpipeline notnativeequivalent untilverified; report numericalerrors, do notrelax .001gate.

Residentgrid256CUs,4waves, existingdevice-scope boundedbarrier. PrefixproducesO andMix; gridbarrierpublishes attentionO tooutputregion. Outputregion owns FP8activation scratch/scales, OAresult, OBresult. Globalbarriers delimitquant/projection dependencies. Last outputbarrier publishesOB to mHCpost; postwrites disjoint residualfeatures, launchcompletionpublishes finaloutput. Sharedmemory should alias across sequentialstages; verify compiledLDS/VGPR, zero spills, oneCTA/CU. TDM tiledloads, WMMAlearnedGEMMs, keepL2prefetch inprefix.

Important baselineboundary: ATOM's attentionpost is normally fusedwithnextFFNpre. Native standalone mhc_post is an extractedoperator boundary, nottheliteralserverlaunchsequence. Report baseline distinction explicitly; prefix-throughOB nativecalls are actualdispatch. Avoidclaimingfullmodelservinggain.

Performance: separate second-round graph timings with same reusedbuffers onbothsides; FlatGraphBatchhelper. Nativeandfusedsame inputs/positions/cache andoutputlayout. Validate fullpipelineagainstnative aswell as fusedversusstagedours, so arithmetic differences are distinguishedfrom fusion/syncbugs.
