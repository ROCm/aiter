# Restore the C16 experiment bundle

The outer archive contains:
- monokernel-c1248-rollout-20261021.tgz: the complete prior working source/artifact snapshot and restore instructions.
- c16-monokernel/: the new isolated C16 implementation, benchmark drivers, JSON results, traces, assembly, diagnostics, and report.
- dependencies/: checkpoint fixtures and native reference harnesses needed by the new experiments but absent from the previous archive.
- ARTIFACT_MANIFEST.json and ENVIRONMENT.json inside c16-monokernel/.

First extract the prior archive into the workspace according to its restore instructions. Copy the contents of outer dependencies/ into the restored dsv4-monokernel tree, preserving relative paths. Copy the outer c16-monokernel directory to /home/vigokhal/dsv4-monokernel/c16-monokernel. Mount that dsv4-monokernel tree as/work in the saved experiment image or existing container vigokhal_dsv4_tp4. The model is not included; provide/data/DeepSeek-V4-Flash. All new kernels load dependencies from the prior/work tree plus the installed/app/ATOM and/app/aiter.

The C16 experiment is single-GPU Flash, with four GPUs used for independent workers. No C16 serving integration, image push, or E2E benchmark was performed in this task. C16 is implemented/tested at the full-layer kernel entry point, not enabled in the HTTP server adapter. Existing C1/2/4/8 serving sources were not edited.

Run four workers, one per GPU:
```
docker exec -d -e HIP_VISIBLE_DEVICES=0 vigokhal_dsv4_tp4 bash /work/c16-monokernel/native_worker.sh 0
docker exec -d -e HIP_VISIBLE_DEVICES=1 vigokhal_dsv4_tp4 bash /work/c16-monokernel/native_worker.sh 1
docker exec -d -e HIP_VISIBLE_DEVICES=2 vigokhal_dsv4_tp4 bash /work/c16-monokernel/native_worker.sh 2
docker exec -d -e HIP_VISIBLE_DEVICES=3 vigokhal_dsv4_tp4 bash /work/c16-monokernel/native_worker.sh 3
```
Wait for all workers before reusing a GPU. Run full-layer validation:
```
HIP_VISIBLE_DEVICES=0 TEST_BATCHES=16 python /work/c16-monokernel/final_swa.py
HIP_VISIBLE_DEVICES=1 TEST_BATCHES=16 RATIO=4 python /work/c16-monokernel/final_compressed.py
HIP_VISIBLE_DEVICES=2 TEST_BATCHES=16 RATIO=128 python /work/c16-monokernel/final_compressed.py
HIP_VISIBLE_DEVICES=3 python /work/c16-monokernel/check_herd.py
```
The above four Python commands can run in parallel inside the container. Repeat with TEST_BATCHES=8 for controls. Run report.py after results exist. A native benchmark failure is recorded in run-*.exit; inspect logs before reporting. Historical diagnostic logs containing failures are retained intentionally. Authoritative completed data: REGION_RESULTS.json, FULL_LAYER_CHECKS.json, HERD_CHECKS.json, final-micro-*.json, native-c*/ paired results listed in REGION_RESULTS.json.

Next priorities: adapt the tuned native C16 QKV/OA/OB GEMM designs; boundary HCA compression; reduce HERD synchronization overhead without scratch aliasing; evaluate whole-model C16 only after integration and accuracy checks. When E2E is requested, start each server once, run each client twice, and report only the second client run.
