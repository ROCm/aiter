"""ValidationAgent — benchmark and validate tuned configs against baselines.

Responsibilities
----------------
- Collect per-shape timing data for three configurations on all requested
  shapes: the *baseline* (reference implementation), the *untuned* Triton
  kernel (autotuned on-the-fly), and the *tuned* Triton kernel (using the
  configs produced by ConfigGeneratorAgent).
- Use ``rocprof --stats`` to capture accurate GPU timing rather than relying
  on Python-side wall-clock measurements.
- Perform a 3-way comparison of the collected timings and classify each shape
  as: regression, neutral, or improvement relative to both the baseline and
  the untuned kernel.
- Aggregate results across all GPUs and shapes into a structured report.
"""

from typing import Any, Dict, List, Optional

from ..remote import RemoteExecutor
from .base import BaseSubagent, SubagentResult


class ValidationAgent(BaseSubagent):
    """Collect timings and classify tuned-config regressions vs baselines.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"fmha"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
    shapes:
        List of shape dicts to benchmark, e.g.
        ``[{"M": 256, "N": 128, "K": 64}, ...]``.
    bench_script:
        Absolute path on the remote host to the benchmark script to invoke.
        The script is expected to accept shape arguments and a ``--mode``
        flag (``baseline`` | ``untuned`` | ``tuned``).
    gpu_ids:
        List of GPU device IDs to benchmark on (e.g. ``[0, 1]``).  When
        multiple IDs are provided the script is executed once per GPU and
        results are aggregated.
    baseline_data:
        Pre-collected baseline timing dict mapping shape keys to elapsed_ms
        values.  When provided, skips re-running the baseline benchmark.
    untuned_data:
        Pre-collected untuned timing dict in the same format as
        *baseline_data*.  When provided, skips re-running the untuned
        benchmark.
    expected_triton_commit:
        Forwarded to :class:`~.base.BaseSubagent`.
    expected_aiter_branch:
        Forwarded to :class:`~.base.BaseSubagent`.
    """

    name: str = "validation"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        shapes: List[Dict[str, Any]],
        bench_script: str,
        gpu_ids: List[int],
        baseline_data: Optional[Dict[str, Any]] = None,
        untuned_data: Optional[Dict[str, Any]] = None,
        expected_triton_commit: Optional[str] = None,
        expected_aiter_branch: Optional[str] = None,
    ) -> None:
        super().__init__(
            executor=executor,
            kernel_name=kernel_name,
            artifact_dir=artifact_dir,
            expected_triton_commit=expected_triton_commit,
            expected_aiter_branch=expected_aiter_branch,
        )
        self.shapes = shapes
        self.bench_script = bench_script
        self.gpu_ids = gpu_ids
        self.baseline_data = baseline_data
        self.untuned_data = untuned_data

    def _execute(self) -> dict:
        """Run benchmarks and produce a 3-way comparison report.

        TODO
        ----
        Implement the following steps in order:

        1. **Collect baseline timings** (skip if *baseline_data* provided)
           - For each shape in *self.shapes* and each GPU in *self.gpu_ids*:
               ``rocprof --stats python <bench_script> --mode baseline
                 --M <M> --N <N> --K <K> --gpu <gpu_id>``
           - Parse ``rocprof`` CSV output (``results.stats.csv``) to extract
             the mean kernel duration in microseconds.
           - Store results in a dict keyed by a canonical shape string
             (e.g. ``"M256_N128_K64"``).

        2. **Collect untuned timings** (skip if *untuned_data* provided)
           - Repeat the collection above with ``--mode untuned``.
           - Untuned mode lets Triton's built-in autotuner select configs
             at runtime; this serves as the no-tuning-effort reference.

        3. **Collect tuned timings**
           - Repeat with ``--mode tuned``; the bench script should load the
             configs produced by ConfigGeneratorAgent automatically.
           - Ensure the config directory is visible to the script by checking
             or setting ``AITER_CONFIG_DIR`` in the remote environment.

        4. **3-way comparison & classification**
           - For each shape, compute:
             - ``speedup_vs_baseline  = baseline_ms  / tuned_ms``
             - ``speedup_vs_untuned   = untuned_ms   / tuned_ms``
           - Classify each shape using thresholds (e.g. ±5 %):
             - ``"regression"``   — tuned is >5 % slower than baseline OR untuned.
             - ``"neutral"``      — within ±5 % of both references.
             - ``"improvement"``  — tuned is >5 % faster than both references.
           - Aggregate across GPUs (mean speedup per shape across all GPU IDs).

        5. **Write artifact & return**
           - Persist the full comparison table via
             ``self._write_json_artifact("validation_report.json", ...)``.
           - Return a dict with keys: ``"regressions"``, ``"improvements"``,
             ``"neutral"``, ``"summary"`` (aggregate pass/fail verdict),
             ``"per_shape_results"``.
        """
        # TODO: implement steps above
        return {"status": "not_implemented"}
