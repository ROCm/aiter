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

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from ..remote import RemoteExecutor
from ..types import ShapeResult
from .base import BaseSubagent
from .baseline_agent import parse_stats_csv

# Classification labels
_IMPROVED = "improved"
_NEUTRAL = "neutral"
_COMPILER_REGRESSION = "compiler_regression"
_TUNING_REGRESSION = "tuning_regression"
_TUNING_REGRESSION_SEVERE = "tuning_regression_severe"


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
        multiple IDs are provided shapes are distributed round-robin across
        GPUs for parallel execution.
    kernel_variant:
        Substring used to identify relevant rows inside the rocprof stats CSV
        (matched against the ``Name`` column).
    baseline_data:
        Pre-collected baseline timing dict mapping shape keys to elapsed_ms
        values.  When provided, skips re-running the baseline benchmark.
    untuned_data:
        Pre-collected untuned timing dict in the same format as
        *baseline_data*.  When provided, skips re-running the untuned
        benchmark.
    threshold:
        Percentage threshold (as a fraction, e.g. ``0.05`` for 5%) used to
        classify shapes as improved or regression vs. baseline.
    tuning_threshold:
        Percentage threshold used to classify tuning-specific regressions vs
        the untuned kernel.  Defaults to the same value as *threshold*.
    shuffle:
        When ``True``, pass ``--shuffle`` to the benchmark script.
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
        shapes: List[Tuple[int, int, int]],
        bench_script: str,
        gpu_ids: List[int],
        kernel_variant: str,
        baseline_data: Optional[Dict[str, Any]] = None,
        untuned_data: Optional[Dict[str, Any]] = None,
        threshold: float = 0.05,
        tuning_threshold: Optional[float] = None,
        shuffle: bool = False,
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
        self.kernel_variant = kernel_variant
        self.baseline_data = baseline_data
        self.untuned_data = untuned_data
        self.threshold = threshold
        self.tuning_threshold = tuning_threshold if tuning_threshold is not None else threshold
        self.shuffle = shuffle

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rocprof_output_prefix(self, m: int, n: int, k: int) -> str:
        """Return the /tmp path prefix for rocprof ``-o`` argument."""
        return f"/tmp/rp_{m}_{n}_{k}"

    def _run_rocprof_for_shape(
        self,
        m: int,
        n: int,
        k: int,
        gpu_id: int,
    ) -> ShapeResult:
        """Run rocprof --stats for a single shape on the given GPU.

        Executes the benchmark script via ``executor.docker_exec()`` with
        ``HIP_VISIBLE_DEVICES`` set to *gpu_id*, reads back the generated
        ``*.stats.csv`` file, and returns a :class:`~..types.ShapeResult`.
        """
        prefix = self._rocprof_output_prefix(m, n, k)
        csv_output = f"{prefix}.csv"
        stats_csv = f"{prefix}.stats.csv"

        shuffle_flag = " --shuffle" if self.shuffle else ""
        rocprof_cmd = (
            f"rocprof --stats -o {csv_output} "
            f"python {self.bench_script} "
            f"--shape {m} {n} {k} --metric time --layout TN{shuffle_flag}"
        )

        self.executor.docker_exec(
            rocprof_cmd,
            env={"HIP_VISIBLE_DEVICES": str(gpu_id)},
        )

        # Read back the generated stats CSV.
        read_result = self.executor.docker_exec(
            f"cat {stats_csv}",
        )
        csv_content = read_result.stdout

        main_ns, reduce_ns = parse_stats_csv(csv_content, self.kernel_variant)
        return ShapeResult(m=m, n=n, k=k, main_ns=main_ns, reduce_ns=reduce_ns)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        shape_result: ShapeResult,
        baseline_key: str,
        untuned_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Classify a tuned shape result against baseline and untuned data.

        Returns a regression dict when a regression is detected, ``None``
        when the shape is improved or neutral.
        """
        if self.baseline_data is None or self.untuned_data is None:
            return None

        tuned_ns = shape_result.total_ns
        baseline_ns = self.baseline_data.get(baseline_key, {}).get("total_ns", 0)
        untuned_ns = self.untuned_data.get(untuned_key, {}).get("total_ns", 0)

        if baseline_ns <= 0 or tuned_ns <= 0:
            return None

        # Compute relative deltas (positive means tuned is slower).
        delta_vs_baseline = (tuned_ns - baseline_ns) / baseline_ns
        delta_vs_untuned = (tuned_ns - untuned_ns) / untuned_ns if untuned_ns > 0 else 0.0

        # IMPROVED: tuned is meaningfully faster than baseline.
        if delta_vs_baseline < -self.threshold:
            return None  # improvement, not a regression entry

        # COMPILER_REGRESSION: untuned already regressed vs baseline AND
        # tuned ≈ untuned (tuning didn't make things worse beyond tuning_threshold).
        if untuned_ns > 0:
            delta_untuned_vs_baseline = (untuned_ns - baseline_ns) / baseline_ns
            if (
                delta_untuned_vs_baseline > self.threshold
                and abs(delta_vs_untuned) <= self.tuning_threshold
            ):
                return {
                    "m": shape_result.m,
                    "n": shape_result.n,
                    "k": shape_result.k,
                    "delta": delta_vs_baseline,
                    "classification": _COMPILER_REGRESSION,
                }

        # TUNING_REGRESSION_SEVERE: tuned is much worse than baseline and
        # untuned was fine (untuned did not regress vs baseline).
        if untuned_ns > 0:
            delta_untuned_vs_baseline = (untuned_ns - baseline_ns) / baseline_ns
            if (
                delta_vs_baseline > self.threshold
                and delta_untuned_vs_baseline <= self.threshold
                and delta_vs_untuned > self.threshold
            ):
                return {
                    "m": shape_result.m,
                    "n": shape_result.n,
                    "k": shape_result.k,
                    "delta": delta_vs_baseline,
                    "classification": _TUNING_REGRESSION_SEVERE,
                }

        # TUNING_REGRESSION: tuned is worse than untuned beyond tuning_threshold.
        if untuned_ns > 0 and delta_vs_untuned > self.tuning_threshold:
            return {
                "m": shape_result.m,
                "n": shape_result.n,
                "k": shape_result.k,
                "delta": delta_vs_untuned,
                "classification": _TUNING_REGRESSION,
            }

        # NEUTRAL or within-threshold improvement vs baseline.
        return None

    def _is_improved(self, shape_result: ShapeResult, baseline_key: str) -> bool:
        """Return True when tuned is meaningfully faster than baseline."""
        if self.baseline_data is None:
            return False
        tuned_ns = shape_result.total_ns
        baseline_ns = self.baseline_data.get(baseline_key, {}).get("total_ns", 0)
        if baseline_ns <= 0 or tuned_ns <= 0:
            return False
        return (tuned_ns - baseline_ns) / baseline_ns < -self.threshold

    # ------------------------------------------------------------------
    # _execute implementation
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Run rocprof benchmarks for all shapes and produce a comparison report.

        Returns
        -------
        dict
            Keys: ``results``, ``regressions``, ``shapes_validated``,
            ``improved_count``, ``regression_count``.
        """
        if not self.gpu_ids:
            return {
                "results": [],
                "regressions": [],
                "shapes_validated": 0,
                "improved_count": 0,
                "regression_count": 0,
            }

        # Distribute shapes round-robin across available GPUs.
        shape_gpu_pairs: List[Tuple[Tuple[int, int, int], int]] = []
        for idx, shape in enumerate(self.shapes):
            gpu_id = self.gpu_ids[idx % len(self.gpu_ids)]
            shape_gpu_pairs.append((shape, gpu_id))

        # Collect timings — parallelise across GPUs when more than one GPU.
        shape_results: List[ShapeResult] = []

        if len(self.gpu_ids) > 1:
            futures_map = {}
            with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as pool:
                for (m, n, k), gpu_id in shape_gpu_pairs:
                    future = pool.submit(self._run_rocprof_for_shape, m, n, k, gpu_id)
                    futures_map[future] = (m, n, k)

                for future in as_completed(futures_map):
                    shape_results.append(future.result())
        else:
            for (m, n, k), gpu_id in shape_gpu_pairs:
                shape_results.append(self._run_rocprof_for_shape(m, n, k, gpu_id))

        # Sort results by (m, n, k) for deterministic output.
        shape_results.sort(key=lambda r: (r.m, r.n, r.k))

        # Build result dicts and perform classification.
        results = []
        regressions = []
        improved_count = 0

        for sr in shape_results:
            shape_key = f"M{sr.m}_N{sr.n}_K{sr.k}"
            results.append(
                {
                    "m": sr.m,
                    "n": sr.n,
                    "k": sr.k,
                    "main_ns": sr.main_ns,
                    "reduce_ns": sr.reduce_ns,
                    "total_ns": sr.total_ns,
                }
            )

            if self.baseline_data is not None and self.untuned_data is not None:
                regression = self._classify(sr, shape_key, shape_key)
                if regression is not None:
                    regressions.append(regression)
                elif self._is_improved(sr, shape_key):
                    improved_count += 1

        # Clean up temporary rocprof files.
        self.executor.docker_exec("rm -f /tmp/rp_*.csv /tmp/rp_*.stats.csv")

        return {
            "results": results,
            "regressions": regressions,
            "shapes_validated": len(shape_results),
            "improved_count": improved_count,
            "regression_count": len(regressions),
        }
