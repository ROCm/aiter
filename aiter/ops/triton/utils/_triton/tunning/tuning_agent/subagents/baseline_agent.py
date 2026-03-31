"""BaselineAgent: runs rocprof --stats benchmarks for a set of (M, N, K) shapes."""

import csv
import io
import os
from dataclasses import asdict
from typing import List, Tuple

from ..remote import RemoteExecutor
from ..types import ShapeResult
from .base import BaseSubagent, SubagentError


def parse_stats_csv(csv_content: str, kernel_variant: str) -> Tuple[int, int]:
    """Parse rocprof --stats CSV output and extract kernel timings.

    Scans *csv_content* for rows whose ``Name`` column contains
    *kernel_variant*.  Among those rows:

    * **Main kernel** — a row that also contains ``"kernel"`` but *not*
      ``"reduce_kernel"``.
    * **Reduce kernel** — a row that contains ``"reduce_kernel"``
      (present only for split-K configurations).

    Parameters
    ----------
    csv_content:
        Raw text content of the ``*.stats.csv`` file produced by rocprof.
    kernel_variant:
        Substring to filter relevant kernel rows (e.g. ``"gemm_afp4wfp4"``).

    Returns
    -------
    Tuple[int, int]
        ``(main_ns, reduce_ns)`` where each value is the integer
        ``AverageNs`` from the matching row, or ``0`` if that row was
        not found.  Returns ``(0, 0)`` when no rows match *kernel_variant*
        at all.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    main_ns: int = 0
    reduce_ns: int = 0
    found_any = False

    for row in reader:
        name = row.get("Name", "")
        if kernel_variant not in name:
            continue
        found_any = True
        avg = int(row.get("AverageNs", 0))
        if "reduce_kernel" in name:
            reduce_ns = avg
        elif "kernel" in name:
            main_ns = avg

    if not found_any:
        return (0, 0)
    return (main_ns, reduce_ns)


class BaselineAgent(BaseSubagent):
    """Subagent that collects rocprof baseline timings for a list of shapes.

    For each ``(M, N, K)`` shape the agent:

    1. Runs ``rocprof --stats`` over the benchmark script on the remote host.
    2. Reads back the generated ``*.stats.csv`` file.
    3. Parses main-kernel and (optionally) reduce-kernel ``AverageNs`` values.
    4. Stores a :class:`~tuning_agent.types.ShapeResult` per shape.

    After all shapes are processed the collected results are serialised to a
    JSON artifact and the summary dict is returned to the base-class ``run``
    machinery.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"gemm_afp4wfp4"``).
    artifact_dir:
        Absolute path on the *remote* host where JSON artifacts are written.
    shapes:
        List of ``(M, N, K)`` integer tuples to benchmark.
    bench_script:
        Absolute path (on the remote host) to the Python benchmark script.
    gpu_id:
        GPU index to use for benchmarking (passed as ``HIP_VISIBLE_DEVICES``).
    kernel_variant:
        Substring used to identify relevant rows inside the rocprof stats CSV
        (matched against the ``Name`` column).
    expected_triton_commit:
        Forwarded to :class:`~tuning_agent.subagents.base.BaseSubagent`.
    expected_aiter_branch:
        Forwarded to :class:`~tuning_agent.subagents.base.BaseSubagent`.
    """

    name = "baseline"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        shapes: List[Tuple[int, int, int]],
        bench_script: str,
        gpu_id: int,
        kernel_variant: str,
        expected_triton_commit=None,
        expected_aiter_branch=None,
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
        self.gpu_id = gpu_id
        self.kernel_variant = kernel_variant

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rocprof_prefix(self, m: int, n: int, k: int) -> str:
        """Return the remote path prefix used as rocprof ``-o`` argument."""
        return os.path.join(
            self.artifact_dir,
            f"rocprof_{self.kernel_name}_M{m}_N{n}_K{k}",
        )

    def _run_rocprof(self, m: int, n: int, k: int) -> str:
        """Execute rocprof for a single shape and return the stats CSV content.

        Parameters
        ----------
        m, n, k:
            Matrix dimensions for the benchmark.

        Returns
        -------
        str
            Content of the ``<prefix>.stats.csv`` file.

        Raises
        ------
        SubagentError
            If rocprof exits with a non-zero code or the stats file cannot be
            read back.
        """
        prefix = self._rocprof_prefix(m, n, k)
        csv_output = f"{prefix}.csv"
        stats_csv = f"{prefix}.stats.csv"

        rocprof_cmd = (
            f"HIP_VISIBLE_DEVICES={self.gpu_id} "
            f"rocprof --stats -o {csv_output} "
            f"python {self.bench_script} "
            f"--shape {m} {n} {k} --metric time --layout TN"
        )

        self.executor.docker_exec(rocprof_cmd, check=True)

        # Read back the generated stats CSV.
        read_result = self.executor.docker_exec(f"cat {stats_csv}", check=True)
        return read_result.stdout

    # ------------------------------------------------------------------
    # _execute implementation
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Benchmark all shapes and return a summary dict.

        Returns
        -------
        dict
            ``{"shapes_collected": <int>, "results_path": <str>}``
        """
        results: List[ShapeResult] = []

        for m, n, k in self.shapes:
            csv_content = self._run_rocprof(m, n, k)
            main_ns, reduce_ns = parse_stats_csv(csv_content, self.kernel_variant)
            results.append(ShapeResult(m=m, n=n, k=k, main_ns=main_ns, reduce_ns=reduce_ns))

        # Serialise results as a list of dicts (dataclasses are not JSON-native).
        # Use an explicit dict to include the total_ns property which asdict() omits.
        serialisable = [
            {"m": r.m, "n": r.n, "k": r.k, "main_ns": r.main_ns, "reduce_ns": r.reduce_ns, "total_ns": r.total_ns}
            for r in results
        ]
        results_path = self._write_json_artifact("baseline_results.json", serialisable)

        return {
            "shapes_collected": len(results),
            "results_path": results_path,
            "results": serialisable,
        }
