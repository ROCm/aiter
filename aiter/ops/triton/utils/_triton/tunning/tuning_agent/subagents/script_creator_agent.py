"""ScriptCreatorAgent — generate missing benchmark/test scripts for the tuning pipeline.

Responsibilities
----------------
- Read existing benchmark, unit-test, and smoke-test scripts as templates to
  understand project conventions (argument parsing, timing loops, etc.).
- Read the kernel source to extract the public API (function signatures,
  argument names and dtypes, expected tensor shapes).
- Generate any scripts identified as missing by DiscoveryAgent, filling in
  kernel-specific details while preserving the template structure.
- Run a quick smoke test of each newly generated script to confirm it imports
  and executes without error before handing off to the tuning phases.
"""

import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from ..remote import RemoteExecutor
from .base import BaseSubagent, SubagentResult


# ---------------------------------------------------------------------------
# Kernel-API parsing helpers
# ---------------------------------------------------------------------------

# Indicators that a kernel is "too complex" for automatic script generation.
_COMPLEXITY_SIGNALS = [
    r"custom_preprocess",
    r"def\s+\w+\s*\(.*\)\s*->.*Tuple",   # multiple return values typed
    r"->.*,",                               # multiple returns in annotation
    r"preprocess_weight",
    r"unusual_layout",
]

# Known dtype categories based on function/module name patterns.
_DTYPE_FP8 = "fp8"
_DTYPE_BF16 = "bf16"
_DTYPE_FP4 = "fp4"
_DTYPE_UNKNOWN = "unknown"


def _detect_dtype_category(source: str, kernel_name: str) -> str:
    """Heuristically determine the dtype category from source and kernel name."""
    name_lower = kernel_name.lower()
    if "fp4" in name_lower or "afp4" in name_lower or "wfp4" in name_lower:
        return _DTYPE_FP4
    if "a8w8" in name_lower or "fp8" in name_lower or "a8" in name_lower:
        return _DTYPE_FP8
    if "a16w16" in name_lower or "bf16" in name_lower:
        return _DTYPE_BF16
    # Fall back to source analysis
    if "fp4" in source.lower():
        return _DTYPE_FP4
    if "fp8" in source.lower() or "e4m3" in source.lower():
        return _DTYPE_FP8
    return _DTYPE_BF16


def _extract_public_function(source: str) -> Optional[Tuple[str, str]]:
    """Return (function_name, parameter_string) for the first public function.

    A "public" function is one whose name does not start with ``_`` and is not
    a Triton JIT kernel (i.e. not decorated with ``@triton.jit``).
    """
    # Find all function definitions, then filter out private and jit-decorated ones.
    pattern = re.compile(
        r"^def\s+([A-Za-z][A-Za-z0-9_]*)\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)",
        re.MULTILINE,
    )
    for m in pattern.finditer(source):
        fn_name = m.group(1)
        params = m.group(2)
        # Skip private helpers
        if fn_name.startswith("_"):
            continue
        # Skip triton.jit decorated functions by checking preceding text
        preceding = source[:m.start()]
        preceding_lines = preceding.rstrip().rsplit("\n", 3)
        if any("@triton.jit" in line or "@triton.autotune" in line for line in preceding_lines):
            continue
        preceding = source[max(0, m.start() - 200) : m.start()]
        if "@triton.jit" in preceding:
            continue
        return fn_name, params.strip()
    return None


def _is_complex_kernel(source: str) -> bool:
    """Return True if the kernel source has signs of unusual complexity."""
    for pattern in _COMPLEXITY_SIGNALS:
        if re.search(pattern, source):
            return True
    # Multiple return values: more than one name before the colon/return type
    # arrow at the top-level function signature.
    if re.search(r"->\s*Tuple\[", source):
        return True
    return False


# ---------------------------------------------------------------------------
# Script content generators
# ---------------------------------------------------------------------------

_UT_SCRIPT_TEMPLATE = """\
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from {module_import_path} import {kernel_fn}
from {test_module_import_path} import (
    {input_generator_fn},
)
############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
{input_generation_code}
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        {kernel_call_code}
        ############################################################

    run_profile(fn)
"""

_UT_SCRIPT_TEMPLATE_FP8 = """\
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from {module_import_path} import {kernel_fn}
from aiter.ops.triton.utils.types import get_fp8_dtypes
############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
_, e4m3_type = get_fp8_dtypes()
dtype = torch.bfloat16
x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(e4m3_type)
weight = torch.randint(-128, 127, (N, K), dtype=torch.int8).to(e4m3_type)
x_scale = torch.ones(M, dtype=dtype, device="cuda")
w_scale = torch.ones(N, dtype=dtype, device="cuda")
y = torch.zeros(M, N, dtype=dtype, device="cuda")
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        {kernel_call_code}
        ############################################################

    run_profile(fn)
"""

_UT_SCRIPT_TEMPLATE_BF16 = """\
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from {module_import_path} import {kernel_fn}
############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
x = torch.randn(M, K, dtype=dtype, device="cuda")
weight = torch.randn(N, K, dtype=dtype, device="cuda")
y = torch.zeros(M, N, dtype=dtype, device="cuda")
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        {kernel_call_code}
        ############################################################

    run_profile(fn)
"""

_UT_SCRIPT_TEMPLATE_FP4 = """\
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from {module_import_path} import {kernel_fn}
############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
# FP4 tensors are packed: 2 fp4 values per byte -> divide K by 2
x = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda")
weight = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")
x_scales = torch.ones(M, dtype=dtype, device="cuda")
w_scales = torch.ones(N, dtype=dtype, device="cuda")
y = torch.zeros(M, N, dtype=dtype, device="cuda")
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        {kernel_call_code}
        ############################################################

    run_profile(fn)
"""


def _derive_module_paths(kernel_source_path: str, kernel_fn: str) -> Dict[str, str]:
    """Derive Python import paths from the kernel source file path.

    Parameters
    ----------
    kernel_source_path:
        Path relative to the workspace root, e.g.
        ``aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py``.
    kernel_fn:
        The public function name.

    Returns
    -------
    dict with keys ``module_import_path`` and ``test_module_import_path``.
    """
    # Normalise: strip leading slash or workspace prefix
    path = kernel_source_path
    for prefix in ("/workspace/aiter/", "aiter/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    # Convert path separators to dots and strip .py extension
    if path.endswith(".py"):
        path = path[:-3]
    module_path = "aiter." + path.replace("/", ".")

    # Derive test module path: replace ops/triton with op_tests/triton_tests
    test_path = module_path.replace("aiter.ops.triton.", "op_tests.triton_tests.")
    # Replace the module name with test_<module>
    parts = test_path.rsplit(".", 1)
    if len(parts) == 2:
        test_module_path = parts[0] + ".test_" + parts[1]
    else:
        test_module_path = "op_tests.triton_tests.test_" + parts[0]

    return {
        "module_import_path": module_path,
        "test_module_import_path": test_module_path,
    }


def _derive_input_generator_fn(kernel_fn: str) -> str:
    """Return the conventional name of the input-generation helper."""
    return f"generate_{kernel_fn}_inputs"


def _build_kernel_call_code(kernel_fn: str, dtype_category: str) -> str:
    """Build a plausible kernel call for the ut script."""
    if dtype_category == _DTYPE_FP8:
        return f"{kernel_fn}(x, weight, x_scale, w_scale, None, dtype, y, config=config)"
    elif dtype_category == _DTYPE_FP4:
        return (
            f"{kernel_fn}(\n"
            f"            x, weight, x_scales, w_scales, dtype, y, config=config\n"
            f"        )"
        )
    else:
        return f"{kernel_fn}(x, weight, None, dtype, y, config=config)"


def _render_ut_script(
    kernel_fn: str,
    kernel_source_path: str,
    dtype_category: str,
) -> str:
    """Render the content of a ``ut_*.py`` script."""
    paths = _derive_module_paths(kernel_source_path, kernel_fn)
    input_gen_fn = _derive_input_generator_fn(kernel_fn)
    kernel_call = _build_kernel_call_code(kernel_fn, dtype_category)

    if dtype_category == _DTYPE_FP8:
        return _UT_SCRIPT_TEMPLATE_FP8.format(
            module_import_path=paths["module_import_path"],
            kernel_fn=kernel_fn,
            kernel_call_code=kernel_call,
        )
    elif dtype_category == _DTYPE_FP4:
        return _UT_SCRIPT_TEMPLATE_FP4.format(
            module_import_path=paths["module_import_path"],
            kernel_fn=kernel_fn,
            kernel_call_code=kernel_call,
        )
    elif dtype_category == _DTYPE_BF16:
        return _UT_SCRIPT_TEMPLATE_BF16.format(
            module_import_path=paths["module_import_path"],
            kernel_fn=kernel_fn,
            kernel_call_code=kernel_call,
        )
    else:
        # Generic fallback using the test-helper pattern
        input_code = (
            f"dtype = torch.bfloat16\n"
            f"inputs = {input_gen_fn}(*input_shape, dtype, output=True)"
        )
        return _UT_SCRIPT_TEMPLATE.format(
            module_import_path=paths["module_import_path"],
            kernel_fn=kernel_fn,
            test_module_import_path=paths["test_module_import_path"],
            input_generator_fn=input_gen_fn,
            input_generation_code=input_code,
            kernel_call_code=f"{kernel_fn}(*inputs, config=config)",
        )


_BENCH_SCRIPT_TEMPLATE = """\
import argparse
import torch

from {module_import_path} import {kernel_fn}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark {kernel_fn}")
    parser.add_argument("--shape", nargs=3, type=int, default=[1024, 1024, 1024],
                        metavar=("M", "N", "K"), help="Matrix dimensions")
    parser.add_argument("--metric", choices=["time", "tflops"], default="tflops",
                        help="Metric to report")
    parser.add_argument("--layout", default="TN",
                        help="Matrix layout (default: TN)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Enable weight shuffle (fp4 kernels)")
    return parser.parse_args()


def main():
    args = parse_args()
    M, N, K = args.shape
    dtype = torch.bfloat16

{input_setup_code}

    import time
    warmup = 10
    repeats = 100
    for _ in range(warmup):
        {kernel_call_code}
    torch.cuda.synchronize() if hasattr(torch.cuda, "synchronize") else None

    start = time.perf_counter()
    for _ in range(repeats):
        {kernel_call_code}
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000 / repeats
    flops = 2 * M * N * K
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    if args.metric == "time":
        print(f"Elapsed: {{elapsed_ms:.3f}} ms")
    else:
        print(f"Performance: {{tflops:.2f}} TFLOPS")


if __name__ == "__main__":
    main()
"""


def _render_bench_script(
    kernel_fn: str,
    kernel_source_path: str,
    dtype_category: str,
) -> str:
    """Render the content of a ``bench_*.py`` script."""
    paths = _derive_module_paths(kernel_source_path, kernel_fn)

    if dtype_category == _DTYPE_FP8:
        input_setup = textwrap.dedent("""\
            from aiter.ops.triton.utils.types import get_fp8_dtypes
            _, e4m3_type = get_fp8_dtypes()
            x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(e4m3_type)
            weight = torch.randint(-128, 127, (N, K), dtype=torch.int8).to(e4m3_type)
            x_scale = torch.ones(M, dtype=dtype, device="cuda")
            w_scale = torch.ones(N, dtype=dtype, device="cuda")
            y = torch.zeros(M, N, dtype=dtype, device="cuda")""")
        kernel_call = f"{kernel_fn}(x, weight, x_scale, w_scale, None, dtype, y)"
    elif dtype_category == _DTYPE_FP4:
        input_setup = textwrap.dedent("""\
            x = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda")
            weight = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")
            x_scales = torch.ones(M, dtype=dtype, device="cuda")
            w_scales = torch.ones(N, dtype=dtype, device="cuda")
            y = torch.zeros(M, N, dtype=dtype, device="cuda")""")
        kernel_call = f"{kernel_fn}(x, weight, x_scales, w_scales, dtype, y)"
    else:  # bf16 or unknown
        input_setup = textwrap.dedent("""\
            x = torch.randn(M, K, dtype=dtype, device="cuda")
            weight = torch.randn(N, K, dtype=dtype, device="cuda")
            y = torch.zeros(M, N, dtype=dtype, device="cuda")""")
        kernel_call = f"{kernel_fn}(x, weight, None, dtype, y)"

    # Indent the input setup for inside main()
    indented_setup = textwrap.indent(input_setup, "    ")

    return _BENCH_SCRIPT_TEMPLATE.format(
        module_import_path=paths["module_import_path"],
        kernel_fn=kernel_fn,
        input_setup_code=indented_setup,
        kernel_call_code=kernel_call,
    )


# ---------------------------------------------------------------------------
# ScriptCreatorAgent
# ---------------------------------------------------------------------------


class ScriptCreatorAgent(BaseSubagent):
    """Generate missing benchmark and test scripts from templates.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"fmha"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
    kernel_source_path:
        Absolute path on the remote host to the kernel source file(s).  Used
        to extract function signatures and argument metadata.
    template_dir:
        Absolute path on the remote host to the directory containing existing
        scripts to use as templates.
    missing_scripts:
        List of script descriptors (as produced by DiscoveryAgent) describing
        which scripts need to be created.  Each element is expected to be a
        dict with at least ``"type"`` (``"bench"`` | ``"ut"`` | ``"smoke"``)
        and ``"target_path"`` keys.
    expected_triton_commit:
        Forwarded to :class:`~.base.BaseSubagent`.
    expected_aiter_branch:
        Forwarded to :class:`~.base.BaseSubagent`.
    """

    name: str = "script_creator"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        kernel_source_path: str,
        template_dir: str,
        missing_scripts: List[Dict[str, Any]],
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
        self.kernel_source_path = kernel_source_path
        self.template_dir = template_dir
        self.missing_scripts = missing_scripts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_kernel_source(self) -> str:
        """Fetch the kernel source file content from the remote host."""
        result = self.executor.docker_exec(
            f"cat /workspace/aiter/{self.kernel_source_path}",
            check=False,
        )
        if result.returncode != 0:
            # Try the path as-is (absolute)
            result = self.executor.docker_exec(
                f"cat {self.kernel_source_path}",
                check=True,
            )
        return result.stdout

    def _read_template(self) -> str:
        """Fetch the boilerplate ut_a8w8_gemm.py template from the remote host."""
        result = self.executor.docker_exec(
            "cat /workspace/aiter/aiter/ops/triton/utils/_triton/tunning/ut_a8w8_gemm.py",
            check=False,
        )
        return result.stdout

    def _write_remote_script(self, target_path: str, content: str) -> None:
        """Write *content* to *target_path* on the remote host via docker exec."""
        # Use a here-document approach via printf to avoid quoting issues.
        escaped = content.replace("\\", "\\\\").replace("'", "'\\''")
        self.executor.docker_exec(
            f"printf '%s' '{escaped}' > {target_path}",
            check=True,
        )
        self.executor.docker_exec(f"chmod +x {target_path}", check=True)

    def _smoke_test_script(self, target_path: str) -> Tuple[bool, str]:
        """Run a quick smoke test of the script with a tiny shape.

        Returns (passed, error_message).
        """
        result = self.executor.docker_exec(
            f"cd /workspace/aiter && python {target_path} 8 8192 8192",
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        error = (result.stderr or result.stdout or "").strip()
        return False, error

    # ------------------------------------------------------------------
    # _execute implementation
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Generate missing scripts and validate them on the remote host.

        Steps
        -----
        1. Read the kernel source file to understand the public API.
        2. Read the reference template script for boilerplate.
        3. For each missing script, generate content and write it remotely.
        4. Smoke-test each generated script with shape 8 8192 8192.
        5. Write a JSON artifact and return the summary dict.
        """
        # Step 1 — read kernel source
        kernel_source = self._read_kernel_source()

        # Step 2 — read boilerplate template (ignore failures; it's advisory)
        _template = self._read_template()  # noqa: F841  used for context

        # Check for complexity
        if _is_complex_kernel(kernel_source):
            escalation_msg = (
                f"Cannot determine input format for kernel {self.kernel_name}. "
                "Please provide a PyTorch reference."
            )
            result_data = {
                "scripts_created": [],
                "smoke_test_passed": False,
                "needs_human_input": True,
                "escalation_message": escalation_msg,
            }
            self._write_json_artifact("script_creation.json", result_data)
            return result_data

        # Extract kernel API
        api = _extract_public_function(kernel_source)
        if api is None:
            escalation_msg = (
                f"Cannot determine input format for kernel {self.kernel_name}. "
                "Please provide a PyTorch reference."
            )
            result_data = {
                "scripts_created": [],
                "smoke_test_passed": False,
                "needs_human_input": True,
                "escalation_message": escalation_msg,
            }
            self._write_json_artifact("script_creation.json", result_data)
            return result_data

        kernel_fn, _params = api
        dtype_category = _detect_dtype_category(kernel_source, self.kernel_name)

        # Step 3 — generate and write each missing script
        scripts_created: List[str] = []
        smoke_results: Dict[str, bool] = {}
        smoke_errors: Dict[str, str] = {}
        all_smoke_passed = True

        for script_desc in self.missing_scripts:
            script_type = script_desc.get("type", "ut")
            target_path = script_desc.get("target_path", "")

            if script_type == "ut":
                content = _render_ut_script(kernel_fn, self.kernel_source_path, dtype_category)
            elif script_type == "bench":
                content = _render_bench_script(kernel_fn, self.kernel_source_path, dtype_category)
            else:
                # Default to ut style for "smoke" or unknown types
                content = _render_ut_script(kernel_fn, self.kernel_source_path, dtype_category)

            self._write_remote_script(target_path, content)
            scripts_created.append(target_path)

            # Step 4 — smoke test
            passed, error = self._smoke_test_script(target_path)
            smoke_results[target_path] = passed
            if not passed:
                all_smoke_passed = False
                smoke_errors[target_path] = error

        # Step 5 — write artifact and return
        result_data: Dict[str, Any] = {
            "scripts_created": scripts_created,
            "smoke_test_passed": all_smoke_passed,
            "needs_human_input": False,
            "escalation_message": None,
        }
        if smoke_errors:
            result_data["smoke_errors"] = smoke_errors

        self._write_json_artifact("script_creation.json", result_data)
        return result_data
