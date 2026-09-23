# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only wiring guards, with no aiter, Torch, FlyDSL, or YAML dependency.

Actionlint checks YAML syntax separately. Literal block checks avoid YAML 1.1's
conversion of ``on`` to a boolean and never skip for a missing optional parser.
Conditions are compared, not evaluated as Python code.
"""

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_PR = (
    "(github.event_name == 'pull_request' && "
    "github.event.pull_request.head.repo.full_name == github.repository && "
    "github.event.pull_request.draft == false) || "
)


def _block(text, key, indent):
    """Read a literal mapping block, ending at its next sibling or parent."""
    match = re.search(
        rf"(?ms)^{' ' * indent}{re.escape(key)}:[^\n]*\n"
        rf".*?(?=^ {{0,{indent}}}\S|\Z)",
        text,
    )
    if match is None:
        raise AssertionError(f"Missing workflow block: {key}")
    return match.group()


class TestFlydslMoeCi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (_ROOT / ".github/workflows/tuning-tests.yaml").read_text()
        cls.events = _block(cls.workflow, "on", 0)
        cls.jobs = _block(cls.workflow, "jobs", 0)

    def test_pull_request_trigger_covers_moe_and_ci_dependencies(self):
        pr = _block(self.events, "pull_request", 2)
        self.assertIn("branches: [main]", pr)
        self.assertIn(
            "types: [opened, synchronize, reopened, ready_for_review, converted_to_draft]",
            pr,
        )
        paths = re.findall(r'^      - "([^"]+)"$', pr, re.MULTILINE)
        for path in (
            "aiter/ops/flydsl/kernels/moe_gemm_2stage/**",
            "aiter/ops/flydsl/kernels/moe_gemm_2stage*.py",
            "aiter/ops/flydsl/kernels/tensor_shim.py",
            "aiter/ops/flydsl/fused_moe_gfx942.py",
            "aiter/ops/flydsl/moe_gemm_2stage.py",
            "aiter/aot/flydsl/**",
            "aiter/fused_moe.py",
            "aiter/fused_moe_registry.py",
            "aiter/jit/**",
            "aiter/configs/**",
            "op_tests/tuning_tests/**",
            ".github/workflows/tuning-tests.yaml",
            ".github/workflows/ci-config.yaml",
            ".github/runner-config.yml",
            ".github/scripts/build_aiter_triton.sh",
            ".github/scripts/install_triton.sh",
            ".github/requirements/**",
            "requirements*.txt",
            "setup.py",
            "pyproject.toml",
        ):
            with self.subTest(path=path):
                self.assertIn(path, paths)

    def test_runtime_jobs_require_trusted_nondraft_pr_or_schedule_manual(self):
        for job, suite, accepts_pr in (
            ("cpu-validation", "level01", True),
            ("whole_graph_cache", "whole_graph_cache", True),
            ("gpu-pipeline", "tune_pipeline", False),
            ("gpu-run-config", "run_config", False),
        ):
            with self.subTest(job=job):
                condition = _block(_block(self.jobs, job, 2), "if", 4)
                condition = " ".join(condition.split("if: >-", 1)[1].split())
                expected = (
                    "github.event_name == 'schedule' || "
                    "(github.event_name == 'workflow_dispatch' && "
                    "(github.event.inputs.suite == 'all' || "
                    f"github.event.inputs.suite == '{suite}'))"
                )
                self.assertEqual(
                    condition, (_TRUSTED_PR if accepts_pr else "") + expected
                )

    def test_complete_test_modules_use_the_real_environment(self):
        for job, module in (
            ("cpu-validation", "test_flydsl_moe_cache"),
            ("whole_graph_cache", "test_flydsl_moe_run_only"),
        ):
            body = _block(self.jobs, job, 2)
            with self.subTest(job=job):
                self.assertIn("needs: [ci_config, workflow-guard]", body)
                self.assertIn("needs.ci_config.outputs.pytorch_py312_image", body)
                self.assertIn(
                    "BUILD_TRITON=0 ./.github/scripts/build_aiter_triton.sh", body
                )
                self.assertRegex(
                    body,
                    rf"bash -lc [^\n]*python3 -m unittest [^\n]*"
                    rf"op_tests\.tuning_tests\.{module} -v",
                )
                self.assertIn("set -euo pipefail", body)
                self.assertNotIn("continue-on-error:", body)
                steps = _block(body, "steps", 4)
                for condition in re.findall(r"(?m)^        if: (.+)$", steps):
                    self.assertEqual(condition, "always()")
        guard = _block(self.jobs, "workflow-guard", 2)
        self.assertIn("runs-on: ubuntu-latest", guard)
        self.assertIn(
            "python3 -m unittest op_tests.tuning_tests.test_flydsl_moe_ci -v", guard
        )
        self.assertIn("runtime coverage: SKIPPED", guard)

    def test_integration_uses_two_reserved_gfx942_devices(self):
        gpu = _block(self.jobs, "whole_graph_cache", 2)
        self.assertIn("runs-on: linux-aiter-oci-mi300x-8", gpu)
        runners = (_ROOT / ".github/runner-config.yml").read_text()
        allocation = _block(runners, "linux-aiter-oci-mi300x-8", 2)
        self.assertIn("gpu_arch: MI300X", allocation)
        self.assertIn("gpu_count: 8", allocation)
        self.assertIn("GPU_ARCHS: gfx942", gpu)
        self.assertIn("AITER_TEST_EXPECTED_GFX: gfx942", gpu)
        self.assertIn('-e AITER_TEST_EXPECTED_GFX="${AITER_TEST_EXPECTED_GFX}"', gpu)
        for variable in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            self.assertIn(f"-e {variable}=0,1", gpu)
        self.assertNotIn("gfx950", gpu)
        self.assertNotIn("strategy:", gpu)

    def test_schedule_manual_and_existing_suites_are_preserved(self):
        self.assertIn('cron: "0 20 * * *"', _block(self.events, "schedule", 2))
        dispatch = _block(self.events, "workflow_dispatch", 2)
        for suite in (
            "all",
            "level01",
            "tune_pipeline",
            "run_config",
            "whole_graph_cache",
        ):
            self.assertIn(f"          - {suite}\n", dispatch)
        cpu = _block(self.jobs, "cpu-validation", 2)
        self.assertIn("runs-on: linux-aiter-mi35x-1", cpu)
        for module in (
            "test_csv_validation",
            "test_tuner_infra",
            "test_mp_tuner_logic",
            "test_config_shape_collision",
            "test_a6w6_tuning",
            "test_mixed_mxfp_tuning",
        ):
            self.assertIn(f"op_tests.tuning_tests.{module}", cpu)
        for job, module in (
            ("gpu-pipeline", "test_tune_pipeline"),
            ("gpu-run-config", "test_run_config"),
        ):
            body = _block(self.jobs, job, 2)
            self.assertIn("runs-on: linux-aiter-mi35x-1", body)
            self.assertIn(
                f"python3 -m unittest op_tests.tuning_tests.{module} -v", body
            )

    def test_no_privileged_pr_trigger_or_persisted_checkout_credentials(self):
        self.assertNotIn("pull_request_target:", self.workflow)
        self.assertEqual(
            _block(self.workflow, "permissions", 0).strip(),
            "permissions:\n  contents: read",
        )
        self.assertNotRegex(self.workflow, r"(?m)^ +permissions:")
        checkouts = re.findall(
            r"(?m)^ +(?:- )?uses: actions/checkout@[^\n]+\n((?:^ {8,}.*\n)*)",
            self.workflow,
        )
        self.assertEqual(len(checkouts), 5)
        for checkout in checkouts:
            self.assertIn("persist-credentials: false", checkout)


if __name__ == "__main__":
    unittest.main()
