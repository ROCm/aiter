import json
import unittest

from monitor import snapshot
from observe_build import pending_builds


class DiagnosticsTests(unittest.TestCase):
    def test_no_build(self):
        self.assertEqual(pending_builds("HANG detected"), set())

    def test_unfinished_build(self):
        self.assertEqual(pending_builds("start build [module_gemm]"), {"module_gemm"})

    def test_finished_build(self):
        self.assertEqual(
            pending_builds("start build [a]\n\x1b[32mfinish build [a], cost 236.5s"),
            set(),
        )

    def test_concurrent_builds(self):
        self.assertEqual(
            pending_builds("start build [a]\nstart build [b]\nfinish build [a]"), {"b"}
        )

    def test_rebuild(self):
        self.assertEqual(
            pending_builds("start build [a]\nfinish build [a]\nstart build [a]"), {"a"}
        )

    def test_snapshot_serializes(self):
        result = snapshot()
        json.dumps(result)
        self.assertGreater(result["time"], 0)
        self.assertIn("cpu.stat", result["cgroup"])


if __name__ == "__main__":
    unittest.main()
