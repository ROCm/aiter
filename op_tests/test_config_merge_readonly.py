# SPDX-License-Identifier: MIT
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas as pd

from aiter.jit import core


class ConfigMergeReadonlyTest(unittest.TestCase):
    def test_merge_preserves_sources(self):
        cases = [
            ("faster_second", [(1, "a", 10), (2, "b", 8)], [(1, "a", 5)], [5, 8]),
            ("tie", [(1, "a", 10)], [(1, "a", 10)], [10]),
            ("tags", [(1, "a", 10)], [(1, "b", 5)], [5, 10]),
            ("clean", [(1, "a", 10)], [(2, "a", 5)], [5, 10]),
        ]
        for label, left, right, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = root / "aiter" / "configs"
                cfg.mkdir(parents=True)
                name = "test_" + uuid.uuid4().hex + "_tuned"
                (cfg / (name.replace("tuned", "untuned") + ".csv")).write_text("M\n")
                sources = [cfg / "left.csv", cfg / "right.csv"]
                for p, rows in zip(sources, (left, right)):
                    pd.DataFrame(rows, columns=["M", "_tag", "us"]).assign(
                        cu_num=304, gfx="gfx942", kernel=p.stem
                    ).to_csv(p, index=False)
                    p.chmod(0o444)
                original = [p.read_bytes() for p in sources]
                real_to_csv = pd.DataFrame.to_csv

                def guarded_to_csv(
                    frame, path, *args, sources=sources, writer=real_to_csv, **kwargs
                ):
                    # Also catch illegal writes when tests happen to run as root.
                    self.assertNotIn(Path(path), sources)
                    return writer(frame, path, *args, **kwargs)

                output = Path("/tmp/aiter_configs") / (name + ".csv")
                try:
                    with mock.patch.object(
                        core, "AITER_ROOT_DIR", str(root)
                    ), mock.patch.object(pd.DataFrame, "to_csv", guarded_to_csv):
                        for _ in range(2):
                            result = core.AITER_CONFIGS.update_config_files(
                                ":".join(map(str, sources)), name
                            )
                            merged = pd.read_csv(result)
                            self.assertEqual(sorted(merged.us.tolist()), expected)
                            if label == "tie":
                                self.assertEqual(merged.kernel.tolist(), ["left"])
                    self.assertEqual([p.read_bytes() for p in sources], original)
                finally:
                    for suffix in ("", ".lock", ".tmp"):
                        Path(str(output) + suffix).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
