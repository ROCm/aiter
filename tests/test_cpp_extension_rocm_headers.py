import pathlib
import tempfile
import unittest
from unittest.mock import patch

from aiter.jit.utils import cpp_extension


class RocmHeaderDiscoveryTest(unittest.TestCase):
    @staticmethod
    def _write_system_headers(root: pathlib.Path, version=(7, 15)) -> pathlib.Path:
        include = root / "include"
        (include / "hip").mkdir(parents=True)
        (include / "rocprim").mkdir()
        (include / "hip" / "hip_version.h").write_text(
            f"#define HIP_VERSION_MAJOR {version[0]}\n"
            f"#define HIP_VERSION_MINOR {version[1]}\n"
        )
        (include / "rocprim" / "rocprim.hpp").write_text("// test header\n")
        return include

    def test_matching_system_headers_are_discovered(self):
        with tempfile.TemporaryDirectory() as tempdir:
            include = self._write_system_headers(pathlib.Path(tempdir))
            with patch.object(
                cpp_extension, "_SYSTEM_ROCM_INCLUDE", str(include)
            ):
                self.assertEqual(
                    cpp_extension._find_matching_system_rocm_include((7, 15)),
                    str(include),
                )

    def test_mismatched_system_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tempdir:
            include = self._write_system_headers(pathlib.Path(tempdir), (7, 14))
            with patch.object(
                cpp_extension, "_SYSTEM_ROCM_INCLUDE", str(include)
            ):
                self.assertIsNone(
                    cpp_extension._find_matching_system_rocm_include((7, 15))
                )

    def test_system_headers_without_rocprim_are_rejected(self):
        with tempfile.TemporaryDirectory() as tempdir:
            include = self._write_system_headers(pathlib.Path(tempdir))
            (include / "rocprim" / "rocprim.hpp").unlink()
            with patch.object(
                cpp_extension, "_SYSTEM_ROCM_INCLUDE", str(include)
            ):
                self.assertIsNone(
                    cpp_extension._find_matching_system_rocm_include((7, 15))
                )

    def test_include_paths_add_matching_system_headers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = pathlib.Path(tempdir)
            runtime = root / "python-sdk"
            (runtime / "include").mkdir(parents=True)
            system_include = self._write_system_headers(root / "system")
            with patch.object(cpp_extension, "ROCM_HOME", str(runtime)), patch.object(
                cpp_extension, "IS_HIP_EXTENSION", True
            ), patch.object(cpp_extension, "ROCM_VERSION", (7, 15)), patch.object(
                cpp_extension, "_find_rocm_devel_include", return_value=None
            ), patch.object(
                cpp_extension, "_SYSTEM_ROCM_INCLUDE", str(system_include)
            ):
                paths = cpp_extension.include_paths(cuda=True)

        self.assertIn(str(system_include), paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
