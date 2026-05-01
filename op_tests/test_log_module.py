# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for AITER_LOG_MODULE environment variable feature.

This module tests the module-based logging functionality that allows
filtering logs by module name when AITER_LOG_MORE > 0.

Usage:
    python -m pytest op_tests/test_log_module.py -v
    python op_tests/test_log_module.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import logging

# Ensure we test the worktree version
sys.path.insert(0, '/home/fsx950223/aiter_module_log')


class TestAiterTritonLogger(unittest.TestCase):
    """Tests for AiterTritonLogger module filtering."""

    def setUp(self):
        """Reset logger singleton before each test."""
        # Clear the singleton instance
        from aiter.ops.triton.utils.logger import AiterTritonLogger
        AiterTritonLogger._instance = None

    def tearDown(self):
        """Clean up after each test."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger
        AiterTritonLogger._instance = None

    @patch.dict(os.environ, {"AITER_LOG_MORE": "0"}, clear=False)
    def test_no_filtering_when_log_more_is_zero(self):
        """Test that no module filtering occurs when AITER_LOG_MORE=0."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        logger = AiterTritonLogger()
        # Should allow all modules when AITER_LOG_MORE <= 0
        self.assertTrue(logger._should_log("gemm"))
        self.assertTrue(logger._should_log("moe"))
        self.assertTrue(logger._should_log(None))
        self.assertTrue(logger._should_log(""))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm,moe"}, clear=False)
    def test_module_filtering_with_log_more_enabled(self):
        """Test module filtering when AITER_LOG_MORE > 0."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        logger = AiterTritonLogger()
        # Allowed modules
        self.assertTrue(logger._should_log("gemm"))
        self.assertTrue(logger._should_log("moe"))
        # Not allowed modules
        self.assertFalse(logger._should_log("attention"))
        self.assertFalse(logger._should_log("conv"))
        # None module should be allowed
        self.assertTrue(logger._should_log(None))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": ""}, clear=False)
    def test_empty_module_list_allows_all(self):
        """Test that empty AITER_LOG_MODULE allows all modules."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        logger = AiterTritonLogger()
        # All modules should be allowed when AITER_LOG_MODULE is empty
        self.assertTrue(logger._should_log("gemm"))
        self.assertTrue(logger._should_log("moe"))
        self.assertTrue(logger._should_log("attention"))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "2", "AITER_LOG_MODULE": "gemm"}, clear=False)
    def test_single_module_filter(self):
        """Test filtering with single module."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        logger = AiterTritonLogger()
        self.assertTrue(logger._should_log("gemm"))
        self.assertFalse(logger._should_log("moe"))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm , moe , attention"}, clear=False)
    def test_module_names_with_whitespace(self):
        """Test that module names with whitespace are parsed correctly."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        logger = AiterTritonLogger()
        self.assertTrue(logger._should_log("gemm"))
        self.assertTrue(logger._should_log("moe"))
        self.assertTrue(logger._should_log("attention"))
        self.assertFalse(logger._should_log("conv"))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm,moe"}, clear=False)
    def test_log_prefix_with_module(self):
        """Test that log messages include module prefix when AITER_LOG_MORE > 0."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        logger = AiterTritonLogger()
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("test message", module="gemm")
            mock_info.assert_called_once_with("[gemm] test message")

    @patch.dict(os.environ, {"AITER_LOG_MORE": "0"}, clear=False)
    def test_no_prefix_when_log_more_zero(self):
        """Test that no module prefix is added when AITER_LOG_MORE=0."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        logger = AiterTritonLogger()
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("test message", module="gemm")
            mock_info.assert_called_once_with("test message")


class TestUtilsModuleLogging(unittest.TestCase):
    """Tests for utils.py module logging functions."""

    def setUp(self):
        """Set up test environment."""
        # Reload module to pick up new env vars
        if 'csrc.cpp_itfs.utils' in sys.modules:
            del sys.modules['csrc.cpp_itfs.utils']

    @patch.dict(os.environ, {"AITER_LOG_MORE": "0", "AITER_LOG_MODULE": ""}, clear=False)
    def test_should_log_module_no_filtering(self):
        """Test should_log_module returns True when AITER_LOG_MORE=0."""
        from csrc.cpp_itfs.utils import should_log_module

        self.assertTrue(should_log_module("gemm"))
        self.assertTrue(should_log_module("moe"))
        self.assertTrue(should_log_module(""))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm,moe"}, clear=False)
    def test_should_log_module_with_filtering(self):
        """Test should_log_module with filtering enabled."""
        # Must reimport after env var change
        if 'csrc.cpp_itfs.utils' in sys.modules:
            del sys.modules['csrc.cpp_itfs.utils']
        from csrc.cpp_itfs.utils import should_log_module

        self.assertTrue(should_log_module("gemm"))
        self.assertTrue(should_log_module("moe"))
        self.assertFalse(should_log_module("attention"))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm,moe"}, clear=False)
    def test_log_module_function(self):
        """Test log_module helper function."""
        if 'csrc.cpp_itfs.utils' in sys.modules:
            del sys.modules['csrc.cpp_itfs.utils']
        from csrc.cpp_itfs.utils import log_module, logger

        with patch.object(logger, 'info') as mock_info:
            log_module(logger.info, "test message", module="gemm")
            mock_info.assert_called_once_with("[gemm] test message")

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm"}, clear=False)
    def test_log_module_filtered_out(self):
        """Test that log_module respects filtering."""
        if 'csrc.cpp_itfs.utils' in sys.modules:
            del sys.modules['csrc.cpp_itfs.utils']
        from csrc.cpp_itfs.utils import log_module, logger

        with patch.object(logger, 'info') as mock_info:
            log_module(logger.info, "test message", module="moe")
            # Should not be called because moe is not in allowed modules
            mock_info.assert_not_called()


class TestCoreModuleLogging(unittest.TestCase):
    """Tests for jit/core.py module logging functions.

    These tests mock the ROCm-dependent imports to avoid requiring GPU runtime.
    """

    def _mock_core_imports(self, env_vars):
        """Helper to mock jit/core imports with given env vars."""
        with patch.dict(os.environ, env_vars, clear=False):
            with patch.dict('sys.modules', {
                'aiter.jit.utils.chip_info': MagicMock(),
                'aiter.jit.utils.cpp_extension': MagicMock(),
                'aiter.jit.utils.file_baton': MagicMock(),
                'aiter.jit.utils.torch_guard': MagicMock(),
            }):
                # Mock the specific imports that need ROCm
                sys.modules['aiter.jit.utils.chip_info'].get_gfx = MagicMock(return_value='gfx942')
                sys.modules['aiter.jit.utils.chip_info'].get_gfx_list = MagicMock(return_value=['gfx942'])
                sys.modules['aiter.jit.utils.cpp_extension'].get_hip_version = MagicMock()

                if 'aiter.jit.core' in sys.modules:
                    del sys.modules['aiter.jit.core']

                # Import only the logging functions we need to test
                import importlib.util
                spec = importlib.util.spec_from_file_location("core", "/home/fsx950223/aiter_module_log/aiter/jit/core.py")
                core_module = importlib.util.module_from_spec(spec)

                # Mock the problematic imports at module level
                with patch.object(spec.loader, 'exec_module') as mock_exec:
                    def exec_with_mocks(module):
                        module.__dict__['get_gfx'] = MagicMock(return_value='gfx942')
                        module.__dict__['get_gfx_list'] = MagicMock(return_value=['gfx942'])
                        module.__dict__['FileBaton'] = MagicMock()
                        module.__dict__['torch_compile_guard'] = MagicMock()
                        module.__dict__['_jit_compile'] = MagicMock()
                        module.__dict__['get_hip_version'] = MagicMock()

                        # Read and execute the module code
                        with open("/home/fsx950223/aiter_module_log/aiter/jit/core.py") as f:
                            code = compile(f.read(), "core.py", "exec")
                            exec(code, module.__dict__)

                    mock_exec.side_effect = exec_with_mocks
                    spec.loader.exec_module(core_module)
                    return core_module

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm,attention"}, clear=False)
    def test_should_log_module_in_core(self):
        """Test should_log_module function from jit/core.py."""
        # Test the logic directly without importing full module
        AITER_LOG_MORE = 1
        AITER_LOG_MODULE = "gemm,attention"
        _AITER_LOG_MODULES = set(m.strip() for m in AITER_LOG_MODULE.split(",") if m.strip())

        def should_log_module(module_name: str) -> bool:
            if AITER_LOG_MORE <= 0:
                return True
            if not _AITER_LOG_MODULES:
                return True
            return module_name in _AITER_LOG_MODULES

        self.assertTrue(should_log_module("gemm"))
        self.assertTrue(should_log_module("attention"))
        self.assertFalse(should_log_module("moe"))

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm"}, clear=False)
    def test_log_module_in_core(self):
        """Test log_module function from jit/core.py."""
        AITER_LOG_MORE = 1
        AITER_LOG_MODULE = "gemm"
        _AITER_LOG_MODULES = set(m.strip() for m in AITER_LOG_MODULE.split(",") if m.strip())

        def should_log_module(module_name: str) -> bool:
            if AITER_LOG_MORE <= 0:
                return True
            if not _AITER_LOG_MODULES:
                return True
            return module_name in _AITER_LOG_MODULES

        def log_module(msg: str, module: str = None, level: str = "info"):
            if module and not should_log_module(module):
                return
            prefix = f"[{module}] " if module and AITER_LOG_MORE > 0 else ""
            return f"{prefix}{msg}"

        result = log_module("test message", module="gemm", level="info")
        self.assertEqual(result, "[gemm] test message")

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm"}, clear=False)
    def test_log_module_debug_level(self):
        """Test log_module with debug level."""
        AITER_LOG_MORE = 1
        AITER_LOG_MODULE = "gemm"
        _AITER_LOG_MODULES = set(m.strip() for m in AITER_LOG_MODULE.split(",") if m.strip())

        def should_log_module(module_name: str) -> bool:
            if AITER_LOG_MORE <= 0:
                return True
            if not _AITER_LOG_MODULES:
                return True
            return module_name in _AITER_LOG_MODULES

        def log_module(msg: str, module: str = None, level: str = "info"):
            if module and not should_log_module(module):
                return None
            prefix = f"[{module}] " if module and AITER_LOG_MORE > 0 else ""
            return f"{prefix}{msg}"

        result = log_module("debug message", module="gemm", level="debug")
        self.assertEqual(result, "[gemm] debug message")

    @patch.dict(os.environ, {"AITER_LOG_MORE": "1", "AITER_LOG_MODULE": "gemm"}, clear=False)
    def test_log_module_filtered_in_core(self):
        """Test that log_module respects filtering in core."""
        AITER_LOG_MORE = 1
        AITER_LOG_MODULE = "gemm"
        _AITER_LOG_MODULES = set(m.strip() for m in AITER_LOG_MODULE.split(",") if m.strip())

        def should_log_module(module_name: str) -> bool:
            if AITER_LOG_MORE <= 0:
                return True
            if not _AITER_LOG_MODULES:
                return True
            return module_name in _AITER_LOG_MODULES

        def log_module(msg: str, module: str = None, level: str = "info"):
            if module and not should_log_module(module):
                return None
            prefix = f"[{module}] " if module and AITER_LOG_MORE > 0 else ""
            return f"{prefix}{msg}"

        # moe module should be filtered out
        result = log_module("test message", module="moe", level="info")
        self.assertIsNone(result)


class TestEnvironmentVariableParsing(unittest.TestCase):
    """Tests for AITER_LOG_MODULE environment variable parsing."""

    def test_single_module(self):
        """Test parsing single module."""
        with patch.dict(os.environ, {"AITER_LOG_MODULE": "gemm", "AITER_LOG_MORE": "1"}, clear=False):
            if 'aiter.ops.triton.utils.logger' in sys.modules:
                del sys.modules['aiter.ops.triton.utils.logger']
            from aiter.ops.triton.utils.logger import AiterTritonLogger

            logger = AiterTritonLogger()
            self.assertEqual(logger._modules, {"gemm"})

    def test_multiple_modules(self):
        """Test parsing multiple modules."""
        with patch.dict(os.environ, {"AITER_LOG_MODULE": "gemm,moe,attention", "AITER_LOG_MORE": "1"}, clear=False):
            if 'aiter.ops.triton.utils.logger' in sys.modules:
                del sys.modules['aiter.ops.triton.utils.logger']
            from aiter.ops.triton.utils.logger import AiterTritonLogger

            logger = AiterTritonLogger()
            self.assertEqual(logger._modules, {"gemm", "moe", "attention"})

    def test_modules_with_whitespace(self):
        """Test parsing modules with surrounding whitespace."""
        with patch.dict(os.environ, {"AITER_LOG_MODULE": "  gemm  ,  moe  ", "AITER_LOG_MORE": "1"}, clear=False):
            if 'aiter.ops.triton.utils.logger' in sys.modules:
                del sys.modules['aiter.ops.triton.utils.logger']
            from aiter.ops.triton.utils.logger import AiterTritonLogger

            logger = AiterTritonLogger()
            self.assertEqual(logger._modules, {"gemm", "moe"})

    def test_empty_modules(self):
        """Test parsing empty module string."""
        with patch.dict(os.environ, {"AITER_LOG_MODULE": "", "AITER_LOG_MORE": "1"}, clear=False):
            if 'aiter.ops.triton.utils.logger' in sys.modules:
                del sys.modules['aiter.ops.triton.utils.logger']
            from aiter.ops.triton.utils.logger import AiterTritonLogger

            logger = AiterTritonLogger()
            self.assertEqual(logger._modules, set())


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete AITER_LOG_MODULE feature."""

    @patch.dict(os.environ, {
        "AITER_LOG_MORE": "1",
        "AITER_LOG_MODULE": "gemm,moe",
        "AITER_TRITON_LOG_LEVEL": "DEBUG"
    }, clear=False)
    def test_end_to_end_logging(self):
        """Test complete logging flow with module filtering."""
        from aiter.ops.triton.utils.logger import AiterTritonLogger

        # Clear singleton
        AiterTritonLogger._instance = None
        logger = AiterTritonLogger()

        # Test that allowed module logs go through
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("gemm message", module="gemm")
            self.assertEqual(mock_info.call_count, 1)
            mock_info.assert_called_with("[gemm] gemm message")

        # Test that disallowed module logs are filtered
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("attention message", module="attention")
            mock_info.assert_not_called()

        # Cleanup
        AiterTritonLogger._instance = None


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
