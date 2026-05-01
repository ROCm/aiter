import os
import logging


# AITER Triton Logger which is singleton object around python logging.
# Note: Python logging is also a singleton object, but we want to read the
# env var AITER_LOG_LEVEL once at the beginning. Another alternative is to do
# this in __init__.py. In fact, that's how CK logger is setup. We can look at
# switching to that at some point
#
# AITER_LOG_LEVEL follows python logging levels
#   DEBUG
#   INFO
#   WARNING
#   ERROR
#   CRITICAL
#
# AITER_LOG_MODULE: comma-separated list of module names to filter logs
#   When AITER_LOG_MORE > 0, only logs from specified modules will be output
#   Example: AITER_LOG_MODULE=gemm,moe,attention
#
# AITER_LOG_MORE: enable detailed logging (0=off, 1=on, 2=verbose)
#   When > 0 and AITER_LOG_MODULE is set, logs are filtered by module


class AiterTritonLogger(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AiterTritonLogger, cls).__new__(cls)
            log_level_str = os.getenv("AITER_TRITON_LOG_LEVEL", "WARNING").upper()
            numeric_level = getattr(logging, log_level_str, logging.WARNING)
            cls._instance._logger = logging.getLogger("AITER_TRITON")
            cls._instance._logger.setLevel(numeric_level)

            # Initialize module filtering
            cls._instance._log_more = int(os.getenv("AITER_LOG_MORE", 0))
            modules_str = os.getenv("AITER_LOG_MODULE", "")
            cls._instance._modules = set()
            if modules_str:
                cls._instance._modules = set(m.strip() for m in modules_str.split(",") if m.strip())

        return cls._instance

    def get_logger(self):
        return self._logger

    def _should_log(self, module=None):
        """Check if log should be output based on module filtering."""
        # If AITER_LOG_MORE <= 0, no module filtering
        if self._log_more <= 0:
            return True
        # If no modules specified, allow all
        if not self._modules:
            return True
        # If no module specified, use generic logging
        if module is None:
            return True
        return module in self._modules

    def debug(self, msg, module=None):
        if self._should_log(module):
            prefix = f"[{module}] " if module and self._log_more > 0 else ""
            self._logger.debug(f"{prefix}{msg}")

    def info(self, msg, module=None):
        if self._should_log(module):
            prefix = f"[{module}] " if module and self._log_more > 0 else ""
            self._logger.info(f"{prefix}{msg}")

    def warning(self, msg, module=None):
        if self._should_log(module):
            prefix = f"[{module}] " if module and self._log_more > 0 else ""
            self._logger.warning(f"{prefix}{msg}")

    def error(self, msg, module=None):
        if self._should_log(module):
            prefix = f"[{module}] " if module and self._log_more > 0 else ""
            self._logger.error(f"{prefix}{msg}")

    def critical(self, msg, module=None):
        if self._should_log(module):
            prefix = f"[{module}] " if module and self._log_more > 0 else ""
            self._logger.critical(f"{prefix}{msg}")
