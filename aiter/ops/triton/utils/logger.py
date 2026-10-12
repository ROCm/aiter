import logging
import os


# AITER Triton Logger which is singleton object around python logging.
# `*args` is forwarded so callers can use lazy %-style formatting rather than
# building an f-string that is thrown away below the configured level.
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
class AiterTritonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            log_level_str = os.getenv("AITER_TRITON_LOG_LEVEL", "WARNING").upper()
            numeric_level = getattr(logging, log_level_str, logging.WARNING)
            cls._instance._logger = logging.getLogger("AITER_TRITON")
            cls._instance._logger.setLevel(numeric_level)

        return cls._instance

    def get_logger(self):
        return self._logger

    def debug(self, msg, *args):
        self._logger.debug(msg, *args)

    def info(self, msg, *args):
        self._logger.info(msg, *args)

    def warning(self, msg, *args):
        self._logger.warning(msg, *args)

    def error(self, msg, *args):
        self._logger.error(msg, *args)

    def critical(self, msg, *args):
        self._logger.critical(msg, *args)
