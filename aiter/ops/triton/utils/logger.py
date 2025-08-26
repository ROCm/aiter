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
class AiterTritonLogger(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AiterTritonLogger, cls).__new__(cls)
            log_level_str = os.getenv("AITER_TRITON_LOG_LEVEL", "WARNING").upper()
            numeric_level = getattr(logging, log_level_str, logging.WARNING)
            cls._instance._logger = logging.getLogger("AITER_TRITON")
            cls._instance._logger.setLevel(numeric_level)

        return cls._instance

    def get_logger(self):
        return self._logger

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)
