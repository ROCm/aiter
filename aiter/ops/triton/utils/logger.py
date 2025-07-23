import os
import logging

#AITER Logger which is singleton object around python logging.
#Note: Python logging is also a singleton object, but we want to read the
#env var AITER_LOG_LEVEL once at the beginning.
#
#AITER_LOG_LEVEL follows python logging levels
#   DEBUG
#   INFO
#   WARNING
#   ERROR
#   CRITICAL
class AiterLogger(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AiterLogger, cls).__new__(cls)
            log_level_str = os.getenv('AITER_LOG_LEVEL', 'WARNING').upper()
            numeric_level = getattr(logging, log_level_str, logging.INFO)
            #print(f"{log_level_str}")
            #print(f"{numeric_level}")
            logging.basicConfig(level=numeric_level)
            cls._instance._logger = logging.getLogger("AITER")
            #print(f"{cls._instance._logger}")

        return cls._instance
    
    def get_logger(self):
        return self._logger

    def debug(self, msg):
        #print(f"{self._logger}")
        self._logger.debug(msg)

    def info(self, msg):
        #print(f"{self._logger}")
        self._logger.info(msg)
    
    def warning(self, msg):
        #print(f"{self._logger}")
        self._logger.warning(msg)
    
    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)
        
        


    
