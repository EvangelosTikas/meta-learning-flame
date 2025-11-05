import logging
from typing import Any


class MLogger(logging.Logger):

    def __init__(self, loggerid: Any = 0, level = logging.DEBUG):
        self.loggerid = loggerid
        self.level    = level
        self.logger   = None
    def create_logger(self, module_name):
        # create logger
        logger = logging.getLogger(module_name)
        logger.setLevel(self.level)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logger.level)

        # create formatter
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        self.logger = logger
        return logger
