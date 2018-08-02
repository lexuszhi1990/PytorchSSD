# -*- coding: utf-8 -*-

import logging
import time
from pathlib import Path

def setup_logger(file_path):
    log_path = '%s-%s.log' % (file_path, "%s"%(time.strftime("%Y-%m-%d-%H-%M")))
    logging.basicConfig(level=logging.DEBUG,
                        handlers = [
                            logging.StreamHandler(),
                            logging.FileHandler(log_path)
                        ])
    logging.info('create log file: %s' % log_path)
