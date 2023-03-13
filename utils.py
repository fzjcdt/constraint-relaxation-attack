import os
import logging
from typing import List


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """

    def __init__(self, path: str):
        if not os.path.exists('./log'):
            os.makedirs('./log')
        self.logger = logging.getLogger()
        self.path = path
        self.setup_file_logger()

    def setup_file_logger(self):
        hdlr = logging.FileHandler(self.path, 'w+')
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message: str):
        self.logger.info(message)

    def new_line(self):
        self.logger.info('')


def get_model_ids(dataset: str, l_norm: str) -> List[str]:
    file_name = os.path.join('model_ids/', dataset + '_' + l_norm + '.txt')
    model_ids = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '' and not line.startswith('#'):
                model_ids.append(line.strip())

    return model_ids
