from src.exception import MyException
from src.logger import logging

import sys

try:
    x = 1 / 'es'
except Exception as e:
    logging.info(e)
    raise MyException(e, sys) from e
