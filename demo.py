from src.exception import MyException
from src.logger import configure_logger


import sys
import os

# try:
#     x = 1 / 'es'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e
from src.pipeline.training_pipeline import TrainPipeline

pipeline= TrainPipeline()
pipeline.run_pipeline()
