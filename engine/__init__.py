import logging
import time
from tqdm import tqdm

from engine.models import lenet


# create logger
logger = logging.getLogger(name="engine.logger")
logger.setLevel(logging.INFO)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('[%(asctime)s]  %(levelname)s: %(message)s', datefmt='%I:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)