import logging
import time
from engine import trainer, tester

# create logger
logger = logging.getLogger(name="engine.logger")
logger.setLevel(lvl=logging.INFO)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(lvl=logging.INFO)
# create formatter
formatter = logging.Formatter('[%(asctime)s]  %(levelname)s: %(message)s', datefmt='%I:%M:%S')
ch.setFormatter(form=formatter)
logger.addHandler(hdlr=ch)