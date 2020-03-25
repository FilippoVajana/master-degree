from typing import List
import results.utils as utils
import results.ood_plots as ood_plt
import results.shifted_plots as shift_plt
import results.train_plots as train_plt

import numpy as np
import logging as log
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["figure.figsize"] = [10, 8]

LOG_MODE = log.INFO
log.basicConfig(level=LOG_MODE,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
LOGGER = log.getLogger()
