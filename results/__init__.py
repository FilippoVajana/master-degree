from typing import List
import results.utils as utils
import results.ood_plots as ood_plt
import results.shifted_plots as shift_plt

import numpy as np
import logging as log
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["figure.figsize"] = [10, 8]
# plt.style.use('tableau-colorblind10')
N = 4
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.coolwarm(np.linspace(0, 1, N)))

LOG_MODE = log.INFO
log.basicConfig(level=LOG_MODE,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
LOGGER = log.getLogger()
