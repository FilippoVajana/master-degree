from results import ood_plt
from results import shift_plt
from results import train_plt
from results import perf_plots
from results import labeldrop_plots
from results import labeldrop_compare_plots
from results import id_vs_ood_plots
from results import tr_labeldrop_plots
from results import tr_labeldrop_compare_plots
from results import mc_drop_plots
from results.utils import *
import datetime as dt
from typing import Dict, List
import pandas as pd
import numpy as np
import natsort
import argparse
import glob
import logging as log
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["figure.figsize"] = [10, 8]

print("numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__)
print("natsort:", natsort.__version__)
print("pandas:", pd.__version__)

log.basicConfig(level=log.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


if __name__ == "__main__":
    ENABLE_SAVE_FIGURES = True
    RESULTS_DIRECTORY = os.path.relpath(os.path.join(os.getcwd(), 'results'))
    log.info("Results directory: %s", RESULTS_DIRECTORY)

    items = [path for path in os.listdir(RESULTS_DIRECTORY)]
    log.info("Items: %s", items)

    run_dirs = [item for item in items if os.path.isdir(
        os.path.join('results', item)) and not("__" in item)]
    log.info("Runs: %s", run_dirs)

    RUN_ID = natsort.natsorted(run_dirs, reverse=True)[0]
    log.info("Run id: %s", RUN_ID)
    RUN_ID = "0325_1845"

    IMGS_PATH = None

    parser = argparse.ArgumentParser(description="Analyze data.")
    parser.add_argument('-data', type=str, default=None,
                        action='store', help='Data folder name.')
    args = parser.parse_args()

    # set data folders
    res_dir_list = list()
    if args.data is not None:
        IMGS_PATH = os.path.relpath(args.data)
        res_dir_list.append(os.path.relpath(args.data))
        log.info(f"Result directory: {os.path.relpath(args.data)}")
    else:
        IMGS_PATH = os.path.join(RESULTS_DIRECTORY, RUN_ID)
        res_dir_list = glob.glob(f"{IMGS_PATH}/lenet5*")
        log.info(f"Result directories: {res_dir_list}")

    # PLOT
    figures = dict()

    # # train data
    # figures["train_accuracy"] = train_plt.plot_train_accuracy(res_dir_list)
    # figures["train_loss"] = train_plt.plot_train_loss(res_dir_list)
    # figures["train_brier"] = train_plt.plot_train_brier(res_dir_list)
    # figures["train_entropy"] = train_plt.plot_train_entropy(res_dir_list)
    # # plt.show()

    # # shifted data
    # figures["rotated.png"] = shift_plt.plot_rotated(res_dir_list)
    # figures["shifted.png"] = shift_plt.plot_shifted(res_dir_list)
    # figures["conf_acc60.png"] = shift_plt.plot_confidence_vs_accuracy_60(
    #     res_dir_list)
    # figures["count_acc60.png"] = shift_plt.plot_confidence_vs_count_60(
    #     res_dir_list)
    # # plt.show()

    # # ood data
    # figures["ood_entropy.png"] = ood_plt.plot_entropy_ood(res_dir_list)
    # figures["ood_confidence.png"] = ood_plt.plot_confidence_ood(res_dir_list)

    # REVIEW: performance shift plots
    # perf_plots.main(run_id="perf_shift")

    # REVIEW: labeldrop sensitivity plots
    # labeldrop_plots.main(run_id="label_drop", model_prefix="lenet5-")
    # labeldrop_plots.main(run_id="label_drop", model_prefix="lenet5mc")

    # REVIEW: labeldrop sensitivity comparison plots
    # labeldrop_compare_plots.main(run_id="label_drop_compare")

    # REVIEW: id vs ood entropy and confidence
    # id_vs_ood_plots.main(run_id="id_vs_ood")

    # REVIEW: labeldrop vs labeldrop TR
    # tr_labeldrop_compare_plots.main(run_id="tr_compare")

    # REVIEW: labeldrop + TR
    tr_labeldrop_plots.main(run_id="tr_labeldrop")

    # REVIEW: mc dropout
    # mc_drop_plots.main(run_id='mc_drop')

    # log.info("cwd: %s", os.getcwd())
    # if ENABLE_SAVE_FIGURES:
    #     log.info("Saving plots")
    #     # create save folder
    #     log.info("image path: %s", IMGS_PATH)
    #     os.makedirs(IMGS_PATH, exist_ok=True)
    #     # save loop
    #     for fn in figures:
    #         figures[fn].savefig(os.path.join(IMGS_PATH, fn))
