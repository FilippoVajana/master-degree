import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *


def main(run_id="tr_compare"):
    N = 4
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
    plt.rcParams.update({'font.size': 14})

    ENABLE_SAVE_FIGURES = True
    RESULTS_DIRECTORY = os.path.relpath(os.path.join(os.getcwd(), 'results'))
    log.info("Results directory: %s", RESULTS_DIRECTORY)
    RUN_ID = run_id

    # set data folders
    res_dir_list = list()
    IMGS_PATH = os.path.join(RESULTS_DIRECTORY, RUN_ID)
    res_dir_list = glob.glob(f"{IMGS_PATH}/lenet5*")
    log.info(f"Result directories: {res_dir_list}")

    # PLOT
    figures = dict()

    # # shifted data
    figures["tr-shifted.png"] = plot_shifted(res_dir_list)
    figures["tr-rotated.png"] = plot_rotated(res_dir_list)

    log.info("cwd: %s", os.getcwd())
    if ENABLE_SAVE_FIGURES:
        log.info("Saving plots")
        # create save folder
        log.info("image path: %s", IMGS_PATH)
        os.makedirs(IMGS_PATH, exist_ok=True)
        # save loop
        for fn in figures:
            figures[fn].savefig(os.path.join(IMGS_PATH, fn), dpi=400)


def plot_shifted(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_shifted")
    # get shifted results
    shifted_df_dict = get_shifted_df(res_dir_list)

    # plot
    formatter = ticker.FormatStrFormatter("%dpx")
    fig, ax1 = plt.subplots()
    fig.suptitle("(MNIST)")
    fig.tight_layout(h_pad=None, w_pad=None, rect=[0.015, 0.03, 0.97, 0.97])

    ax2 = ax1.twinx()
    xticks = range(0, 16, 2)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 14)

    for k in shifted_df_dict:
        ax1.plot(xticks, shifted_df_dict[k]['accuracy'], label=k)
        ax2.plot(xticks, shifted_df_dict[k]['brier_score'], label=k, linestyle='-.', alpha=0.5)

    ax1.xaxis.set_major_formatter(formatter)
    # ax1.set_xlabel("Traslazione")
    ax1.set_ylabel("Accuratezza")
    ax2.set_ylabel("Brier score")
    ax1.legend(loc='upper right', labels=["MC LeNet5", "TL LS-15%", "TL LS-45%"])
    return fig


def plot_rotated(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_rotated")
    # get rotated results
    rotated_df_dict = get_rotated_df(res_dir_list)

    # plot
    formatter = ticker.FormatStrFormatter("%d°")
    fig, ax1 = plt.subplots()
    fig.suptitle("(MNIST)")
    fig.tight_layout(h_pad=None, w_pad=None, rect=[0.015, 0.03, 0.97, 0.97])

    ax2 = ax1.twinx()
    xticks = range(0, 195, 15)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 180)

    for k in rotated_df_dict:
        ax1.plot(xticks, rotated_df_dict[k]['accuracy'], label=k)
        ax2.plot(xticks, rotated_df_dict[k]['brier_score'], label=k, linestyle='-.', alpha=0.5)

    ax1.xaxis.set_major_formatter(formatter)
    # ax1.set_xlabel("Rotazione")
    ax1.set_ylabel("Accuratezza")
    ax2.set_ylabel("Brier score")
    ax1.legend(loc='upper right', labels=["MC LeNet5", "TL LS-15%", "TL LS-45%"])
    return fig
