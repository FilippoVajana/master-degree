import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *


def main(run_id="perf_shift", model_prefix="lenet5"):
    N = 7
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
    plt.rcParams.update({'font.size': 14})

    ENABLE_SAVE_FIGURES = True
    RESULTS_DIRECTORY = os.path.relpath(os.path.join(os.getcwd(), 'results'))
    log.info("Results directory: %s", RESULTS_DIRECTORY)
    RUN_ID = run_id

    # set data folders
    res_dir_list = list()
    IMGS_PATH = os.path.join(RESULTS_DIRECTORY, RUN_ID)
    res_dir_list = glob.glob(f"{IMGS_PATH}/{model_prefix}*")
    log.info(f"Result directories: {res_dir_list}")

    # PLOT
    figures = dict()

    # # shifted data
    model_prefix = model_prefix.replace('-', '')
    figures[f"ld-{model_prefix}-comp-rotated.png"] = plot_rotated(res_dir_list)
    figures[f"ld-{model_prefix}-comp-count60.png"] = plot_confidence_vs_count_60(res_dir_list)

    log.info("cwd: %s", os.getcwd())
    if ENABLE_SAVE_FIGURES:
        log.info("Saving plots")
        # create save folder
        log.info("image path: %s", IMGS_PATH)
        os.makedirs(IMGS_PATH, exist_ok=True)
        # save loop
        for fn in figures:
            figures[fn].savefig(os.path.join(IMGS_PATH, fn), dpi=200)


def plot_confidence_vs_count_60(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_confidence_vs_count_60")
    # load rotated 60° dataframes
    df_dict = {os.path.basename(path): load_csv(os.path.join(
        path, 'mnist_rotate60.csv')) for path in res_dir_list}
    res_df = pd.DataFrame()

    confidence_range = np.arange(0, 1, .01)
    for k in df_dict:
        # select data based on confidence value
        count_list = list()
        for cv in confidence_range:
            count_df = df_dict[k].loc[df_dict[k]['t_confidence'] > cv]
            ratio = count_df.iloc[:, 0].count(
            ) / df_dict[k]['t_confidence'].count()
            count_list.append(ratio)

        # save grouped data
        res_df[k] = pd.Series(count_list, index=list(confidence_range))

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Confidenza vs Conteggio (Rotazione 60°)")
    fig.tight_layout(h_pad=None, w_pad=None, rect=[0.04, 0.03, 1, 0.97])

    x_formatter = ticker.FormatStrFormatter("%.2f")
    y_formatter = ticker.PercentFormatter(xmax=1.0)

    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 1)

    for k in res_df:
        ax1.scatter(res_df[k].index, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Frazione di campioni con $p(y|x) > \tau$")
    ax1.set_xlabel(r"Confidenza ($\tau$)")
    ax1.legend(labels=["LeNet5", "LS-15%", "LS-45%", "MC LeNet5", "MC LS-15%", "MC LS-45%"])
    return fig


def plot_rotated(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_rotated")
    # get rotated results
    rotated_df_dict = get_rotated_df(res_dir_list)

    # plot
    formatter = ticker.FormatStrFormatter("%d°")
    fig, ax1 = plt.subplots()
    fig.suptitle("Rotazione (MNIST modificato)")
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
    ax1.set_xlabel("Rotazione")
    ax1.set_ylabel("Accuratezza")
    ax2.set_ylabel("Brier score")
    ax1.legend(loc='upper right', labels=["LeNet5", "LS-15%", "LS-45%", "MC LeNet5", "MC LS-15%", "MC LS-45%"])
    return fig
