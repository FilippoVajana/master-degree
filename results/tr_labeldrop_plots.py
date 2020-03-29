import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *


def main(run_id="tr_labeldrop"):
    N = 3
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, N)))
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
    figures["tr-entropy.png"] = plot_entropy_ood(res_dir_list)
    figures["tr-acc60.png"] = plot_confidence_vs_accuracy_60(res_dir_list)

    log.info("cwd: %s", os.getcwd())
    if ENABLE_SAVE_FIGURES:
        log.info("Saving plots")
        # create save folder
        log.info("image path: %s", IMGS_PATH)
        os.makedirs(IMGS_PATH, exist_ok=True)
        # save loop
        for fn in figures:
            figures[fn].savefig(os.path.join(IMGS_PATH, fn), dpi=200)


def plot_confidence_vs_accuracy_60(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_confidence_vs_accuracy_60")
    # load rotated 60° dataframes
    df_dict = {os.path.basename(path): load_csv(os.path.join(
        path, 'mnist_rotate60.csv')) for path in res_dir_list}
    res_df = pd.DataFrame()

    X_MAX = 0.5
    confidence_range = np.arange(0, X_MAX, .007)
    for k in df_dict:
        # select data based on confidence value
        acc_list = list()
        for cv in confidence_range:
            acc_df = df_dict[k].loc[df_dict[k]['t_confidence'] > cv]
            accuracy = get_accuracy(acc_df)
            acc_list.append(accuracy)

        # save grouped data
        res_df[k] = pd.Series(acc_list, index=list(confidence_range))

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Confidenza vs Accuratezza (Rotazione 60°)")
    formatter = ticker.FormatStrFormatter("%.2f")

    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, X_MAX)

    for k in res_df:
        ax1.scatter(res_df[k].index, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Accuratezza campioni con $p(y|x) > \tau$")
    ax1.set_xlabel(r"Confidenza ($\tau$)")
    ax1.legend(labels=["MC LeNet5", "LS 15%", "LS 45%"])
    return fig


def plot_entropy_ood(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_entropy_ood")
    # load nomnist dataframe
    df_dict = {
        os.path.basename(path): load_csv(os.path.join(path, 'nomnist.csv'))['t_entropy']
        for path in res_dir_list
    }
    res_df = pd.DataFrame()

    # count examples based on entropy value
    ent_range = np.arange(2.0, 2.31, 0.0025)
    for k in df_dict:
        count_list = list()
        for ev in ent_range:
            count_df = df_dict[k].loc[df_dict[k] < ev]
            ratio = count_df.count() / df_dict[k].count()
            count_list.append(ratio)

        # save grouped data
        res_df[k] = pd.Series(count_list, index=list(ent_range))

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Entropia (notMNIST)")
    x_formatter = ticker.FormatStrFormatter("%.2f")
    y_formatter = ticker.PercentFormatter(xmax=1.0)

    ax1.set_xlim(min(ent_range), max(ent_range))
    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')

    for k in res_df:
        ax1.scatter(res_df[k].index, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Frazione di campioni con $H < \tau$")
    ax1.set_xlabel("Entropia (Nats)")
    plt.legend(loc='upper left', labels=["MC LeNet5", "LS 15%", "LS 45%"])
    return fig
