import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *


def main(run_id="id_vs_ood"):
    N = 3
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
    figures["id_ood_entropy2.png"] = plot_entropy2(res_dir_list)
    figures["id_ood_confidence.png"] = plot_confidence(res_dir_list)

    # plt.show()
    log.info("cwd: %s", os.getcwd())
    if ENABLE_SAVE_FIGURES:
        log.info("Saving plots")
        # create save folder
        log.info("image path: %s", IMGS_PATH)
        os.makedirs(IMGS_PATH, exist_ok=True)
        # save loop
        for fn in figures:
            figures[fn].savefig(os.path.join(IMGS_PATH, fn), dpi=400)


def plot_entropy2(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_entropy_ood")
    # load nomnist dataframe
    df_ood_dict = {
        os.path.basename(path): load_csv(os.path.join(path, 'nomnist.csv'))['t_entropy']
        for path in res_dir_list
    }
    df_id_dict = {
        f"{os.path.basename(path)}": load_csv(os.path.join(path, 'mnist.csv'))['t_entropy']
        for path in res_dir_list
    }

    # plot
    fig = plt.figure()
    ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharey=True)
    fig.tight_layout(h_pad=None, w_pad=None, rect=[0.025, 0.03, 1, 0.94])
    # fig.suptitle("Entropia")
    x_formatter = ticker.FormatStrFormatter("%.2f")

    ax1.set_xlim(2, 2.5)
    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax2.grid(True)
    ax2.tick_params(grid_linestyle='dotted')

    id_keys = list(df_id_dict.keys())
    ood_keys = list(df_ood_dict.keys())

    ax1.hist(df_id_dict[id_keys[0]], linewidth=1.5, alpha=0.35, label='MNIST')
    ax1.hist(df_ood_dict[ood_keys[0]], histtype='step', linewidth=1, edgecolor='black', linestyle='dashed', alpha=.65, label='notMNIST')

    ax2.hist(df_id_dict[id_keys[1]], linewidth=1.5, alpha=0.35, label='MNIST')
    ax2.hist(df_ood_dict[ood_keys[1]], histtype='step', linewidth=1, edgecolor='black', linestyle='dashed', alpha=.65, label='notMNIST')

    ax1.set_title('LeNet5')
    ax1.set_ylabel(r"Numero di campioni")
    ax1.set_xlabel("Entropia")

    ax2.set_title('LeNet5 MC dropout')
    ax2.set_xlabel("Entropia")
    # ax2.legend(labels=['notMNIST', 'MNIST'])
    ax2.legend()
    return fig


def plot_confidence(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_confidence_ood")
    # load nomnist dataframe
    df_ood_dict = {
        os.path.basename(path):
        load_csv(os.path.join(path, 'nomnist.csv')) for path in res_dir_list
    }
    df_id_dict = {
        os.path.basename(path):
        load_csv(os.path.join(path, 'mnist.csv')) for path in res_dir_list
    }

    ood_res_df = pd.DataFrame()
    id_res_df = pd.DataFrame()

    X_MIN = 0
    confidence_range = np.arange(X_MIN, 1, .01)

    # add OOD
    for k in df_ood_dict:
        # select data based on confidence value
        count_list = list()
        for cv in confidence_range:
            count_df = df_ood_dict[k].loc[df_ood_dict[k]['t_confidence'] > cv]
            ratio = count_df.iloc[:, 0].count(
            ) / df_ood_dict[k]['t_confidence'].count()
            count_list.append(ratio)

        # save grouped data
        ood_res_df[k] = pd.Series(count_list, index=list(confidence_range))

    # add ID
    for k in df_id_dict:
        # select data based on confidence value
        count_list = list()
        for cv in confidence_range:
            count_df = df_id_dict[k].loc[df_id_dict[k]['t_confidence'] > cv]
            ratio = count_df.iloc[:, 0].count(
            ) / df_id_dict[k]['t_confidence'].count()
            count_list.append(ratio)

        # save grouped data
        id_res_df[k] = pd.Series(count_list, index=list(confidence_range))

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.tight_layout(h_pad=None, w_pad=None, rect=[0.03, 0.03, 1, 0.94])
    fig.suptitle("(MNIST e notMNIST)")
    x_formatter = ticker.FormatStrFormatter("%.2f")
    y_formatter = ticker.PercentFormatter(xmax=1.0)

    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(X_MIN, 1)
    xticks = confidence_range

    for k in ood_res_df:
        ax1.scatter(xticks, ood_res_df[k], label=k, s=8)
    for k in id_res_df:
        ax1.plot(id_res_df[k], label=k, linestyle="-.", linewidth=1.5, alpha=.5)

    ax1.set_ylabel(r"Frazione di campioni con $p(y|x) > \tau$")
    ax1.set_xlabel(r"Confidenza ($\tau$)")
    plt.legend(labels=["LeNet5 ID", "MC LeNet5 ID", "LeNet5 OOD", "MC LeNet5 OOD"])
    return fig
