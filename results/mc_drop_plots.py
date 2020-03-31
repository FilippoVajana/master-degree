import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *
from typing import Tuple


def main(run_id="mc_drop", model_prefix="lenet5"):
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
    res_dir_list = glob.glob(f"{IMGS_PATH}/{model_prefix}*")
    log.info(f"Result directories: {res_dir_list}")

    # PLOT
    figures = dict()

    figures[f"mc-rotated.png"] = plot_rotated(res_dir_list)
    figures[f"mc-id-ood.png"] = plot_id_ood(res_dir_list)

    log.info("cwd: %s", os.getcwd())
    if ENABLE_SAVE_FIGURES:
        log.info("Saving plots")
        # create save folder
        log.info("image path: %s", IMGS_PATH)
        os.makedirs(IMGS_PATH, exist_ok=True)
        # save loop
        for fn in figures:
            figures[fn].savefig(os.path.join(IMGS_PATH, fn), dpi=400)


def get_rotated_df(models: List[str]) -> Dict[str, pd.DataFrame]:
    '''Returns a dictionary of dataframes for Accuracy and Brier Score with rotated data.
    '''
    df_dict = dict()

    for model_dir in models:
        sr_list = list()
        # get base data
        df_base = load_csv(os.path.join(model_dir, 'mnist.csv'))
        df_base = df_base[['t_mc_mean', 't_mc_std']].median()

        sr = pd.Series(data={
            'mc_mean': df_base['t_mc_mean'], 'mc_std': df_base['t_mc_std']
        }, name='train')
        sr_list.append(sr)

        # get rotated data
        df_rot = load_rotated(model_dir)
        for df_k in df_rot:
            df_base = df_rot[df_k][['t_mc_mean', 't_mc_std']].median()
            sr = pd.Series(data={
                'mc_mean': df_base['t_mc_mean'], 'mc_std': df_base['t_mc_std']
            }, name=df_k)

            sr_list.append(sr)

        # merge series
        df = pd.DataFrame(columns=['mc_mean', 'mc_std'])
        for sr in sr_list:
            df = df.append(sr, ignore_index=False)

        # add to df dictionary
        df_dict[os.path.basename(model_dir)] = df

    # log.debug(f"Rotated df:\n {df_dict}")
    return df_dict


def plot_id_ood(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot MC dropout mean and std for ID and OOD data")
    # load nomnist dataframe
    ood_df_dict = {
        f"ood_{os.path.basename(path)}": load_csv(os.path.join(path, 'nomnist.csv'))[['t_mc_mean', 't_mc_std']]
        for path in res_dir_list
    }
    # load mnist dataframe
    id_df_dict = {
        f"id_{os.path.basename(path)}": load_csv(os.path.join(path, 'mnist.csv'))[['t_mc_mean', 't_mc_std']]
        for path in res_dir_list
    }

    # plot
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey='row')
    ax11 = axs[0, 0]
    ax12 = axs[0, 1]
    ax21 = axs[1, 0]
    ax22 = axs[1, 1]

    ax11.set_xlim((0, 1))
    ax21.set_xlim((0, 1))
    ax12.set_xlim((0, 0.4))
    ax22.set_xlim((0, 0.4))

    fig.tight_layout(h_pad=None, w_pad=None, rect=[0.025, 0.03, 1, 0.94])
    fig.suptitle("Monte Carlo dropout")
    for ax in axs.flatten():
        ax.grid(True)
        ax.tick_params(grid_linestyle='dotted')

    for k in ood_df_dict:
        ax11.hist(ood_df_dict[k]['t_mc_mean'], linewidth=1.5, linestyle='solid', alpha=0.35)
        ax12.hist(ood_df_dict[k]['t_mc_std'], linewidth=1.5, linestyle='solid', alpha=0.35)

    for k in id_df_dict:
        ax21.hist(id_df_dict[k]['t_mc_mean'], linewidth=1.5, linestyle='solid', alpha=0.35)
        ax22.hist(id_df_dict[k]['t_mc_std'], linewidth=1.5, linestyle='solid', alpha=0.35)

    ax11.set_ylabel('notMNIST')
    ax21.set_ylabel('MNIST')
    ax21.set_xlabel('Confidenza media')
    ax22.set_xlabel('Deviazione standard')
    ax11.legend(labels=['MC LeNet5', 'MC LeNet5 LS-45%'])

    return fig


def plot_rotated(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_rotated")
    # get rotated results
    rotated_df_dict = get_rotated_df(res_dir_list)

    # plot
    formatter = ticker.FormatStrFormatter("%d°")
    fig, ax1 = plt.subplots()
    fig.tight_layout(h_pad=None, w_pad=None, rect=[0.015, 0.03, 1, 0.97])
    fig.suptitle("Monte Carlo dropout (MNIST)")

    xticks = range(0, 195, 15)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 180)
    # ax1.set_ylim(0, 1)

    for k in rotated_df_dict:
        mean = rotated_df_dict[k]['mc_mean']
        std = rotated_df_dict[k]['mc_std']
        ax1.plot(xticks, mean, label=k)
        ax1.fill_between(xticks, mean + std, mean - std, alpha=0.15, linestyle='dashed', edgecolor='black')

    ax1.xaxis.set_major_formatter(formatter)
    ax1.set_ylabel("Confidenza")
    ax1.set_xlabel("Rotazione")
    ax1.legend(labels=["MC LeNet5", "MC LeNet5 LS-45%"])
    return fig
