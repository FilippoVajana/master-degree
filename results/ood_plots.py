import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *


def plot_entropy_ood(res_dir_list: List[str]):
    R.LOGGER.info("plot_entropy_ood")
    # load nomnist dataframe
    df_dict = {
        os.path.basename(path): load_csv(os.path.join(path, 'nomnist.csv'))['t_entropy']
        for path in res_dir_list
    }
    res_df = pd.DataFrame()

    # count examples based on entropy value
    ent_range = np.arange(1.9, 2.25, 0.005)
    for k in df_dict:
        count_list = list()
        for ev in ent_range:
            count_df = df_dict[k].loc[df_dict[k] > ev]
            ratio = count_df.count() / df_dict[k].count()
            count_list.append(ratio)

        # save grouped data
        res_df[k] = pd.Series(count_list, index=list(ent_range))

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Entropy \n(Out-of-distribution)")
    x_formatter = ticker.FormatStrFormatter("%.2f")
    y_formatter = ticker.PercentFormatter(xmax=1.0)

    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(min(ent_range), max(ent_range))

    for k in res_df:
        ax1.scatter(res_df[k].index, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Fraction of examples with $H > \tau$")
    ax1.set_xlabel("Entropy (Nats)")
    side_legend(ax1)
    return fig


def plot_confidence_ood(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_confidence_ood")
    # load nomnist dataframe
    df_dict = {os.path.basename(path): load_csv(
        os.path.join(path, 'nomnist.csv')) for path in res_dir_list}
    res_df = pd.DataFrame()

    X_MIN = 0
    confidence_range = np.arange(X_MIN, 1, .01)
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
    fig.suptitle("Confidence \n(Out-of-distribution)")
    x_formatter = ticker.FormatStrFormatter("%.2f")
    y_formatter = ticker.PercentFormatter(xmax=1.0)

    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(X_MIN, 1)
    xticks = confidence_range

    for k in res_df:
        ax1.scatter(xticks, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Fraction of examples with $p(y|x) > \tau$")
    ax1.set_xlabel(r"Confidence ($\tau$)")
    side_legend(ax1)
    return fig
