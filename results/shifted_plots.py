import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *


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
    fig.suptitle("Confidence vs Count \n(Rotated 60°)")
    x_formatter = ticker.FormatStrFormatter("%.2f")
    y_formatter = ticker.PercentFormatter(xmax=1.0)

    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.yaxis.set_major_formatter(y_formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 1)

    for k in res_df:
        ax1.scatter(res_df[k].index, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Fraction of examples with $p(y|x) > \tau$")
    ax1.set_xlabel(r"Confidence ($\tau$)")
    side_legend(ax1)
    return fig


def plot_confidence_vs_accuracy_60(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_confidence_vs_accuracy_60")
    # load rotated 60° dataframes
    df_dict = {os.path.basename(path): load_csv(os.path.join(
        path, 'mnist_rotate60.csv')) for path in res_dir_list}
    res_df = pd.DataFrame()

    X_MAX = .55
    confidence_range = np.arange(0, X_MAX, .01)
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
    fig.suptitle("Confidence vs Accuracy \n(Rotated 60°)")
    formatter = ticker.FormatStrFormatter("%.2f")

    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, X_MAX)

    for k in res_df:
        ax1.scatter(res_df[k].index, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Accuracy on examples $p(y|x) > \tau$")
    ax1.set_xlabel(r"Confidence ($\tau$)")
    side_legend(ax1)
    return fig


def plot_shifted(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_shifted")
    # get shifted results
    shifted_df_dict = get_shifted_df(res_dir_list)

    # plot
    fig = plt.figure()
    (ax1, ax2) = fig.subplots(nrows=2)
    fig.suptitle("Translated\n(MNIST)")
    formatter = ticker.FormatStrFormatter("%dpx")

    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax2.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax2.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 14)
    ax2.set_xlim(0, 14)
    ax1.set_ylim(0, 1)
    xticks = range(0, 16, 2)

    for k in shifted_df_dict:
        ax1.plot(xticks, shifted_df_dict[k]['accuracy'], label=k)
        ax2.plot(xticks, shifted_df_dict[k]['brier_score'], label=k)

    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Brier score")
    ax2.set_xlabel("Pixels translation")
    bottom_legend(ax2, len(shifted_df_dict.keys()))
    return fig


def plot_rotated(res_dir_list: List[str]) -> plt.Figure:
    R.LOGGER.info("plot_rotated")
    # get rotated results
    rotated_df_dict = get_rotated_df(res_dir_list)

    # plot
    fig = plt.figure()
    (ax1, ax2) = fig.subplots(nrows=2)
    fig.suptitle("Rotated\n(MNIST)")
    formatter = ticker.FormatStrFormatter("%d°")

    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax2.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax2.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 180)
    ax2.set_xlim(0, 180)
    ax1.set_ylim(0, 1)
    xticks = range(0, 195, 15)

    for k in rotated_df_dict:
        ax1.plot(xticks, rotated_df_dict[k]['accuracy'], label=k)
        ax2.plot(xticks, rotated_df_dict[k]['brier_score'], label=k)

    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Brier score")
    ax2.set_xlabel("Degrees of rotation")
    bottom_legend(ax2, len(rotated_df_dict.keys()))
    return fig
