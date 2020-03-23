import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
import results as R
from results.utils import *


def get_union_df(results: list, df_name: str) -> pd.DataFrame:
    '''Loads and concats dataframes filtered by name from result folders.
    '''
    # load dataframes
    df_list = list()
    for path in results:
        df_path = os.path.join(str(os.path.relpath(path)), df_name)
        df = load_csv(df_path)

        # rename columns
        df = df.rename(
            lambda cn: f"{os.path.basename(path)}_{cn}", axis='columns')
        df_list.append(df)

    res_df = pd.concat(df_list, axis=1)
    # log.debug(res_df.columns)
    return res_df


def plot_train_accuracy(res_dir_list: List[str]) -> plt.Figure:
    # load dataframes
    df_union = get_union_df(res_dir_list, "train_logs.csv")
    df_union.drop(list(df_union.filter(regex='epoch')), axis=1, inplace=True)

    # select accuracy columns
    tr_acc_df = pd.DataFrame(df_union.filter(regex=r"t_mean_accuracy$"))
    tr_acc_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)
    va_acc_df = pd.DataFrame(df_union.filter(regex=r"v_mean_accuracy$"))
    va_acc_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)

    # plot
    fig = plt.figure()
    fig.suptitle("Accuracy Value")
    (ax1, ax2) = fig.subplots(nrows=2, sharex=True)
    ax1 = tr_acc_df.plot(xticks=range(0, 20, 1), ax=ax1)
    plt.gca().set_prop_cycle(None)
    ax2 = va_acc_df.plot(xticks=range(0, 20, 1), ax=ax2,
                         legend=False, linestyle='-.')

    ax1.grid(True)
    ax2.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax2.tick_params(grid_linestyle='dotted')

    ax1.set_ylabel("Training Accuracy")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    return fig


def plot_train_loss(res_dir_list: List[str]) -> plt.Figure:
    # load dataframes
    df_union = get_union_df(res_dir_list, "train_logs.csv")
    df_union.drop(list(df_union.filter(regex='epoch')), axis=1, inplace=True)

    # select accuracy columns
    tr_loss_df = pd.DataFrame(df_union.filter(regex=r"t_mean_loss$"))
    tr_loss_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)
    va_loss_df = pd.DataFrame(df_union.filter(regex=r"v_mean_loss$"))
    va_loss_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)

    # plot
    fig = plt.figure()
    fig.suptitle("Loss Value")
    (ax1, ax2) = fig.subplots(nrows=2, sharex=True)
    ax1 = tr_loss_df.plot(xticks=range(0, 20, 2), ax=ax1)
    plt.gca().set_prop_cycle(None)
    ax2 = va_loss_df.plot(xticks=range(0, 20, 2), ax=ax2,
                          legend=False, linestyle='-.')

    ax1.grid(True)
    ax2.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax2.tick_params(grid_linestyle='dotted')

    ax1.set_ylabel("Training Loss")
    ax2.set_ylabel("Validation Loss")
    ax2.set_xlabel("Epoch")

    return fig


def plot_train_brier(res_dir_list: List[str]) -> plt.Figure:
    # load dataframes
    df_union = get_union_df(res_dir_list, "train_logs.csv")
    df_union.drop(list(df_union.filter(regex='epoch')), axis=1, inplace=True)

    # select accuracy columns
    tr_brier_df = pd.DataFrame(df_union.filter(regex=r"t_mean_brier$"))
    tr_brier_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)
    va_brier_df = pd.DataFrame(df_union.filter(regex=r"_v_mean_brier"))
    va_brier_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)
    ov_brier_df = pd.DataFrame(df_union.filter(
        regex=r"ov_mean_brier$")).round(decimals=2)
    ov_brier_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)

    # plot
    fig = plt.figure()
    fig.suptitle("Brier Score")
    (ax1, ax2, ax3) = fig.subplots(nrows=3, sharex=True)

    ax1 = tr_brier_df.plot(xticks=range(0, 20, 2), ax=ax1)
    plt.gca().set_prop_cycle(None)
    ax2 = va_brier_df.plot(xticks=range(0, 20, 2), ax=ax2,
                           legend=False, linestyle='-.')
    plt.gca().set_prop_cycle(None)
    ax3 = ov_brier_df.plot(xticks=range(0, 20, 2), ax=ax3,
                           legend=False, linestyle='dashed')

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax2.tick_params(grid_linestyle='dotted')
    ax3.tick_params(grid_linestyle='dotted')

    ax1.set_ylabel("Training Brier")
    ax2.set_ylabel("Validation Brier")
    ax3.set_ylabel("OOD Validation Brier")
    ax3.set_xlabel("Epoch")

    return fig


def plot_train_entropy(res_dir_list: List[str]) -> plt.Figure:
    # load dataframes
    df_union = get_union_df(res_dir_list, "train_logs.csv")
    df_union.drop(list(df_union.filter(regex='epoch')), axis=1, inplace=True)

    # select accuracy columns
    tr_ent_df = pd.DataFrame(df_union.filter(regex=r"t_mean_entropy$"))
    tr_ent_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)
    va_ent_df = pd.DataFrame(df_union.filter(regex=r"_v_mean_entropy"))
    va_ent_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)
    ov_ent_df = pd.DataFrame(df_union.filter(regex=r"ov_mean_entropy$"))
    ov_ent_df.rename(columns=lambda cn: str(cn).split('_')[0], inplace=True)

    # plot
    fig = plt.figure()
    fig.suptitle("Entropy Value")
    (ax1, ax2, ax3) = fig.subplots(nrows=3, sharex=True)

    ax1 = tr_ent_df.plot(xticks=range(0, 20, 2), ax=ax1)
    plt.gca().set_prop_cycle(None)
    ax2 = va_ent_df.plot(xticks=range(0, 20, 2), ax=ax2,
                         legend=False, linestyle='-.')
    plt.gca().set_prop_cycle(None)
    ax3 = ov_ent_df.plot(xticks=range(0, 20, 2), ax=ax3,
                         legend=False, linestyle='dashed')

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax2.tick_params(grid_linestyle='dotted')
    ax3.tick_params(grid_linestyle='dotted')

    ax1.set_ylabel("Training Entropy")
    ax2.set_ylabel("Validation Entropy")
    ax3.set_ylabel("OOD Validation Entropy")
    ax3.set_xlabel("Epoch")

    return fig
