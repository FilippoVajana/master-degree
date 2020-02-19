import argparse
import glob
import logging as log
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import natsort
import numpy as np
import pandas as pd
from typing import *

print("numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__)
print("natsort:", natsort.__version__)
print("pandas:", pd.__version__)

log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

LENET5_VANILLA_PATH = os.path.join(os.getcwd(), "data-results", "lenet5")


def load_csv(filename: str):
    '''Loads data from csv file and return as DataFrame.
    '''
    data = pd.read_csv(filename, index_col=[0])
    log.info(f"Loaded csv file: {filename}")
    return data

###########
# TRAINING #

###########
# TESTING #


# DATAFRAME HELPERS
def get_union_df(results: list, df_name: str) -> pd.DataFrame:
    '''Loads and concats dataframes filtered by name from result folders.
    '''
    # load dataframes
    df_list = list()
    for path in results:
        df_path = os.path.join(str(os.path.relpath(path)), df_name)
        df = load_csv(df_path)

        # rename columns
        df = df.rename(lambda cn: f"{os.path.basename(path)}_{cn}", axis='columns')
        df_list.append(df)

    res_df = pd.concat(df_list, axis=1)
    log.debug(res_df.columns)
    return res_df


def load_rotated(directory: str) -> Dict[str, pd.DataFrame]:
    '''Returns a dictionary of dataframes with keys equals to rotation degrees range.
    '''
    # get rotated filenames
    paths = glob.glob(f'{directory}/*rotate*.csv')
    paths = natsort.natsorted(paths)

    # load as pandas
    df_dict = dict()
    re_exp = r'[a-z]*_rotate(\d+).csv'
    rotation_regex = re.compile(re_exp)
    for df_p in paths:
        # get rotation value
        m = rotation_regex.search(df_p)
        rotation = m.groups()[0]
        df_dict[rotation] = load_csv(df_p)

    return df_dict


def load_shifted(directory: str) -> Dict[str, pd.DataFrame]:
    '''Returns a dictionary of dataframes with keys equals to shifted pixels range.
    '''
    # get shifted filenames
    paths = glob.glob(f'{directory}/*shift*.csv')
    paths = natsort.natsorted(paths)

    # load as pandas
    df_dict = dict()
    shift_regex = re.compile(r'[a-z]*_shift(\d+).csv')
    for df_p in paths:
        # get shift value
        m = shift_regex.search(df_p)
        shift = m.groups()[0]
        df_dict[shift] = load_csv(df_p)

    return df_dict


def get_rotated_df(results: List[str]) -> Dict[str, pd.DataFrame]:
    '''Returns a dictionary of dataframes for Accuracy and Brier Score with rotated data.
    '''
    df_dict = dict()

    for res_dir in results:
        sr_list = list()
        # get base data
        df_base = load_csv(os.path.join(res_dir, 'mnist.csv'))
        sr = pd.Series(data={'accuracy': get_accuracy(
            df_base), 'brier_score': get_brier(df_base)}, name='train')
        sr_list.append(sr)

        # get rotated data
        df_rot = load_rotated(res_dir)
        for df_k in df_rot:
            data_dict = {
                'accuracy': get_accuracy(df_rot[df_k]),
                'brier_score': get_brier(df_rot[df_k])
            }
            sr_list.append(pd.Series(data=data_dict, name=df_k))

        # merge series
        df = pd.DataFrame(columns=['accuracy', 'brier_score'])
        for sr in sr_list:
            df = df.append(sr, ignore_index=False)

        # add to df dictionary
        df_dict[os.path.basename(res_dir)] = df

    log.debug(f"Rotated df:\n {df_dict}")
    return df_dict


def get_shifted_df(results: List[str]) -> Dict[str, pd.DataFrame]:
    '''Returns a dictionary of dataframes for Accuracy and Brier Score with shifted data.
    '''
    df_dict = dict()

    for res_dir in results:
        sr_list = list()
        # get base data
        df_base = load_csv(os.path.join(res_dir, 'mnist.csv'))
        sr = pd.Series(data={'accuracy': get_accuracy(
            df_base), 'brier_score': get_brier(df_base)}, name='train')
        sr_list.append(sr)

        # get rotated
        df_sh = load_shifted(res_dir)
        for df_k in df_sh:
            data_dict = {
                'accuracy': get_accuracy(df_sh[df_k]),
                'brier_score': get_brier(df_sh[df_k])
            }
            sr_list.append(pd.Series(data=data_dict, name=df_k))

        # merge series
        df = pd.DataFrame(columns=['accuracy', 'brier_score'])
        for sr in sr_list:
            df = df.append(sr, ignore_index=False)

        # add to df dictionary
        df_dict[os.path.basename(res_dir)] = df

    log.debug(f"Shifted df:\n {df_dict}")
    return df_dict


# METRICS HELPERS
def get_accuracy(df: pd.DataFrame):
    accuracy_row = df['t_good_pred']
    accuracy = accuracy_row.sum() / accuracy_row.count()
    return accuracy


def get_brier(df: pd.DataFrame):
    brier_row = df['t_brier']
    return brier_row.mean()


# PLOTTING
def plot_rotated(res_dir_list: List[str]):
    # get rotated results
    rotated_df_dict = get_rotated_df(res_dir_list)

    # plot
    fig = plt.figure()
    (ax1, ax2) = fig.subplots(nrows=2, sharex=True)
    fig.suptitle("Rotated MNIST")
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
    ax1.legend()
    ax2.set_ylabel("Brier score")
    ax2.set_xlabel("Intensity of Skew")
    #return ax1, ax2



def plot_shifted(res_dir_list: List[str]):
    # get shifted results
    shifted_df_dict = get_shifted_df(res_dir_list)

    # plot
    fig = plt.figure()
    (ax1, ax2) = fig.subplots(nrows=2, sharex=True)
    fig.suptitle("Translated MNIST")
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
    ax1.legend()
    ax2.set_ylabel("Brier score")
    ax2.set_xlabel("Intensity of Skew")
    #return ax1, ax2


def plot_confidence_vs_accuracy_60(res_dir_list: List[str]):
    # load rotated 60° dataframes
    df_dict = {os.path.basename(path):load_csv(os.path.join(path, 'mnist_rotate60.csv')) for path in res_dir_list}
    res_df = pd.DataFrame()

    confidence_range = np.arange(0, 1, .01)
    for k in df_dict:
        # select data based on confidence value
        acc_list = list()
        for cv in confidence_range:
            acc_df = df_dict[k].loc[df_dict[k]['t_confidence'] >= cv]
            accuracy = get_accuracy(acc_df)
            acc_list.append(accuracy)

        # save grouped data
        res_df[k] = pd.Series(acc_list, index=list(confidence_range))

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Confidence vs Acc Rotated 60°")
    formatter = ticker.FormatStrFormatter("%.1f")

    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')
    ax1.set_xlim(0, 1)
    xticks = confidence_range

    for k in res_df:
        ax1.plot(xticks, res_df[k], label=k)

    ax1.set_ylabel(r"Accuracy on examples $p(y|x) \geq \tau$")
    ax1.set_xlabel(r"$\tau$")
    ax1.legend()
    #return ax1



def plot_count_vs_confidence_60(dataset_name: str):
    # load rotated 60° dataframe
    df = load_csv(os.path.join(LENET5_VANILLA_PATH, "mnist_rotate60.csv"))

    confidence_range = np.arange(0, 1, 0.01)
    count_df = df['t_confidence']

    # select data based on confidence value
    conf_count_dict = {}
    for cv in confidence_range:
        count_df = count_df.loc[df['t_confidence'] >= cv]
        count = count_df.count()
        conf_count_dict[cv] = count

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Count vs Acc Rotated 60°")
    formatter = ticker.FormatStrFormatter("%.1f")

    ax1.xaxis.set_major_formatter(formatter)

    ax1.grid(True)

    ax1.tick_params(grid_linestyle='dotted')

    ax1.set_xlim(0, 1)

    xticks = confidence_range

    ax1.plot(xticks, list(conf_count_dict.values()))

    ax1.set_ylabel(r"Number of examples $p(y|x) \geq \tau$")
    ax1.set_xlabel(r"$\tau$")

    return ax1


def plot_entropy_ood(dataset_name: str):
    # load nomnist dataframe
    df = load_csv(os.path.join(LENET5_VANILLA_PATH, "nomnist.csv"))
    ent_df = df['t_entropy']

    # count examples based on entropy value
    ent_range = np.arange(ent_df.min(), ent_df.max()*1,
                          (ent_df.max() - ent_df.min())/25)

    ent_count_dict = {}
    for ev in ent_range:
        count_df = ent_df.loc[df['t_entropy'] >= ev]
        count = count_df.count()
        ent_count_dict[ev] = count

    # ent_df.hist()
    # print(ent_df.describe())

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Entropy on OOD")
    formatter = ticker.FormatStrFormatter("%.1f")

    ax1.xaxis.set_major_formatter(formatter)

    ax1.grid(True)

    ax1.tick_params(grid_linestyle='dotted')

    #ax1.set_xlim(0, .9)
    #ax1.set_ylim(0, ent_df.count())

    xticks = ent_range

    ax1.plot(xticks, list(ent_count_dict.values()))

    ax1.set_ylabel("Number of examples")
    ax1.set_xlabel("Entropy (Nats)")

    return ax1


def plot_confidence_ood(dataset_name: str):
    # load rotated 60° dataframe
    df = load_csv(os.path.join(LENET5_VANILLA_PATH, "nomnist.csv"))

    confidence_range = np.arange(0, 1, 0.01)
    count_df = df['t_confidence']

    # select data based on confidence value
    conf_count_dict = {}
    for cv in confidence_range:
        count_df = count_df.loc[df['t_confidence'] >= cv]
        count = count_df.count()
        conf_count_dict[cv] = count

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Confidence on OOD")
    formatter = ticker.FormatStrFormatter("%.1f")

    ax1.xaxis.set_major_formatter(formatter)

    ax1.grid(True)

    ax1.tick_params(grid_linestyle='dotted')

    ax1.set_xlim(0, 1)

    xticks = confidence_range

    ax1.plot(xticks, list(conf_count_dict.values()))

    ax1.set_ylabel(r"Number of examples $p(y|x) \geq \tau$")
    ax1.set_xlabel(r"$\tau$")

    return ax1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze data.")
    parser.add_argument('-data', type=str, default=None,
                        action='store', help='Data folder name.')
    args = parser.parse_args()

    # results data
    RESULTS_DIRECTORY = os.path.relpath('data-results')
    log.debug(f"Results directory: {RESULTS_DIRECTORY}")

    # set data folders
    res_dir_list = list()
    if args.data is not None:
        res_dir_list.append(os.path.relpath(args.data))
        log.debug(f"Result directory: {os.path.relpath(args.data)}")
    else:
        res_dir_list = [os.path.join(RESULTS_DIRECTORY, path) for path in os.listdir(
            RESULTS_DIRECTORY) if os.path.isdir(os.path.join(RESULTS_DIRECTORY, path))]
        log.debug(f"Result directories: {res_dir_list}")

    # get mnist results
    # mnist_df = get_union_df(res_dir_list, "mnist.csv")

    # get rotated results
    # rotated_df = get_rotated_df(res_dir_list)

    # get shifted results
    # shifted_df = get_shifted_df(res_dir_list)

    # plot
    # plot_shifted(res_dir_list)
    plot_confidence_vs_accuracy_60(res_dir_list)

    plt.show()
