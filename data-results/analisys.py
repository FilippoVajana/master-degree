import pandas as pd
import natsort
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import os
import fnmatch
import re
import numpy as np
print("numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__)
print("natsort:", natsort.__version__)
print("pandas:", pd.__version__)


LENET5_VANILLA_PATH = os.path.join(os.getcwd(), "data-results", "lenet5")


def load_csv(filename: str):
    '''Loads data from csv file and return as DataFrame.
    '''
    data = pd.read_csv(filename, index_col=[0])
    return data


# LENET5_VANILLA

# TRAINING


# TESTING
# dataframe helpers
def load_rotated():
    '''Returns a dictionary of pandas datasets for tests with rotated data.
    '''
    # get filenames
    files = [fn for fn in os.listdir(LENET5_VANILLA_PATH) if os.path.isfile(
        os.path.join(LENET5_VANILLA_PATH, fn))]
    paths = fnmatch.filter(files, '*rotate*')
    paths = natsort.natsorted(paths)

    # load as pandas
    degree_regex = re.compile(r'\d+')
    dataframes = {}
    for filename in paths:
        df = load_csv(os.path.join(LENET5_VANILLA_PATH, filename))
        name = degree_regex.findall(filename)[0]
        dataframes[name] = df

    return dataframes


def load_shifted():
    '''Returns a dictionary of pandas datasets for tests with shifted data. tests with shifted data.
    '''
    # get filenames
    files = [fn for fn in os.listdir(LENET5_VANILLA_PATH) if os.path.isfile(
        os.path.join(LENET5_VANILLA_PATH, fn))]
    paths = fnmatch.filter(files, '*shift*')
    paths = natsort.natsorted(paths)

    # load as pandas
    pixel_regex = re.compile(r'\d+')
    dataframes = {}
    for filename in paths:
        df = load_csv(os.path.join(LENET5_VANILLA_PATH, filename))
        name = pixel_regex.findall(filename)[0]
        dataframes[name] = df

    return dataframes


# metrics helpers
def get_accuracy(df: pd.DataFrame):
    accuracy_row = df['t_good_pred']
    accuracy = accuracy_row.sum() / accuracy_row.count()
    return accuracy


def get_brier(df: pd.DataFrame):
    brier_row = df['t_brier']
    return brier_row.mean()


# plotting
def plot_rotated(dataset_name: str):
    accuracy_dict = {}
    brier_dict = {}

    # load dataframes
    original_df = load_csv(os.path.join(LENET5_VANILLA_PATH, "mnist.csv"))
    rotated_dict = load_rotated()

    # get accuracy
    accuracy_dict["0"] = get_accuracy(original_df)
    for key in rotated_dict.keys():
        accuracy_dict[str(key)] = get_accuracy(rotated_dict[key])

    # get brier
    brier_dict["0"] = get_brier(original_df)
    for key in rotated_dict.keys():
        brier_dict[str(key)] = get_brier(rotated_dict[key])

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

    ax1.plot(xticks, list(accuracy_dict.values()))
    ax2.plot(xticks, list(brier_dict.values()))

    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Brier score")
    ax2.set_xlabel("Intensity of Skew")

    return ax1, ax2


def plot_shifted(dataset_name: str):
    accuracy_dict = {}
    brier_dict = {}

    # load dataframes
    original_df = load_csv(os.path.join(LENET5_VANILLA_PATH, "mnist.csv"))
    shifted_dict = load_shifted()

    # get accuracy
    accuracy_dict["0"] = get_accuracy(original_df)
    for key in shifted_dict.keys():
        accuracy_dict[str(key)] = get_accuracy(shifted_dict[key])

    # get brier
    brier_dict["0"] = get_brier(original_df)
    for key in shifted_dict.keys():
        brier_dict[str(key)] = get_brier(shifted_dict[key])

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

    ax1.plot(xticks, list(accuracy_dict.values()))
    ax2.plot(xticks, list(brier_dict.values()))

    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Brier score")
    ax2.set_xlabel("Intensity of Skew")

    return ax1, ax2


def plot_confidence_vs_accuracy_60(dataset_name: str):
    # load rotated 60° dataframe
    df = load_csv(os.path.join(LENET5_VANILLA_PATH, "mnist_rotate60.csv"))

    confidence_range = np.arange(0, 1, .01)
    acc_conf_df = df[['t_good_pred', 't_confidence']]

    # select data based on confidence value
    acc_conf_dict = {}
    for cv in confidence_range:
        acc_df = acc_conf_df.loc[df['t_confidence'] >= cv]
        accuracy = get_accuracy(acc_df)
        acc_conf_dict[cv] = accuracy

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

    ax1.plot(xticks, list(acc_conf_dict.values()))

    ax1.set_ylabel(r"Accuracy on examples $p(y|x) \geq \tau$")
    ax1.set_xlabel(r"$\tau$")

    return ax1


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
    plot_rotated("mnist")
    plot_shifted("mnist")
    plot_confidence_vs_accuracy_60("mnist")
    plot_count_vs_confidence_60("mnist")
    plot_entropy_ood("not-mnist")
    plot_confidence_ood("not-mnist")

    plt.show()
