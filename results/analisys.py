import pandas as pd
import natsort
import matplotlib.pyplot as plt
import matplotlib
import os
import fnmatch
import re
import numpy as np
print("numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__)
print("natsort:", natsort.__version__)
print("pandas:", pd.__version__)


LENET5_VANILLA_PATH = os.path.join(os.getcwd(), "results", "lenet5")


def load_csv(filename: str):
    '''Loads data from csv file and return as DataFrame.
    '''
    data = pd.read_csv(filename, index_col=[0])
    return data

# LENET5_VANILLA
# TRAINING


# load train logs dataframe
lenet5V_train_df = load_csv(os.path.join(
    LENET5_VANILLA_PATH, "train_logs.csv"))
lenet5V_train_df.head()

# LENET5_VANILLA
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
    degree_regex = re.compile('\d+')
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
    pixel_regex = re.compile('\d+')
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
    accuracy_dict["R0"] = get_accuracy(original_df)
    for key in rotated_dict.keys():
        accuracy_dict[f"R{key}"] = get_accuracy(rotated_dict[key])

    # get brier
    brier_dict["R0"] = get_brier(original_df)
    for key in rotated_dict.keys():
        brier_dict[f"R{key}"] = get_brier(rotated_dict[key])

    # plot
    xlabels = [f"{v}°" for v in range(0, 195, 15)]

    plt.subplot(211)
    plt.scatter(x=xlabels, y=accuracy_dict.values())
    plt.title(str.upper(dataset_name))
    plt.xlabel("Intensity of Skew")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")

    plt.subplot(212)
    plt.scatter(x=xlabels, y=brier_dict.values())
    plt.title(str.upper(dataset_name))
    plt.xlabel("Intensity of Skew")
    plt.xticks(rotation=45)
    plt.ylabel("Brier")

    plt.subplots_adjust(hspace=1.5)
    plt.show()


def plot_shifted(dataset_name: str):
    accuracy_dict = {}
    brier_dict = {}

    # load dataframes
    original_df = load_csv(os.path.join(LENET5_VANILLA_PATH, "mnist.csv"))
    shifted_dict = load_shifted()

    # get accuracy
    accuracy_dict["R0"] = get_accuracy(original_df)
    for key in shifted_dict.keys():
        accuracy_dict[f"R{key}"] = get_accuracy(shifted_dict[key])

    # get brier
    brier_dict["R0"] = get_brier(original_df)
    for key in shifted_dict.keys():
        brier_dict[f"P{key}"] = get_brier(shifted_dict[key])

    # plot
    xlabels = [f"{v}px" for v in range(0, 16, 2)]

    plt.subplot(211)
    plt.scatter(x=xlabels, y=accuracy_dict.values())
    plt.title(str.upper(dataset_name))
    plt.xlabel("Intensity of Skew")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")

    plt.subplot(212)
    plt.scatter(x=xlabels, y=brier_dict.values())
    plt.title(str.upper(dataset_name))
    plt.xlabel("Intensity of Skew")
    plt.xticks(rotation=45)
    plt.ylabel("Brier")

    plt.subplots_adjust(hspace=1.5)
    plt.show()


def plot_confidence_vs_accuracy_60(dataset_name: str):
    # load rotated 60° dataframe
    df = load_csv(os.path.join(LENET5_VANILLA_PATH, "mnist_rotate60.csv"))

    confidence_range = np.arange(0, 1, 0.1)
    acc_conf_df = df[['t_good_pred', 't_confidence']]
    # print(acc_conf_df.head())

    # select data based on confidence value
    acc_conf_dict = {}
    for cv in confidence_range:
        acc_df = acc_conf_df.loc[df['t_confidence'] >= cv]
        accuracy = get_accuracy(acc_df)
        acc_conf_dict[cv] = accuracy

    # plot
    xlabels = [f"{v/10}" for v in range(0, 10, 1)]

    plt.subplot(211)
    plt.scatter(x=xlabels, y=acc_conf_dict.values())
    plt.title(str.upper(dataset_name))
    plt.xlabel("Confidence")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.show()


def plot_count_vs_confidence_60(dataset_name: str):
    # load rotated 60° dataframe
    df = load_csv(os.path.join(LENET5_VANILLA_PATH, "mnist_rotate60.csv"))

    confidence_range = np.arange(0, 1, 0.1)
    count_df = df['t_confidence']

    # select data based on confidence value
    conf_count_dict = {}
    for cv in confidence_range:
        count_df = count_df.loc[df['t_confidence'] >= cv]
        count = count_df.count()
        conf_count_dict[cv] = count

    # plot
    xlabels = [f"{v/10}" for v in range(0, 10, 1)]

    plt.subplot(211)
    plt.scatter(x=xlabels, y=conf_count_dict.values())
    plt.title(str.upper(dataset_name))
    plt.xlabel("Confidence")
    plt.xticks(rotation=45)
    plt.ylabel("Number of examples")

    plt.show()


def plot_entropy_ood(dataset_name: str):
    # load nomnist dataframe
    df = load_csv(os.path.join(LENET5_VANILLA_PATH, "nomnist.csv"))
    print(df.head())

    ent_df = df['t_entropy']
    print(ent_df.describe())

    # count examples based on entropy value
    ent_range = np.arange(ent_df.min(), ent_df.max()*1.1, 1)
    ent_count_dict = {}
    for ev in ent_range:
        count_df = ent_df.loc[df['t_entropy'] >= ev]
        count = count_df.count()
        ent_count_dict[ev] = count

    # plot
    xlabels = [f"{v}" for v in ent_range]

    plt.subplot(211)
    plt.hist(ent_df)
    plt.title(str.upper(dataset_name))
    plt.xlabel("Entropy (Nats)")
    plt.xticks(rotation=45)
    plt.ylabel("Number of examples")

    plt.show()


if __name__ == "__main__":
    plot_rotated("mnist")
    plot_shifted("mnist")
    plot_confidence_vs_accuracy_60("mnist")
    plot_count_vs_confidence_60("mnist")
    plot_entropy_ood("not-mnist")
