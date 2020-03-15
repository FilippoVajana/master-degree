import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np

from results import *
from results.utils import *


def plot_entropy_ood(res_dir_list: List[str]):
    # load nomnist dataframe
    df_dict = {
        os.path.basename(path): load_csv(os.path.join(path, 'nomnist.csv'))['t_entropy']
        for path in res_dir_list
    }
    res_df = pd.DataFrame()

    # count examples based on entropy value
    ent_range = np.arange(1.9, 2.2, 0.005)
    for k in df_dict:
        count_list = list()
        for ev in ent_range:
            count_df = df_dict[k].loc[df_dict[k] <= ev]
            count_list.append(count_df.count())

        # save grouped data
        res_df[k] = pd.Series(count_list, index=list(ent_range))

    # plot
    fig = plt.figure()
    ax1 = fig.subplots(nrows=1)
    fig.suptitle("Entropy on OOD")
    formatter = ticker.FormatStrFormatter("%.2f")

    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True)
    ax1.tick_params(grid_linestyle='dotted')

    for k in res_df:
        ax1.scatter(ent_range, res_df[k], label=k, s=8)

    ax1.set_ylabel(r"Number of examples $H \leq \tau$")
    ax1.set_xlabel("Entropy (Nats)")
    ax1.legend()
    return fig
