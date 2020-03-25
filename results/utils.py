import datetime as dt
import pandas as pd
import results as res
from typing import Dict, List
import os
import glob
import natsort
import re


def get_id() -> str:
    '''Returns run id as a time string.
    '''
    time = dt.datetime.now()
    t_id = time.strftime("%d%m_%H%M")
    res.LOGGER.debug(f"Created ID: {t_id}")
    return t_id


def load_csv(filename: str):
    '''Loads data from csv file and return as DataFrame.
    '''
    data = pd.read_csv(filename, index_col=[0])
    res.LOGGER.info(f"Loaded csv file: {filename}")
    return data


def side_legend(ax):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


def bottom_legend(ax, count=4):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=count, frameon=False)


def get_accuracy(df: pd.DataFrame):
    accuracy_row = df['t_good_pred']
    accuracy = accuracy_row.sum() / accuracy_row.count() if accuracy_row.count() >= 0 else 0
    return accuracy


def get_brier(df: pd.DataFrame):
    brier_row = df['t_brier']
    return brier_row.mean()


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

    # log.debug(f"Shifted df:\n {df_dict}")
    return df_dict


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

    # log.debug(f"Rotated df:\n {df_dict}")
    return df_dict
