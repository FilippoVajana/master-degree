import datetime as dt
import pandas as pd
import results as res


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
