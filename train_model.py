import argparse
import datetime as dt
import os
import shutil
import logging as log
log.basicConfig(level=log.DEBUG, format='[%(asctime)s] %(message)s.', datefmt='%H:%M:%S')

import GPUtil
from torch import save
from torch.cuda import is_available

import engine

MODELS = {
    'LeNet5': engine.LeNet5()
}
# getattr(engine, config['model'])()

RUN_CFG_PATH = './runconfig.json'
RUNS_DIR = './runs/'
TRAINS_DIR = './runs/train'
TESTS_DIR = './runs/test'

def get_id():
    # get run id as a time string
    time = dt.datetime.now()
    t_id = time.strftime("%d%m_%H%M") 
    log.debug(f"Created ID: {t_id}")

    return t_id

def save_empty_cfg(path: str):
    engine.RunConfig().save(path)


def create_run_folders():
    '''Creates "runs/" root directory plus "train/" and "test/" subdirectories.
    Sets the "TRAINS_DIR" and "TESTS_DIR" global variables.
    '''    
    # create train root folder
    tr_path = os.path.join(RUNS_DIR, 'train')
    log.info(f"Creating train root: {tr_path}")
    os.makedirs(tr_path, exist_ok=True)
    TRAINS_DIR = tr_path

    # create test root folder
    te_path = os.path.join(RUNS_DIR, 'test')
    log.info(f"Creating test root: {te_path}")
    os.makedirs(te_path, exist_ok=True)
    TESTS_DIR = te_path


def create_train_run_folder(model_name: str, run_id=None):
    '''Creates a dedicated folder for the train run.
    Returns the folder's path.
    '''
    if run_id is None:
        run_id = get_id()

    # create train run folder
    r_path = os.path.join(TRAINS_DIR, model_name, run_id)
    log.info(f"Creating train run folder: {r_path}")
    os.makedirs(r_path)
    return r_path


def get_device():
    device = "cpu"  # default device
    # check cuda device availability
    if is_available():
        gpu = GPUtil.getFirstAvailable()  # get best GPU
        device = f"cuda:{gpu[0]}"
    print("Selected device: ", device)
    return device


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DNN models.")
    parser.add_argument('-cfg', type=str, action='store',
                        default=RUN_CFG_PATH, help='Load configuration file.')
    parser.add_argument('-out', type=str, action='store',
                        default=RUNS_DIR, help='Output directory.')
    args = parser.parse_args()

    RUN_CFG_PATH = args.cfg
    RUNS_DIR = args.out

    # create folders
    create_run_folders() # train/ and test/ folders  

    # get model
    run_cfg = engine.RunConfig().load(RUN_CFG_PATH)
    model = MODELS[run_cfg.model]
    run_cfg.model = model  # swaps model classname with proper model instance
    log.info(f"Loaded model {run_cfg.model.__class__.__name__}")

    # create folder for train run
    tr_run_folder = create_train_run_folder(run_cfg.model.__class__.__name__)

    #TRAIN MODEL
    ############
    device = get_device()
    tr_model, tr_df = model.start_training(
        run_cfg, device)

    # save model dict
    state_dict_path = os.path.join(
        tr_run_folder, f"{run_cfg.model.__class__.__name__}.pt")
    save(tr_model, state_dict_path)

    # save training logs
    train_logs_path = os.path.join(tr_run_folder, 'train_logs.csv')
    tr_df.to_csv(train_logs_path, index=True)

    
    #TEST MODEL
    ############
    # TODO