import argparse
import datetime as dt
import os
import shutil

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


def get_id():
    # get run id as a time string
    time = dt.datetime.now()
    return time.strftime("%d%m_%H%M")  # Hour_Minute_Day_Month


def create_run(root: str):
    path = os.path.join(root, get_id())
    os.makedirs(path, exist_ok=True)
    return path


def save_empty_cfg(path: str):
    engine.RunConfig().save(path)


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

    # load reference RunConfig json
    run_cfg = engine.RunConfig().load(RUN_CFG_PATH)

    # create folders
    os.makedirs(os.path.join(RUNS_DIR, run_cfg.model))

    # get model
    model = MODELS[run_cfg.model]
    run_cfg.model = model  # swaps model classname with proper model instance

    # create train result folder
    dirs_name = [int(d.name) for d in os.scandir(os.path.join(
        RUNS_DIR, run_cfg.model.__class__.__name__)) if os.path.isdir(d.path)]

    if len(dirs_name) == 0:
        results_path = os.path.join(
            RUNS_DIR, run_cfg.model.__class__.__name__, "0")
    else:
        dirs_name.sort()
        last = dirs_name[-1]
        results_path = os.path.join(
            RUNS_DIR, run_cfg.model.__class__.__name__, f"{last + 1}")

    os.makedirs(results_path, exist_ok=True)

    # train model
    device = get_device()
    trained_model, train_dataframe = model.start_training(
        run_cfg, device)

    # save model dict
    state_dict_path = os.path.join(
        results_path, f"{run_cfg.model.__class__.__name__}.pt")
    save(trained_model, state_dict_path)

    # save training logs
    train_logs_path = os.path.join(results_path, 'train_logs.csv')
    train_dataframe.to_csv(train_logs_path, index=True)
