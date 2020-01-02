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
    if is_available():  # check cuda device availability
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

    # get model
    model = MODELS[run_cfg.model]
    run_cfg.model = model  # swaps model classname with proper model instance

    # create train result folder
    # results_path = create_run(os.path.join(
    #     RUNS_DIR, run_cfg.model.__class__.__name__))

    results_path = os.path.join(RUNS_DIR, run_cfg.model.__class__.__name__)
    # if os.path.exists(results_path):
    #     shutil.rmtree(results_path)
    os.makedirs(results_path, exist_ok=True)

    # train model
    trained_model, training_logs = model.start_training(run_cfg, get_device())

    # save model dict
    tm_path = os.path.join(
        results_path, f"{run_cfg.model.__class__.__name__}.pt")
    save(trained_model, tm_path)

    # save training logs
    tl_path = os.path.join(results_path, 'train_logs.csv')
    training_logs.to_csv(tl_path)
