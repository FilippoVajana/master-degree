import os
import datetime as dt
import mlflow
from torch import save
import argparse

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
    return time.strftime("%d%m_%H%M") # Hour_Minute_Day_Month

def create_run(root: str):
    path = os.path.join(root, get_id())
    os.mkdir(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DNN models.")
    parser.add_argument('-cfg', type=str, action='store', default=RUN_CFG_PATH, help='Load configuration file.')
    parser.add_argument('-out', type=str, action='store', default=RUNS_DIR, help='Output directory.')
    args = parser.parse_args()

    RUN_CFG_PATH = args.cfg
    RUNS_DIR = args.out

    # save reference RunConfig json
    # engine.RunConfig().save(run_cfg_path)

    # load reference RunConfig json
    run_cfg = engine.RunConfig().load(RUN_CFG_PATH)

    # get model
    model = MODELS[run_cfg.model]
    run_cfg.model = model # swaps model classname with model instance

    # init run 
    run_path = create_run(RUNS_DIR)
    results_path = os.path.join(run_path, run_cfg.model.__class__.__name__)
    os.mkdir(results_path)

    # train model
    mlflow.start_run(run_name=run_cfg.model.__class__.__name__)
    trained_model, training_logs = model.start_training(run_cfg)
    mlflow.end_run()

    # save model dict
    tm_path = os.path.join(results_path, f"{run_cfg.model.__class__.__name__}.pt")
    save(trained_model, tm_path)

    # save training logs
    tl_path = os.path.join(results_path, 'training logs.xlsx')
    training_logs.to_excel(tl_path, engine='xlsxwriter')
