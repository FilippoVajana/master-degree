import os
import shutil
import datetime as dt
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

def save_empty_cfg(path: str):
    engine.RunConfig().save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DNN models.")
    parser.add_argument('-cfg', type=str, action='store', default=RUN_CFG_PATH, help='Load configuration file.')
    parser.add_argument('-out', type=str, action='store', default=RUNS_DIR, help='Output directory.')
    args = parser.parse_args()

    RUN_CFG_PATH = args.cfg
    RUNS_DIR = args.out    

    # load reference RunConfig json
    run_cfg = engine.RunConfig().load(RUN_CFG_PATH)

    # get model
    model = MODELS[run_cfg.model]
    run_cfg.model = model # swaps model classname with model instance

    # init run 
    # run_path = create_run(RUNS_DIR)
    results_path = os.path.join(RUNS_DIR, run_cfg.model.__class__.__name__)
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    # train model    
    trained_model, training_logs = model.start_training(run_cfg)

    # save model dict
    tm_path = os.path.join(results_path, f"{run_cfg.model.__class__.__name__}.pt")
    save(trained_model, tm_path)

    # save training logs
    tl_path = os.path.join(results_path, 'train_logs.xlsx')
    training_logs.to_excel(tl_path, engine='xlsxwriter')
