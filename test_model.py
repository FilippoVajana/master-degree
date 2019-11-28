import os
import datetime as dt
import argparse
import mlflow
import torch

import engine
import engine.tester as tester

MODELS = {
    'LeNet5': engine.LeNet5()
}
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
    parser = argparse.ArgumentParser(description="Test DNN models.")
    parser.add_argument('-m', type=str, action='store', help='Load model file.')
    parser.add_argument('-n', type=str, action='store', help='Model class name')
    parser.add_argument('-d', type=str, action='store', help='Test data directory.')
    args = parser.parse_args()

    # load model
    model = MODELS[args.n]
    path = os.path.normpath(os.path.join(os.getcwd(), args.m))
    model.load_state_dict(torch.load(path))

    # init dataloader
    dataloader = engine.dataloader.ImageDataLoader(
            data_folder=args.d,
            batch_size=1,
            shuffle=False,
            train_mode=False,
            max_items=10
        ).dataloader

    # test model
    mlflow.start_run(run_name=model.__class__.__name__)

    t = tester.Tester(model)
    log = t.test(dataloader)
    print(log)

    mlflow.end_run()
    