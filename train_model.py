import engine
from torch.cuda import is_available
from torch import save
import GPUtil
import argparse
import datetime as dt
import os
import shutil
import logging as log
log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')


# getattr(engine, config['model'])()
MODELS = {
    'LeNet5': engine.LeNet5()
}
RUN_CFG_PATH = './runconfig.json'
RUNS_ROOT = './runs/'


def get_id():
    # get run id as a time string
    time = dt.datetime.now()
    t_id = time.strftime("%d%m_%H%M")
    log.debug(f"Created ID: {t_id}")
    return t_id


def create_run_folders(model_name: str):
    '''Creates "runs/" root directory, "run_id/" folder and "train/" and "test/" run subdirectories.    
    '''
    # get run id
    r_id = get_id()

    # create train root folder
    tr_path = os.path.join(RUNS_ROOT, r_id, model_name, 'train')
    os.makedirs(tr_path, exist_ok=True)
    log.info(f"Created train root: {tr_path}")

    # create test root folder
    te_path = os.path.join(RUNS_ROOT, r_id, model_name, 'test')
    os.makedirs(te_path, exist_ok=True)
    log.info(f"Created test root: {te_path}")

    return tr_path, te_path


def get_device():
    device = "cpu"  # default device
    # check cuda device availability
    if is_available():
        gpu = GPUtil.getFirstAvailable()  # get best GPU
        device = f"cuda:{gpu[0]}"
    log.info(f"Set compute device: {device}")
    return device


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DNN models.")
    parser.add_argument('-cfg', type=str, action='store',
                        default=RUN_CFG_PATH, help='Load configuration file.')
    args = parser.parse_args()

    # load config file
    run_cfg = engine.RunConfig().load(args.cfg)
    log.info(f"Loaded run configuration file: {args.cfg}")

    # get model
    model = MODELS[run_cfg.model]
    run_cfg.model = model  # swaps model classname with proper model instance
    log.info(f"Loaded model: {run_cfg.model.__class__.__name__}")

    # set compute device
    device = get_device()

    # create run folders
    train_dir, test_dir = create_run_folders(
        model_name=run_cfg.model.__class__.__name__)

    # TRAIN MODEL
    ############
    log.info("Start training phase")
    train_model, train_df = model.start_training(
        run_cfg, device)

    # save model dict
    state_dict_path = os.path.join(
        train_dir, f"{run_cfg.model.__class__.__name__}.pt")
    save(train_model, state_dict_path)

    # save training logs
    train_logs_path = os.path.join(train_dir, 'train_logs.csv')
    train_df.to_csv(train_logs_path, index=True)

    # TEST MODEL
    ############
    # TODO
    # log.info("Start testing phase")
