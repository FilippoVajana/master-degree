import argparse
import datetime as dt
import logging as log
import os

import GPUtil
from torch import save
from torch.cuda import is_available

import engine

log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)-7s: %(message)s', datefmt='%H:%M:%S')

# getattr(engine, config['model'])()
MODELS = {
    'LeNet5': engine.LeNet5()
}
RUN_CFG_PATH = './runconfig.json'
RUNS_ROOT = './runs/'


def get_id():
    '''Returns run id as a time string.
    '''
    time = dt.datetime.now()
    t_id = time.strftime("%d%m_%H%M")
    log.debug(f"Created ID: {t_id}")
    return t_id


def create_run_folders(model_name: str, train=True, test=False):
    '''Creates "runs/" root directory, "run_id/" folder and 
    "train/" and "test/" run subdirectories.
    '''
    tr_path, te_path = str(), str()
    # get run id
    r_id = get_id()

    if train == True:
        # create train root folder
        tr_path = os.path.join(RUNS_ROOT, r_id, model_name, 'train')
        os.makedirs(tr_path, exist_ok=True)
        log.info(f"Created train root: {tr_path}")

    if test == True:
        # create test root folder
        te_path = os.path.join(RUNS_ROOT, r_id, model_name, 'test')
        os.makedirs(te_path, exist_ok=True)
        log.info(f"Created test root: {te_path}")

    return tr_path, te_path


def get_device():
    '''Returns the best available compute device.
    '''
    device = "cpu"  # default device
    # check cuda device availability
    if is_available():
        gpu = GPUtil.getFirstAvailable()  # get best GPU
        device = f"cuda:{gpu[0]}"
    log.info(f"Set compute device: {device}")
    return device


def do_train(model, device, config, directory):
    '''Trains the input model.
    The methods saves train logs as .csv and the final model state dict.     
    '''
    log.info("Started training phase")
    train_model, train_df = model.start_training(config, device)

    # save model dict
    state_dict_path = os.path.join(
        directory, f"{config.model.__class__.__name__}.pt")
    save(train_model, state_dict_path)
    log.info("Saved model state dict: %s", state_dict_path)

    # save training logs
    train_logs_path = os.path.join(directory, 'train_logs.csv')
    train_df.to_csv(train_logs_path, index=True)
    log.info("Saved train logs: %s", train_logs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DNN models.")
    parser.add_argument('-cfg', type=str, action='store',
                        default=RUN_CFG_PATH, help='Load configuration file.')
    parser.add_argument('-o', '--outdir', type=str,
                        action='store', help='Train output directory.')
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

    ###############
    # TRAIN MODEL #

    # create run folders
    train_dir = str()
    if args.outdir is None:
        train_dir, _ = create_run_folders(
            model_name=run_cfg.model.__class__.__name__)
    else:
        train_dir = os.path.join(args.outdir, run_cfg.model.__class__.__name__)
        os.makedirs(train_dir)
        log.info(f"Created train folder: {args.outdir}")

    # performs training
    do_train(model, device, run_cfg, train_dir)
