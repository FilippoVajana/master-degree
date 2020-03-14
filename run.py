import datetime as dt
import argparse
import logging as log
import os
import torch.cuda as tcuda
import GPUtil
import glob
import engine
import train_model as trm
import test_model as tem
import numpy as np
import copy
from typing import List, Dict

log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

RUN_ROOT = './runs'
RUN_CFG = 'default_runcfg.json'


def get_id() -> str:
    '''Returns run id as a time string.
    '''
    time = dt.datetime.now()
    t_id = time.strftime("%m%d_%H%M")
    log.debug(f"Created ID: {t_id}")
    return t_id


def get_device() -> str:
    '''Returns the best available compute device.
    '''
    device = "cpu"  # default device
    # check cuda device availability
    if tcuda.is_available():
        gpu = GPUtil.getFirstAvailable()  # get best GPU
        device = f"cuda:{gpu[0]}"
    log.info(f"Set compute device: {device}")
    return device


def create_run_folder(model_name: str, run_id=None):
    '''Creates "runs/run_id/model_name" folder.
    '''
    # get run id
    r_id = get_id() if run_id is None else run_id

    # create run folder
    path = os.path.join(RUN_ROOT, r_id, model_name.lower())
    os.makedirs(path, exist_ok=True)
    return path


def create_labdrop_configs(reference_cfg: engine.RunConfig, dropout_probs: np.ndarray) -> Dict[str, engine.RunConfig]:
    dl_configs = dict()
    for val in dropout_probs:
        cfg = copy.copy(reference_cfg)
        # hack the config
        delattr(cfg, 'models')
        setattr(cfg, 'model', engine.LeNet5())
        cfg.dirty_labels = float("{0:.2f}".format(val))
        # save LD config
        key = f"{cfg.model.__class__.__name__}-labdrop{cfg.dirty_labels}"
        dl_configs[key] = cfg
        log.info(f"Created Label Dropout config: {key}")
    return dl_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Test models.")
    parser.add_argument('-dirty', default=False,
                        action='store_true', help='Train with Label Drop.')
    parser.add_argument('-short', default=False,
                        action='store_true', help='Train with less examples.')
    parser.add_argument('-cfg', default=RUN_CFG,
                        action='store', help='Run configuration file.')
    args = parser.parse_args()

    ENABLE_DIRTY_LABELS = args.dirty
    ENABLE_SHORT_TRAIN = args.short
    RUN_CFG = args.cfg

    # get device
    r_device = get_device()
    # get run id
    r_id = get_id()

    # load cfg objects
    run_configurations = dict()
    with engine.RunConfig.load(os.path.join(RUN_ROOT, RUN_CFG)) as ref_cfg:
        for model in ref_cfg.models:
            # swaps model classname with proper model instance
            model = getattr(engine, model)()
            cfg = copy.copy(ref_cfg)
            delattr(cfg, 'models')
            setattr(cfg, 'model', model)  # HACK: stinky code!
            key = model.__class__.__name__
            run_configurations[key] = cfg
            log.info(f"Loaded configuration for {key}")

    if ENABLE_DIRTY_LABELS:
        with engine.RunConfig.load(os.path.join(RUN_ROOT, RUN_CFG)) as ref_cfg:
            values = np.arange(0.05, 0.15, 0.05)
            configs = create_labdrop_configs(ref_cfg, values)
            run_configurations.update(configs)

    # create run folders
    for k in run_configurations:
        cfg = run_configurations[k]
        run_dir = create_run_folder(
            model_name=k, run_id=r_id)

        if ENABLE_SHORT_TRAIN:
            cfg.max_items = 1500

        # training
        trm.do_train(model=cfg.model, device=r_device,
                     config=cfg, directory=run_dir)

        # testing
        pt_path = glob.glob(
            f"{run_dir}/{cfg.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=cfg.model.__class__.__name__,
                    state_dict_path=pt_path, device=r_device, directory=run_dir)
