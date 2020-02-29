import datetime as dt
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
RUN_CONFIGS = [
    'LeNet5_runcfg.json',
    'LeNet5SimpleLLDropout_runcfg.json',
    'LeNet5SimpleDropout_runcfg.json',
    'LeNet5ConcreteDropout_runcfg.json'
]

#RUN_CONFIGS = ['LeNet5_runcfg.json']


def get_id() -> str:
    '''Returns run id as a time string.
    '''
    time = dt.datetime.now()
    t_id = time.strftime("%d%m_%H%M")
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


def create_labeldropout_configs(cfg: engine.RunConfig, dropout: np.ndarray) -> Dict[str, engine.RunConfig]:
    dl_configs = dict()
    for drop_v in dropout:
        dl_cfg = copy.copy(cfg)
        dl_cfg.dirty_labels = float("{0:.2f}".format(drop_v))
        key = f"{dl_cfg.model.__class__.__name__}labdrop{dl_cfg.dirty_labels}"
        dl_configs[key] = dl_cfg
        log.info(f"Created Label Dropout config: {key}")
    return dl_configs


if __name__ == '__main__':
    ENABLE_DIRTY_LABELS = True
    ENABLE_SHORT_TRAIN = False

    # load cfg objects
    run_configurations = dict()
    for cfg in RUN_CONFIGS:
        cfg = engine.RunConfig.load(os.path.join(RUN_ROOT, cfg))
        cfg.model = getattr(engine, cfg.model)()
        key = cfg.model.__class__.__name__
        run_configurations[key] = cfg
        log.info(f"Loaded configuration for {key}")

    if ENABLE_DIRTY_LABELS:
        lenet5_cfg = run_configurations["LeNet5"]
        dl_values = [0.10, 0.20]
        run_configurations.update(
            create_labeldropout_configs(lenet5_cfg, dl_values))

    # get device
    r_device = get_device()

    # get run id
    r_id = get_id()

    # create run folders
    for k in run_configurations:
        cfg = run_configurations[k]
        run_dir = create_run_folder(
            model_name=k, run_id=r_id)

        # training
        if ENABLE_SHORT_TRAIN:
            cfg.max_items = 1500
        trm.do_train(model=cfg.model, device=r_device,
                     config=cfg, directory=run_dir)

        # testing
        pt_path = glob.glob(
            f"{run_dir}/{cfg.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=cfg.model.__class__.__name__,
                    state_dict_path=pt_path, device=r_device, directory=run_dir)
