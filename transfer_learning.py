import argparse
import datetime as dt
import logging as log
import os
from copy import copy
from typing import Dict, List
import glob
import GPUtil
import numpy as np
import torch.cuda as tcuda
import train_model as trm
import test_model as tem
import torch
import numpy as np

import engine

log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


RUN_ROOT = './runs'
REFERENCE_CONFIG = 'transfer_runcfg.json'


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
    log.info(f"Created run folder: {path}")
    return path


def create_tl_labdrop_configs(reference_cfg: engine.RunConfig, dropout_probs: np.ndarray, labeldrop_ver=2) -> Dict[str, engine.RunConfig]:
    dl_configs = dict()
    for val in dropout_probs:
        cfg = copy(reference_cfg)
        # hack the config
        cfg.models = None
        if labeldrop_ver == 2:
            setattr(cfg, 'model', engine.LeNet5LabelDrop())
        else:
            setattr(cfg, 'model', engine.LeNet5())
        cfg.dirty_labels = float("{0:.2f}".format(val))
        # save LD config
        key = f"{cfg.model.__class__.__name__}tl-labdrop{cfg.dirty_labels}"
        dl_configs[key] = cfg
        log.info(f"Created Label Dropout config: {key}")
    return dl_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train LeNet5 and then perform Transfer Learning with LabelDrop.")
    parser.add_argument('-cfg', default=REFERENCE_CONFIG,
                        action='store', help='Run configuration file.')
    parser.add_argument('-short', default=False,
                        action='store_true', help='Train with less examples.')
    args = parser.parse_args()

    ENABLE_SHORT_TRAIN = args.short
    cfg_path = os.path.join(RUN_ROOT, args.cfg)
    REFERENCE_CONFIG = engine.RunConfig.load(cfg_path)
    log.info(f"Loaded reference config: {cfg_path}")

    # get device
    DEVICE = get_device()

    # get run id
    R_ID = get_id()

    # TODO: create mapping Model->RunConfig
    # load cfg objects
    train_configs: Dict[str, engine.RunConfig] = dict()
    for model in REFERENCE_CONFIG.models:
        # swaps model classname with proper model instance
        model = getattr(engine, model)()
        cfg = copy(REFERENCE_CONFIG)
        cfg.models = None
        setattr(cfg, 'model', model)  # HACK: stinky code!
        key = model.__class__.__name__
        train_configs[key] = cfg
        log.info(f"Loaded configuration for {key}")

    # TRAIN & TEST LENET5 VANILLA
    log.info("TRAIN & TEST LENET5 VANILLA")
    VANILLA_MODELS = list()
    for m_name in train_configs:
        m_config = train_configs[m_name]
        run_dir = create_run_folder(
            model_name=m_name, run_id=R_ID)

        if ENABLE_SHORT_TRAIN:
            m_config.max_items = 1500

        # train
        trained_model = trm.do_train(model=m_config.model, device=DEVICE,
                                     config=m_config, directory=run_dir)
        VANILLA_MODELS.append(trained_model)

        # test
        pt_path = glob.glob(
            f"{run_dir}/{m_config.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=m_config.model.__class__.__name__,
                    state_dict_path=pt_path, device=DEVICE, directory=run_dir)


    # PREPARE FOR TRANSFER LEARNING
    log.info("PREPARE FOR TRANSFER LEARNING")
    TL_MODELS = list()
    for vm in VANILLA_MODELS:
        pm = copy(vm)
        # reset last fully connected layer
        in_features = pm.fc3.in_features
        out_features = pm.fc3.out_features
        pm.fc3 = torch.nn.Linear(
            in_features=in_features, out_features=out_features)
        TL_MODELS.append(pm)

    # CREATE LABELDROP CONFIGS
    log.info("CREATE LABELDROP CONFIGS")
    tl_configs: Dict[torch.nn.Module, engine.RunConfig] = dict()
    tl_range = np.arange(0.10, 0.50, 0.10)
    for model in TL_MODELS:
        ldrop_configs = create_tl_labdrop_configs(REFERENCE_CONFIG, tl_range)
        tl_configs.update(ldrop_configs)

    # TRAIN & TEST TR_LENET5
    log.info("TRAIN & TEST TR_LENET5")
    for m_name in tl_configs:
        tlm_config = tl_configs[m_name]
        run_dir = create_run_folder(
            model_name=m_name, run_id=R_ID)

        if ENABLE_SHORT_TRAIN:
            tlm_config.max_items = 1500

        # train
        model: torch.nn.Module = trm.do_train(model=tlm_config.model, device=DEVICE,
                                              config=tlm_config, directory=run_dir)
        TL_MODELS.append(model)

        # test
        pt_path = glob.glob(
            f"{run_dir}/{tlm_config.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=tlm_config.model.__class__.__name__,
                    state_dict_path=pt_path, device=DEVICE, directory=run_dir)
