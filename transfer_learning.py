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


def create_labeldropout_configs(reference_cfg: engine.RunConfig, dropout_probs: np.ndarray, labeldrop_ver=2) -> Dict[str, engine.RunConfig]:
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
    train_configs: Dict[str, engine.RunConfig] = dict()
    ref_config = engine.RunConfig.load(os.path.join(RUN_ROOT, RUN_CFG))
    for model in ref_config.models:
        # swaps model classname with proper model instance
        model = getattr(engine, model)()
        cfg = copy(ref_config)
        cfg.models = None
        setattr(cfg, 'model', model)  # HACK: stinky code!
        key = model.__class__.__name__
        train_configs[key] = cfg
        log.info(f"Loaded configuration for {key}")

    # TRAIN & TEST LENET5 VANILLA
    log.info("TRAIN & TEST LENET5 VANILLA")
    vanilla_models: List[torch.nn.Module] = list()
    for m_name in train_configs:
        ref_config = train_configs[m_name]
        run_dir = create_run_folder(
            model_name=m_name, run_id=r_id)

        if ENABLE_SHORT_TRAIN:
            ref_config.max_items = 1500

        # train
        trained_model: torch.nn.Module = trm.do_train(model=ref_config.model, device=r_device,
                                                      config=ref_config, directory=run_dir)
        vanilla_models.append(trained_model)

        # test
        pt_path = glob.glob(
            f"{run_dir}/{ref_config.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=ref_config.model.__class__.__name__,
                    state_dict_path=pt_path, device=r_device, directory=run_dir)

    # PREPARE FOR TRANSFER LEARNING
    log.info("PREPARE FOR TRANSFER LEARNING")
    tl_models: List[torch.nn.Module] = list()
    for vm in vanilla_models:
        prepared_model = copy(vm)
        # TODO: prepare model
        tl_models.append(prepared_model)

    # CREATE LABELDROP CONFIGS
    log.info("CREATE LABELDROP CONFIGS")
    tl_configs: Dict[str, engine.RunConfig] = dict()
    ldrop_configs = create_labeldropout_configs(
        ref_config, np.arange(0.10, 0.50, 0.10))
    tl_configs.update(ldrop_configs)

    # TRAIN & TEST TR_LENET5
    log.info("TRAIN & TEST TR_LENET5")
    for m_name in tl_configs:
        ref_config = tl_configs[m_name]
        run_dir = create_run_folder(
            model_name=m_name, run_id=r_id)

        if ENABLE_SHORT_TRAIN:
            ref_config.max_items = 1500

        # train
        model: torch.nn.Module = trm.do_train(model=ref_config.model, device=r_device,
                                              config=ref_config, directory=run_dir)
        tl_models.append(model)

        # test
        pt_path = glob.glob(
            f"{run_dir}/{ref_config.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=ref_config.model.__class__.__name__,
                    state_dict_path=pt_path, device=r_device, directory=run_dir)
