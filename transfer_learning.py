import argparse
import datetime as dt
import logging as log
import os
from copy import copy
from typing import Dict, List, Tuple
import glob
import GPUtil
import numpy as np
import torch.nn
import torch.cuda as tcuda
import train_model as trm
import test_model as tem
from torch.nn import Module
import numpy as np

import engine

log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


RUN_ROOT = './runs'
REFERENCE_CONFIG = 'transfer_runcfg.json'
DO_SHORT = True
DO_TEST = False
MAX_ITEMS = 1250
DEVICE = 'cpu'
R_ID = '0000_0000'


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


def train_and_test(models: Dict[str, Tuple[Module, engine.RunConfig]], do_test=True, max_items=-1) -> Dict[str, Tuple[Module, engine.RunConfig]]:
    trained: Dict[str, (Module, engine.RunConfig)] = dict()
    for m_name in models:
        model = models[m_name][0]
        config = models[m_name][1]
        log.info(f"Train & Test: {m_name}")

        run_dir = create_run_folder(
            model_name=m_name, run_id=R_ID)

        # train
        t_model = trm.do_train(model=model, device=DEVICE,
                               config=config, directory=run_dir)
        # test
        if do_test:
            model_cls = model.__class__.__name__
            pt_path = glob.glob(f"{run_dir}/{model_cls}.pt")[0]
            tem.do_test(model_cls=model_cls,
                        state_dict_path=pt_path, device=DEVICE, directory=run_dir, max_items=max_items)
        # save trained model
        trained[m_name] = (t_model, config)
    return trained


def prepare_for_tl(tr_models: Dict[str, Tuple[Module, engine.RunConfig]]) -> Dict[str, Tuple[Module, engine.RunConfig]]:
    prepared = dict()
    for m_name in tr_models:
        log.info(f"Prepare {m_name} to Transfer Learning")
        model = tr_models[m_name][0]
        config = tr_models[m_name][1]

        # set transfer learning flag
        model.do_transferlearn = True

        # disable all layers
        for param in model.parameters():
            param.requires_grad = False

        # reset fully connected layers
        # in_features = model.fc1.in_features
        # out_features = model.fc1.out_features
        # model.fc1 = torch.nn.Linear(in_features=in_features, out_features=out_features)

        # in_features = model.fc2.in_features
        # out_features = model.fc2.out_features
        # model.fc2 = torch.nn.Linear(in_features=in_features, out_features=out_features)

        in_features = model.fc3.in_features
        out_features = model.fc3.out_features
        model.fc3 = torch.nn.Linear(in_features=in_features, out_features=out_features)

        # add to models dict
        name = f"{m_name}TL"
        prepared[name] = (model, config)
    return prepared


def prepare_for_labeldrop(tl_models: Dict[str, Tuple[Module, engine.RunConfig]], dropout_probs: np.ndarray) -> Dict[str, Tuple[Module, engine.RunConfig]]:
    labeldrop = dict()
    for m_name in tl_models:
        log.info(f"Prepare {m_name} for LabelDrop")
        for p in dropout_probs:
            model = copy(tl_models[m_name][0])
            config = copy(tl_models[m_name][1])
            config.dirty_labels = float("{0:.2f}".format(p))

            # add to models dict
            name = f"{m_name}-{config.dirty_labels}"
            labeldrop[name] = (model, config)
    return labeldrop


if __name__ == '__main__':
    DEVICE = get_device()
    R_ID = get_id()

    parser = argparse.ArgumentParser(
        description="Train LeNet5 and then perform Transfer Learning with LabelDrop.")
    parser.add_argument('-cfg', default=REFERENCE_CONFIG,
                        action='store', help='Run configuration file.')
    parser.add_argument('-short', default=False,
                        action='store_true', help='Train with less examples.')
    parser.add_argument('-test', default=False,
                        action='store_true', help='Do testing.')
    args = parser.parse_args()

    DO_SHORT = args.short
    DO_TEST = args.test
    MAX_ITEMS = MAX_ITEMS if DO_SHORT else -1

    cfg_path = os.path.join(RUN_ROOT, args.cfg)
    REFERENCE_CONFIG = engine.RunConfig.load(cfg_path)
    log.info(f"Loaded reference config: {cfg_path}")

    # load and hack train configurations
    train_configs: Dict[str, Tuple[Module, engine.RunConfig]] = dict()
    for m_name in REFERENCE_CONFIG.models:
        log.info(f"Load & Hack configuration for {m_name}")
        config = copy(REFERENCE_CONFIG)
        # swaps model classname with proper model instance
        model = getattr(engine, m_name)()
        delattr(config, "models")
        setattr(config, 'model', model)  # HACK: stinky code!
        if DO_SHORT:
            config.max_items = MAX_ITEMS
        train_configs[m_name] = (model, config)

    # TRAIN & TEST LENET5 VANILLA
    models_dict = train_and_test(train_configs, DO_TEST, MAX_ITEMS)  # trained base model

    # PREPARE FOR TRANSFER LEARNING
    models_dict = prepare_for_tl(models_dict)

    # CREATE LABELDROP CONFIGS
    tl_range = np.arange(0.05, 0.40, 0.10)
    models_dict = prepare_for_labeldrop(models_dict, tl_range)

    # TRAIN & TEST TR_LENET5
    RESULTS = train_and_test(models_dict, do_test=DO_TEST, max_items=MAX_ITEMS)
