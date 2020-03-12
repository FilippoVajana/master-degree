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


def create_tl_labdrop_configs(reference_cfg: engine.RunConfig, dropout_probs: np.ndarray) -> Dict[str, engine.RunConfig]:
    dl_configs = dict()
    for val in dropout_probs:
        cfg = copy(reference_cfg)
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
    ENABLE_TESTING = True
    cfg_path = os.path.join(RUN_ROOT, args.cfg)
    REFERENCE_CONFIG = engine.RunConfig.load(cfg_path)
    log.info(f"Loaded reference config: {cfg_path}")

    # get device
    DEVICE = get_device()

    # get run id
    R_ID = get_id()

    # load and hack train configurations
    train_configs: Dict[Module, engine.RunConfig] = dict()
    for m_name in REFERENCE_CONFIG.models:
        config = copy(REFERENCE_CONFIG)
        # swaps model classname with proper model instance
        model = getattr(engine, m_name)()
        delattr(config, "models")
        setattr(config, 'model', model)  # HACK: stinky code!
        train_configs[model] = config
        log.info(f"Loaded configuration for {model.__class__.__name__}")

    # TRAIN & TEST LENET5 VANILLA
    TRAINED_MODELS: Dict[Module, engine.RunConfig] = dict()
    log.info("TRAIN & TEST LENET5 VANILLA")
    for model in train_configs:
        config = train_configs[model]
        name = model.__class__.__name__
        run_dir = create_run_folder(
            model_name=name, run_id=R_ID)

        if ENABLE_SHORT_TRAIN:
            config.max_items = 1500

        # train
        t_model = trm.do_train(model=model, device=DEVICE,
                               config=config, directory=run_dir)
        # test
        if ENABLE_TESTING:
            pt_path = glob.glob(
                f"{run_dir}/{model.__class__.__name__}.pt")[0]
            tem.do_test(model_name=model.__class__.__name__,
                        state_dict_path=pt_path, device=DEVICE, directory=run_dir)
        # save trained model
        TRAINED_MODELS[t_model] = config

    # PREPARE FOR TRANSFER LEARNING
    log.info("PREPARE FOR TRANSFER LEARNING")
    TRLEARN_MODELS: Dict[Module, engine.RunConfig] = dict()

    # TODO: train the last layer only
    def prepare_model(model: Module):
        # mantains the gradients locked inside GenericTrainer
        # model.do_transfer_learn = True

        # # disable all layers
        # for param in model.parameters():
        #     param.requires_grad = False

        # reset last fully connected layer
        in_features = model.fc3.in_features
        out_features = model.fc3.out_features
        model.fc3 = torch.nn.Linear(
            in_features=in_features, out_features=out_features)
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        return model

    for t_model in TRAINED_MODELS:
        t_cfg = TRAINED_MODELS[t_model]
        p_model = prepare_model(copy(t_model))
        TRLEARN_MODELS[p_model] = t_cfg

    # CREATE LABELDROP CONFIGS
    log.info("CREATE LABELDROP CONFIGS")
    # Module -> (model_folder, config)
    tl_configs: Dict[Module, Tuple[str, engine.RunConfig]] = dict()
    tl_range = np.arange(0.10, 0.50, 0.10)
    for model in TRLEARN_MODELS:
        config = TRLEARN_MODELS[model]
        ldrop_configs = create_tl_labdrop_configs(config, tl_range)
        for name in ldrop_configs:
            tl_configs[copy(model)] = (name, ldrop_configs[name])

    # TRAIN & TEST TR_LENET5
    FINAL_MODELS: List[Module] = list()
    log.info("TRAIN & TEST TR_LENET5")
    for model in tl_configs:
        name = tl_configs[model][0]
        config = tl_configs[model][1]

        run_dir = create_run_folder(
            model_name=name, run_id=R_ID)

        if ENABLE_SHORT_TRAIN:
            config.max_items = 1500

        # train
        t_model: torch.nn.Module = trm.do_train(model=model, device=DEVICE,
                                                config=config, directory=run_dir)
        FINAL_MODELS.append(copy(t_model))

        # test
        if ENABLE_TESTING:
            pt_path = glob.glob(
                f"{run_dir}/{t_model.__class__.__name__}.pt")[0]
            tem.do_test(model_name=t_model.__class__.__name__,
                        state_dict_path=pt_path, device=DEVICE, directory=run_dir)
