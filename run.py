import datetime as dt
import logging as log
import os
import torch.cuda as tcuda
import GPUtil
import glob
import engine
import train_model as trm
import test_model as tem

log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

RUN_ROOT = './runs'
RUN_CONFIGS = ['LeNet5_runcfg.json']


def get_id():
    '''Returns run id as a time string.
    '''
    time = dt.datetime.now()
    t_id = time.strftime("%d%m_%H%M")
    log.debug(f"Created ID: {t_id}")
    return t_id


def get_device():
    '''Returns the best available compute device.
    '''
    device = "cpu"  # default device
    # check cuda device availability
    if tcuda.is_available():
        gpu = GPUtil.getFirstAvailable()  # get best GPU
        device = f"cuda:{gpu[0]}"
    log.info(f"Set compute device: {device}")
    return device


def create_run_folders(model_name: str, train=True, test=True, run_id=None):
    '''Creates "runs/" root directory, "run_id/" folder and 
    "train/" and "test/" run subdirectories.
    '''
    tr_path, te_path = str(), str()
    # get run id
    r_id = get_id() if run_id is None else run_id

    if train == True:
        # create train root folder
        tr_path = os.path.join(RUN_ROOT, r_id, model_name, 'train')
        os.makedirs(tr_path, exist_ok=True)
        log.info(f"Created train root: {tr_path}")

    if test == True:
        # create test root folder
        te_path = os.path.join(RUN_ROOT, r_id, model_name, 'test')
        os.makedirs(te_path, exist_ok=True)
        log.info(f"Created test root: {te_path}")

    return tr_path, te_path


if __name__ == '__main__':
    # get cfg paths
    cfg_regex = f"{RUN_ROOT}/*_runcfg.json"
    cfg_paths = glob.glob(cfg_regex)
    log.debug(f"glob result: {cfg_paths}")

    # load cfg objects
    cfg_list = list()
    for p in cfg_paths:
        cfg_obj = engine.RunConfig.load(p)
        cfg_obj.model = getattr(engine, cfg_obj.model)()
        cfg_list.append(cfg_obj)
        log.info(
            f"Loaded configuration for {cfg_obj.model.__class__.__name__}")

    # get device
    r_device = get_device()

    # get run id
    r_id = get_id()

    # create run folders
    for cfg in cfg_list:
        train_dir, test_dir = create_run_folders(
            model_name=cfg.model.__class__.__name__, run_id=r_id)

        # performs training
        trm.do_train(model=cfg.model, device=r_device,
                     config=cfg, directory=train_dir)

        # performs testing
        pt_path = glob.glob(
            f"{train_dir}/{cfg.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=cfg.model.__class__.__name__,
                    state_dict_path=pt_path, device=r_device, directory=test_dir)
