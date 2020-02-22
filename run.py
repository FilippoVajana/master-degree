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
RUN_CONFIGS = [
    'LeNet5_runcfg.json',
    'LeNet5SimpleLLDropout_runcfg.json',
    'LeNet5SimpleDropout_runcfg.json'
]


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
    os.makedirs(path)
    return path


if __name__ == '__main__':
    # detect models config json
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
        run_dir = create_run_folder(
            model_name=cfg.model.__class__.__name__, run_id=r_id)

        # performs training
        trm.do_train(model=cfg.model, device=r_device,
                     config=cfg, directory=run_dir)

        # performs testing
        pt_path = glob.glob(
            f"{run_dir}/{cfg.model.__class__.__name__}.pt")[0]
        tem.do_test(model_name=cfg.model.__class__.__name__,
                    state_dict_path=pt_path, device=r_device, directory=run_dir)
