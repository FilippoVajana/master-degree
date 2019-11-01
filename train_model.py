import mlflow

import engine

MODELS = {
    'LeNet5': engine.LeNet5()
}


if __name__ == '__main__':
    run_cfg_path = './runconfig.json'

    # save reference RunConfig json
    # engine.RunConfig().save(run_cfg_path)

    # load reference RunConfig json
    run_cfg = engine.RunConfig().load(run_cfg_path)

    # get model
    model = MODELS[run_cfg.model]
    run_cfg.model = model # swaps model classname with model instance

    # train model
    mlflow.start_run(run_name='LeNet5-dev')
    model.start_training(run_cfg)
    