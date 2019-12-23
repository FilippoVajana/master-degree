import os
import argparse
import torch
import pandas as pd
import numpy as np
import engine
import engine.tester as tester

MODELS = {
    'LeNet5': engine.LeNet5()
}

RUNS_DICT = {
    'LeNet5': "./runs/LeNet5"
}

DATA_DICT = {
    'mnist': './data/mnist',
    'no-mnist': './data/no-mnist'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test DNN models.")
    parser.add_argument('-m', type=str, action='store',
                        help='Model file path.')
    parser.add_argument('-n', type=str, action='store',
                        help='Model class name')
    parser.add_argument('-d', type=str, action='store',
                        help='Test data directory.')
    args = parser.parse_args()

    # load model
    model = MODELS[args.n]
    model.load_state_dict(torch.load(args.m, map_location=torch.device('cpu')))

    ### OOD ###
    # init OOD dataloader
    dataloader = engine.dataloader.ImageDataLoader(
        data_folder=DATA_DICT[args.d],
        batch_size=1,
        shuffle=False,
        train_mode=False,
        max_items=100
    ).dataloader

    # test model
    t = tester.Tester(model)
    log = t.test(dataloader)

    # save test results
    for m in log.keys():
        if m == "input_tensor":
            continue
        try:
            df = pd.DataFrame(np.vstack(log[m]))
            path = os.path.join(RUNS_DICT['LeNet5'], f'{args.d}_test_{m}.csv')
            df.to_csv(path)
        except Exception:
            continue
