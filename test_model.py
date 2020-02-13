import argparse
import os
import datetime
import GPUtil
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from skimage import transform
from torch.cuda import is_available

import engine
from engine.tester import Tester

MODELS = {
    'LeNet5': engine.LeNet5()
}

RUN_DIRECTORY_DICT = {
    'LeNet5': "./runs/LeNet5"
}

DATA_DICT = {
    'mnist': './data/mnist',
    'no-mnist': './data/no-mnist'
}

DEVICE = "cpu"


def get_device():
    device = "cpu"  # default device
    # check cuda device availability
    if is_available():
        gpu = GPUtil.getFirstAvailable()  # get best GPU
        device = f"cuda:{gpu[0]}"
    print("Selected device: ", device)
    return device


def test_regular_data(model, dataset_name):
    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=32,
        shuffle=False,
        transformation=None
    ).build(train_mode=False, max_items=-1, validation_ratio=0)

    # test model
    tester = Tester(model, device=DEVICE, is_ood=False)
    df = tester.test(dataloader[0])
    return df


def test_ood_data(model, dataset_name):
    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=32,
        shuffle=False,
        transformation=None
    ).build(train_mode=False, max_items=-1, validation_ratio=0)

    # test model
    tester = Tester(model, device=DEVICE, is_ood=True)
    df = tester.test(dataloader[0])
    return df


def test_rotated_data(model, dataset_name, rotation_value=45):
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((rotation_value, rotation_value)),
        transforms.ToTensor()
    ])

    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=32,
        shuffle=False,
        transformation=transformation
    ).build(train_mode=False, max_items=-1, validation_ratio=.0)

    # test model
    tester = Tester(model, device=DEVICE, is_ood=False)
    df = tester.test(dataloader[0])
    return df


def test_shifted_data(model, dataset_name, shift_value=45):
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(0, translate=(shift_value, shift_value)),
        transforms.ToTensor()
    ])

    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=32,
        shuffle=False,
        transformation=transformation
    ).build(train_mode=False, max_items=-1, validation_ratio=.0)

    # test model
    tester = Tester(model, device=DEVICE, is_ood=False)
    df = tester.test(dataloader[0])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test DNN models.")
    parser.add_argument('-m', type=str, action='store', default='./runs/LeNet5/LeNet5.pt',
                        help='Model file path.')
    parser.add_argument('-n', type=str, action='store', default='LeNet5',
                        help='Model class name')
    parser.add_argument('-d', type=str, action='store',
                        help='Test data directory.')
    parser.add_argument('-save', action='store_true')
    args = parser.parse_args()

    # get compute device
    DEVICE = get_device()

    # load model
    model = MODELS[args.n]
    if DEVICE is "cpu":
        model.load_state_dict(torch.load(
            args.m, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.m))

    # compute test run id    
    now = datetime.datetime.now()
    TEST_RUN_ID = now.strftime("%j_%H%M%S")

    # prepare test folder
    TEST_DIRECTORY = os.path.join(RUN_DIRECTORY_DICT['LeNet5'], "test")
    os.makedirs(TEST_DIRECTORY, exist_ok=True)

    # TEST LOOP
    results_df_dict = dict()  # {filename.csv, pd.DataFrame}

    # test In-Distribution
    if True:
        try:
            print("Testing MNIST")
            # results_df_dict.append((df, os.path.join(TEST_DIRECTORY, "mnist.csv")))
            df = test_regular_data(model, "mnist")            
            results_df_dict["mnist.csv"] = df
        except Exception as exc:
            print(exc)

    # test In-Distribution Rotated
    rotation_range = range(15, 180 + 15, 15)
    if True:
        for rotation_value in rotation_range:
            try:
                print(f"Testing Rotated {rotation_value} MNIST")
                df = test_rotated_data(model, "mnist", rotation_value)
                # results_df_dict.append((rotated_df, os.path.join(TEST_DIRECTORY, f"mnist_rotate{rotation_value}.csv")))
                results_df_dict[f"mnist_rotate{rotation_value}.csv"] = df
            except Exception as exc:
                print(exc)

    # test In-Distribution shifted
    if True:
        shift_range = range(2, 14 + 2, 2)
        img_size = 28
        for shift_value in shift_range:
            shift_value /= img_size
            try:
                print(f"Testing Shifted {int(shift_value * img_size)}px MNIST")
                df = test_shifted_data(model, "mnist", shift_value)                
                # results_df_dict.append((df, os.path.join(TEST_DIRECTORY, f"mnist_shift{int(shift_value * img_size)}.csv")))
                results_df_dict[f"mnist_shift{int(shift_value * img_size)}.csv"] = df
            except Exception as exc:
                print(exc)

    # test Out-of-Distribution
    if True:
        try:
            print("Testing OOD")
            df = test_ood_data(model, "no-mnist")            
            # results_df_dict.append((df, os.path.join(TEST_DIRECTORY, "nomnist.csv")))
            results_df_dict["nomnist.csv"] = df
        except Exception as exc:
            print(exc)

    # save results    
    if args.save:
        # create run folder
        r_path = os.path.join(TEST_DIRECTORY, TEST_RUN_ID)
        os.makedirs(r_path)
        for key in results_df_dict.keys():
            path = os.path.join(r_path, key)
            print("Saving to ", path)
            results_df_dict[key].to_csv(path, index=True)
