import os
import argparse
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import engine
from engine.tester import Tester
from skimage import transform

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


def test_regular_data(model, dataset_name):
    # get dataloader
    dataloader = engine.ImageDataLoader(
        data_folder=DATA_DICT[dataset_name],
        batch_size=32,
        shuffle=False,
        transformation=None
    ).build(train_mode=False, max_items=-1, validation_ratio=0)

    # test model
    tester = Tester(model, device="cuda", is_ood=False)
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
    tester = Tester(model, device="cuda", is_ood=True)
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
    tester = Tester(model, device="cuda", is_ood=False)
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
    tester = Tester(model, device="cuda", is_ood=False)
    df = tester.test(dataloader[0])
    return df


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

    # prepare test folder
    df_path = os.path.join(RUNS_DICT['LeNet5'], "test")
    os.makedirs(df_path, exist_ok=True)

    # test In-Distribution
    if True:
        try:
            print("Testing MNIST")
            mnist_df = test_regular_data(model, "mnist")
            mnist_df.to_csv(df_path + os.sep + "mnist.csv", index=True)
        except Exception as exc:
            print(exc)

    # test In-Distribution Rotated
    rotation_range = range(15, 180 + 15, 15)
    if True:
        for rotation_value in rotation_range:
            try:
                print(f"Testing Rotated {rotation_value} MNIST")
                rotated_df = test_rotated_data(model, "mnist", rotation_value)
                rotated_df.to_csv(df_path + os.sep +
                                  f"mnist_rotate{rotation_value}.csv", index=True)
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
                rotated_df = test_shifted_data(model, "mnist", shift_value)
                rotated_df.to_csv(df_path + os.sep +
                                  f"mnist_shift{int(shift_value * img_size)}.csv", index=True)
            except Exception as exc:
                print(exc)

    # test Out-of-Distribution
    if True:
        try:
            print("Testing OOD")
            ood_df = test_ood_data(model, "no-mnist")
            ood_df.to_csv(df_path + os.sep + f"nomnist.csv", index=True)
        except Exception as exc:
            print(exc)
