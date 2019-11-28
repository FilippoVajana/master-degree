import os

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
        ])

    trainset = torchvision.datasets.MNIST(root='./data/mnist/archive', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data/mnist/archive', train=False, download=True, transform=transform)

    trainld = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    testld = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)   

    return trainld, testld

def save_data(arr: np.ndarray, name: str, path: str):
    size = arr.size * arr.itemsize / 1e6
    tqdm.write(f"Saving {name} ndarray [{size} MB]")
    np.save(os.path.join(path, name), arr)


if __name__ == "__main__":
    CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    # make dirs
    DATA_ROOT = "./data/mnist"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    # build loaders
    train_dl, test_dl =  get_mnist()

    # train data
    train_img_arr = []
    train_lab_arr = []

    tqdm.write("Reading train data")
    for data in tqdm(iter(train_dl)):
        image, label = data
        img_class = CLASSES[label.item()]
        train_img_arr.append(image.numpy())
        train_lab_arr.append(int(img_class))
    
    save_data(np.asarray(train_img_arr), "images", TRAIN_DIR)
    save_data(np.asarray(train_lab_arr), "labels", TRAIN_DIR)

    # test data
    test_img_arr = []
    test_lab_arr = []

    tqdm.write("Reading test data")
    for data in tqdm(iter(test_dl)):
        image, label = data
        img_class = CLASSES[label.item()]
        test_img_arr.append(image.numpy())
        test_lab_arr.append(int(img_class))
    
    save_data(np.asarray(test_img_arr), "images", TEST_DIR)
    save_data(np.asarray(test_lab_arr), "labels", TEST_DIR)