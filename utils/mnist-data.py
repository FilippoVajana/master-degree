import os

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import imageio


def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor()
        ])

    trainset = torchvision.datasets.MNIST(root='./data/mnist/archive', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data/mnist/archive', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)   

    return trainloader, testloader

def saveimg(img_data, filename, images_dir, label):
    img_path = os.path.join(images_dir, f"{filename}.png")
    label_path = os.path.join(images_dir, "labels", f"{filename}.txt")

    npimg = img_data.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    
    imageio.imwrite(img_path, npimg)

    with open(label_path, "w") as l_file:
        l_file.write(label)
    
    pass


if __name__ == "__main__":
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    # make dirs
    cifar10_data_path = "./data/mnist"
    train_dir = os.path.join(cifar10_data_path, "train")
    test_dir = os.path.join(cifar10_data_path, "test")
    if not os.path.exists(train_dir) : os.makedirs(train_dir)
    if not os.path.exists(test_dir) : os.makedirs(test_dir)    

    # build loaders
    trainloader, testloader =  get_cifar10()

    # iterate over train data
    i = 0
    labels_folder = os.path.join(train_dir, "labels")
    os.mkdir(labels_folder)

    for data in iter(trainloader):
        image, label = data
        i_class = classes[label.item()]

        i = i + 1
        # if i > 1 : break

        saveimg(image.squeeze(dim=0), f"img_{i}", train_dir, i_class)

    # iterate over test data
    i = 0
    labels_folder = os.path.join(test_dir, "labels")
    os.mkdir(labels_folder)
    
    for data in iter(testloader):
        image, label = data
        i_class = classes[label.item()]

        i = i + 1
        # if i > 1 : break

        saveimg(image.squeeze(dim=0), f"img_{i}", test_dir, i_class)





