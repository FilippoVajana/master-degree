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

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10/archive', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10/archive', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)   

    return trainloader, testloader

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    pass

def saveimg(img_data, filename, images_dir, label):
    img_path = os.path.join(images_dir, f"{filename}.png")
    label_path = os.path.join(images_dir, "labels", f"{filename}.txt")

    npimg = img_data.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    
    imageio.imwrite(img_path, npimg)

    with open(label_path, "w") as l_file:
        l_file.write(label)
    
    pass

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))

# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == "__main__":
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # make dirs
    cifar10_data_path = "./data/cifar10"
    train_dir = os.path.join(cifar10_data_path, "train")
    test_dir = os.path.join(cifar10_data_path, "test")
    if not os.path.exists(train_dir) : os.makedirs(train_dir)
    if not os.path.exists(test_dir) : os.makedirs(test_dir)    

    # build loaders
    trainloader, testloader =  get_cifar10()

    # iterate over train data
    i = 0
    for data in iter(trainloader):
        image, label = data
        i_class = classes[label.item()]

        i = i + 1
        # if i > 1 : break

        saveimg(image.squeeze(), f"img_{i}", train_dir, i_class)

    # iterate over test data
    i = 0
    for data in iter(testloader):
        image, label = data
        i_class = classes[label.item()]

        i = i + 1
        # if i > 1 : break

        saveimg(image.squeeze(), f"img_{i}", test_dir, i_class)





