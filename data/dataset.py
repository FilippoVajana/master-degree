import os

import torch
import torchvision
from torch.utils.data import Dataset

import cachetools
import imageio


class CustomDataset(Dataset):
    def __init__(self, root, train_data=True):
        self.root = root

        if train_data:
            self.mode = "TRAIN"
            self.ex_dir = os.path.join(self.root, "train")
            self.l_dir = os.path.join(self.ex_dir, "labels")
        else:
            self.mode="TEST"
            self.ex_dir = os.path.join(self.root, "test")
            self.l_dir = os.path.join(self.ex_dir, "labels")

        # init data model
        self.data = self.join_data()

        # init cache system
        self.cache = cachetools.LRUCache(maxsize=len(self.data))

    
    def get_images(self, directory):
        items = list(map(lambda x : os.path.join(directory, x), os.listdir(directory)))
        imgs = [os.path.normpath(i) for i in items if i.endswith(".png")]
        imgs.sort()
        return imgs

    def get_labels(self, directory):
        items = list(map(lambda x : os.path.join(directory, x), os.listdir(directory)))
        labels = [os.path.normpath(i) for i in items if i.endswith(".txt")]
        labels.sort()
        return labels
        
    def join_data(self):
        ex = self.get_images(self.ex_dir)
        l = self.get_labels(self.l_dir)
        return tuple(zip(ex,l))

    def __len__(self):        
        return len(self.data)

    @cachetools.cachedmethod(lambda self: self.cache)
    def __getitem__(self, index):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        # get image and label paths
        d = self.data[index]

        # load image
        img = imageio.imread(d[0])

        # load label value
        lab = open(d[1], "r").read()

        return (transform(img), lab)
        
        