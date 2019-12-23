import os
import numpy as np
import torch
from torch.utils.data import Dataset

import cachetools


class CustomDataset(Dataset):
    def __init__(self, data_path: str):
        # init data
        self.images, self.labels = self.load_data(data_path)

        # init cache system
        self.cache = cachetools.LRUCache(maxsize=len(self))

    def load_data(self, path: str):
        images = np.load(os.path.join(path, "images.npy"))
        labels = np.load(os.path.join(path, "labels.npy"))
        return images, labels

    def __len__(self):
        return len(self.images)

    @cachetools.cachedmethod(lambda self: self.cache)
    def __getitem__(self, index):
        image = torch.Tensor(self.images[index])
        label = int(self.labels[index])

        return (image, label)
