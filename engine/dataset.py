import os
import numpy as np
import torchvision
from torch.utils.data import Dataset

import cachetools

class CustomDataset(Dataset):
    def __init__(self, data_path: str):
        # init data
        images, labels = self.load_data(data_path)
        self.data = (images, labels)

        # init cache system
        self.cache = cachetools.LRUCache(maxsize=len(images))


    def load_data(self, path: str):
        images = np.load(os.path.join(path, "images.npy"))
        labels = np.load(os.path.join(path, "labels.npy"))
        return images, labels


    def __len__(self):
        return len(self.data[0])


    @cachetools.cachedmethod(lambda self: self.cache)
    def __getitem__(self, index):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        image = self.data[0][index]
        label = self.data[1][index]

        return (transform(image), int(label))
