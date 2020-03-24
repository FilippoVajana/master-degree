import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, data_path: str, transformation: None):
        self.transformation = transformation
        self.flip_transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor()])

        # init data
        self.images, self.labels = self.__load_data(data_path)

    def __load_data(self, path: str):
        images = np.load(os.path.join(path, "images.npy"))
        labels = np.load(os.path.join(path, "labels.npy"))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.Tensor(self.images[index])
        image = self.flip_transformation(image)

        if self.transformation:
            image = self.transformation(image)

        label = int(self.labels[index])
        return (image, label)
