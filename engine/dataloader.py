import os
from torch.utils.data import DataLoader as TorchDataLoader

from .dataset import CustomDataset


class ImageDataLoader():
    """
    Wrapper for PyTorch DataLoader class.
    Attributes
    -------
    cfg_file_path: string
        Path to the configuration file.
    """

    def __init__(self, data_folder: str, batch_size: int, shuffle: bool, train_mode: bool, max_items: int):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_mode = train_mode
        self.max_items = max_items

        if self.train_mode:
            data = os.path.join(self.data_folder, 'train')
        else:
            data = os.path.join(self.data_folder, 'test')

        self.dataset = self._build_dataset(data)
        self.dataloader = self._build_dataloader(self.batch_size, self.shuffle)

    def _build_dataset(self, data_folder):
        dataset = CustomDataset(data_folder)

        if (len(dataset) > self.max_items) and (self.max_items > 0):
            dataset.images = dataset.images[:self.max_items]
            dataset.labels = dataset.labels[:self.max_items]

        return dataset

    def _build_dataloader(self, batch_size, shuffle):
        dataloader = TorchDataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
