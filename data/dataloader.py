import os
from torch.utils.data import DataLoader as TorchDataLoader

from data.dataset import CustomDataset


class ImageDataLoader():
    """
    Wrapper for PyTorch DataLoader class.
    Attributes
    -------
    cfg_file_path: string
        Path to the configuration file.
    """
    def __init__(self, data_folder: str, batch_size: int, shuffle: bool, train_mode: bool):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_mode = train_mode

        if self.train_mode:
            data = os.path.join(self.data_folder, 'train')
            self.dataset = self._build_dataset(self.data_folder, data)
        else:
            data = os.path.join(self.data_folder, 'test')
            self.dataset = self._build_dataset(self.data_folder, data)        

        self.dataloader = self._build_dataloader(self.batch_size, self.shuffle)

    def _build_dataset(self, data_folder, train_data):
        dataset = CustomDataset(root=data_folder, train_data=train_data)
        return dataset

    def _build_dataloader(self, batch_size, shuffle):
        dataloader = TorchDataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
