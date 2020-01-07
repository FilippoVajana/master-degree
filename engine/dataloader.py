import os
import torch
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

    def __init__(self, data_folder: str, batch_size: int, shuffle: bool, transformation=None):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: add images transformation option
        self.transformation = None

    def build(self, train_mode: bool, max_items: int, validation_ratio=0.0):
        # check validation_ratio
        if validation_ratio < 0 or validation_ratio > 1:
            raise ValueError("validation_ratio must be in [0,1] interval")

        # check type of dataloader required
        if train_mode:
            self.data_folder = os.path.join(self.data_folder, 'train')
        else:
            self.data_folder = os.path.join(self.data_folder, 'test')
            validation_ratio = 0.0

        # build base dataset
        base_set = CustomDataset(self.data_folder, self.transformation)

        # check items count
        if (len(base_set) > max_items) and (max_items > 0):
            base_set.images = base_set.images[:max_items]
            base_set.labels = base_set.labels[:max_items]

        # split base dataset into Subset
        main_set_len = int(len(base_set) - (len(base_set) * validation_ratio))
        val_set_len = len(base_set) - main_set_len
        main_set, val_set = torch.utils.data.random_split(
            base_set, [main_set_len, val_set_len])

        # build dataloaders
        self.main_dl = TorchDataLoader(
            main_set, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_dl = TorchDataLoader(val_set, batch_size=1, shuffle=False)

        return (self.main_dl, self.val_dl)
