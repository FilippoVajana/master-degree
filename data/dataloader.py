import os
import jsonpickle

from data.dataset import CustomDataset
from torch.utils.data import DataLoader as TorchDataLoader


class DataLoaderConfig():
    """
    Configuration model for ImageDataLoader.
    """

    def __init__(self):
        self.id = ""
        self.data_folder = ""
        self.batch_size = 1
        self.train_data = False
        self.shuffle = False

    @staticmethod
    def load(cfg_path):
        """
        Loads a configuration from file.
        
        Parameters
        ----------
        cfg_path : string
            Path to the configuration file.
        
        Returns
        -------
        DataLoaderConfiguration
            A configuration instance.
        """

        with open(cfg_path, "r") as f:
            json_str = f.read()
            cfg_obj = jsonpickle.decode(json_str)
            return cfg_obj

    def save(self, save_path):
        """
        Saves the current configuration instance.
        
        Parameters
        ----------
        save_path : string
            Destination path.
        
        Returns
        -------
        string
            Configuration file path.
        """
        with open(save_path, "w") as f:
            json_obj = jsonpickle.encode(self)
            f.write(json_obj)
        return save_path


class ImageDataLoader():
    """
    Wrapper for PyTorch DataLoader class.
    
    Attributes
    -------
    cfg_file_path: string
        Path to the configuration file.
    """
    def __init__(self, cfg_file_path : str):
        self.cfg = DataLoaderConfig().load(cfg_file_path)
        self.dataset = self._build_dataset(self.cfg.data_folder, self.cfg.train_data)
        self.dataloader = self._build_dataloader(self.cfg.batch_size, self.cfg.shuffle)

    def _build_dataset(self, data_folder, train_data):
        dataset = CustomDataset(root=data_folder, train_data=train_data)
        return dataset

    def _build_dataloader(self, batch_size, shuffle):
        dataloader = TorchDataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader