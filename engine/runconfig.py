import torch
import jsonpickle

class RunConfig():
    def __init__(self):
        # dataloaders params
        self.data_folder = "./data/mnist/train"
        self.batch_size = 1
        self.shuffle = True

        # trainer params
        self.model = torch.nn.Module()
        self.device = "cpu"
        self.epochs = 10
        self.optimizer_args = {
            'lr': 0.1,
            'weight_decay':1e-4,
            'betas':(0.9, 0.999),
            'eps':1e-08
            }

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