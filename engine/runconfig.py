import jsonpickle


class RunConfig():
    def __init__(self):
        # dataloaders params
        self.data_folder = "./data/mnist/"
        self.max_items = -1
        self.batch_size = 256
        self.shuffle = True

        # trainer params
        self.models = [
            'LeNet5',
            'LeNet5SimpleDropout',
            'LeNet5ConcreteDropout'
        ]
        self.epochs = 20
        self.optimizer_args = {
            'lr': 0.01,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
        self.dirty_labels = 0.0

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
