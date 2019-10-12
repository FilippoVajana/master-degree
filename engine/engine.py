from engine import *
from configparser import SafeConfigParser
import numpy as np
from data.dataset import *


class Engine():
    def __init__(self, root_dir, config_file):
        self.root_dir = root_dir

        # read config file using configparser        
        self.config = self.__read_config(config_file)
        if self.config == None: 
            self.__save_dummy_config(config_file)

        # build training/testing dataloaders

        # init model object

        # init model trainer

        raise NotImplementedError()

    
    def __read_config(self, path):
        parser = SafeConfigParser()       
        logger.info(f'Reading config file at {path}')
        found = parser.read(path)
        if len(found) == 0:
            logger.error(f'Unable to read config file at {path}')    
            return None
        else:
            return parser    
        

    def __save_dummy_config(self, path):
        logger.info(f'Saving dummy config file at {path}')

        parser = SafeConfigParser()
        parser.add_section('training')
        parser.set('training', 'model', 'LeNet5')
        parser.set('training', 'device', 'cpu')
        parser.set('training', 'data', './data/')
        parser.set('training', 'epochs', '1')

        parser.add_section('testing')
        parser.set('testing', 'model', 'LeNet5')
        parser.set('testing', 'device', 'cpu')
        parser.set('testing', 'data', './data/')

        with open(path, 'w') as file:
            parser.write(file)


    def train_model(self):
        raise NotImplementedError()

    def test_model(self):
        raise NotImplementedError()