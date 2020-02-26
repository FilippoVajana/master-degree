import torch
import torch.nn
import torch.optim
import torch.nn.functional as F

from ..runconfig import RunConfig
from ..dataloader import ImageDataLoader
from engine.models.lenet5 import LeNet5Trainer


class LeNet5ConcreteDropout(torch.nn.Module):
    def __init__(self):
        super(LeNet5ConcreteDropout, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)

        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.max_pool1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.max_pool2(x))

        #x = x.view(-1, 16*5*5)
        x = self.flat(x)

        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc1(x))

        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))

        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x  # softmax values must be evaluated during inference.

    def start_training(self, cfg: RunConfig, device):
        # init dataloader
        dataloader = ImageDataLoader(
            data_folder=cfg.data_folder,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle
        ).build(train_mode=True, max_items=cfg.max_items, validation_ratio=.2)

        # init model trainer
        trainer = LeNet5Trainer(cfg, device)

        # run training
        model, data = trainer.train(
            epochs=cfg.epochs,
            train_dataloader=dataloader[0],
            validation_dataloader=dataloader[1]
        )

        return model, data
