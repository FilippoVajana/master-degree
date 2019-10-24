import torch
import torch.nn   
import torch.optim 
import torch.nn.functional as F

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.avg_pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.avg_pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=86)
        self.fc3 = torch.nn.Linear(in_features=86, out_features=10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.avg_pool1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.avg_pool2(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x # softmax values must be evaluated during inference.

    # TODO: add train() method


# TODO: implement trainer class