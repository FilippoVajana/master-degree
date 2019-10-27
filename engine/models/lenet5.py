import torch
import torch.nn   
import torch.optim 
import torch.nn.functional as F

from ..trainer import GenericTrainer

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.avg_pool1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.avg_pool2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x # softmax values must be evaluated during inference.


"""
LeNet and MLP were trained for 20 epochs using the Adam optimizer (Kingma &
Ba, 2014) and used ReLU activation functions. For stochastic methods, we averaged 300 sample
predictions to yield a predictive distribution, and the ensemble model used 10 instances trained from
independent random initializations. The LeNet architecture (LeCun et al.,
1998) applies two convolutional layers 3x3 kernels of 32 and 64 filters respectively) followed by two
fully-connected layers with one hidden layer of 128 activations; dropout was applied before each
fully-connected layer. We employed hyperparameter tuning (See Section A.7) to select the training
batch size, learning rate, and dropout rate.
"""

class LenetTrainer(GenericTrainer):
    def __init__(self, model, device):
        super().__init__(model, device)