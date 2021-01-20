import torch
import torch.nn as nn
import torch.nn.functional as F


class MISLnet(nn.Module):
    def __init__(self):
        super(MISLnet, self).__init__()

        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[3, 1, 5, 5]), requires_grad=True)
        self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=4)
        self.conv2 = nn.Conv2d(96, 64, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 128, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(6272, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)

    def set_constrain(self):
        """
        Constrained Convolutional Layer

        """
        self.const_weight.data[:, :, 2, 2] = 0
        for i in range(3):
            self.const_weight.data[i, 0, :, :] = (
                self.const_weight.data[i, 0, :, :] / self.const_weight.data[i, 0, :, :].sum()
            )
            self.const_weight.data[i, 1, :, :] = (
                self.const_weight.data[i, 1, :, :] / self.const_weight.data[i, 1, :, :].sum()
            )
            self.const_weight.data[i, 2, :, :] = (
                self.const_weight.data[i, 2, :, :] / self.const_weight.data[i, 2, :, :].sum()
            )
        self.const_weight.data[:, :, 2, 2] = -1

    def forward(self, x):
        # Constrained-CNN
        self.set_constrain()
        x = F.conv2d(x, self.const_weight)
        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.avg_pool(torch.tanh(x))
        # Fully Connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        logist = self.fc3(x)
        output = F.softmax(logist, dim=1)
        return output