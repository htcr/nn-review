import torch
import torch.nn as nn
import torch.nn.functional as F

class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 36)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, num_class)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNetBN(nn.Module):
    def __init__(self, num_class=10):
        super(LeNetBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(16*4*4, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, num_class)

    def forward(self, x):
        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 36)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

