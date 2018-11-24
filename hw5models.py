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


