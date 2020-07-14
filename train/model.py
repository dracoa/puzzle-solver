import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channel = 3
        self.conv = nn.Conv2d(3, self.channel, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.channel * 48 * 48, 64)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, self.channel * 48 * 48)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
