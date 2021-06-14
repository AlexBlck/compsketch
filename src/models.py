from torch import nn
import torch.nn.functional as F


class Synth(nn.Module):
    def __init__(self):
        super(Synth, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv3 = nn.Conv2d(512, 832, 3, padding=1)
        self.pool = nn.MaxPool2d(3, 2)

        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.bn3 = nn.BatchNorm2d(num_features=832)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        return x
