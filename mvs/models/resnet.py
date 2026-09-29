import torch.nn as nn
import torch.nn.functional as F


class Basic(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o)
        self.c2, self.b2 = nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o)
        self.sc = nn.Sequential() if stride == 1 and i == o else \
            nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        widths, layers, i = (64, 128, 256, 512), [], 64
        for k, o in enumerate(widths):
            layers += [Basic(i, o, 1 if k == 0 else 2), Basic(o, o, 1)]
            i = o
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, classes)

    def forward(self, x, y):
        return F.cross_entropy(self.fc(self.layers(self.stem(x)).mean((2, 3))).float(), y)
