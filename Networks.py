import torch
import torch.nn as nn
import torch.nn.functional as F

class conditionalCapsDcganD(nn.Module):
    def __init__(self, args):
        super(conditionalCapsDcganD, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        self.fc = nn.Linear(in_features=169, out_features=1)

    def forward(self, x, args):
        batch_size = x.size(0)
        x = self.main(x)
        x = x.view(batch_size, x.size(2)*x.size(3))
        if args['image_size'] > 64:
            x = self.fc(x)
        x = F.sigmoid(x)
        return x.squeeze(1)

