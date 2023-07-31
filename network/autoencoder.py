import torch
import torch.nn as nn

from dataloader.again_reader import AgainReader


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, 1, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 3, 3, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 8, 3, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 16, 3, 1, dilation=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 3, 5, 1, dilation=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return e, d


if __name__ == '__main__':
    pass