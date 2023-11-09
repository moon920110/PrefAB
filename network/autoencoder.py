import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, 1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 3, 1)
        self.conv3 = nn.Conv2d(8, 3, 3, 1)

        self.conv_t1 = nn.ConvTranspose2d(3, 8, 3, 1)
        self.conv_t2 = nn.ConvTranspose2d(16, 16, 3, 1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_t3 = nn.ConvTranspose2d(32, 3, 5, 1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x1_down = self.max_pool(x1)
        x2 = torch.relu(self.conv2(x1_down))
        x2_down = self.max_pool(x2)
        e = torch.relu(self.conv3(x2_down))  # 3 x 76 x 116 = 26448

        x4 = torch.relu(self.conv_t1(e))
        x4_up = self.upsample(x4)
        x4_skip_connect = torch.cat((x4_up, x2), dim=1)
        x5 = torch.relu(self.conv_t2(x4_skip_connect))
        x5_up = self.upsample(x5)
        s5_skip_connect = torch.cat((x5_up, x1), dim=1)

        d = torch.sigmoid(self.conv_t3(s5_skip_connect))

        # e = self.encoder(x)
        # d = self.decoder(e)
        return e, d


if __name__ == '__main__':
    pass