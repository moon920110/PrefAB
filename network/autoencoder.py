import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 5, )
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 3, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(3)

        # self.conv_t1 = nn.ConvTranspose2d(3, 8, 3, 2, 1, 1)
        # self.bn_t1 = nn.BatchNorm2d(8)
        # self.conv_t2 = nn.ConvTranspose2d(16, 16, 5)
        # self.bn_t2 = nn.BatchNorm2d(16)
        # self.conv_t3 = nn.ConvTranspose2d(32, 3, 5, 2, 2, 1)
        # self.bn_t3 = nn.BatchNorm2d(3)

        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.bn1(self.lrelu(self.conv1(x)))
        x2 = self.bn2(self.lrelu(self.conv2(x1)))
        e = self.bn3(self.lrelu(self.conv3(x2)))

        # x4 = self.bn_t1(self.lrelu(self.conv_t1(e)))
        # x4_skip_connect = torch.cat((x4, x2), dim=1)
        # x5 = self.bn_t2(self.lrelu(self.conv_t2(x4_skip_connect)))
        # s5_skip_connect = torch.cat((x5, x1), dim=1)
        # d = torch.sigmoid(self.bn_t3(self.conv_t3(s5_skip_connect)))

        return e, None#, d


if __name__ == '__main__':
    pass