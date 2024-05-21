import torch
import torch.nn as nn
from torchinfo import summary
from diffdrr.drr import DRR, Registration
from monai.transforms import ScaleIntensity
from diffdrr.visualization import plot_drr
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import math
from thop import profile


class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]


class PRegNet(nn.Module):
    def __init__(self, in_channel=2, channels=16, num_classes=6):
        super(PRegNet, self).__init__()
        self.encoder_mov = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.encoder_mov2 = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.encoder_mov3 = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.encoder_mov4 = Encoder(in_channel=in_channel, first_out_channel=channels)
        self.encoder_fixed = Encoder(in_channel=2, first_out_channel=channels)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channels * 8, num_classes)  # L4 * 8, L3 * 4, L2 * 2

        # self.encoder2_mov = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.encoder2_fixed = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc2 = nn.Linear(channels * 4, num_classes - 3)

    def forward(self, x1, x2, mask):
        x = torch.cat((x1, x2, mask), dim=1)
        # encode stage
        L1, L2, L3, L4 = self.encoder_mov(x)
        # _, _, _, L4 = self.encoder_mov1(x)
        # _, _, L3, _ = self.encoder_mov2(x)
        # _, L2, _, _ = self.encoder_mov3(x)
        # L1, _, _, _ = self.encoder_mov4(x)
        # F1, F2, F3, F4 = self.encoder_fixed(x2)
        # l2 = torch.cat((M2, F2), dim=1)
        # print(L3.shape)
        # out1 = self.avgpool1(L1)
        # out2 = self.avgpool1(L2)
        # out3 = self.avgpool1(L3)
        out = self.avgpool1(L4)
        # out = torch.cat((out1, out2, out3, out4), dim=1)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        # trans_out = self.avgpool2(l2)
        # trans_out = trans_out.view(trans_out.size(0), -1)
        # trans_out = self.fc2(trans_out)
        # out = torch.cat((rot_out, trans_out), dim=1)
        return out


if __name__ == '__main__':
    device = torch.device("cuda:0")
    img_size = 256
    model = PRegNet(in_channel=3, channels=16, num_classes=6).to(device)
    f_x = torch.randn((1, 1, img_size, img_size)).to(device)
    f_mask = torch.randn((1, 1, img_size, img_size)).to(device)
    m_x = torch.randn((1, 1, img_size, img_size)).to(device)
    y = model(m_x, f_x, f_mask)
    summary(model, input_data=(m_x, f_x, f_mask))
    flops, params = profile(model, inputs=(m_x, f_x, f_mask))
    # print(flops)
    print('flops:%.4sG' % (flops / 1024 / 1024 / 1024))
