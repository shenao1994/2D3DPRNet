import torch
import torch.nn as nn
from monai.networks.nets import resnet50, densenet121, seresnet50, seresnext50, UNet
import math
from matplotlib import pyplot as plt
# import cv2
import numpy as np
from thop import profile
from torchinfo import summary


class SRNet(nn.Module):

    def __init__(self, n_channel, out_channel, backboneName='resnet'):
        super(SRNet, self).__init__()
        # self._3D_conv = nn.Sequential(
        #     nn.Conv3d(1, 4, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv3d(4, 8, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv3d(8, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv3d(16, 8, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv3d(8, 3, 3, 1, 1),
        #     nn.ReLU()
        # )
        self.backboneName = backboneName
        if backboneName == 'resnet':
            self.resnet = resnet50(spatial_dims=2, n_input_channels=n_channel, num_classes=out_channel)
        elif backboneName == 'seresnet':
            self.seresnet = seresnet50(spatial_dims=2, in_channels=n_channel, num_classes=out_channel)
        elif backboneName == 'densenet':
            self.densenet = densenet121(spatial_dims=2, in_channels=n_channel, out_channels=out_channel)
        elif backboneName == 'seresnext':
            # self.backbone = seresnet50(spatial_dims=2, in_channels=n_channel, num_classes=out_channel)
            self.seresnext = seresnext50(spatial_dims=2, in_channels=n_channel, num_classes=out_channel)

    def forward(self, x1, x2, mask):
        # w1_x1 = x1 * seg1
        # w2_x2 = x2 * seg2
        # cat_x1 = torch.cat((x1, seg1), dim=1)
        # cat_x2 = torch.cat((x2, seg2), dim=1)
        # print(x1.shape)
        # print(x2.shape)
        # fig, axs = plt.subplots(2, 2)
        # # 在第一个子图中显示张量1
        # axs[0, 0].imshow(x1[0, 0, ].cpu().numpy())
        # axs[0, 0].set_title('Tensor 1')
        #
        # # 在第二个子图中显示张量2
        # axs[0, 1].imshow(mask1[0, 0, ].cpu().numpy())
        # axs[0, 1].set_title('Tensor 2')
        #
        # # 在第三个子图中显示张量3
        # axs[1, 0].imshow(x2[0, 0, ].cpu().numpy())
        # axs[1, 0].set_title('Tensor 3')
        #
        # # 在第四个子图中显示张量4
        # axs[1, 1].imshow(mask2[0, 0, ].cpu().numpy())
        # axs[1, 1].set_title('Tensor 4')
        # plt.show()
        # x = torch.cat((x1, mask1, x2, mask2, x3, x4), dim=1)
        x = torch.cat((x1, x2, mask), dim=1)
        if self.backboneName == 'resnet':
            out = self.resnet(x)
        elif self.backboneName == 'seresnet':
            out = self.seresnet(x)
        elif self.backboneName == 'densenet':
            out = self.densenet(x)
        elif self.backboneName == 'seresnext':
            # self.backbone = seresnet50(spatial_dims=2, in_channels=n_channel, num_classes=out_channel)
            out = self.seresnext(x)
        # out = self.resnet(x)
        # out = self.seresnet(x)
        # out = self.seresnext(x)
        # out = self.densenet(x)
        return out


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    device = torch.device("cuda:0")
    img_size = 256
    model = SRNet(3, 6).to(device)
    f_x = torch.randn((1, 1, img_size, img_size)).to(device)
    f_mask = torch.randn((1, 1, img_size, img_size)).to(device)
    m_x = torch.randn((1, 1, img_size, img_size)).to(device)
    y = model(m_x, f_x, f_mask)
    summary(model, input_data=(m_x, f_x, f_mask))
    flops, params = profile(model, inputs=(m_x, f_x, f_mask))
    # print(flops)
    print('flops:%.4sG' % (flops / 1024 / 1024 / 1024))
