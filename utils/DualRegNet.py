import torch
import torch.nn as nn
from monai.networks.nets import resnet50, densenet121, seresnet50, seresnext50, UNet
import math
from matplotlib import pyplot as plt
# import cv2
import numpy as np
from thop import profile
from torchinfo import summary


class DRegNet(nn.Module):

    def __init__(self, n_channel, out_channel):
        super(DRegNet, self).__init__()
        # self.inplane_backbone = densenet121(spatial_dims=2, in_channels=n_channel, out_channels=out_channel - 3)
        self.outplane_backbone = densenet121(spatial_dims=2, in_channels=n_channel, out_channels=out_channel - 3)

    def forward(self, x1, x2, mask1, x3, x4, mask2):
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

        ap_x = torch.cat((x1, x2, mask1), dim=1)
        la_x = torch.cat((x3, x4, mask2), dim=1)
        # inplane_out = self.inplane_backbone(ap_x)
        outplane_out = self.outplane_backbone(la_x)
        # out = torch.cat([outplane_out, inplane_out], dim=1)
        # indices1 = [0, 1, 5, 3, 4, 2]
        # 使用这个索引数组进行广播，交换列
        # out = out[:, indices1]
        # indices2 = [0, 1, 2, 3, 5, 4]
        # out = out[:, indices2]
        return outplane_out


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    device = torch.device("cuda:0")
    img_size = 256
    model = DRegNet(2, 6).to(device)
    ap_x = torch.randn((4, 1, img_size, img_size)).to(device)
    ap_y = torch.randn((4, 1, img_size, img_size)).to(device)
    la_x = torch.randn((4, 1, img_size, img_size)).to(device)
    la_y = torch.randn((4, 1, img_size, img_size)).to(device)
    y = model(ap_x, ap_y, la_x, la_y)
    print(y.shape)
    # summary(model, input_data=(ap_x, ap_y, la_x, la_y))
    # flops, params = profile(model, inputs=(ap_x, ap_y, la_x, la_y))
    # print(flops)
    # print('flops:%.4sG' % (flops / 1024 / 1024 / 1024))
