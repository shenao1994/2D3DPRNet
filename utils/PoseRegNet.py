import torch
import torch.nn as nn
from monai.networks.nets import resnet50
import math
from matplotlib import pyplot as plt
import cv2
import numpy as np


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, inchannel, block, layers):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.outlayer1 = nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=4, padding=1, output_padding=3, bias=False)
        self.outbn1 = nn.BatchNorm2d(512)
        self.outrelu1 = nn.ReLU(inplace=True)

        self.outlayer2 = nn.ConvTranspose2d(512, 64, kernel_size=3, stride=4, padding=1, output_padding=3, bias=False)
        self.outbn2 = nn.BatchNorm2d(64)
        self.outrelu2 = nn.ReLU(inplace=True)

        self.outlayer3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.outbn3 = nn.BatchNorm2d(3)
        self.outrelu3 = nn.ReLU(inplace=True)

        self.out4 = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=True)

        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.outlayer1(x)
        print(x.shape)
        x = self.outbn1(x)
        x = self.outrelu1(x)

        x = self.outlayer2(x)
        print(x.shape)
        x = self.outbn2(x)
        x = self.outrelu2(x)

        x = self.outlayer3(x)
        print(x.shape)
        x = self.outbn3(x)
        x = self.outrelu3(x)

        x = self.out4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class PoseRegNet(nn.Module):

    def __init__(self, n_channel):
        super(PoseRegNet, self).__init__()

        self.resnet = ResNet(n_channel, Bottleneck, [3, 4, 6, 3])

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out = self.resnet(x)
        return out


def load_data():
    drr_path = '../Data/pose_reg/weng/drr.png'
    coor_2d_path = '../Data/pose_reg/weng/idx2d.txt'
    coor_3d_path = '../Data/pose_reg/weng/idx3d.txt'
    drr_img = cv2.imread(drr_path)
    # print(drr_img.shape)
    coor_arr = np.zeros(drr_img.shape)
    coor_2d_list = []
    coor_3d_list = []
    for line in open(coor_2d_path):
        # print([int(line.strip('\n')) % 1536, int(line.strip('\n')) // 1536])
        coor_2d_list.append([int(line.strip('\n')) % 1536, int(line.strip('\n')) // 1536])
    for line in open(coor_3d_path):
        # print(line.strip('\n').split(' ')[0])
        coor_3d_list.append([float(line.strip('\n').split(' ')[0]), float(line.strip('\n').split(' ')[1]),
                             float(line.strip('\n').split(' ')[2])])
    print(coor_arr.shape)
    print(np.array(coor_2d_list).shape)
    print(np.array(coor_3d_list).shape)
    coor_2d_arr = np.array(coor_2d_list)
    coor_3d_arr = np.array(coor_3d_list)
    for coor2d, coor3d in zip(coor_2d_arr, coor_3d_arr):
        # print(coor2d)
        # print(coor3d)
        coor_arr[coor2d[0]][coor2d[1]][0] = coor3d[0]
        coor_arr[coor2d[0]][coor2d[1]][1] = coor3d[1]
        coor_arr[coor2d[0]][coor2d[1]][2] = coor3d[2]
    plt.subplot(131)
    plt.xticks([]), plt.yticks([])
    plt.imshow(coor_arr[:, :, 0], cmap='jet')
    plt.subplot(132)
    plt.xticks([]), plt.yticks([])
    plt.imshow(coor_arr[:, :, 1], cmap='jet')
    plt.subplot(133)
    plt.xticks([]), plt.yticks([])
    plt.imshow(coor_arr[:, :, 2], cmap='jet')
    plt.tight_layout()
    plt.savefig('coor_img.png', dpi=300)
    plt.show()
    coor_tensor = torch.tensor(coor_arr, dtype=torch.float32).to(device)
    drr_tensor = torch.tensor(drr_img, dtype=torch.float32).to(device)
    # print(coor_tensor.shape)
    # print(drr_tensor.shape)
    return coor_tensor, drr_tensor


if __name__ == '__main__':
    device = torch.device("cuda:0")
    coor, drr = load_data()
    coor = coor.permute(2, 0, 1)
    drr = drr.permute(2, 0, 1)
    coor = torch.unsqueeze(coor, 0)
    drr = torch.unsqueeze(drr, 0)
    print(coor.shape)
    # img1 = torch.rand(1, 1, 64, 64).to(device)
    # img2 = torch.rand(1, 1, 64, 64).to(device)
    # # features2 = torch.rand(1, 1, 6).to(device)
    model = PoseRegNet(6).to(device)
    pred = model(coor, drr)
    print(pred.shape)
