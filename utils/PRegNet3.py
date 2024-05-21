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
import os
from diffdrr.pose import convert


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


class DeResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(DeResBlock, self).__init__()
        self.deblock = nn.Sequential(
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose2d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.deblock(x) + x
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


class DeConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        # self.main = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)
        self.decon = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernal_size, stride, padding),  # [, 256, 24, 24]
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.decon(x)
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


class Decoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=128, first_out_channel=128):
        super(Decoder, self).__init__()

        c = first_out_channel
        # self.conv0 = ConvInsBlock(in_channel, c, 3, 1)
        self.deconv0 = DeConvInsBlock(in_channel, c // 2, 3, 1)

        self.deconv1 = nn.Sequential(
            # nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 20
            # nn.ConvTranspose2d(in_channels=c, out_channels=c // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(c // 2, c // 4, 3, stride=2, padding=1, output_padding=1),
            DeResBlock(c // 4)
        )

        self.deconv2 = nn.Sequential(
            # nn.Conv2d(c // 4, c // 8, kernel_size=3, stride=2, padding=1),  # 40
            nn.ConvTranspose2d(c // 4, c // 8, 3, stride=2, padding=1, output_padding=1),  # [, 128, 48, 48]
            DeResBlock(c // 8)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(c // 8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 80
            DeResBlock(16)
        )

        self.outde = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        out0 = self.deconv0(x)  # 1
        out1 = self.deconv1(out0)  # 1/2
        # print(out1.shape)
        out2 = self.deconv2(out1)  # 1/4
        # print(out2.shape)
        out3 = self.deconv3(out2)  # 1/8
        # print(out3.shape)
        out = self.outde(out3)
        # print(out.shape)
        return out


# pose intensity regNet
class PIRegNet(nn.Module):
    def __init__(self, in_channel=2, channels=16, num_classes=6):
        super(PIRegNet, self).__init__()
        self.encoder1 = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.encoder2 = Encoder(in_channel=1, first_out_channel=channels)
        # self.decoder2 = Decoder(in_channel=channels * 8, first_out_channel=channels * 8)
        # self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=channels)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(8 * channels, num_classes - 3)
        self.fc2 = nn.Linear(8 * channels, num_classes - 3)
        self.outMov = nn.Conv2d(1, 1, 3, 1, 1)
        # self.encoder2 = Encoder(in_channel=in_channel, first_out_channel=channels)
        # # self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc2 = nn.Linear(8 * channels, num_classes)
        # self.pointout = nn.Linear(8 * channels, 4)

    def forward(self, l1_mov, x2, mask, ct, scaler, offset, poseSC, use_regular=False):
        # encode stage
        x = torch.cat((l1_mov, x2, mask), dim=1)
        l1, l2, l3, l4 = self.encoder1(x)
        # F1, F2, F3, F4 = self.encoder_fixed(fixed)
        out1 = self.avgpool1(l4)
        out1_f = out1.view(out1.size(0), -1)
        out_rot = self.fc1(out1_f)
        out_trans = self.fc2(out1_f)
        # print(out1.shape)
        eu_out = torch.cat((out_rot, out_trans), dim=1)
        if use_regular:
            # moving_l2 = gene_drr(ct_name, out1, scalar)
            offset_mov = generate_drr(offset, scaler, poseSC, eu_out, ct, eu_out.shape[0])
            mov_out = self.outMov(offset_mov.float())
            # plot_drr(l2_mov, ticks=False)
            # plot_drr(x2, ticks=False)
            # plot_drr(pred_out, ticks=False)
            # plt.show()
            return out_rot, out_trans, mov_out
        else:
            return out1


def generate_drr(offset_info, intScale, pose_scale, T_tensor, ct_tensor, batchSize):
    vert_fold = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_ct'
    pred_drrs = torch.tensor([], dtype=torch.float64, device='cuda')
    for num in range(batchSize):
        # print(T_trans_tensor)
        # print(T_rot_tensor)
        mov_outputs = T_tensor.detach().cpu().numpy()
        mov_outputs = pose_scale.inverse_transform(mov_outputs)
        mov_outputs = torch.from_numpy(mov_outputs).to('cuda')
        ct_Dir = ct_tensor[num]
        single_rot = mov_outputs[num, :3]
        single_trans = mov_outputs[num, 3:]
        # print(single_trans)
        vert_name = os.path.split(ct_Dir)[-1]
        vert_path = os.path.join(vert_fold, vert_name)
        # print(vert_path)
        offset_x = offset_info.loc[offset_info['img_save_fold'] == vert_path]['tx'].item()
        offset_y = offset_info.loc[offset_info['img_save_fold'] == vert_path]['ty'].item()
        offset_z = offset_info.loc[offset_info['img_save_fold'] == vert_path]['tz'].item()
        single_trans = torch.unsqueeze(single_trans, dim=0)
        single_rot = torch.unsqueeze(single_rot, dim=0)
        origin_img = nib.load(ct_Dir)
        spacing = origin_img.header.get_zooms()
        spacing = np.array((spacing[0], spacing[1], spacing[2]), dtype=np.float64)
        # print(spacing)
        bx, by, bz = torch.tensor(origin_img.shape) * torch.tensor(spacing) / 2
        drr_mov = DRR(
            origin_img.get_fdata(),
            spacing,
            sdr=500,
            height=256,
            delx=2.0,
        ).to('cuda')
        rotation = torch.tensor([[torch.pi / 2, 0, torch.pi]], device='cuda') + single_rot
        translation = torch.tensor([[bx - offset_x, by + offset_y, bz + offset_z]], device='cuda') + single_trans
        reg = Registration(
            drr_mov,
            rotation.clone(),
            translation.clone(),
            parameterization="euler_angles",
            convention="ZYX",
        )
        gene_drr = reg()
        # print(reg())
        gene_drr = intScale(gene_drr)
        pred_drrs = torch.cat([pred_drrs, gene_drr], dim=0)
        del drr_mov
    return pred_drrs


if __name__ == '__main__':
    device = torch.device("cuda:0")
    img_size = 256
    model = PIRegNet(in_channel=3, channels=16, num_classes=6).to(device)
    f_x = torch.randn((1, 1, img_size, img_size)).to(device)
    m_x = torch.randn((1, 1, img_size, img_size)).to(device)
    gt_mask = torch.randn((1, 1, img_size, img_size)).to(device)
    y = model(m_x, f_x, gt_mask)
    summary(model, input_data=(f_x, m_x))
