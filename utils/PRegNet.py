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

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
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
        self.encoder1 = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=channels)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(8 * channels, num_classes)

        self.encoder2 = Encoder(in_channel=in_channel, first_out_channel=channels)
        # self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=channels)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(8 * channels, num_classes)
        self.pointout = nn.Linear(8 * channels, 4)

    def forward(self, l1_mov, x2, mask, ct, scaler, offset, poseSC, use_regular=False):
        # encode stage
        x = torch.cat((l1_mov, x2, mask), dim=1)
        l1, l2, l3, l4 = self.encoder1(x)
        # F1, F2, F3, F4 = self.encoder_fixed(fixed)
        out1 = self.avgpool1(l4)
        out1_f = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1_f)
        if use_regular:
            # moving_l2 = gene_drr(ct_name, out1, scalar)
            l2_mov = generate_drr(offset, scaler, poseSC, out1, ct, out1.shape[0])
            # plot_drr(l2_mov, ticks=False)
            # plot_drr(x2, ticks=False)
            # plot_drr(mask, ticks=False)
            # plt.show()
            x2 = torch.cat((l2_mov.float(), x2, mask), dim=1)
            l1, l2, l3, l4 = self.encoder2(x2)
            out = self.avgpool2(l4)
            out_feats = out.view(out.size(0), -1)
            # out = torch.concat((out, out1_f), dim=1)
            out = self.fc2(out_feats)
            point_out = self.pointout(out_feats)
            return out, point_out
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
        # print(ct_Dir)
        # print(single_trans)
        vert_name = os.path.split(ct_Dir)[-1]
        vert_path = os.path.join(vert_fold, vert_name)
        print(vert_path)
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
    model = PRegNet(in_channel=3, channels=16, num_classes=6).to(device)
    f_x = torch.randn((1, 1, img_size, img_size)).to(device)
    m_x = torch.randn((1, 1, img_size, img_size)).to(device)
    gt_mask = torch.randn((1, 1, img_size, img_size)).to(device)
    y = model(m_x, f_x, gt_mask)
    summary(model, input_data=(f_x, m_x))
