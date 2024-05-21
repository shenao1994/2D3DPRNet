import torch
import torch.nn as nn
from monai.networks.nets import resnet50, densenet121, seresnet50
import math
from matplotlib import pyplot as plt
# import cv2
import numpy as np
from diffdrr.drr import DRR, Registration
from monai.transforms import ScaleIntensity
import nibabel as nib
import os


class unSRNet(nn.Module):

    def __init__(self, n_channel, out_channel):
        super(unSRNet, self).__init__()
        # self.resnet = resnet50(spatial_dims=2, n_input_channels=n_channel, num_classes=out_channel)
        self.seresnet = seresnet50(spatial_dims=2, in_channels=n_channel, num_classes=out_channel)
        # self.densenet = densenet121(spatial_dims=2, in_channels=n_channel, out_channels=out_channel)

    def forward(self, x1, x2, mask, ct, scaler, offset):
        x = torch.cat((x1, x2, mask), dim=1)
        # out = self.resnet(x)
        pose_out = self.seresnet(x)
        # out = self.densenet(x)
        out = get_drr(offset, scaler, pose_out[:, :3], pose_out[:, 3:], ct, pose_out.shape[0])
        return out


def get_drr(offset_info, intScale, T_rot_tensor, T_trans_tensor, ct_tensor, batchSize):
    vert_fold = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_ct'
    pred_drrs = torch.tensor([], dtype=torch.float64, device='cuda')
    for num in range(batchSize):
        # print(T_trans_tensor)
        # print(T_rot_tensor)
        ct_Dir = ct_tensor[num]
        single_rot = T_rot_tensor[num, :]
        single_trans = T_trans_tensor[num, :]
        # print(ct_Dir)
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
        print(reg())
        gene_drr = intScale(gene_drr)
        pred_drrs = torch.cat([pred_drrs, gene_drr], dim=0)
        del drr_mov
    return pred_drrs


if __name__ == '__main__':
    print(torch.cuda.is_available())
