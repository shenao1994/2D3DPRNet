import math
import pandas as pd
import os
import numpy as np
import monai
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, ToTensord, \
    EnsureChannelFirstd, ScaleIntensity, Resized, Resize, LoadImage
from monai.data import CSVSaver
from monai.losses import SSIMLoss
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from utils.PRegNet import PRegNet
# from utils.PRegNet2 import PRegNet
from utils.SpatialRegNet import SRNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
from diffdrr.drr import DRR, Registration
from diffdrr.metrics import NormalizedCrossCorrelation2d
import SimpleITK as sitk
from diffdrr.visualization import plot_drr
from monai.losses import BendingEnergyLoss, DiceLoss
import nibabel as nib
from sklearn.metrics import mean_absolute_error
from utils.MNCC_loss import MaskedNCC, GradNCC
from utils.unsRegNet import unSRNet
from torchvision.ops import masks_to_boxes
from utils.PRegNet3 import PIRegNet
import json
from diffdrr.pose import random_rigid_transform, convert
from create_transforms import get_transform_parameters
import random
import cv2
from typing import Tuple
from utils.so3Geodesic_loss import GeodesicSE3


def get_data(gt_fold, mov_fold, mask_fold, ct_fold, pose_data):
    ap_mov_list = []
    ap_gt_list = []
    ap_mask_list = []
    ct_list = []
    pose_list = []
    verb_list = ['L1', 'L2', 'L3', 'L4', 'L5']
    for verb in verb_list:
        for index, row in pose_data.iterrows():
            name = '_'.join(row['name'].split('_')[:-2])
            num = int(row['name'].split('_')[-2])
            gt_ap_mask_path = os.path.join(mask_fold, name + '_' + verb + '_' + str(num) + '_ap.nii.gz')
            # gt_lal_mask_path = os.path.join(mask_fold, name + '_' + verb + '_' + str(num) + '_lal.nii.gz')
            # gt_lar_mask_path = os.path.join(mask_fold, name + '_' + verb + '_' + str(num) + '_lar.nii.gz')
            gt_ap_drr_path = os.path.join(gt_fold, name + '_' + str(num) + '_ap.nii.gz')
            mov_ap_drr_path = os.path.join(mov_fold, name + '_' + verb + '_ap.nii.gz')
            vertebra_path = os.path.join(ct_fold, name + '_' + verb + '.nii.gz')
            # print(gt_ap_mask_path)
            # if os.path.exists(gt_ap_mask_path) and num < 20 and os.path.getsize(mov_ap_drr_path) / 1024 >= 125:
            if os.path.exists(gt_ap_mask_path) and os.path.getsize(mov_ap_drr_path) / 1024 >= 29:
                ap_mov_list.append(mov_ap_drr_path)
                # la_mov_list.append(os.path.join(mov_fold, name + '_' + verb + '_lar.nii.gz'))
                ap_gt_list.append(gt_ap_drr_path)
                # la_gt_list.append(os.path.join(gt_fold, name + '_' + str(num) + '_lar.nii.gz'))
                ap_mask_list.append(gt_ap_mask_path)
                # la_mask_list.append(gt_lar_mask_path)
                ct_list.append(vertebra_path)
                pose_list.append([float(row['rx']), float(row['ry']), float(row['rz']),
                                  float(row['tx']), float(row['ty']), float(row['tz'])])
    print(len(ap_mov_list))
    print(len(ap_gt_list))
    print(len(ap_mask_list))
    print(len(pose_list))
    print(len(ct_list))
    # print(pose_list)
    # print(ap_mov_list)
    # print(ap_gt_list)
    # print(ap_mask_list)
    # print(ct_list)
    return ap_mov_list, ap_gt_list, ap_mask_list, ct_list, np.array(pose_list)


def get_verse_data(gt_fold, mov_fold, mask_fold, ct_fold, pose_data):
    ap_mov_list = []
    ap_gt_list = []
    ap_mask_list = []
    ct_list = []
    pose_list = []
    verb_list = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    for verb in verb_list:
        for index, row in pose_data.iterrows():
            name = '_'.join(row['name'].split('_')[:-2])
            # print(name)
            num = int(row['name'].split('_')[-2])
            gt_ap_mask_path = os.path.join(mask_fold, name + '_seg_' + verb + '_' + str(num) + '_ap.nii.gz')
            # print(gt_ap_mask_path)
            # gt_lal_mask_path = os.path.join(mask_fold, name + '_' + verb + '_' + str(num) + '_lal.nii.gz')
            # gt_lar_mask_path = os.path.join(mask_fold, name + '_' + verb + '_' + str(num) + '_lar.nii.gz')
            gt_ap_drr_path = os.path.join(gt_fold, name + '_' + str(num) + '_ap.nii.gz')
            mov_ap_drr_path = os.path.join(mov_fold, name + '_' + verb + '_ap.nii.gz')
            vertebra_path = os.path.join(ct_fold, name + '_' + verb + '.nii.gz')
            # print(gt_ap_mask_path)
            # if os.path.exists(gt_ap_mask_path) and num < 20 and os.path.getsize(mov_ap_drr_path) / 1024 >= 125:
            if os.path.exists(gt_ap_mask_path) and os.path.getsize(mov_ap_drr_path) / 1024 >= 10:
                ap_mov_list.append(mov_ap_drr_path)
                # la_mov_list.append(os.path.join(mov_fold, name + '_' + verb + '_lar.nii.gz'))
                ap_gt_list.append(gt_ap_drr_path)
                # la_gt_list.append(os.path.join(gt_fold, name + '_' + str(num) + '_lar.nii.gz'))
                ap_mask_list.append(gt_ap_mask_path)
                # la_mask_list.append(gt_lar_mask_path)
                ct_list.append(vertebra_path)
                pose_list.append([float(row['rx']), float(row['ry']), float(row['rz']),
                                  float(row['tx']), float(row['ty']), float(row['tz'])])
    print(len(ap_mov_list))
    print(len(ap_gt_list))
    print(len(ap_mask_list))
    print(len(pose_list))
    print(len(ct_list))
    # print(pose_list)
    # print(ap_mov_list)
    # print(ap_gt_list)
    # print(ap_mask_list)
    # print(ct_list)
    return ap_mov_list, ap_gt_list, ap_mask_list, ct_list, np.array(pose_list)


def model_engine():
    # drr路径
    mov_drr_path = 'Data/total_data/256/verse/drr_initial'
    gt_path = 'Data/total_data/256/verse/drr_gt2'
    mask_path = 'Data/total_data/256/verse/gt_mask2'
    vertebra_ct_path = 'Data/total_data/256/verse/vertebra_ct'
    # 位姿路径
    excel_path = 'Data/total_data/256/verse/total_pose2_ap.csv'
    pose_data = pd.read_csv(excel_path)
    ap_mov_list, ap_gt_list, ap_mask_list, ct_list, pose_arr = \
        get_verse_data(gt_path, mov_drr_path, mask_path, vertebra_ct_path, pose_data)
    SS = StandardScaler()
    pose_arr = SS.fit_transform(pose_arr)
    # print(ap_mask_list[-20:])
    # print(la_mask_list[-20:])
    pickle.dump(SS, open('reg_model/MICCAI/pose_scaler2.pkl', 'wb'))
    start_num = 0
    data_num = -1
    ap_gt_train, ap_gt_test, ap_mask_train, ap_mask_test, \
    ap_mov_train, ap_mov_test, ct_train, ct_test, pose_train, pose_test = \
        train_test_split(ap_gt_list[start_num:data_num], ap_mask_list[start_num:data_num],
                         ap_mov_list[start_num:data_num], ct_list[start_num:data_num],
                         pose_arr[start_num:data_num], test_size=0.2, random_state=1)
    train_files = [{"ap_gt_img": ap_gt_img, "ap_gt_mask": ap_gt_mask,
                    "ap_mov_img": ap_mov_img, "ct_img": ct_img, "pose": pose}
                   for ap_gt_img, ap_gt_mask, ap_mov_img, ct_img, pose in
                   zip(ap_gt_train, ap_mask_train, ap_mov_train, ct_train, pose_train)]
    # 将验证集变成dict形式
    val_files = [{"ap_gt_img": ap_gt_img, "ap_gt_mask": ap_gt_mask,
                  "ap_mov_img": ap_mov_img, "ct_img": ct_img, "pose": pose}
                 for ap_gt_img, ap_gt_mask, ap_mov_img, ct_img, pose in
                 zip(ap_gt_test, ap_mask_test, ap_mov_test, ct_test, pose_test)]
    # print(val_files)
    pre_model_path = 'reg_model/MICCAI/RegNet_2level_pre_model1.pth'
    model_path = 'reg_model/MICCAI/unsup_model1.pth'
    # [: 10]
    # training(train_files, val_files, model_path, SS, True)
    evaluta_model(val_files[1900:2000], model_path, SS)


def training(train_files, val_files, save_dir, StdS=None, useCT=False):
    input_size = 128
    keys = ["ap_gt_img", "ap_gt_mask",
            "ap_mov_img"]
    train_transforms = Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            # EnsureChannelFirstd(keys=keys),
            # AddChanneld(keys=keys),
            # Resized(keys=keys[:-1], spatial_size=(input_size, input_size), mode="bilinear", align_corners=True),
            # 像素归一化
            ScaleIntensityd(keys=["ap_gt_img", "ap_mov_img"]),
            # NormalizeIntensityd(keys=keys),
            # 尺寸归一化
            # ConcatItemsd(keys=keys, name="inputs"),
            # 转换为tensor形式
            # ToTensord(keys=keys),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            # Resized(keys=keys[:-1], spatial_size=(input_size, input_size), mode="bilinear", align_corners=True),
            # 像素归一化
            ScaleIntensityd(keys=["ap_gt_img", "ap_mov_img"]),
            # ConcatItemsd(keys=keys, name="inputs"),
            # NormalizeIntensityd(keys=keys),
            # ToTensord(keys=keys),
        ]
    )
    # 每隔2次验证一次
    val_interval = 2
    # 保存最佳的MSE
    best_metric = 1000
    # 验证集MSE最低时的epoch次数
    best_metric_epoch = -1
    # 用于保存训练集的loss
    epoch_loss_values = list()
    # 用于保存测试集的loss
    val_loss_list = []
    # 用于保存每次验证时的MSE
    metric_values = list()
    writer = SummaryWriter()
    device_ids = [0, 1, 2, 3]  # 可用GPU
    # 学习率
    lr = 0.001
    # 批处理大小
    batch_size = 8
    print(len(train_files))
    # 加载训练集
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size * len(device_ids), num_workers=0)
    # train_data = monai.utils.misc.first(train_loader)
    # create a validation data loader
    # 加载测试集
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    # 模型加载
    model = PIRegNet(in_channel=3, channels=16, num_classes=6).to(device)
    # model = model.cuda(device=device_ids[0])
    # model = SRNet(3, 6, backboneName='densenet').to(device)
    # model = unSRNet(3, 6).to(device)
    # Create Net, MSE and Adam optimizer 损失函数和优化器
    MSE_loss = torch.nn.MSELoss()
    geodesic = GeodesicSE3()
    # Theta_loss = torch.nn.MSELoss()
    NCC_loss = MaskedNCC()
    # GNCC_loss = GradNCC
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    info_path = 'Data/total_data/256/verse/dataset_record2.csv'
    info_data = pd.read_csv(info_path)
    # resize_transform = Resize(spatial_size=(128, 128), mode='bilinear', align_corners=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['reg_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
        epoch_num = 100 - start_epoch
    else:
        start_epoch = 0
        epoch_num = 100
    # start a typical PyTorch training
    # checkpoint_interval = 100
    # 训练循环
    # rot_tensor = ((-torch.pi / 9 - torch.pi / 9) * torch.rand(batch_size, 3) + torch.pi / 9).to(device)
    # trans_tensor = ((-10 - 11) * torch.rand(batch_size, 3) + 11).to(device)
    # val_rot_tensor = ((-torch.pi / 9 - torch.pi / 9) * torch.rand(batch_size, 3) + torch.pi / 9).to(device)
    # val_trans_tensor = ((-10 - 11) * torch.rand(batch_size, 3) + 11).to(device)
    # rot_tensor = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    # trans_tensor = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    # val_rot_tensor = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    # val_trans_tensor = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    # scaler = pickle.load(open('reg_model/pose_scaler2.pkl', 'rb'))
    # mov_drrs, init_trans = get_drr(trans_tensor, rot_tensor, batch_size, device)
    scaleInen = ScaleIntensity()
    # scale_mov_drrs = scaleInen(mov_drrs)
    params = []
    train_losses = []
    for epoch in range(start_epoch, epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        # print(scheduler.get_last_lr())
        model.train()
        epoch_loss = 0
        step = 0
        # for i, (inputs, labels, imgName) in enumerate(train_loader):
        for batch_data in train_loader:
            step += 1
            # print(batch_data['mov_ct_meta_dict'])
            # spacing = batch_data['mov_ct_meta_dict']['pixdim'][0][1:4]
            # gt_imgs, mov_cts = batch_data["gt_img"].to(device), batch_data["mov_ct"].to(device)
            gt_ap_imgs, gt_ap_masks, mov_ap_imgs = \
                batch_data["ap_gt_img"].to(device), batch_data["ap_gt_mask"].to(device), \
                batch_data["ap_mov_img"].to(device),
            pose_param = batch_data["pose"].to(device)
            ct_path = batch_data["ct_img"]
            # gt_T = StdS.inverse_transform(pose_param.cpu().numpy())
            # gt_T = torch.from_numpy(gt_T).to(device)
            # pred_imgs = get_drr(gt_T[:, 3:], gt_T[:, :3], batch_size, device)
            # print(batch_data['gt_img_meta_dict']['filename_or_obj'])
            # print(pose_param.shape)
            # print(mov_drrs.shape)
            # plot_drr(scale_mov_drrs, ticks=False)
            # print(gt_imgs.shape)
            # plot_drr(gt_imgs, ticks=False)
            # plt.show()
            # print(scalar.inverse_transform(pose_param.cpu().numpy()))
            # gt_imgs = torch.permute(gt_imgs, (0, 1, 3, 2))
            # print(gt_ap_imgs.shape)
            # resized_gt_aps = resize_transform(gt_ap_imgs[0, :])
            # resized_gt_aps = torch.unsqueeze(resized_gt_aps, 0)
            # resized_gt_masks = resize_transform(gt_ap_masks[0, :])
            # resized_gt_masks = torch.unsqueeze(resized_gt_masks, 0)
            # resized_mov_aps = resize_transform(mov_ap_imgs[0, :])
            # resized_mov_aps = torch.unsqueeze(resized_mov_aps, 0)
            # plot_drr(resized_gt_masks, ticks=False)
            # plot_drr(resized_mov_aps, ticks=False)
            # plt.show()
            # inputs1 = torch.cat((resized_gt_aps, resized_gt_masks, resized_mov_aps), dim=1)
            # plot_drr(mov_ap_imgs, ticks=False)
            # plot_drr(gt_ap_imgs, ticks=False)
            # plot_drr(gt_ap_masks, ticks=False)
            # plt.show()
            # outputs = model(mov_ap_imgs, gt_ap_imgs, gt_ap_masks)
            # spacing = batch_data['ct_img_meta_dict']['pixdim'][0][1:4]
            # print(spacing)
            # if useCT:
            # bbx = masks_to_boxes(torch.squeeze(gt_ap_masks, dim=1))
            # bbx = scaleInen(bbx)
            # print(bbx)
            output_rots, output_trans, out_mov = model(mov_ap_imgs, gt_ap_imgs, gt_ap_masks, ct_path, scaleInen,
                                                       info_data, StdS, True)
            out_mov = scaleInen(out_mov)
            # mov_outputs = outputs.detach().cpu().numpy()
            # mov_outputs = StdS.inverse_transform(mov_outputs)
            # mov_outputs = torch.from_numpy(mov_outputs).to(device)
            # print(mov_outputs)
            # pred_drrs = get_drr(info_data, scaleInen, mov_outputs[:, :3], mov_outputs[:, 3:],
            #                     ct_path, mov_ap_imgs.shape[0], device)  # 生成预测后的drr图像
            # plot_drr(pred_drrs, ticks=False)
            # plot_drr(mov_ap_imgs, ticks=False)
            # plot_drr(gt_ap_imgs, ticks=False)
            # plt.show()
            # gt_ap_imgs = gt_ap_imgs.type(torch.FloatTensor).to(device)
            # gt_ap_masks = gt_ap_masks.type(torch.FloatTensor).to(device)
            # ncc_loss = NCC_loss(pred_drrs.float(), gt_ap_imgs, gt_ap_masks)
            # print(ncc_loss)
            # print(pred_drrs.shape)
            # else:
            #     outputs = model(mov_ap_imgs, gt_ap_imgs, gt_ap_masks)
            # out_rot = np.array([[math.pi / 2, 0, math.pi]]) + mov_outputs[:, :3]
            # out_trans = np.array([init_trans]) + mov_outputs[:, 3:]
            # scale_pred_drrs = scaleInen(pred_drrs)
            # print(mov_outputs[:, 0].tolist())
            # plot_drr(pred_drrs, ticks=False)
            # plt.show()
            # print(mov_outputs)
            # print(outputs)
            # y_pred_transform = y_pred_transform.squeeze()
            # print(outputs.shape)
            # trans_tensor[:, :3] = mov_outputs[:, :3]
            # rot_tensor[:, :3] = mov_outputs[:, 3:]
            # print(y_ordinal_encoding.shape)
            # 计算损失函数
            # print(outputs.shape)
            # loss_trans = MSE_loss(output_trans, pose_param.float()[:, 3:])
            rot_pose = convert(pose_param.float()[:, :3], pose_param.float()[:, 3:], parameterization='euler_angles', convention="ZYX")
            pred_pose = convert(output_rots, output_trans, parameterization='euler_angles', convention="ZYX")
            # print(pred_pose.matrix)
            loss_geodesic = geodesic(pred_pose, rot_pose)
            # loss_ssm = MSE_loss(out_mov, gt_ap_imgs * gt_ap_masks)
            # print(outpoints)
            # loss_theta = Theta_loss(outpoints, bbx)
            # print(loss_mse)
            # plot_drr(gt_imgs, ticks=False)
            # plt.show()
            # plot_drr(scale_pred_drrs, ticks=False)
            # plt.show()
            loss_ncc = NCC_loss(out_mov, gt_ap_imgs, gt_ap_masks)
            # print('ncc{}'.format(ncc_loss.mean()))
            # loss = 1 - ncc_loss.mean() + loss_mse
            # loss = 1 - ncc_loss.mean()
            loss = 0.5 * (1 - loss_ncc) + 0.5 * loss_geodesic
            # print(loss_ncc)
            # print(loss_mse)
            optimizer.zero_grad()
            # 损失回传
            loss.mean().backward()
            # 更新优化器参数
            optimizer.step()
            epoch_loss += loss.mean().item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.mean().item():.4f}")
            epoch_len = len(train_loader) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.mean().item(), epoch_len * epoch + step)
            # if step == 1:
            #     params.append([i for i in [out_rot[0, 0].tolist(), out_rot[0, 1].tolist(),
            #                                out_rot[0, 2].tolist(), out_trans[0, 0].tolist(),
            #                                out_trans[0, 1].tolist(), out_trans[0, 2].tolist()]])
            #     train_losses.append(loss.item())
        epoch_loss /= step
        # print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        # scheduler.step()
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            val_step = 0
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.float32, device=device)
                y_bbx = torch.tensor([], dtype=torch.float32, device=device)
                pred_bbx = torch.tensor([], dtype=torch.float32, device=device)
                for val_data in val_loader:
                    val_step += 1
                    val_gt_ap_imgs, val_gt_ap_masks, val_mov_ap_imgs = \
                        val_data["ap_gt_img"].to(device), val_data["ap_gt_mask"].to(device), \
                        val_data["ap_mov_img"].to(device)
                    val_pose = val_data["pose"].to(device)
                    val_ct_path = val_data["ct_img"]
                    val_pose = val_pose.type(torch.FloatTensor).to(device)
                    # val_gts_outputs = val_pose.cpu().numpy()
                    # val_gts_outputs = scalar.inverse_transform(val_gts_outputs)
                    # val_gts_outputs = torch.from_numpy(val_gts_outputs).to(device)
                    # val_gts = get_drr(val_gts_outputs[:, 3:], val_gts_outputs[:, 3:], batch_size, device)
                    # val_mov_drrs = get_drr(val_trans_tensor, val_rot_tensor, batch_size, device)
                    # val_gts = torch.permute(val_gts, (0, 1, 3, 2))
                    # for i in range(val_gt_ap_imgs.shape[0]):
                    # val_resized_gt_aps = resize_transform(val_gt_ap_imgs[0, :])
                    # val_resized_gt_aps = torch.unsqueeze(val_resized_gt_aps, 0)
                    # val_resized_gt_masks = resize_transform(val_gt_ap_masks[0, :])
                    # val_resized_gt_masks = torch.unsqueeze(val_resized_gt_masks, 0)
                    # val_resized_mov_aps = resize_transform(val_mov_ap_imgs[0, :])
                    # val_resized_mov_aps = torch.unsqueeze(val_resized_mov_aps, 0)
                    # val_outputs = model(val_mov_ap_imgs, val_gt_ap_imgs, val_gt_ap_masks)
                    # plot_drr(resized_gt_masks, ticks=False)
                    # plot_drr(resized_mov_aps, ticks=False)
                    # plt.show()
                    # val_inputs1 = torch.cat((val_resized_gt_aps, val_resized_gt_masks, val_resized_mov_aps), dim=1)
                    # print(val_inputs1.shape)
                    # if useCT:
                    # val_inputs2 = torch.cat((val_gt_ap_imgs, val_gt_ap_masks), dim=1)
                    # val_filepath = val_data['ap_mov_img_meta_dict']['filename_or_obj'][0].split('\\')[-1]
                    # val_name = '_'.join(val_filepath.split('_')[:-1])
                    # val_ct_path = os.path.join(vertebra_ct_path, val_name + '.nii.gz')
                    # val_outputs = model(val_inputs1, val_inputs2, StdS, val_ct_path)
                    # val_outputs = model(val_mov_ap_imgs, val_gt_ap_imgs, val_gt_ap_masks)
                    # val_bbx = masks_to_boxes(torch.squeeze(val_gt_ap_imgs, dim=1))
                    # val_bbx = scaleInen(val_bbx)
                    val_rot_outputs, val_trans_outputs, val_outMovs = model(val_mov_ap_imgs, val_gt_ap_imgs,
                                                                            val_gt_ap_masks, val_ct_path,
                                                                            scaleInen, info_data, StdS, True)
                    val_outMovs = scaleInen(val_outMovs)
                    # val_mov_outputs = val_outputs.detach().cpu().numpy()
                    # val_mov_outputs = StdS.inverse_transform(val_mov_outputs)
                    # val_mov_outputs = torch.from_numpy(val_mov_outputs).to(device)
                    # val_pred_drrs = get_drr(info_data, scaleInen, val_mov_outputs[:, :3], val_mov_outputs[:, 3:],
                    #                         val_ct_path, val_mov_ap_imgs.shape[0], device)  # 生成预测后的drr图像
                    # plot_drr(pred_drrs, ticks=False)
                    # plot_drr(mov_ap_imgs, ticks=False)
                    # plot_drr(gt_ap_imgs, ticks=False)
                    # plt.show()
                    # val_ncc_loss = NCC_loss(val_pred_drrs.float(), val_gt_ap_imgs, val_gt_ap_masks)
                    # else:
                    #     # val_outputs = model(val_inputs1)
                    #     val_outputs = model(val_mov_ap_imgs, val_gt_ap_imgs, val_gt_ap_masks)
                    # val_mov_outputs = val_outputs.cpu().numpy()
                    # val_mov_outputs = scalar.inverse_transform(val_mov_outputs)
                    # val_mov_outputs = torch.from_numpy(val_mov_outputs).to(device)
                    # val_pred_drrs, _ = get_drr(val_mov_outputs[:, 3:], val_mov_outputs[:, 3:], batch_size, device)
                    # val_scale_pred_drrs = scaleInen(val_pred_drrs)
                    # val_trans_tensor[:, :3] = val_mov_outputs[:, :3]
                    # val_rot_tensor[:, :3] = val_mov_outputs[:, 3:]
                    val_pose_gt = convert(val_pose.float()[:, :3], val_pose.float()[:, 3:], parameterization='euler_angles', convention="ZYX")
                    val_pose_se3 = convert(val_rot_outputs, val_trans_outputs, parameterization='euler_angles', convention="ZYX")
                    val_loss_geodesic = geodesic(val_pose_se3, val_pose_gt)
                    val_vect_ouputs = torch.cat((val_rot_outputs, val_trans_outputs), dim=1)
                    y_pred = torch.cat([y_pred, val_vect_ouputs], dim=0)
                    y = torch.cat([y, val_pose], dim=0)
                    # pred_bbx = torch.cat([pred_bbx, val_outpoints], dim=0)
                    # y_bbx = torch.cat([y_bbx, val_bbx], dim=0)
                    # 验证集的损失函数
                    # print(y_pred.shape, y.shape)
                    # val_mse = MSE_loss(y_pred, y)
                    # val_theta = Theta_loss(pred_bbx, y_bbx)
                    val_loss_ncc = NCC_loss(val_outMovs, val_gt_ap_imgs, val_gt_ap_masks)
                    # val_loss = 1 - val_ncc_loss.mean() + val_mse
                    val_loss = (1 - val_loss_ncc) * 0.5 + val_loss_geodesic * 0.5
                    val_epoch_loss += val_loss.mean().item()
                    # print(f"val_loss: {val_loss.item():.4f}")
                    val_epoch_len = len(val_loader) // val_loader.batch_size
                    writer.add_scalar("val_loss", val_loss.mean().item(), val_epoch_len * epoch + val_step)
                val_epoch_loss /= val_step
                print(f"val_loss: {val_epoch_loss:.4f}")
                val_loss_list.append(val_epoch_loss)
                # ncc = torch_mean_absolute_error(y_pred.cpu().numpy(), y.cpu().numpy())
                mse = mean_squared_error(y_pred.cpu().numpy(), y.cpu().numpy(), squared=False)
                # print(mse)
                metric_values.append(mse)
                # 保存MSE最优时的模型
                if mse <= best_metric:
                    best_metric = mse
                    best_metric_epoch = epoch + 1
                    checkpoint = {'reg_model': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'epoch': epoch + 1
                                  }
                    torch.save(checkpoint, save_dir)
                    print("saved new best metric reg_model")
                print(
                    "current epoch: {} current MSE: {:.4f} best MSE: {:.4f} at epoch {}".format(
                        epoch + 1, mse, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mse", mse, epoch + 1)
    # df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    # df["loss"] = train_losses
    # df.to_csv("{}_training_params.csv".format(save_dir.split('.p')[0]), index=False)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    # 绘制损失值曲线
    plt.figure('train', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    val_x = [val_interval * (i + 1) for i in range(len(val_loss_list))]
    val_y = val_loss_list
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.plot(val_x, val_y)
    plt.legend(['training loss', 'validation loss'])
    plt.subplot(1, 2, 2)
    # 绘制验证集的MSE和训练次数的关系曲线
    plt.title("Validation: MSE")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    # 保存曲线图像
    plt.savefig(save_dir + '_training info.png', format='png')
    # 将数值保存到表格中
    val_epoch = pd.DataFrame(data=metric_values)  # 数据有三列，列名分别为one,two,three
    val_epoch.to_csv(save_dir.split('.p')[0] + '_mse_per_epoch.csv', encoding='gbk')
    train_loss = pd.DataFrame(data=epoch_loss_values)  # 数据有三列，列名分别为one,two,three
    train_loss.to_csv(save_dir.split('.p')[0] + '_train_loss_per_epoch.csv', encoding='gbk')
    val_per_loss = pd.DataFrame(data=val_loss_list)  # 数据有三列，列名分别为one,two,three
    val_per_loss.to_csv(save_dir.split('.p')[0] + '_val_loss_per_epoch.csv', encoding='gbk')
    plt.show()


def evaluta_model(test_files, model_name, StdS):
    keys = ["ap_gt_img", "ap_gt_mask",
            "ap_mov_img"]
    input_size = 256
    # print(modal_key[:-1])
    test_transforms = Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True),
            # Resized(keys=keys, spatial_size=(input_size, input_size), mode="bilinear", align_corners=True),
            # 像素归一化
            ScaleIntensityd(keys=["ap_gt_img", "ap_mov_img"]),
            # ConcatItemsd(keys=keys, name="inputs"),
            # NormalizeIntensityd(keys=keys),
        ]
    )
    # create a validation data loader
    print(len(test_files))
    info_path = 'Data/total_data/256/verse/dataset_record2.csv'
    info_data = pd.read_csv(info_path)
    test_ds = monai.data.CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=0)
    model = PIRegNet(in_channel=3, channels=16, num_classes=6).to(device)
    # model = SRNet(3, 6, 'densenet').to(device)
    scaleInen = ScaleIntensity()
    # Evaluate the reg_model on test dataset #
    # print(os.path.basename(model_name).split('.')[0])
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['reg_model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epochs = checkpoint['epoch']
    # reg_model.load_state_dict(torch.load(log_dir))
    model.eval()
    with torch.no_grad():
        saver = CSVSaver(output_dir="result/",
                         filename=os.path.basename(model_name).split('.')[0] + '_val.csv')
        y_true = []
        y_pred = []
        # pred_list = []
        for test_data in test_loader:
            # test_mov, test_gt = test_data["mov_img"].to(device), test_data["gt_img"].to(device)
            test_gt_ap_imgs, test_gt_ap_masks, test_mov_ap_imgs = \
                test_data["ap_gt_img"].to(device), test_data["ap_gt_mask"].to(device), \
                test_data["ap_mov_img"].to(device),
            test_pose = test_data["pose"].to(device)
            test_ct = test_data["ct_img"]
            test_pose = test_pose.type(torch.FloatTensor).to(device)
            # pred = model(test_mov_ap_imgs, test_gt_ap_imgs, test_gt_ap_masks)
            pred_rot, pred_trans, pred_mov = model(test_mov_ap_imgs, test_gt_ap_imgs, test_gt_ap_masks, test_ct, scaleInen, info_data, StdS, True)
            pred = torch.cat((pred_rot, pred_trans), dim=1)
            # print(pred_mov.shape)
            # plot_drr(pred_mov)
            # plot_drr(test_gt_ap_imgs)
            # plt.figure('train', (10, 6))
            # plt.subplot(1, 2, 1)
            # plt.imshow(pred_mov[0, :].unsqueeze().cpu().numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(test_gt_ap_imgs[0, :].unsqueeze().cpu().numpy())
            # plt.show()
            # print(test_pose[0])
            # print(pred.shape)
            for j in range(len(pred)):
                y_true.append(test_pose[j].cpu().numpy())
                y_pred.append(pred[j].cpu().numpy())
                # pred_list.append(pred[j].item())
        # saver.finalize()
        y_pred_transform = StdS.inverse_transform(y_pred)
        y_true = StdS.inverse_transform(y_true)
        theta_pred = y_pred_transform[:, 0]
        theta_gt = y_true[:, 0]
        phi_pred = y_pred_transform[:, 1]
        phi_gt = y_true[:, 1]
        gamma_pred = y_pred_transform[:, 2]
        gamma_gt = y_true[:, 2]
        x_pred = y_pred_transform[:, 3]
        x_gt = y_true[:, 3]
        y_pred = y_pred_transform[:, 4]
        y_gt = y_true[:, 4]
        z_pred = y_pred_transform[:, 5]
        z_gt = y_true[:, 5]
        mae = mean_absolute_error(y_pred_transform, y_true, multioutput='raw_values')
        trans_mae = np.abs(y_pred_transform[:, 3:] - y_true[:, 3:])
        # print(np.average(trans_mae, axis=1).shape)
        ave_mae = np.average(trans_mae, axis=1)
        cal_SR_method(y_pred_transform, y_true)
        print('SR为:{:.2f}%'.format(len(ave_mae[np.where(ave_mae <= 1.2)]) / len(ave_mae) * 100))
        print('x轴旋转方向上误差:{:.4f}°, y轴旋转方向上误差:{:.4f}°, z轴旋转方向上误差:{:.4f}°, '
              '\nx平移方向上误差:{:.4f} mm, y平移方向上误差:{:.4f} mm, z平移方向上误差:{:.4f} mm'.
              format(mae[1] / math.pi * 180, mae[0] / math.pi * 180, mae[2] / math.pi * 180,
                     mae[3], mae[5], mae[4]))
        # print('MAE=' + str(mae))
        # 绘制拟合曲线
        # plt.figure('train', (12, 6))
        # plt.subplot(2, 3, 1)
        # plt.plot([i for i in range(len(theta_gt))], theta_gt)
        # # plt.scatter([i for i in range(len(y_true))], y_pred, c='g')
        # plt.plot([i for i in range(len(theta_pred))], theta_pred)
        # curve_label = ['theta gt', 'theta pred']
        # plt.legend(curve_label, loc='best')
        # plt.ylim(0, 10)
        # plt.subplot(2, 3, 2)
        # plt.plot([i for i in range(len(phi_gt))], phi_gt)
        # plt.plot([i for i in range(len(phi_pred))], phi_pred)
        # curve_label = ['phi gt', 'phi pred']
        # plt.legend(curve_label, loc='best')
        # plt.ylim(-5, 5)
        # plt.subplot(2, 3, 3)
        # plt.plot([i for i in range(len(gamma_gt))], gamma_gt)
        # plt.plot([i for i in range(len(gamma_pred))], gamma_pred)
        # curve_label = ['gamma gt', 'gamma pred']
        # plt.legend(curve_label, loc='best')
        # plt.ylim(-5, 6)
        # plt.subplot(2, 3, 4)
        # plt.plot([i for i in range(len(x_gt))], x_gt)
        # plt.plot([i for i in range(len(x_pred))], x_pred)
        # curve_label = ['x gt', 'x pred']
        # plt.legend(curve_label, loc='best')
        # plt.ylim(-5, 120)
        # plt.subplot(2, 3, 5)
        # plt.plot([i for i in range(len(y_gt))], y_gt)
        # plt.plot([i for i in range(len(y_pred))], y_pred)
        # curve_label = ['y gt', 'y pred']
        # plt.legend(curve_label, loc='best')
        # plt.ylim(-5, 120)
        # plt.subplot(2, 3, 6)
        # plt.plot([i for i in range(len(z_gt))], z_gt)
        # plt.plot([i for i in range(len(z_pred))], z_pred)
        # curve_label = ['z gt', 'z pred']
        # plt.legend(curve_label, loc='best')
        # plt.ylim(-5, 120)
        # # plt.savefig('results/MSE Curve.jpg', dpi=300)
        # plt.show()


def get_drr(offset_info, intScale, T_rot_tensor, T_trans_tensor, ct_tensor, batchSize, dl_device):
    # ctDir = 'Data/spatial-data/weng_fang_qi/weng_fang_qi_L4.nii.gz'
    vert_fold = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_ct'
    pred_drrs = torch.tensor([], dtype=torch.float64, device=device)
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
        # print(spacing)
        # spacing = origin_arr[1]['pixdim']
        # print(spacing)
        spacing = np.array((spacing[0], spacing[1], spacing[2]), dtype=np.float64)
        # print(spacing)
        bx, by, bz = torch.tensor(origin_img.shape) * torch.tensor(spacing) / 2
        drr_mov = DRR(
            origin_img.get_fdata(),
            spacing,
            sdr=500,
            height=256,
            delx=2.0,
        ).to(dl_device)

        # gt_rot = torch.tensor([[true_params["rx"], true_params["ry"], true_params["rz"]]], device=device)
        # gt_trans = torch.tensor([[true_params["tx"], true_params["ty"], true_params["tz"]]], device=device)
        rotation = torch.tensor([[torch.pi / 2, 0, torch.pi]], device=dl_device) + single_rot
        translation = torch.tensor([[bx - offset_x, by + offset_y, bz + offset_z]], device=dl_device) + single_trans
        # print(rotation)
        # print(T_rot_tensor.shape)
        # print(T_trans_tensor.shape)
        # print(translation)
        # gene_drr = drr_mov(rotation, translation, parameterization="euler_angles", convention="ZYX")
        # gene_drr = torch.permute(gene_drr, (0, 1, 3, 2))
        # print(gene_drr)
        reg = Registration(
            drr_mov,
            rotation.clone(),
            translation.clone(),
            parameterization="euler_angles",
            convention="ZYX",
        )
        gene_drr = reg()
        gene_drr = intScale(gene_drr)
        pred_drrs = torch.cat([pred_drrs, gene_drr], dim=0)
        del drr_mov
    return pred_drrs


def torch_mean_absolute_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def cal_SR_method(pred_pose_arr, gt_pose_arr):
    points_fold = 'Data/total_data/256/verse/landmarks'
    points_path = 'Data/total_data/256/verse/total_pose1_ap_points.csv'
    points_data = pd.read_csv(points_path)
    mTRE = []
    for pred_6pose, gt_6pose in zip(pred_pose_arr, gt_pose_arr):
        for index, row in points_data.iterrows():
            gt_name = row['name']
            vert = row['vert']
            num = int(gt_name.split('_')[-2])
            # print(mask3d_path)
            json_path = os.path.join(points_fold, gt_name + '_' + vert + '_3d_p.json')
            with open(json_path, 'r') as f:
                points_js = json.load(f)
                # print(points_js['markups'][0]['controlPoints'][0]['position'])
                points_arr = points_js['markups'][0]['controlPoints']
                for pts_list in points_arr:
                    pts = torch.tensor([[pts_list['position']]], dtype=torch.float64)
                    gt_rot = torch.tensor(np.array(gt_6pose[:3]))
                    gt_trans = torch.tensor(np.array(gt_6pose[3:]))
                    gt_pose = convert(gt_rot, gt_trans, parameterization='euler_angles', convention="ZYX")
                    transformed_pts = gt_pose(pts)
                    pred_rot = torch.tensor(np.array(pred_6pose[:3]))
                    pred_trans = torch.tensor(np.array(pred_6pose[3:]))
                    pred_pose = convert(pred_rot, pred_trans, parameterization='euler_angles', convention="ZYX")
                    pred_transformed_pts = pred_pose(pts)
                    dis = np.sqrt(
                        np.sum(np.square(pred_transformed_pts.squeeze().numpy() - transformed_pts.squeeze().numpy())))
                    mTRE.append(dis)
    arr_std = np.std(mTRE, ddof=1)
    print(f'mTRE为:{np.mean(mTRE):.2f}±{arr_std:.2f}')
    mtre_arr = np.array(mTRE)
    print(len(mtre_arr))
    print('SR为:{:.2f}%'.format(len(mtre_arr[np.where(mtre_arr < 20)]) / len(mtre_arr) * 100))


def test_single_sample():
    reader = LoadImage(ensure_channel_first=True, image_only=False)
    # 2647 642 552 3688
    sample_num = 642
    # sample_num = random.randint(1, 4700)
    vert_list = ['L2', 'L3', 'L4']
    random_number = random.randint(0, 2)
    vert = vert_list[0]
    model_path = 'reg_model/MICCAI/paper_model/densenet_data2_model.pth'
    info_path = 'Data/total_data/256/verse/dataset_record2.csv'
    excel_path = 'Data/total_data/256/verse/total_pose2_ap.csv'
    gt_fold = 'Data/total_data/256/verse/drr_gt2'
    mask_fold = 'Data/total_data/256/verse/gt_mask2'
    vert3d_fold = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_ct'
    pose_data = pd.read_csv(excel_path)
    case_name = pose_data.iloc[sample_num, 0]
    gt_T = np.array([pose_data.iloc[sample_num, 1], pose_data.iloc[sample_num, 2], pose_data.iloc[sample_num, 3],
                     pose_data.iloc[sample_num, 4], pose_data.iloc[sample_num, 5], pose_data.iloc[sample_num, 6]])
    data_info = pd.read_csv(info_path)
    num = int(case_name.split('_')[-2])
    name = '_'.join(case_name.split('_')[:-2])
    gt_img_path = os.path.join(gt_fold, case_name + '.nii.gz')
    gt_mask_path = os.path.join(mask_fold, name + '_seg_' + vert + '_' + str(num) + '_ap.nii.gz')
    vert3d_path = os.path.join(vert3d_fold, name + '_' + vert + '.nii.gz')
    offset_x = data_info.loc[data_info['img_save_fold'] == vert3d_path]['tx'].item()
    offset_y = data_info.loc[data_info['img_save_fold'] == vert3d_path]['ty'].item()
    offset_z = data_info.loc[data_info['img_save_fold'] == vert3d_path]['tz'].item()
    gt_img = reader(gt_img_path)
    gt_mask = reader(gt_mask_path)
    scaleInen = ScaleIntensity()
    # print(xray[0].shape)
    gt_img = scaleInen(gt_img[0])
    gt_mask_tensor = torch.unsqueeze(gt_mask[0], dim=0).to(device)
    gt_img_tensor = torch.unsqueeze(gt_img, dim=0).to(device)
    # print(gt_img.shape)
    # img_path = data_info.iloc[sample_num, 0]
    # vert_path = data_info.iloc[sample_num, 2]
    # mask_path = data_info.iloc[sample_num, 3]
    # offset_arr = [data_info.iloc[sample_num, 5], data_info.iloc[sample_num, 6], data_info.iloc[sample_num, 7]]
    offset_arr = [offset_x, offset_y, offset_z]
    # print(vert3d_path)
    # print(data_info.iloc[200, 4])
    scaler = pickle.load(open('reg_model/MICCAI/pose_scaler2.pkl', 'rb'))
    # gt_img, gt_T = generate_drr(img_path, offset_arr, False, False)
    mov_img, _ = generate_drr(vert3d_path, offset_arr, True, False)
    # gt_mask, _ = generate_drr(mask_path, offset_arr, True, True)
    # ct_tensor = torch.tensor([vert_path], device=device)
    ct_arr = np.array([vert3d_path])
    # plot_drr(gt_img, ticks=False)
    # plot_drr(gt_mask, ticks=False)
    # plot_drr(mov_img, ticks=False)
    # plt.show()
    # print(offset_arr)
    model = SRNet(3, 6, 'densenet').to(device)
    # model = PRegNet(in_channel=3, channels=16, num_classes=6).to(device)
    scaleInen = ScaleIntensity()
    # Evaluate the reg_model on test dataset #
    # print(os.path.basename(model_name).split('.')[0])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['reg_model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epochs = checkpoint['epoch']
    # reg_model.load_state_dict(torch.load(log_dir))
    model.eval()
    with torch.no_grad():
        pred = model(mov_img, gt_img_tensor, gt_mask_tensor)
        # pred, outpoints = model(mov_img, gt_img_tensor, gt_mask_tensor, ct_arr, scaleInen, data_info, scaler, True)
        # print(pred)
        y_pred_transform = scaler.inverse_transform(pred.cpu().numpy())
        # print(y_pred_transform)
        print(gt_T)
        # y_true = scaler.inverse_transform(gt_T)
        # y_pred_transform = y_pred_transform.squeeze(axis=0)
        gt_T = np.expand_dims(gt_T, axis=0)
        mae = mean_absolute_error(y_pred_transform, gt_T, multioutput='raw_values')
        # print(mae)
        trans_mae = np.abs(y_pred_transform[:, 3:] - gt_T[:, 3:])
        print(y_pred_transform[0, :])
        # print(np.average(trans_mae, axis=1).shape)
        ave_mae = np.average(trans_mae, axis=1)
        # cal_SR_method(y_pred_transform, gt_T)
        print('SR为:{:.2f}%'.format(len(ave_mae[np.where(ave_mae <= 2.0)]) / len(ave_mae) * 100))
        print('x轴旋转方向上误差:{:.4f}°, y轴旋转方向上误差:{:.4f}°, z轴旋转方向上误差:{:.4f}°, '
              '\nx平移方向上误差:{:.4f} mm, y平移方向上误差:{:.4f} mm, z平移方向上误差:{:.4f} mm'.
              format(mae[1] / math.pi * 180, mae[0] / math.pi * 180, mae[2] / math.pi * 180,
                     mae[3], mae[5], mae[4]))
        pred_img, _ = generate_drr(vert3d_path, offset_arr, True, False, pred_pose=y_pred_transform[0, :])
        gt_vert_img, _ = generate_drr(vert3d_path, offset_arr, True, False, input_gt_pose=gt_T[0, :])
        pred_img = torch.permute(pred_img, (0, 1, 3, 2))
        pred_drr = pred_img.squeeze().cpu().numpy()
        gt_vert_img = torch.permute(gt_vert_img, (0, 1, 3, 2))
        gt_vert_drr = gt_vert_img.squeeze().cpu().numpy()
        # gt_drr = gt_drr.squeeze().cpu().numpy()
        # gray = cv2.cvtColor(pred_drr, cv2.COLOR_BGR2GRAY)
        test_mov = torch.permute(mov_img, (0, 1, 3, 2))
        test_mov = test_mov.squeeze().cpu().numpy()
        mov_normalized = cv2.normalize(test_mov, None, 0, 255.0,
                                       cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mov_thresh = cv2.Canny(np.uint8(mov_normalized), 1, 50)
        img_normalized = cv2.normalize(pred_drr, None, 0, 255.0,
                                       cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        thresh = cv2.Canny(np.uint8(img_normalized), 1, 50)

        gt_normalized = cv2.normalize(gt_vert_drr, None, 0, 255.0,
                                      cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        gt_thresh = cv2.Canny(np.uint8(gt_normalized), 1, 50)
        test_gt = torch.permute(gt_img_tensor, (0, 1, 3, 2))
        overlay_gt = overlay(test_gt.squeeze().cpu().numpy(), gt_thresh, alpha=0.1)

        overlay_initial = overlay(test_gt.squeeze().cpu().numpy(), mov_thresh, color=(0, 255, 0), alpha=0.1)
        print(overlay_gt.shape)
        overlay_pred = overlay(overlay_gt, thresh, color=(0, 255, 0), alpha=1)
        plt.figure('show', (12, 5), dpi=300)
        plt.subplot(1, 2, 1)
        plt.title("Initial")
        plt.imshow(overlay_initial)
        # plt.imshow(mov_thresh, cmap='RdPu', alpha=0.3)
        # plt.xticks([])
        # plt.yticks([])
        # # plt.subplot(1, 2, 2)
        # # plt.title("GT")
        # # plt.imshow(test_gt.squeeze().cpu().numpy(), cmap='gray')
        # # plt.imshow(gt_thresh, cmap='Blues', alpha=0.4)
        # # plt.xticks([])
        # # plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.title("Pred")
        plt.imshow(overlay_pred)
        # plt.imshow(gt_thresh, cmap='YlOrRd', alpha=0.3)
        # plt.imshow(thresh, cmap='PuBu', alpha=0.3)
        # plt.savefig('results/test_{n}.jpg'.format(n=num), dpi=300)
        # plt.xticks([])
        # plt.yticks([])
        plt.show()


def overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
        resize: Tuple[int, int] = (1024, 1024)
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.

    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.

    Returns:
        image_combined: The combined image.

    """
    print(image.shape)
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # rgb_gt = norm_image * 255
    # rgb_gt = rgb_gt.astype(np.uint8)
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    # if resize is not None:
    #     image = cv2.resize(image.transpose(1, 2, 0), resize)
    #     image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def cal_CR():
    sample_num = 400
    model_path = 'reg_model/MICCAI/paper_model/densenet_data2_model.pth'
    info_path = 'Data/total_data/256/verse/dataset_record2.csv'
    excel_path = 'Data/total_data/256/verse/total_pose2_ap.csv'
    pose_data = pd.read_csv(excel_path)
    data_info = pd.read_csv(info_path)
    img_path = data_info.iloc[sample_num, 0]
    vert_path = data_info.iloc[sample_num, 2]
    mask_path = data_info.iloc[sample_num, 3]
    offset_arr = [data_info.iloc[sample_num, 5], data_info.iloc[sample_num, 6], data_info.iloc[sample_num, 7]]
    print(vert_path)
    # print(data_info.iloc[200, 4])
    scaler = pickle.load(open('reg_model/MICCAI/pose_scaler2.pkl', 'rb'))
    gt_img, gt_T = generate_drr(img_path, offset_arr, False, False)
    mov_img, _ = generate_drr(vert_path, offset_arr, True, False)
    gt_mask, _ = generate_drr(mask_path, offset_arr, True, True)
    ct_arr = np.array([vert_path])
    # plot_drr(gt_img, ticks=False)
    # plot_drr(gt_mask, ticks=False)
    # plot_drr(mov_img, ticks=False)
    # plt.show()
    # print(offset_arr)
    model = SRNet(3, 6, 'densenet').to(device)
    # model = PRegNet(in_channel=3, channels=16, num_classes=6).to(device)
    scaleInen = ScaleIntensity()
    # Evaluate the reg_model on test dataset #
    # print(os.path.basename(model_name).split('.')[0])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['reg_model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epochs = checkpoint['epoch']
    # reg_model.load_state_dict(torch.load(log_dir))
    model.eval()
    with torch.no_grad():
        # plot_drr(gt_img, ticks=False)
        # plot_drr(gt_mask, ticks=False)
        # plot_drr(mov_img, ticks=False)
        # plt.show()
        outputs = model(mov_img, gt_img, gt_mask)
        # outputs, outpoints = model(mov_img, gt_img, gt_mask, ct_arr, scaleInen, data_info, scaler, True)
        # print(pred)
        y_pred_transform = scaler.inverse_transform(outputs.cpu().numpy())
        print(y_pred_transform)
        print(gt_T)
        # y_true = scaler.inverse_transform(gt_T)
        # y_pred_transform = y_pred_transform.squeeze(axis=0)
        gt_T = np.expand_dims(gt_T, axis=0)
        mae = mean_absolute_error(y_pred_transform, gt_T, multioutput='raw_values')
        # print(mae)
        # trans_mae = np.abs(y_pred_transform[3:] - gt_T[3:])
        # print(np.average(trans_mae, axis=1).shape)
        # ave_mae = np.average(trans_mae, axis=1)
        # cal_SR_method(y_pred_transform, gt_T)
        # print('SR为:{:.2f}%'.format(len(ave_mae[np.where(ave_mae <= 1.2)]) / len(ave_mae) * 100))
        print('x轴旋转方向上误差:{:.4f}°, y轴旋转方向上误差:{:.4f}°, z轴旋转方向上误差:{:.4f}°, '
              '\nx平移方向上误差:{:.4f} mm, y平移方向上误差:{:.4f} mm, z平移方向上误差:{:.4f} mm'.
              format(mae[1] / math.pi * 180, mae[0] / math.pi * 180, mae[2] / math.pi * 180,
                     mae[3], mae[5], mae[4]))


def generate_drr(img_path, offset, isVert=False, isMask=False, input_gt_pose=None, pred_pose=None):
    img = nib.load(img_path)
    # np.random.seed(24)
    # Make the ground truth X-ray
    SDR = 500.0
    volume = img.get_fdata()
    spacing = img.header.get_zooms()
    shape = img.shape
    spacing = np.array((spacing[0], spacing[1], spacing[2]), dtype=np.float64)
    bx, by, bz = torch.tensor(shape) * torch.tensor(spacing) / 2
    true_params = {
        "sdr": SDR,
        "rx": torch.pi / 2,  # 沿y轴旋转，正数是逆时针旋转，0是右，torch.pi是左
        "ry": 0,
        "rz": torch.pi,  # 沿z轴旋转
        "tx": bx,
        "ty": by,  # 沿z轴平移
        "tz": bz,  # 沿y轴平移
    }
    # 15 60 prost
    # 2 180 xreg
    # 5 90 psss
    error_rot_range = 60
    error_trans_range = 15
    # err_rx, err_ry, err_rz, err_tx, err_ty, err_tz = get_transform_parameters(error_rot_range, error_trans_range)
    err_rx, err_ry, err_rz, err_tx, err_ty, err_tz = 0, 0, 0, 0, 0, 0
    # print(err_rx / math.pi * 180, err_ry / math.pi * 180, err_rz / math.pi * 180, err_tx, err_ty, err_tz)
    rx, ry, rz, tx, ty, tz = get_transform_parameters(18, 20)
    # print(rx, ry, rz, tx, ty, tz)
    # gt_pose = torch.tensor([[rx, ry, rz, tx, ty, tz]], device=device)
    gt_pose = np.array([rx, ry, rz, tx, ty, tz])
    # gt_pose = np.array([0.103010065, 0.004991424,     0.019786016,     31.70281632,     15.99929525,     17.14376815])
    # rx, ry, rz, tx, ty, tz = gt_pose[0], gt_pose[1], gt_pose[2], gt_pose[3], gt_pose[4], gt_pose[5]
    # print(gt_pose)
    drr_detector = DRR(
        volume,
        spacing,
        sdr=SDR,
        height=256,
        delx=2.0,
        bone_attenuation_multiplier=10.5
    ).to(device)
    if isVert and isMask and pred_pose is None and input_gt_pose is None:
        rot = torch.tensor([[true_params["rx"] + rx, true_params["ry"] + ry, true_params["rz"] + rz]], device=device)
        trans = torch.tensor([[true_params["tx"] + tx - offset[0], true_params["ty"] + ty + offset[1],
                               true_params["tz"] + tz + offset[2]]], device=device)
    elif isVert and isMask is False and pred_pose is None and input_gt_pose is None:
        rot = torch.tensor([[true_params["rx"], true_params["ry"], true_params["rz"]]], device=device)
        trans = torch.tensor([[true_params["tx"] - offset[0], true_params["ty"] + offset[1],
                               true_params["tz"] + offset[2]]], device=device)
    elif isVert and isMask is False and pred_pose is not None and input_gt_pose is None:
        rot = torch.tensor([[true_params["rx"] + pred_pose[0] + err_rx, true_params["ry"] + pred_pose[1] + err_ry,
                             true_params["rz"] + pred_pose[2] + err_rz]], device=device)
        trans = torch.tensor([[true_params["tx"] - offset[0] + pred_pose[3] + err_tx,
                               true_params["ty"] + offset[1] + pred_pose[4] + err_ty,
                               true_params["tz"] + offset[2] + pred_pose[5] + err_tz]], device=device)
    elif isVert and isMask is False and pred_pose is None and input_gt_pose is not None:
        rot = torch.tensor([[true_params["rx"] + input_gt_pose[0], true_params["ry"] + input_gt_pose[1],
                             true_params["rz"] + input_gt_pose[2]]], device=device)
        trans = torch.tensor([[true_params["tx"] - offset[0] + input_gt_pose[3],
                               true_params["ty"] + offset[1] + input_gt_pose[4],
                               true_params["tz"] + offset[2] + input_gt_pose[5]]], device=device)
    else:
        rot = torch.tensor([[true_params["rx"] + rx, true_params["ry"] + ry, true_params["rz"] + rz]], device=device)
        trans = torch.tensor([[true_params["tx"] + tx, true_params["ty"] + ty, true_params["tz"] + tz]], device=device)
    reg = Registration(
        drr_detector,
        rot.float().clone(),
        trans.float().clone(),
        parameterization="euler_angles",
        convention="ZYX",
    )
    gene_drr = reg()
    # drr = drr_detector(rot.float(), trans.float(), parameterization="euler_angles", convention="ZYX")
    # print(drr.shape)
    scaleInen = ScaleIntensity()
    drr = scaleInen(gene_drr)
    if isMask:
        drr = torch.where(drr > 0, 1, 0)
    del drr_detector
    return drr, gt_pose


if __name__ == '__main__':
    # 判断是否有GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model_engine()
    # test_single_sample()
    cal_CR()
