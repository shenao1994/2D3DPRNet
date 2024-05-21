import os

import numpy as np
import torch
import torchgeometry as tgm
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import torch
import torch.nn.functional as F
from diffdrr.drr import DRR
from diffdrr.data import load_example_ct
from diffdrr.visualization import plot_drr
import csv
import pandas as pd
from tqdm import tqdm
from PIL import Image
from monai.transforms import LoadImage
import nibabel as nib
import time
from diffdrr.pose import convert, random_rigid_transform, RigidTransform


def test_single_drr():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ct_path = 'F:/BaiduNetdiskDownload/verse2020_dataset/01_training/verse075/verse075.nii.gz'
    # ct_path = 'F:/BaiduNetdiskDownload/verse2020_dataset/02_validation/verse765/verse765_CT-ax.nii.gz'
    ct_path = 'Data/total_data/cropped/vertebra_mask/dukemei_L3_seg.nii.gz'
    volume, spacing, true_params = get_true_drr(ct_path)
    T1 = time.time()
    drr_target = DRR(
        volume,
        spacing,
        sdr=true_params["sdr"],
        height=256,
        delx=2.0,
        bone_attenuation_multiplier=10.5
    ).to(device)

    gt_rot = torch.tensor([[true_params["rx"], true_params["ry"], true_params["rz"]]], device=device)
    gt_trans = torch.tensor([[true_params["tx"], true_params["ty"],
                              true_params["tz"]]], device=device)
    gt_drr = drr_target(gt_rot, gt_trans, parameterization="euler_angles", convention="ZYX")
    T2 = time.time()
    print('drr生成时间:%s秒' % (T2 - T1))
    gt_drr = torch.permute(gt_drr, (0, 1, 3, 2))
    # gt_drr = gt_drr.squeeze().cpu().numpy()
    # gt_drr = np.where(gt_drr != 0, 1, 0)
    # gt_img = sitk.GetImageFromArray(gt_drr)
    plot_drr(gt_drr)
    plt.show()
    # print(save_path)
    # sitk.WriteImage(gt_img, 'Data/total_data/256/dukemei_L3_seg_drr.nii.gz')


def generate_verse_initial_drr():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # volume, spacing = load_example_ct()
    # ct_path = 'Data/total_data/origin/cao_fei/crop_L3.nii.gz'
    # ct_path = 'Data/total_data/vertebra_ct/cao_fei_L3.nii.gz'
    posture = 'ap'
    data_record_path = 'Data/total_data/256/verse/dataset_record2.csv'
    save_fold = 'Data/total_data/256/verse/drr_initial'
    data_info = pd.read_csv(data_record_path)
    for index, row in data_info.iterrows():
        # ct_path = row['name']
        if row['orient'] == 'LAS':
            vertebra_path = row['img_save_fold']
            bx = row['tx']
            by = row['ty']
            bz = row['tz']
            # ct_path = 'F:/BaiduNetdiskDownload/verse2020_dataset/vertebra_ct/verse075_L1.nii.gz'
            vert_img = nib.load(vertebra_path)
            volume, spacing, true_params = get_true_drr(vert_img)
            T1 = time.time()
            drr_target = DRR(
                volume,
                spacing,
                sdr=true_params["sdr"],
                height=256,
                delx=2.0,
                bone_attenuation_multiplier=10.5
            ).to(device)
            print([true_params["tx"], true_params["ty"], true_params["tz"]])
            gt_rot = torch.tensor([[true_params["rx"], true_params["ry"], true_params["rz"]]], device=device)
            gt_trans = torch.tensor([[true_params["tx"] - bx, true_params["ty"] + by, true_params["tz"] + bz]],
                                    device=device)
            # pose = convert(gt_rot, gt_trans, parameterization='euler_angles', convention="ZYX")
            # print(pose)
            # extrinsic = pose.matrix[:, :3]
            # extrinsic = random_rigid_transform()
            # print(extrinsic)
            # gt_drr = drr_target(extrinsic.to(device),
            #                     parameterization="matrix")
            gt_drr = drr_target(gt_rot, gt_trans, parameterization="euler_angles", convention="ZYX")
            T2 = time.time()
            print('drr生成时间:%s秒' % (T2 - T1))
            gt_drr = torch.permute(gt_drr, (0, 1, 3, 2))
            gt_drr = gt_drr.squeeze().cpu().numpy()
            # gt_drr = np.where(gt_drr != 0, 1, 0)
            gt_img = sitk.GetImageFromArray(gt_drr)
            save_path = os.path.join(save_fold, vertebra_path.split('\\')[-1].split('.')[0] + '_' + posture + '.nii.gz')
            print(save_path)
            sitk.WriteImage(gt_img, save_path)
            # plot_drr(gt_drr)
            # plt.show()


def generate_verse_transform_data():
    n_drrs = 50
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_fold = 'Data/total_data/256/verse/drr_gt2'
    posture = 'ap'
    csv_path = "Data/total_data/256/verse/total_pose2_{}.csv".format(posture)
    data_record_path = 'F:/BaiduNetdiskDownload/Verse20_preprocess/dataset_record2.csv'
    data_info = pd.read_csv(data_record_path)
    print("Writing initializations...")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "name",
                "rx",
                "ry",
                "rz",
                "tx",
                "ty",
                "tz",
            ]
        )
        # print(os.listdir(data_path)[103:])
        #     print(data_info.iloc[170:, ])
        data_info = data_info.drop_duplicates(['name'], keep='first')
        print(data_info.head(5))
        # return
        for index, row in data_info.iterrows():
            if row['orient'] == 'LAS':
                ct_path = row['name']
                origin_img = nib.load(ct_path)
                print(ct_path)
                if origin_img.shape[-1] <= 650:
                    volume, spacing, true_params = get_true_drr(origin_img)
                    drr_moving = DRR(
                        volume,
                        spacing,
                        sdr=500,
                        height=256,
                        delx=2.0,
                        bone_attenuation_multiplier=10.5
                    ).to(device)
                    # save_ini_drr(volume, spacing, true_params, device, save_fold, file)
                    # plt.imsave('Data/gt_drr/chazhilin_gt.png', gt_drr, cmap='gray')
                    # print(os.path.split(ct_path)[-1].split('.')[0])
                    name = os.path.split(ct_path)[-1].split('.')[0]
                    for i in range(n_drrs):
                        # sdr, theta, phi, gamma, bx, by, bz = get_initial_parameters(true_params)
                        rx, ry, rz, tx, ty, tz = get_transform_parameters()
                        writer.writerow([name + '_' + str(i) + '_{}'.format(posture), rx, ry, rz, tx, ty, tz])
                        mov_rot = torch.tensor(
                            [[true_params["rx"] + rx, true_params["ry"] + ry, true_params["rz"] + rz]],
                            device=device)
                        mov_trans = torch.tensor(
                            [[true_params["tx"] + tx, true_params["ty"] + ty, true_params["tz"] + tz]],
                            device=device)
                        moving_drr = drr_moving(mov_rot, mov_trans, parameterization="euler_angles", convention="ZYX")
                        moving_drr = torch.permute(moving_drr, (0, 1, 3, 2))
                        moving_drr = moving_drr.squeeze().cpu().numpy()
                        # plt.imsave('Data/moving_drr/chazhilin_moving_{num}.png'.format(num=str(i)), moving_drr, cmap='gray')
                        moving_img = sitk.GetImageFromArray(moving_drr)
                        save_path = os.path.join(save_fold, name + '_' + str(i) + '_{}.nii.gz'.format(posture))
                        # print(save_path)
                        sitk.WriteImage(moving_img, save_path)
                    del drr_moving
                else:
                    pass


def generate_verse_drr_mask():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pose_path = 'Data/total_data/256/verse/total_pose2_ap.csv'
    data_info_path = 'Data/total_data/256/verse/dataset_record2.csv'
    # Data_path = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_mask'
    save_fold = 'Data/total_data/256/verse/gt_mask2'
    posture = 'ap'
    pose_data = pd.read_csv(pose_path)
    data_info = pd.read_csv(data_info_path)
    # print(pose_data.iloc[560:, ])
    # return
    # for file in os.listdir(Data_path):
    for _, row1 in data_info.iterrows():
        seg_path = row1['seg_save_path']
        offset_x = row1['tx']
        offset_y = row1['ty']
        offset_z = row1['tz']
        for _, row2 in pose_data.iterrows():
            name = '_'.join(row2['name'].split('_')[:-2])
            # print(name)
            if name == os.path.split(seg_path)[-1].split('_seg')[0]:
                mask = nib.load(seg_path)
                mask_name = os.path.split(seg_path)[-1].split('.')[0]
                if int(row2['name'].split('_')[-2]) == 0:
                    # mask_path = os.path.join(Data_path, file)
                    volume, spacing, true_params = get_true_drr(mask)
                    drr_detector = DRR(
                        volume,
                        spacing,
                        sdr=500,
                        height=256,
                        delx=2.0,
                        bone_attenuation_multiplier=10.5
                    ).to(device)
                # ct_path = os.path.join(Data_path, name + '.nii.gz')
                if int(row2['name'].split('_')[-2]) < 50:
                    rx = float(row2['rx'])
                    ry = float(row2['ry'])
                    rz = float(row2['rz'])
                    tx = float(row2['tx'])
                    ty = float(row2['ty'])
                    tz = float(row2['tz'])
                    # print(tx, ty, tz)
                    print(offset_x, offset_y, offset_z)
                    gt_rot = torch.tensor(
                        [[true_params['rx'] + rx, true_params['ry'] + ry, true_params['rz'] + rz]],
                        device=device)
                    gt_trans = torch.tensor([[true_params['tx'] + tx - offset_x, true_params['ty'] + ty + offset_y,
                                              true_params['tz'] + tz + offset_z]], device=device)
                    gt_drr = drr_detector(gt_rot, gt_trans, parameterization="euler_angles", convention="ZYX")
                    gt_drr = torch.permute(gt_drr, (0, 1, 3, 2))
                    gt_drr = gt_drr.squeeze().cpu().numpy()
                    gt_drr = np.where(gt_drr != 0, 1, 0)
                    # plt.imshow(gt_drr, cmap='gray')
                    # plt.show()
                    # plt.imsave('Data/moving_drr/chazhilin_moving_{num}.png'.format(num=str(i)), moving_drr, cmap='gray')
                    gt_img = sitk.GetImageFromArray(gt_drr)
                    num = row2['name'].split('_')[-2]
                    # print(file.split('.')[0] + '_' + num + '_{}.nii.gz'.format(posture))
                    # save_path = os.path.join(save_fold, name + '_' + num + '_{}.nii.gz'.format(posture))  # 保存drr图像时的路径
                    save_path = os.path.join(save_fold,
                                             mask_name + '_' + num + '_{}.nii.gz'.format(posture))  # 保存mask图像时的路径
                    print(save_path)
                    sitk.WriteImage(gt_img, save_path)
                    del gt_img
                if int(row2['name'].split('_')[-2]) == 49:
                    del drr_detector


def get_true_drr(img_path):
    """Get parameters for the fixed DRR."""
    # volume, spacing = load_example_ct()
    # img_path = 'Data/crop_3d_img/weng_fang_qi.nii.gz'
    # img_path = 'Data/spatial-data/weng_fang_qi/weng_fang_qi_L4.nii.gz'
    # reader = LoadImage(ensure_channel_first=True, image_only=False)
    np.random.seed(88)
    # Make the ground truth X-ray
    SDR = 500.0
    # ctDir = 'Data/total_data/ct_mask/cao_fei/cao_fei.nii.gz'
    # ctDir = 'Data/gncc_data/CT_L4.nii.gz'
    # origin_arr = reader(img_path)
    img = nib.load(img_path)
    volume = img.get_fdata()
    # volume = np.swapaxes(volume, 0, 2)[::-1].copy()
    # origin_img = sitk.ReadImage(img_path)
    # direct = tuple((1, 1))
    # img_arr = sitk.GetArrayFromImage(origin_img)
    # img_arr = np.swapaxes(img_arr, 2, 0)
    # x, y, z = nib.aff2axcodes(origin_img.affine)
    # print('\nImage orientation:', x, y, z)
    # voxel_spacing = origin_img.affine[:3, :3]
    # image_origin = origin_img.affine[:3, 3]
    # print('\nVoxel spacing:\n', voxel_spacing)
    # print('\nImage origin:\n', image_origin)
    # image_origin = torch.from_numpy(image_origin)
    # print(torch.unsqueeze(image_origin, dim=0))
    # lps2volume = RigidTransform(torch.eye(3))
    spacing = img.header.get_zooms()
    # spacing = origin_img.GetSpacing()
    # spacing = origin_arr[1]['pixdim']
    # print(img.shape)
    shape = img.shape
    # (-0.035726371975637966, 0.002852628550054323, -0.9993575380647652, 0.9961868475216886, -0.07954403227316194,
    #  -0.03584007691295757, -0.07959517052094489, -0.9968272716173457, 6.860136174157218e-08)
    # print(spacing)
    # print(shape)
    # shape = (512, 512, 601)
    spacing = np.array((spacing[0], spacing[1], spacing[2]), dtype=np.float64)
    bx, by, bz = torch.tensor(shape) * torch.tensor(spacing) / 2
    true_params = {
        "sdr": SDR,
        "rx": torch.pi / 2,  # 沿y轴旋转，正数是逆时针旋转，0是右，torch.pi是左
        "ry": 0,  # 沿x轴旋转
        "rz": torch.pi,  # 沿z轴旋转
        "tx": bx,
        "ty": by,  # 沿z轴平移
        "tz": bz,  # 沿y轴平移
    }
    return volume, spacing, true_params


def get_transform_parameters(rd_seed, rot_range, trans_range):
    np.random.seed(rd_seed)
    # rot_range = 18
    # trans_range = 20.0
    """Get starting parameters for the moving DRR by perturbing the true params."""
    rx = np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    ry = np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    rz = np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    tx = np.random.uniform(-trans_range, trans_range)
    ty = np.random.uniform(-trans_range, trans_range)
    tz = np.random.uniform(-trans_range, trans_range)
    return rx, ry, rz, tx, ty, tz


def write_initial_parameters(transform_params, sample_name, pose_path, n_drrs=10000):
    print("Writing initializations...")
    with open(pose_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "name"
                "rx",
                "ry",
                "rz",
                "tx",
                "ty",
                "tz",
            ]
        )
        for i in range(n_drrs):
            # sdr, theta, phi, gamma, bx, by, bz = get_initial_parameters(true_params)
            rx, ry, rz, tx, ty, tz = get_transform_parameters(transform_params)
            writer.writerow([sample_name + '_' + str(i), rx, ry, rz, tx, ty, tz])


def generate_transform_data():
    n_drrs = 50
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = 'Data/total_data/ct'
    save_fold = 'Data/total_data/256/drr_gt'
    # posture = 'ap'
    csv_path = "Data/total_data/256/total_pose2.csv"
    # ct_path = os.path.join(data_path, 'dingjunmei.nii.gz')
    # ct_path = 'Data/tuodao/peizongping/peizongping.nii.gz'
    posture_list = ['ap', 'lar', 'lal']
    print("Writing initializations...")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "name",
                "rx",
                "ry",
                "rz",
                "tx",
                "ty",
                "tz",
            ]
        )
        # print(os.listdir(data_path)[103:])
        for file in os.listdir(data_path):
            low_seed = np.random.randint(500, 1000)
            high_seed = np.random.randint(2000, 3000)
            ct_path = os.path.join(data_path, file)
            volume, spacing, true_params = get_true_drr(ct_path)
            drr_moving = DRR(
                volume,
                spacing,
                sdr=500,
                height=256,
                delx=2.0,
                bone_attenuation_multiplier=10.5
            ).to(device)
            # save_ini_drr(volume, spacing, true_params, device, save_fold, file)
            for i in range(n_drrs):
                seed = np.random.randint(low_seed, high_seed)
                # sdr, theta, phi, gamma, bx, by, bz = get_initial_parameters(true_params)
                rx, ry, rz, tx, ty, tz = get_transform_parameters(seed, 9, 30)
                writer.writerow([file.split('.')[0] + '_' + str(i), rx, ry, rz, tx, ty, tz])
                for posture in posture_list:
                    if posture == 'ap':
                        mov_rot = torch.tensor([[true_params["rx"] + rx, true_params["ry"] + ry, true_params["rz"] + rz]],
                                               device=device)
                        mov_trans = torch.tensor([[true_params["tx"] + tx, true_params["ty"] + ty, true_params["tz"] + tz]],
                                                 device=device)
                    elif posture == 'lar':
                        mov_rot = torch.tensor([[true_params["rx"] + rx - torch.pi / 2, true_params["ry"] + ry, true_params["rz"] + rz]],
                                               device=device)
                        mov_trans = torch.tensor([[true_params["tx"] + tx, true_params["ty"] + ty, true_params["tz"] + tz]],
                                                 device=device)
                    else:
                        mov_rot = torch.tensor([[true_params["rx"] + rx + torch.pi / 2, true_params["ry"] + ry, true_params["rz"] + rz]],
                                               device=device)
                        mov_trans = torch.tensor([[true_params["tx"] + tx, true_params["ty"] + ty, true_params["tz"] + tz]],
                                                 device=device)
                    moving_drr = drr_moving(mov_rot, mov_trans, parameterization="euler_angles", convention="ZYX")
                    moving_drr = torch.permute(moving_drr, (0, 1, 3, 2))
                    moving_drr = moving_drr.squeeze().cpu().numpy()
                    # plt.imsave('Data/moving_drr/chazhilin_moving_{num}.png'.format(num=str(i)), moving_drr, cmap='gray')
                    moving_img = sitk.GetImageFromArray(moving_drr)
                    save_path = os.path.join(save_fold, file.split('.')[0] + '_' + str(i) + '_{}.nii.gz'.format(posture))
                    sitk.WriteImage(moving_img, save_path)
            del drr_moving


def read_pose_generate_other_drrs():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pose_path = 'Data/total_data/256/verse/total_pose1_ap.csv'
    Data_path = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_mask'
    save_fold = 'Data/total_data/256/verse/gt_mask'
    posture = 'ap'
    pose_data = pd.read_csv(pose_path)
    # print(pose_data.iloc[560:, ])
    # return
    # for file in os.listdir(Data_path):
    verb_list = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    for verb in verb_list:
        for index, row in pose_data.iterrows():
            name = '_'.join(row['name'].split('_')[:-2])
            mask_name = name + '_' + verb
            mask_path = os.path.join(Data_path, mask_name + '_seg.nii.gz')
            if int(row['name'].split('_')[-2]) == 0 and os.path.exists(mask_path):
                # mask_path = os.path.join(Data_path, file)
                volume, spacing, true_params = get_true_drr(mask_path)
                drr_detector = DRR(
                    volume,
                    spacing,
                    sdr=1000,
                    height=256,
                    delx=1.2,
                    bone_attenuation_multiplier=10.5
                ).to(device)
            # ct_path = os.path.join(Data_path, name + '.nii.gz')
            # if int(row['name'].split('_')[-2]) < 20 and os.path.exists(mask_path):
            if os.path.exists(mask_path):
                rx = float(row['rx'])
                ry = float(row['ry'])
                rz = float(row['rz'])
                tx = float(row['tx'])
                ty = float(row['ty'])
                tz = float(row['tz'])
                gt_rot = torch.tensor(
                    [[true_params["rx"] + rx, true_params["ry"] + ry, true_params["rz"] + rz]],
                    device=device)
                gt_trans = torch.tensor([[true_params["tx"] + tx, true_params["ty"] + ty, true_params["tz"] + tz]],
                                        device=device)
                gt_drr = drr_detector(gt_rot, gt_trans, parameterization="euler_angles", convention="ZYX")
                gt_drr = torch.permute(gt_drr, (0, 1, 3, 2))
                gt_drr = gt_drr.squeeze().cpu().numpy()
                gt_drr = np.where(gt_drr != 0, 1, 0)
                # plt.imshow(gt_drr, cmap='gray')
                # plt.show()
                # plt.imsave('Data/moving_drr/chazhilin_moving_{num}.png'.format(num=str(i)), moving_drr, cmap='gray')
                gt_img = sitk.GetImageFromArray(gt_drr)
                num = row['name'].split('_')[-2]
                # print(file.split('.')[0] + '_' + num + '_{}.nii.gz'.format(posture))
                # save_path = os.path.join(save_fold, name + '_' + num + '_{}.nii.gz'.format(posture))  # 保存drr图像时的路径
                save_path = os.path.join(save_fold, mask_name.split('.')[0] + '_' + num + '_{}.nii.gz'.format(
                    posture))  # 保存mask图像时的路径
                print(save_path)
                sitk.WriteImage(gt_img, save_path)
                del gt_img
            if int(row['name'].split('_')[-2]) == 19 and os.path.exists(mask_path):
                del drr_detector


def save_ini_drr(volume, spacing, true_params, op_device, out_fold, filename):
    drr_target = DRR(
        volume,
        spacing,
        sdr=true_params["sdr"],
        height=256,
        delx=2.0,
        bone_attenuation_multiplier=10.5,
    ).to(op_device)
    gt_rot = torch.tensor([[true_params["rx"], true_params["ry"], true_params["rz"]]], device=op_device)
    gt_trans = torch.tensor([[true_params["tx"], true_params["ty"], true_params["tz"]]], device=op_device)
    gt_drr = drr_target(gt_rot, gt_trans, parameterization="euler_angles", convention="ZYX")
    gt_drr = torch.permute(gt_drr, (0, 1, 3, 2))
    # plot_drr(gt_drr)
    # plt.show()
    gt_drr = gt_drr.squeeze().cpu().numpy()
    # gt_drr = np.where(gt_drr != 0, 1, 0)
    gt_img = sitk.GetImageFromArray(gt_drr)
    save_path = os.path.join(out_fold, filename.split('.')[0] + '_lar.nii.gz')
    sitk.WriteImage(gt_img, save_path)
    del drr_target


def concat_pose_data():
    pose_fold = 'Data/total_data'
    total_poses = pd.DataFrame()
    for file in os.listdir(pose_fold):
        if file.endswith('.csv'):
            pose_data = pd.read_csv(os.path.join(pose_fold, file))
            # print(pose_data)
            total_poses = pd.concat([total_poses, pose_data])
    print(total_poses)
    total_poses.to_csv('Data/total_data/total_poses.csv')


def delete_off_vision_file():
    # L1_del_list = ['cao_fei', 'ye_suo_ming', 'chen_jing', 'qin_guo_min', 'ren_rui_fa', 'xu_ye', 'ye_suo_ming',
    #                'zhang_shou_song']
    # L5_del_list = ['chen_tao_ming', 'huang_xiao_bo', 'liu_xiao_ying', 'zhu_ying']
    file_fold = 'Data/total_data/mov_drr_temp'
    for file in os.listdir(file_fold):
        file_size = os.path.getsize(os.path.join('Data/total_data/mov_drr_temp', file))
        if (file_size / 1024) < 30:
            os.remove(os.path.join(file_fold, file))
        # print("文件大小为：", file_size / 1024, "KB")


def generate_bbx_points():
    pose_path = 'Data/total_data/256/verse/total_pose1_ap.csv'
    pose_data = pd.read_csv(pose_path)
    info_path = 'Data/total_data/256/verse/dataset_record2.csv'
    info_data = pd.read_csv(info_path)
    gt_fold = 'Data/total_data/256/verse/drr_gt/verse145_4_ap.nii.gz'
    gt_name = os.path.split(gt_fold)[-1].split('.')[0]
    # print(gt_name)
    ini_img_path = 'Data/total_data/256/verse/drr_initial/verse145_L3_ap.nii.gz'
    # ver3d_path = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_ct/verse145_seg_L3.nii'
    vert3d_fold = 'F:/BaiduNetdiskDownload/Verse20_preprocess/vertebra_ct'
    sdr = 500
    delx = 2.0
    dely = 2.0
    height = width = 256
    file_name = os.path.split(ini_img_path)[-1]
    name = '_'.join(file_name.split('_')[:-2])
    vert = file_name.split('_')[-2]
    vert3d_path = os.path.join(vert3d_fold, name + '_' + vert + '.nii.gz')
    print(vert3d_path)
    offset_x = info_data.loc[info_data['img_save_fold'] == vert3d_path]['tx'].item()
    offset_y = info_data.loc[info_data['img_save_fold'] == vert3d_path]['ty'].item()
    offset_z = info_data.loc[info_data['img_save_fold'] == vert3d_path]['tz'].item()
    gt_rx = pose_data.loc[pose_data['name'] == gt_name]['rx'].item()
    gt_ry = pose_data.loc[pose_data['name'] == gt_name]['ry'].item()
    gt_rz = pose_data.loc[pose_data['name'] == gt_name]['rz'].item()
    gt_tx = pose_data.loc[pose_data['name'] == gt_name]['tx'].item()
    gt_ty = pose_data.loc[pose_data['name'] == gt_name]['ty'].item()
    gt_tz = pose_data.loc[pose_data['name'] == gt_name]['tz'].item()
    # gt_img = nib.load(vert3d_path)
    # print(pose)
    # extrinsic = pose.matrix[:, :3]
    # ini_img = nib.load(gt_fold)
    ini_img = sitk.ReadImage(gt_fold)
    ini_img_arr = sitk.GetArrayFromImage(ini_img)
    vert_img = nib.load(vert3d_path)
    spacing = vert_img.header.get_zooms()
    shape = vert_img.shape
    spacing_arr = np.array((spacing[0], spacing[1], spacing[2]), dtype=np.float64)
    ini_bx, ini_by, ini_bz = torch.tensor(shape) * torch.tensor(spacing_arr) / 2
    gt_rot = torch.tensor([[torch.pi / 2 + gt_rx, 0 + gt_ry, torch.pi + gt_rz]])
    gt_trans = torch.tensor([[ini_bx + gt_tx - offset_x, ini_by + gt_ty + offset_y, ini_bz + gt_tz + offset_z]])
    print(gt_rot, gt_trans)
    vert_arr = vert_img.get_fdata()
    image_origin = vert_img.affine[:3, 3]
    # image_origin = np.array((-image_origin[0], -image_origin[1], image_origin[2]), dtype=np.float64)
    # print(image_origin)
    # print(spacing)
    # image_origin = torch.tensor([[np.array(image_origin)]])
    # xmin, xmax, ymin, ymax, zmin, zmax = getbbx_3D(vert_arr)
    # xmin, xmax, ymin, ymax, zmin, zmax = xmin * spacing[0], xmax * spacing[0], ymin * spacing[1], \
    #                                      ymax * spacing[1], zmin * spacing[2], zmax * spacing[2]
    # pts = torch.tensor([[[xmin, ymin, zmin], [xmax, ymax, zmax], [xmin, ymax, zmin],
    #                      [xmin, ymax, zmax], [xmin, ymin, zmax], [xmax, ymax, zmin],
    #                      [xmax, ymin, zmin], [xmax, ymin, zmax]]])
    # print(pts)
    # print(pts + image_origin)
    # physic_pts = pts + image_origin
    nx = torch.arange(96, dtype=torch.float32) * spacing[0]
    ny = torch.arange(87, dtype=torch.float32) * spacing[1]
    nz = torch.arange(52, dtype=torch.float32) * spacing[2]

    pts = torch.meshgrid(nx, ny, nz, indexing="xy")
    pts = torch.stack(pts, dim=-1).reshape(1, -1, 3)
    print(pts)
    flip_xz = torch.tensor(
        [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    translate = torch.tensor(
        [
            [1.0, 0.0, 0.0, -sdr],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose = convert(gt_rot, gt_trans, parameterization='euler_angles', convention="ZYX")
    # print(pose.matrix[:, :3])
    # pose = random_rigid_transform()
    flip_xz = RigidTransform(flip_xz)
    translate = RigidTransform(translate)
    extrinsic = (
        pose.inverse().compose(translate).compose(flip_xz)
    )
    transformed_pts = extrinsic(pts.float())
    intrinsic = torch.tensor(
        [
            [2 * sdr / delx, 0.0, 0.0 / delx - height / 2],
            [0.0, 2 * sdr / dely, 0.0 / dely - width / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    # intrinsic = torch.tensor(
    #     [
    #         [2 * sdr, 0.0, 0.0],
    #         [0.0, 2 * sdr, 0.0],
    #         [0.0, 0.0, 1.0],
    #     ]
    # )
    proj = torch.einsum("ij,bnj->bni", intrinsic, transformed_pts)
    z = proj[..., -1].unsqueeze(-1).clone()
    x = proj / z
    proj_pts = x[..., :2]
    print(proj_pts)

    # img = sitk.ReadImage(img_path)
    # img_arr = sitk.GetArrayFromImage(img)
    # img_arr = (img_arr > 0)
    # ini_img_arr = ini_img.get_fdata()
    # rows = np.any(ini_img_arr, axis=1)
    # cols = np.any(ini_img_arr, axis=0)
    # ymin, ymax = np.argmax(rows), ini_img_arr.shape[0] - 1 - np.argmax(np.flipud(rows))
    # xmin, xmax = np.argmax(cols), ini_img_arr.shape[1] - 1 - np.argmax(np.flipud(cols))
    # print(xmin, xmax, ymin, ymax)
    # x = [xmin, xmax, xmin, xmax]
    # y = [ymin, ymin, ymax, ymax]
    point_x = proj_pts.squeeze().cpu().numpy()[:, 0]
    point_y = proj_pts.squeeze().cpu().numpy()[:, 1]
    print(point_x)
    print(point_y)
    plt.scatter(-point_x, -point_y, c='red', s=50)
    plt.imshow(ini_img_arr)
    plt.show()


def getbbx_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(r)[0][[0, -1]]
    ymin, ymax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


def generate_total_initial_drr():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # volume, spacing = load_example_ct()
    # ct_path = 'Data/total_data/origin/cao_fei/crop_L3.nii.gz'
    # ct_path = 'Data/total_data/vertebra_ct/cao_fei_L3.nii.gz'
    posture = 'ap'
    data_record_path = 'Data/total_data/total_data_info.csv'
    data_fold = 'Data/total_data/cropped/vertebra_ct_no0'
    save_fold = 'Data/total_data/256/drr_initial_no0'
    data_info = pd.read_csv(data_record_path)
    for index, row in data_info.iterrows():
        # ct_path = row['name']
        name = row['name']
        vertebra_path = os.path.join(data_fold, name + '.nii.gz')
        bx = row['offsetx']
        by = row['offsety']
        bz = row['offsetz']
        # ct_path = 'F:/BaiduNetdiskDownload/verse2020_dataset/vertebra_ct/verse075_L1.nii.gz'
        # vert_img = nib.load(vertebra_path)
        volume, spacing, true_params = get_true_drr(vertebra_path)
        T1 = time.time()
        drr_target = DRR(
            volume,
            spacing,
            sdr=true_params["sdr"],
            height=256,
            delx=2.0,
            bone_attenuation_multiplier=10.5
        ).to(device)
        # print([true_params["tx"], true_params["ty"], true_params["tz"]])
        gt_rot = torch.tensor([[true_params["rx"], true_params["ry"], true_params["rz"]]], device=device)
        gt_trans = torch.tensor([[true_params["tx"] - bx, true_params["ty"] + by, true_params["tz"] + bz]],
                                device=device)
        # pose = convert(gt_rot, gt_trans, parameterization='euler_angles', convention="ZYX")
        # print(pose)
        # extrinsic = pose.matrix[:, :3]
        # extrinsic = random_rigid_transform()
        # print(extrinsic)
        # gt_drr = drr_target(extrinsic.to(device),
        #                     parameterization="matrix")
        gt_drr = drr_target(gt_rot, gt_trans, parameterization="euler_angles", convention="ZYX")
        T2 = time.time()
        print('drr生成时间:%s秒' % (T2 - T1))
        gt_drr = torch.permute(gt_drr, (0, 1, 3, 2))
        gt_drr = gt_drr.squeeze().cpu().numpy()
        # gt_drr = np.where(gt_drr != 0, 1, 0)
        gt_img = sitk.GetImageFromArray(gt_drr)
        save_path = os.path.join(save_fold, vertebra_path.split('\\')[-1].split('.')[0] + '_' + posture + '.nii.gz')
        print(save_path)
        sitk.WriteImage(gt_img, save_path)


def read_pose_generate_total_masks():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pose_path = 'Data/total_data/256/total_pose2.csv'
    mask_path = 'Data/total_data/cropped/vertebra_mask'
    save_fold = 'Data/total_data/256/gt_mask'
    data_info_path = 'Data/total_data/total_data_info.csv'
    pose_data = pd.read_csv(pose_path)
    data_info = pd.read_csv(data_info_path)
    # print(pose_data.iloc[560:, ])
    # return
    # for file in os.listdir(Data_path):
    posture_list = ['ap', 'lar', 'lal']
    for _, row1 in data_info.iterrows():
        seg_name = row1['name']
        seg_path = os.path.join(mask_path, seg_name + '_seg.nii.gz')
        offset_x = row1['offsetx']
        offset_y = row1['offsety']
        offset_z = row1['offsetz']
        for _, row2 in pose_data.iterrows():
            name = '_'.join(row2['name'].split('_')[:-1])
            # print(name)
            if name == '_'.join(seg_name.split('_')[:-1]):
                # mask_name = os.path.split(seg_path)[-1].split('.')[0]
                if int(row2['name'].split('_')[-1]) == 0:
                    # mask_path = os.path.join(Data_path, file)
                    volume, spacing, true_params = get_true_drr(seg_path)
                    drr_detector = DRR(
                        volume,
                        spacing,
                        sdr=500,
                        height=256,
                        delx=2.0,
                        bone_attenuation_multiplier=10.5
                    ).to(device)
                # ct_path = os.path.join(Data_path, name + '.nii.gz')
                if int(row2['name'].split('_')[-1]) < 50:
                    rx = float(row2['rx'])
                    ry = float(row2['ry'])
                    rz = float(row2['rz'])
                    tx = float(row2['tx'])
                    ty = float(row2['ty'])
                    tz = float(row2['tz'])
                    # print(tx, ty, tz)
                    print(offset_x, offset_y, offset_z)
                    for posture in posture_list:
                        if posture == 'ap':
                            gt_rot = torch.tensor(
                                [[true_params['rx'] + rx, true_params['ry'] + ry, true_params['rz'] + rz]],
                                device=device)
                            gt_trans = torch.tensor([[true_params['tx'] + tx - offset_x, true_params['ty'] + ty + offset_y,
                                                      true_params['tz'] + tz + offset_z]], device=device)
                        elif posture == 'lar':
                            gt_rot = torch.tensor(
                                [[true_params['rx'] + rx - torch.pi / 2, true_params['ry'] + ry, true_params['rz'] + rz]],
                                device=device)
                            gt_trans = torch.tensor([[true_params['tx'] + tx - offset_x, true_params['ty'] + ty + offset_y,
                                                      true_params['tz'] + tz + offset_z]], device=device)
                        else:
                            gt_rot = torch.tensor(
                                [[true_params['rx'] + rx + torch.pi / 2, true_params['ry'] + ry, true_params['rz'] + rz]],
                                device=device)
                            gt_trans = torch.tensor([[true_params['tx'] + tx - offset_x, true_params['ty'] + ty + offset_y,
                                                      true_params['tz'] + tz + offset_z]], device=device)
                        gt_drr = drr_detector(gt_rot, gt_trans, parameterization="euler_angles", convention="ZYX")
                        gt_drr = torch.permute(gt_drr, (0, 1, 3, 2))
                        gt_drr = gt_drr.squeeze().cpu().numpy()
                        gt_drr = np.where(gt_drr != 0, 1, 0)
                        # plt.imshow(gt_drr, cmap='gray')
                        # plt.show()
                        # plt.imsave('Data/moving_drr/chazhilin_moving_{num}.png'.format(num=str(i)), moving_drr, cmap='gray')
                        gt_img = sitk.GetImageFromArray(gt_drr)
                        num = row2['name'].split('_')[-1]
                        # print(file.split('.')[0] + '_' + num + '_{}.nii.gz'.format(posture))
                        # save_path = os.path.join(save_fold, name + '_' + num + '_{}.nii.gz'.format(posture))  # 保存drr图像时的路径
                        save_path = os.path.join(save_fold,
                                                 seg_name + '_' + num + '_{}.nii.gz'.format(posture))  # 保存mask图像时的路径
                        print(save_path)
                        sitk.WriteImage(gt_img, save_path)
                        del gt_img
                if int(row2['name'].split('_')[-1]) == 49:
                    del drr_detector


if __name__ == '__main__':
    # test_single_drr()
    # generate_verse_transform_data()
    # generate_verse_initial_drr()
    # generate_verse_drr_mask()
    # generate_bbx_points()
    generate_transform_data()
    # generate_total_initial_drr()
    # read_pose_generate_other_drrs()
    # read_pose_generate_total_masks()
    # concat_pose_data()
    # delete_off_vision_file()
