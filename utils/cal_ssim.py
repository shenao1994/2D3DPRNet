from skimage.metrics import structural_similarity
import SimpleITK as sitk
import numpy as np
import os


def cal_ssim():
    mov_path = 'E:/pythonWorkplace/xreg/data/tuodao/dual_view/new_tuodao_1122/L3_2VIEW_DRR/drr_remap_000_000.nii'
    fixed_path = 'E:/pythonWorkplace/xreg/data/tuodao/L3/AP/L3_resized.nii.gz'
    mask_path = ''
    mov_img = sitk.ReadImage(mov_path)
    mov_arr = sitk.GetArrayFromImage(mov_img)
    fixed_img = sitk.ReadImage(fixed_path)
    resized_fix = resample_img(fixed_img, new_width=256)
    fixed_arr = sitk.GetArrayFromImage(resized_fix)
    mask = sitk.ReadImage(mask_path)
    resized_mask = resample_img(mask, new_width=256)
    mask_arr = sitk.GetArrayFromImage(resized_mask)
    fixed_arr = (fixed_arr - np.min(fixed_arr)) / (np.max(fixed_arr) - np.min(fixed_arr))
    # plt.imshow(mov_arr)
    # plt.show()
    print(ssim(mov_arr, fixed_arr * mask_arr))


def ssim(imageA, imageB):
    """
    考虑到人眼对于图像细节的敏感程度，
    比MSE更能反映图像的相似度。
    SSIM计算公式较为复杂，包含对于亮度、对比度、结构等因素的综合考虑。
    @param imageA: 图片1
    @param imageB: 图片2
    @return: SSIM计算公式较为复杂包含对于亮度、对比度、结构等因素的综合考虑。
    """
    # ssim_val = cv2.SSIM(imageA, imageB)
    # ssim_val = structural_similarity(imageA, imageB, data_range=255, multichannel=True)
    ssim_val = structural_similarity(imageA, imageB, data_range=1)
    return ssim_val


def resample_img(input_img, new_width=None, save_path=''):
    # image_file_reader = sitk.ImageFileReader()
    # # only read DICOM images
    # image_file_reader.SetImageIO("GDCMImageIO")
    # image_file_reader.SetFileName(input_file_name)
    # image_file_reader.ReadImageInformation()
    image_size = list(input_img.GetSize())
    if len(image_size) == 3 and image_size[2] == 1:
        input_img = input_img[:, :, 0]
    # input_img.Set(image_size)
    image = input_img
    if new_width:
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_spacing = [(original_size[0]) * original_spacing[0] / new_width] * 2
        new_size = [
            new_width,
            int((original_size[1]) * original_spacing[1] / new_spacing[1]),
        ]
        image = sitk.Resample(
            image1=image,
            size=new_size,
            transform=sitk.Transform(),
            interpolator=sitk.sitkLinear,
            outputOrigin=image.GetOrigin(),
            outputSpacing=new_spacing,
            outputDirection=image.GetDirection(),
            defaultPixelValue=0,
            outputPixelType=image.GetPixelID(),
        )
    if not os.path.exists(save_path) and save_path != '':
        sitk.WriteImage(image, save_path)
    return image


if __name__ == '__main__':
    cal_ssim()
