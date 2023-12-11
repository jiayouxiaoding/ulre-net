# 导入必要的库
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 定义一个函数，计算两幅图像的PSNR和SSIM
def calculate_psnr_ssim (img1, img2):
    # 将图像转换为灰度
    img1 = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor (img2, cv2.COLOR_BGR2GRAY)
    # 计算PSNR
    psnr = peak_signal_noise_ratio (img1, img2)
    # 计算SSIM
    ssim = structural_similarity (img1, img2)
    # 返回结果
    return psnr, ssim

# 定义一个函数，计算一个文件夹中图片的PSNR和SSIM
def calculate_folder_psnr_ssim (folder1, folder2):
    # 获取文件夹中的图片文件名
    filenames1 = os.listdir (folder1)
    filenames2 = os.listdir (folder2)
    # 检查文件夹中的图片数量是否相同
    if len (filenames1) != len (filenames2):
        print ("The number of images in the two folders are not equal.")
        return
    # 初始化一个列表，存储每张图片的PSNR和SSIM
    psnr_list = []
    ssim_list = []
    # 遍历每张图片
    for filename1, filename2 in zip (filenames1, filenames2):
        # 读取图片
        img1 = cv2.imread (os.path.join (folder1, filename1))
        img2 = cv2.imread (os.path.join (folder2, filename2))
        # 计算PSNR和SSIM
        psnr, ssim = calculate_psnr_ssim (img1, img2)
        # 将结果添加到列表中
        psnr_list.append (psnr)
        ssim_list.append (ssim)
    # 计算平均PSNR和SSIM
    mean_psnr = np.mean (psnr_list)
    mean_ssim = np.mean (ssim_list)
    # 返回结果
    return mean_psnr, mean_ssim

# 定义两个文件夹的路径，分别存放原始图片和重建图片
folder1 = "/data0/dsy/project/ULRE-Net/data/test/DICM"
folder2 = "/data0/dsy/project/ULRE-Net/data/result_up2_0.75_epoch3/DICM"
# 调用函数，计算文件夹中图片的PSNR和SSIM
mean_psnr, mean_ssim = calculate_folder_psnr_ssim (folder1, folder2)
# 打印结果
print (f"The mean PSNR of the images in the folder is {mean_psnr:.2f} dB.")
print (f"The mean SSIM of the images in the folder is {mean_ssim:.2f}.")
