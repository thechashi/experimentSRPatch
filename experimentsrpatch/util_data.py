# =============================================================================
# import pdb
# from pathlib import Path
# import math
# import random
# 
# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import metrics
# import torch
# =============================================================================
import torch
import torch.nn.functional as F
import utilities as ut
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
# =============================================================================
# from . import imresize as kernelganresize
# from . import image_common
# =============================================================================


# =============================================================================
# def get_noisy_image(noise_type, img):
#     """
#     This function will create noisy image based on the noise type given
#     :param noise_type: The type of noise required (supported guass, salt and poisson)
#     :param img: image
#     :return: noisy image
#     """
#     height, width = img.shape
#     noisy_image = img.copy()
#     if noise_type == "guass":
#         print("guassian noise added")
#         img_mean = 0
#         img_var = 0.1
#         guass = np.random.normal(img_mean, img_var, (height, width))
#         noisy_image += guass
# 
#     elif noise_type == "poisson":
#         print("poisson noise added")
#         img, img_max, img_min = image_common.min_max_normalize(img)
#         vals = len(np.unique(img))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy_image = np.random.poisson(img * vals) / float(vals)
#         noisy_image = noisy_image * (img_max - img_min) + img_min
# 
#     elif noise_type == "salt_and_pepper":
#         print("salt and pepper noise added")
#         s_vs_p = 0.5
#         amount = 0.004
#         noisy_image = img.copy()
#         # Salt mode
#         num_salt = np.ceil(amount * img.size * s_vs_p)
#         coords = [np.random.randint(0, i - 1, int(num_salt))
#                   for i in img.shape]
#         noisy_image[tuple(coords)] = 1
# 
#         # Pepper mode
#         num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
#         coords = [np.random.randint(0, i - 1, int(num_pepper))
#                   for i in img.shape]
#         noisy_image[tuple(coords)] = 0
#     return noisy_image
# 
# 
# def get_mean_var(img):
#     """
#     This function will return mean and variance
#     :param img: image
#     :return: mean and variance
#     """
#     mean = np.mean(img)
#     var = np.var(img)
#     return mean, var
# 
# 
# def noise_param_search(hr_image, lr_image, conf):
#     """This function will check for best mean and sigma that can produce
#         an image which will have atleast 0.95 ssim after adding noise
#         and it will return the image.
#         :param hr_image: high resolution image
#         :param lr_image: low resolution image
#         :param conf: configurator
#     """
#     noise_patches = noise_creation(hr_image, lr_image, conf)
#     if noise_patches:
#         conf["preprocess_logging"].info("noise patches found adding noise to lr")
#         if len(noise_patches) == 1:
#             lr_image = lr_image + noise_patches[0]
#         else:
#             lr_image = lr_image + noise_patches[random.randint(0, len(noise_patches) - 1)]
#     return lr_image
# 
# 
# def noise_creation(hr_img, lr_img, conf):
#     """
#     This function will collect noise patches. noise patches are calculated by adding noise
#     if after adding patch the ssim is greater than 0.95 those patches will be added
#     :param hr_img: high resolution image
#     :param lr_img: low resolution image
#     :param conf: configurator
#     :return: noise patches
#     """
#     noise_patches = []
#     height, width = hr_img.shape
#     height_down, width_down = lr_img.shape
#     lr_img_org = lr_img.copy()
# 
#     for index_h, img_h in enumerate(range(0, height - height_down, height_down)):
#         for index_w, img_w in enumerate(range(0, width-width_down, width_down)):
#             patch = hr_img[index_h: index_h+height_down, index_w: index_w+width_down]
#             patch = patch - np.mean(patch)
#             lr_img_noise = lr_img + patch
#             vmax = np.max(lr_img)
#             if vmax == 0:
#                 continue
#             ssim_lr = metrics.structural_similarity(lr_img_noise, lr_img_org, data_range=vmax)
#             conf["preprocess_logging"].info(f"this is for patch no {index_h} and {index_w}: {ssim_lr}")
#             if ssim_lr > 0.7:
#                 noise_patches.append(patch)
# 
#     return noise_patches
# 
# 
# class BicubicDownsample:
#     """This class handles the downsample of the image using the cubic kernel"""
# 
#     def __init__(self):
#         use_cuda = torch.cuda.is_available()
#         self.device = torch.device("cuda:0" if use_cuda else "cpu")
# 
#     def adjusted_dims(self, img):
#         """
#         This function adjusts dimensions of image to a multiple of 4. This is
#         done to avoid image dimensions to remain consistent after downsample and
#         upsample.
#         param img: image to pad the dimensions before sending to GAN
#         :return: img is padded to the dimension value of multiple of 4 using
#         reflection padding.
#         """
#         height, width = img.shape
#         pad_height = height % 4
#         pad_width = width % 4
#         img = np.pad(img, ((0, 4 - pad_height), (0, 4 - pad_width)), mode="reflect")
#         return img
# 
#     def calculate_weights_indices(
#         self, in_length, out_length, scale, kernel_width, antialiasing
#     ):
#         """Some operations of making data set. Reference from `https://github.com/xinntao/BasicSR`"""
#         if (scale < 1) and antialiasing:
#             # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
#             kernel_width = kernel_width / scale
# 
#             # Output-space coordinates
#         x = torch.linspace(1, out_length, out_length)
# 
#         # Input-space coordinates. Calculate the inverse mapping such that 0.5
#         # in output space maps to 0.5 in input space, and 0.5+scale in output
#         # space maps to 1.5 in input space.
#         u = x / scale + 0.5 * (1 - 1 / scale)
# 
#         # What is the left-most pixel that can be involved in the computation?
#         left = torch.floor(u - kernel_width / 2)
# 
#         # What is the maximum number of pixels that can be involved in the
#         # computation?  Note: it's OK to use an extra pixel here; if the
#         # corresponding weights are all zero, it will be eliminated at the end
#         # of this function.
#         P = math.ceil(kernel_width) + 2
# 
#         # The indices of the input pixels involved in computing the k-th output
#         # pixel are in row k of the indices matrix.
#         indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
#             0, P - 1, P
#         ).view(1, P).expand(out_length, P)
# 
#         # The weights used to compute the k-th output pixel are in row k of the
#         # weights matrix.
#         distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
#         # apply cubic kernel
#         if (scale < 1) and antialiasing:
#             weights = scale * self.cubic(distance_to_center * scale)
#         else:
#             weights = self.cubic(distance_to_center)
#         # Normalize the weights matrix so that each row sums to 1.
#         weights_sum = torch.sum(weights, 1).view(out_length, 1)
#         weights = weights / weights_sum.expand(out_length, P)
# 
#         # If a column in weights is all zero, get rid of it. only consider the first and last column.
#         weights_zero_tmp = torch.sum((weights == 0), 0)
#         if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
#             indices = indices.narrow(1, 1, P - 2)
#             weights = weights.narrow(1, 1, P - 2)
#         if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
#             indices = indices.narrow(1, 0, P - 2)
#             weights = weights.narrow(1, 0, P - 2)
#         weights = weights.contiguous()
#         indices = indices.contiguous()
#         sym_len_s = -indices.min() + 1
#         sym_len_e = indices.max() - in_length
#         indices = indices + sym_len_s - 1
#         return weights, indices, int(sym_len_s), int(sym_len_e)
# 
#     def cubic(self, x):
#         absx = torch.abs(x)
#         absx2 = absx ** 2
#         absx3 = absx ** 3
#         return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
#             -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
#         ) * (((absx > 1) * (absx <= 2)).type_as(absx))
# 
#     def imresize(self, img, scale, antialiasing=True):
#         # Now the scale should be the same for H and W
#         # input: img: CHW RGB [0,1]
#         # output: CHW RGB [0,1] w/o round
# 
#         in_C, in_H, in_W = img.size()
#         _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
#         kernel_width = 4
# 
#         # Return the desired dimension order for performing the resize.  The
#         # strategy is to perform the resize first along the dimension with the
#         # smallest scale factor.
#         # Now we do not support this.
# 
#         # get weights and indices
#         weights_H, indices_H, sym_len_Hs, sym_len_He = self.calculate_weights_indices(
#             in_H, out_H, scale, kernel_width, antialiasing
#         )
#         weights_W, indices_W, sym_len_Ws, sym_len_We = self.calculate_weights_indices(
#             in_W, out_W, scale, kernel_width, antialiasing
#         )
#         # process H dimension
#         # symmetric copying
#         img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
#         img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)
# 
#         sym_patch = img[:, :sym_len_Hs, :]
#         if self.in_device:
#             inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(self.device)
#         else:
#             inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
# 
#         sym_patch_inv = sym_patch.index_select(1, inv_idx)
#         img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)
# 
#         sym_patch = img[:, -sym_len_He:, :]
#         if self.in_device:
#             inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(self.device)
#         else:
#             inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
#         sym_patch_inv = sym_patch.index_select(1, inv_idx)
#         img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)
# 
#         out_1 = torch.FloatTensor(in_C, out_H, in_W)
#         kernel_width = weights_H.size(1)
#         for i in range(out_H):
#             idx = int(indices_H[i][0])
#             out_1[0, i, :] = (
#                 img_aug[0, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
#             )
#             # out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
#             # out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
#         # process W dimension
#         # symmetric copying
#         out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
#         out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)
# 
#         sym_patch = out_1[:, :, :sym_len_Ws]
#         inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
#         sym_patch_inv = sym_patch.index_select(2, inv_idx)
#         out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)
# 
#         sym_patch = out_1[:, :, -sym_len_We:]
#         inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
#         sym_patch_inv = sym_patch.index_select(2, inv_idx)
#         out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)
# 
#         out_2 = torch.FloatTensor(in_C, out_H, out_W)
#         kernel_width = weights_W.size(1)
#         for i in range(out_W):
#             idx = int(indices_W[i][0])
#             out_2[0, :, i] = out_1_aug[0, :, idx : idx + kernel_width].mv(weights_W[i])
#             # out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
#             # out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])
# 
#         return torch.clamp(out_2, 0, 1)
# 
# 
#     def downsampling_opt(self, image, conf):
#         """
#         This function will do downample based on mean of the image
#         :param image: image matrix
#         :param conf: configurator
#         :return: downsampled image, cleaned hr image
#         """
#         stats = conf["stats"]
#         img_sum = np.sum(image - np.mean(image))
#         if img_sum < 0.02:
#             # downsample using interpolate bicubic
#             hr_image = kernelganresize(image, 1 / conf["cleanup_factor"], kernel="cubic")
#             del image
#             lr_image = image_common.t_interpolate(hr_image, "bicubic", 1 / conf["upscale_factor"])
#             lr_image = np.clip(lr_image, stats["min"], stats["max"])
#         else:
#             height, width = image.shape
#             hr_image, img_max, img_min = image_common.min_max_normalize(image)
#             del image
#             # Adjust the dimensions if the height and width is not divisible by 4
#             # This is mainly done to get back the same original shape if upsample
#             # is done. Mainly needed for CutBlur because the LR image is upsampled
#             # using Bicubic and then blur is introduced.
#             if conf["adjust_dim"]:
#                 if (height % 4 != 0) or (width % 4 != 0):
#                     hr_image = self.adjusted_dims(hr_image)
# 
# 
#             # converting hr image to tensor and moving to device
#             height, width = hr_image.shape
#             hr_image = hr_image.reshape((1, height, width))
#             hr_image = torch.from_numpy(hr_image)
#             hr_image = hr_image.to(self.device)
# 
#             # hr_cleanup
#             hr_image = self.imresize(hr_image, 1.0 / conf["cleanup_factor"], True)
#             _, w, h = hr_image.size()
#             w = w - w % conf["upscale_factor"]
#             h = h - h % conf["upscale_factor"]
#             hr_image = hr_image[:, :w, :h]
# 
#             # lr downsample using tencent downsample
#             hr_image = hr_image.to(self.device)
#             lr_image = self.imresize(hr_image, 1.0 / conf["upscale_factor"], True)
#             hr_image = hr_image.cpu().numpy()[0, :, :]
#             lr_image = lr_image.cpu().numpy()[0, :, :]
# 
#             hr_image = (hr_image * (img_max - img_min)) + img_min
#             lr_image = (lr_image * (img_max - img_min)) + img_min
#             lr_image = np.clip(lr_image, stats["min"], stats["max"])
# 
#         return lr_image, hr_image
# 
#     def process_for_lr(self, conf, img):
#         r"""The low resolution data set is preliminarily processed."""
#         img_name = Path(conf["hr_dir_img"]).name.split('.')[0]
#         conf["preprocess_logging"].info(f"processing image {img_name}")
#         self.in_device = True
# 
#         img_hr = img.copy()
#         lr_img, hr_img = self.downsampling_opt(img, conf)
#         del img
# 
#         if conf["debug_pic"]:
#             conf["hr_dir_img"] = str(conf["hr_dir_img"]) + ".png"
#             plt.imsave(conf["hr_dir_img"], hr_img, cmap="gray")
#         else:
#             np.savez_compressed(conf["hr_dir_img"], hr_img)
# 
#         if conf["noise_data"]:
#             lr_img = noise_param_search(img_hr, lr_img, conf)
#         if conf["debug_pic"]:
#             img_bicub = kernelganresize(lr_img, 4, kernel="cubic")
#             bicub_path = str(conf["lr_dir_img"]) + "bicubic.png"
#             conf["lr_dir_img"] = str(conf["lr_dir_img"]) + ".png"
#             plt.imsave(conf["lr_dir_img"], lr_img, cmap="gray")
#             plt.imsave(bicub_path, img_bicub, cmap="gray")
#         else:
#             np.savez_compressed(conf["lr_dir_img"], lr_img)
# 
#     def resizer(self, conf, img):
#         r"""The low resolution data set is preliminarily processed."""
#         self.in_device = True
#         height, width = img.shape
#         img, img_max, img_min = image_common.min_max_normalize(img)
#         img = img.reshape(1, height, width)
#         img = torch.from_numpy(img).to(self.device)
# 
#         # Remove noise
#         img = self.imresize(img, 1.0, True)
#         img = img.to(self.device)
# 
#         img = self.imresize(img, conf["scale_factor"], True)
#         # reconvert the image from gpu to cpu and to numpy
#         img = img.cpu().numpy()
#         img = img[0, :, :]
#         img = (img * (img_max - img_min)) + img_min
#         return img
# =============================================================================
    
def t_interpolate(image, mode, scale_factor):
    """
    :param image: the image matrix
    :param mode: the type of interpolation
    :param scale_factor: the amount of size the image has to be scaled
    :return: image: interpolated image
    """
    c, h, w = image.shape
    image = image.reshape((1, c, h, w))
    image = torch.tensor(image, dtype=torch.float32)
    if mode == "nearest":
        image = F.interpolate(
            image,
            scale_factor=scale_factor,
            mode=mode,
            recompute_scale_factor=False,
        )
    else:
        image = F.interpolate(
            image,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=True,
            recompute_scale_factor=False,
        )
    image = image.numpy()
    image = image[0, :, :, :]
    return image
from PIL import Image
if __name__ == "__main__":
    img_path = 'data/diff_sizes/test2_4000.jpg'
    
    i1 = ut.load_image(img_path)
    total_image = 1
    for i in range(total_image):
        
        print(i1.shape)
        c, h, w = i1.shape 
        scale_factor = 0.25
        img = t_interpolate(i1, mode='bicubic', scale_factor=scale_factor)
        print(img.shape)
    
        output = torch.tensor(img).int()
        output_folder = 'data/diff_sizes/'
        file_name = img_path.split("/")[-1].split("_")[0]  + "_" + str(int(h*scale_factor))
        ut.save_image(output, output_folder, int(h*scale_factor), int(w*scale_factor), 1, output_file_name=file_name, add_date=False)
        i1 = img