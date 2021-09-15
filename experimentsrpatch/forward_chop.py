"""
Iterative forward chopping
"""
import math
import numpy as np
import utilities as ut
import modelloader as md
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import click
import subprocess
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image


def forward_chop_iterative(
    x, model=None, shave=10, min_size=1024, device="cuda", print_result=True
):
    """
    Forward chopping in an iterative way

    Parameters
    ----------
    x : tensor
        input image.
    model : nn.Module, optional
        SR model. The default is None.
    shave : int, optional
        patch shave value. The default is 10.
    min_size : int, optional
        total patch size (dimension x dimension) . The default is 1024.
    device : int, optional
        GPU or CPU. The default is 'cuda'.
    print_result : bool, optional
        print result or not. The default is True.

    Returns
    -------
    output : tensor
        output image.
    total_time : float
        total execution time.
    total_crop_time : float
        total cropping time.
    total_shift_time : float
        total GPU to CPU shfiting time.
    total_clear_time : float
        total GPU clearing time.

    """
    dim = int(math.sqrt(min_size))  # getting patch dimension
    b, c, h, w = x.size()  # current image batch, channel, height, width
    device = device
    patch_count = 0
    output = torch.tensor(np.zeros((b, c, h * 4, w * 4)))
    total_time = 0
    total_crop_time = 0
    total_shift_time = 0
    total_clear_time = 0
    # =============================================================================
    #     if device == "cuda":
    #         torch.cuda.synchronize()
    #         x = x.to(device)
    # =============================================================================

    new_i_s = 0
    for i in range(0, h, dim - 2 * shave):
        new_j_s = 0
        new_j_e = 0
        for j in range(0, w, dim - 2 * shave):
            patch_count += 1
            h_s, h_e = i, min(h, i + dim)  # patch height start and end
            w_s, w_e = j, min(w, j + dim)  # patch width start and end
            # subprocess.run("gpustat", shell=True)
            # =============================================================================
            #             print(
            #                 "Patch no: {} : {}-{}x{}-{}\n".format(patch_count, h_s, h_e, w_s, w_e)
            #             )
            # =============================================================================

            lr = x[:, :, h_s:h_e, w_s:w_e]
            print(lr.shape)
            if device == "cuda":
                torch.cuda.synchronize()
                lr = lr.to(device)
            with torch.no_grad():
                # EDSR processing
                start = time.time()
                torch.cuda.synchronize()
                sr = model(lr)
                torch.cuda.synchronize()
                end = time.time()
                processing_time = end - start
                total_time += processing_time
            # =============================================================================
            #             print('Processing time: ', processing_time)
            # =============================================================================

            shift_start = time.time()
            torch.cuda.synchronize()
            sr = sr.cpu()
            torch.cuda.synchronize()
            shift_end = time.time()
            shift_time = shift_end - shift_start

            # new cropped patch's dimension (h and w)
            n_h_s, n_h_e, n_w_s, n_w_e = 0, 0, 0, 0

            n_h_s = 0 if h_s == 0 else (shave * 4)
            n_h_e = ((h_e - h_s) * 4) if h_e == h else (((h_e - h_s) - shave) * 4)
            new_i_e = new_i_s + n_h_e - n_h_s

            n_w_s = 0 if w_s == 0 else (shave * 4)
            n_w_e = ((w_e - w_s) * 4) if w_e == w else (((w_e - w_s) - shave) * 4)
            new_j_e = new_j_e + n_w_e - n_w_s

            # corpping image in
            crop_start = time.time()
            sr_small = sr[:, :, n_h_s:n_h_e, n_w_s:n_w_e]
            crop_end = time.time()
            crop_time = crop_end - crop_start
            total_crop_time += crop_time

            # =============================================================================
            #             shift_start = time.time()
            #             if device == "cuda":
            #                 torch.cuda.synchronize()
            #                 sr_small = sr_small.to("cpu")
            #                 torch.cuda.synchronize()
            #             shift_end = time.time()
            #             shift_time = shift_end - shift_start
            # =============================================================================
            total_shift_time += shift_time
            output[:, :, new_i_s:new_i_e, new_j_s:new_j_e] = sr_small
            del sr_small
            clear_start = time.time()
            if device == "cuda":
                ut.clear_cuda(lr, sr)
            clear_end = time.time()
            clear_time = clear_end - clear_start
            total_clear_time += clear_time
            if w_e == w:
                break
            new_j_s = new_j_e

        new_i_s = new_i_e

        if h_e == h:
            break
    # =============================================================================
    #     if print_result == True:
    #         print("Patch dimension: {}x{}".format(dim, dim))
    #         print("Total pacthes: ", patch_count)
    #         print("Total EDSR Processing time: ", total_time)
    #         print("Total crop time: ", total_crop_time)
    #         print("Total shift time: ", total_shift_time)
    #         print("Total clear time: ", total_clear_time)
    # =============================================================================
    return output, total_time, total_crop_time, total_shift_time, total_clear_time


def save_image(model_name, img, scale=4, output_path="results/result_imagex4.png"):
    b, c, h, w = img.shape
    if model_name in ["EDSR"]:
        img = img.int()
    save_time = ut.timer()
    fig = plt.figure(figsize=((4 * h) / 1000, (4 * w) / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    # fig = plt.figure(figsize=(4*h, 4*w))
    ax.imshow(img[0].permute((1, 2, 0)))
    fig.savefig(
        output_path,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
        dpi=1000,
    )
    save_time = save_time.toc()
    return save_time


def get_file_name(img_path):
    path_list = list(img_path.split("/"))
    file_name = list(path_list[-1].split("."))[0]
    return file_name


def predict(
    context, batch, d_input, stream, bindings, p_output, d_output
):  # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(p_output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    print(f"lr_shape: {batch.shape}"
          f"lr input : {batch}")

    print(f"hr_shape: {p_output.shape}"
          f"hr output: {p_output}")

    return p_output


def trt_forward_chop_iterative(
    x,
    trt_engine_path=None,
    shave=10,
    min_size=1024,
    device="cuda",
    print_result=True,
    scale=4,
    use_fp16=False,
):
    """
    Forward chopping in an iterative way

    Parameters
    ----------
    x : tensor
        input image.
    model : nn.Module, optional
        SR model. The default is None.
    shave : int, optional
        patch shave value. The default is 10.
    min_size : int, optional
        total patch size (dimension x dimension) . The default is 1024.
    device : int, optional
        GPU or CPU. The default is 'cuda'.
    print_result : bool, optional
        print result or not. The default is True.

    Returns
    -------
    output : tensor
        output image.
    total_time : float
        total execution time.
    total_crop_time : float
        total cropping time.
    total_shift_time : float
        total GPU to CPU shfiting time.
    total_clear_time : float
        total GPU clearing time.

    """
    dim = int(math.sqrt(min_size))  # getting patch dimension
    b, c, h, w = x.size()  # current image batch, channel, height, width
    device = device
    patch_count = 0
    output = torch.tensor(np.zeros((b, c, h * 4, w * 4))).numpy()
    total_time = 0
    total_crop_time = 0
    total_shift_time = 0
    total_clear_time = 0

    f = open(trt_engine_path, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    new_i_s = 0
    stream = cuda.Stream()
    for i in range(0, h, dim - 2 * shave):
        new_j_s = 0
        new_j_e = 0
        for j in range(0, w, dim - 2 * shave):
            patch_count += 1
            h_s, h_e = i, min(h, i + dim)  # patch height start and end
            w_s, w_e = j, min(w, j + dim)  # patch width start and end
            lr = x[:, :, h_s:h_e, w_s:w_e]
            lr = lr.numpy()
            print(f"shape of lr:{lr.shape}")

            # EDSR processing
            start = time.time()
            # torch.cuda.synchronize()
            USE_FP16 = use_fp16
            target_dtype = np.float16 if USE_FP16 else np.float32
            ba, ch, ht, wt = lr.shape
            lr = np.ascontiguousarray(lr, dtype=np.float32)

            # need to set input and output precisions to FP16 to fully enable it
            p_output = np.empty([b, c, ht * scale, wt * scale], dtype=target_dtype)

            # allocate device memory
            d_input = cuda.mem_alloc(1 * lr.nbytes)
            d_output = cuda.mem_alloc(1 * p_output.nbytes)

            bindings = [int(d_input), int(d_output)]

            sr = predict(context, lr, d_input, stream, bindings, p_output, d_output)

            output_sr = torch.tensor(sr).int()
            output_folder = "output_images"
            file_name = "data/test7.jpg".split("/")[-1].split(".")[0]
            ut.save_image(output_sr[0], output_folder, ht, wt, 4,
                          output_file_name=file_name + f"{i}_{j}_x4")

            # torch.cuda.synchronize()
            end = time.time()
            processing_time = end - start
            total_time += processing_time

            # new cropped patch's dimension (h and w)
            n_h_s, n_h_e, n_w_s, n_w_e = 0, 0, 0, 0

            n_h_s = 0 if h_s == 0 else (shave * 4)
            n_h_e = ((h_e - h_s) * 4) if h_e == h else (((h_e - h_s) - shave) * 4)
            new_i_e = new_i_s + n_h_e - n_h_s

            n_w_s = 0 if w_s == 0 else (shave * 4)
            n_w_e = ((w_e - w_s) * 4) if w_e == w else (((w_e - w_s) - shave) * 4)
            new_j_e = new_j_e + n_w_e - n_w_s

            # corpping image in
            crop_start = time.time()
            sr_small = sr[:, :, n_h_s:n_h_e, n_w_s:n_w_e]
            crop_end = time.time()
            crop_time = crop_end - crop_start
            total_crop_time += crop_time
            output[:, :, new_i_s:new_i_e, new_j_s:new_j_e] = sr_small
            del sr_small
            clear_start = time.time()
            if device == "cuda":
                ut.clear_cuda(None, None)
            clear_end = time.time()
            clear_time = clear_end - clear_start
            total_clear_time += clear_time
            if w_e == w:
                print("first break")
                break
            new_j_s = new_j_e

        new_i_s = new_i_e

        if h_e == h:
            print("second break")
            break
    return output, total_time, total_crop_time, total_shift_time, total_clear_time


# =============================================================================
# @click.command()
# @click.option("--model_name", default="RRDB", help="Upsampler model name")
# @click.option("--img_path", default="data/slices/0.npz", help="Input image path")
# @click.option("--patch_dimension", default=293, help="Dimension of a square patch")
# @click.option("--shave", default=10, help="Overlapping value of two patches")
# @click.option("--scale", default=4, help="Scaling amount of the HR image")
# @click.option(
#     "--print_result",
#     default=True,
#     help="Print the timing of each segment of an upsampling or not",
# )
# @click.option("--device", default="cuda", help="GPU or CPU")
# def main(model_name, img_path, patch_dimension, shave, scale, print_result, device):
#     # Loading image
#     input_image = ut.get_grayscale_image_tensor(img_path)
#     c, h, w = input_image.shape
#     input_image = input_image.reshape((1, c, h, w))
#     # Loading model
#     if model_name == "RRDB":
#         model = md.load_rrdb(device)
#     elif model_name == "EDSR":
#         model = md.load_edsr()
#     else:
#         print("Unknown model")
#     model.eval()
#     total_time = ut.timer()
#     out_tuple = forward_chop_iterative(
#         input_image,
#         shave=shave,
#         min_size=patch_dimension * patch_dimension,
#         model=model,
#         device=device,
#         print_result=print_result,
#     )
#     model.cpu()
#     del model
#     output_image = out_tuple[0]
#     total_time = total_time.toc()
#     #print("Total forward chopping time: ", total_time)
#     if print_result:
#         for i in out_tuple[1:]:
#             print(i)
#         print(total_time)
# =============================================================================
# =============================================================================
#         print("\nSaving...\n")
#         output_file_name = get_file_name(img_path)
#         save_time = ut.timer()
#         ut.save_image(
#             output_image[0],
#             "results/",
#             input_height=h,
#             input_width=w,
#             scale=4,
#             output_file_name=output_file_name,
#         )
#         save_time = save_time.toc()
#         print("Saving time: {}".format(save_time))
# =============================================================================


def trt_helper_rrdb_piterative_experiment(img_dimension, patch_dimension):
    # Loading model and image
    img = ut.load_image("data/test7.jpg").numpy()
    # img = img.f.arr_0
    img = np.resize(img, (img_dimension, img_dimension))
    # =============================================================================
    #     img = img[np.newaxis, :, :]
    # =============================================================================
    img2 = np.zeros((3, img.shape[0], img.shape[1]))
    img2[0, :, :] = img
    img2[1, :, :] = img
    img2[2, :, :] = img
    img = img2
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    input_image = img
    b, c, h, w = input_image.shape
    # input_image = input_image.reshape((1, c, h, w))
    # Loading model
    # =============================================================================
    #     model = md.load_edsr("cuda")
    #
    #     model.eval()
    # =============================================================================
    total_time = ut.timer()
    # =============================================================================
    #     out_tuple = forward_chop_iterative(
    #         input_image,
    #         shave=10,
    #         min_size=patch_dimension * patch_dimension,
    #         model=model,
    #         device="cuda",
    #         print_result=True,
    #     )
    # =============================================================================
    out_tuple = trt_forward_chop_iterative(
        input_image,
        trt_engine_path="inference_models/edsr.trt",
        shave=10,
        min_size=patch_dimension * patch_dimension,
        device="cuda",
        print_result=True,
    )
    # =============================================================================
    #     model.cpu()
    #     del model
    # =============================================================================
    output_image = out_tuple[0]
    print(output_image)
    total_time = total_time.toc()

    for i in out_tuple[1:]:
        print(i)
    print(total_time)

    output = torch.tensor(output_image).int()
    output_folder = "output_images"
    file_name = "data/test7.jpg".split("/")[-1].split(".")[0]
    ut.save_image(output[0], output_folder, patch_dimension, patch_dimension, 4, output_file_name=file_name + "_x4")

def helper_rrdb_piterative_experiment(img_dimension, patch_dimension):
    # Loading model and image
    img = None
    model = None
    img = np.load("data/slices/0.npz")
    img = img.f.arr_0
    img = np.resize(img, (img_dimension, img_dimension))
    img = img[np.newaxis, :, :]
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    model = md.load_rrdb(device="cuda")

    input_image = img
    b, c, h, w = input_image.shape
    # input_image = input_image.reshape((1, c, h, w))
    # Loading model
    model = md.load_rrdb("cuda")

    model.eval()
    total_time = ut.timer()
    out_tuple = forward_chop_iterative(
        input_image,
        shave=10,
        min_size=patch_dimension * patch_dimension,
        model=model,
        device="cuda",
        print_result=True,
    )
    model.cpu()
    del model
    output_image = out_tuple[0]
    total_time = total_time.toc()

    for i in out_tuple[1:]:
        print(i)
    print(total_time)


if __name__ == "__main__":
    # main()
    output = trt_helper_rrdb_piterative_experiment(int(sys.argv[1]), int(sys.argv[2]))
