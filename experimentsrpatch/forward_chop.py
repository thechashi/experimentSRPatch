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

# =============================================================================
#     print(f"lr_shape: {batch.shape}"
#           f"lr input : {batch}")
# 
#     print(f"hr_shape: {p_output.shape}"
#           f"hr output: {p_output}")
# =============================================================================

    return p_output

def trt_forward_chop_iterative_v2(
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
    

    Parameters
    ----------
    x : 4d array
        input image.
    trt_engine_path : str, optional
        path of the trt engine. The default is None.
    shave : int, optional
        shave value. The default is 10.
    min_size : int, optional
        total size of the image. The default is 1024.
    device : str, optional
        device cuda or cpu. The default is "cuda".
    print_result : bool, optional
        print result or not. The default is True.
    scale : int, optional
        hr = scale * lr. The default is 4.
    use_fp16 : bool, optional
        choose precision. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    patch_count = 0
    row_count = 0
    column_count = 0
    
    dim = int(math.sqrt(min_size))  # getting patch dimension
    b, c, img_height, img_width = x.size()  # current image batch, channel, height, width
    
    device = device
    output = torch.tensor(np.zeros((b, c, img_height * 4, img_width * 4))).numpy()

    f = open(trt_engine_path, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()


    new_i_s = 0

    new_i_s = 0 # new patch height start 
    for patch_height_start in range(0, img_height, dim - 2 * shave):
        row_count += 1
        right_most = False
        bottom_most = False
        left_increased = 0
        top_increased = 0
        new_j_s = 0
        new_j_e = 0
        for patch_width_start in range(0, img_width, dim - 2 * shave):
            column_count += 1
            patch_count += 1
            patch_height_end = min(img_height, patch_height_start + dim)
            patch_width_end = min(img_width, patch_width_start + dim)

            if img_height < patch_height_start + dim:
                bottom_most = True
                old_patch_height_start = patch_height_start
                patch_height_start = img_height - dim
                patch_height_start = (
                    (img_height - dim) if (img_height - dim) >= 0 else 0
                )
                top_increased = old_patch_height_start - patch_height_start

            if img_width < patch_width_start + dim:
                right_most = True
                old_patch_width_start = patch_width_start
                patch_width_start = img_width - dim
                patch_width_start = (img_width - dim) if (img_width - dim) >= 0 else 0
                left_increased = old_patch_width_start - patch_width_start

            left_crop, top_crop, right_crop, bottom_crop = (
                0,
                0,
                shave * scale,
                shave * scale,
            )

            if patch_width_start != 0:
                if right_most == True:
                    left_crop = (shave + left_increased) * scale
                else:
                    left_crop = shave * scale

            if patch_height_start != 0:
                if bottom_most == True:
                    top_crop = (shave + top_increased) * scale
                else:
                    top_crop = shave * scale

            if patch_width_end == img_width:
                right_crop = 0

            if patch_height_end == img_height:
                bottom_crop = 0

            # =============================================================================
            #             print('Patch no: {}, Row: {}, Column: {}\n'.format(patch_count, row_count, column_count))
            #             print('{}x{}:{}x{}'.format(patch_height_start, patch_height_end, patch_width_start, patch_width_end ))
            #             print('SR Patch size: {}x{}'.format(dim*scale, dim*scale))
            # =============================================================================

            h_s, h_e, w_s, w_e = (
                0 + top_crop,
                dim * scale - bottom_crop,
                0 + left_crop,
                dim * scale - right_crop,
            )
            # =============================================================================
            #             print('hs, he, ws, we', h_s, h_e, w_s, w_e)
            # =============================================================================
            if dim >= img_height and dim >= img_width:
                h_s, h_e, w_s, w_e = 0, img_height * scale, 0, img_width * scale
            elif dim < img_height and dim >= img_width:
                w_s, w_e = 0, img_width * scale
            elif dim >= img_height and dim < img_width:
                h_s, h_e = 0, img_height * scale

            lr = x[:, :, patch_height_start:patch_height_end, patch_width_start:patch_width_end]
            print('x.shape: ',x.shape)
            print('lr.shape', lr.shape)
            ba, ch, ht, wt = lr.shape

            lr = lr.numpy()

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
            stream = cuda.Stream()
            sr = predict(context, lr, d_input, stream, bindings, p_output, d_output)
            
            new_i_e = new_i_s + h_e - h_s
            new_j_e = new_j_s + w_e - w_s
            patch_crop_positions = [h_s, h_e, w_s, w_e]
            SR_positions = [new_i_s, new_i_e, new_j_s, new_j_e]
            
            # torch.cuda.synchronize()
            end = time.time()
            processing_time = end - start


            sr_small = sr[:, :, h_s:h_e, w_s:w_e]
            output[:, :, new_i_s:new_i_e, new_j_s:new_j_e] = sr_small
            del sr_small
            clear_start = time.time()
            if device == "cuda":
                ut.clear_cuda(None, None)

            new_j_s = new_j_e
            if patch_width_end == img_width:
                break
        new_i_s = new_i_e
        column_count = 0
        if patch_height_end == img_height:
            break

    if patch_count == 0:
        raise Exception("Shave size too big for given patch dimension")

    return output

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
    extra = x.clone().detach()
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
            ba, ch, ht, wt = lr.shape
            print('\nx: {}\n'.format(x))
            print('\nlr: {}\n'.format(lr))
            input_lr = torch.tensor(lr).int()
            output_folder = "output_images"
            file_name = "data/test7.jpg".split("/")[-1].split(".")[0]
            ut.save_image(input_lr[0].int(), output_folder, ht, wt, 4,
                          output_file_name=file_name + f"input_{i}_{j}_x4")
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
                break
            new_j_s = new_j_e

        new_i_s = new_i_e

        if h_e == h:
            break
    return output, total_time, total_crop_time, total_shift_time, total_clear_time


def trt_helper_upsampler_piterative_experiment(model_name, trt_engine_path, img_path, patch_dimension, shave=10):
    """
    Driver function to run a trtengine

    Parameters
    ----------
    model_name : str
        name of the model (i.e. "EDSR", "RRDB")
    trt_engine_path : str
        path of the trt engine.
    img_path : str
        path of the image file.
    patch_dimension : int
        patch_size.
    shave : int, optional
        patch overlapping size. The default is 10.

    Returns
    -------
    None.

    """
    # Loading model and image
    if model_name in ['EDSR']:
        img = ut.load_image(img_path)
        input_image = img.unsqueeze(0)
    elif model_name in ["RRDB"]:
        img = ut.npz_loader(img_path)
        input_image = img.unsqueeze(0)
    else:
        print('Unknown model!')
        return

    b, c, h, w = input_image.shape
    
    total_time = ut.timer()
    out_tuple = trt_forward_chop_iterative_v2(
        input_image,
        trt_engine_path=trt_engine_path,
        shave=shave,
        min_size=patch_dimension * patch_dimension,
        device="cuda",
        print_result=True,
    )
    output_image = out_tuple[0]
    total_time = total_time.toc()

# =============================================================================
#     for i in out_tuple[1:]:
#         print(i)
# =============================================================================
    print('Total executing time: ', total_time)
    output = torch.tensor(output_image).int()
    output_folder = "output_images"
    file_name = img_path.split("/")[-1].split(".")[0]
    ut.save_image(output, output_folder, h, w, 4, output_file_name=file_name + "_output_x4")

def helper_rrdb_piterative_experiment(img_dimension, patch_dimension):
    """
    Driver function for running pytorch model inference

    Parameters
    ----------
    img_dimension : int
        image one side dimension.
    patch_dimension : int
        patch size.

    Returns
    -------
    None.

    """
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
    output = trt_helper_upsampler_piterative_experiment("EDSR", "inference_models/edsr.trt", "data/test9_400.jpg", int(sys.argv[1]))
# =============================================================================
#     #img = ut.load_image("data/test9_400.jpg").numpy()
#     img = ut.npz_loader("data/slices/0.npz")
#     print(type(img))
#     print(img.shape)
#     print(img.size)
# 
# =============================================================================
