"""
Source code for batch patch iterative forward chopping
"""
import sys
import time
import numpy as np
import torchvision
import torch
import subprocess
import matplotlib.pyplot as plt
import modelloader as md
import utilities as ut


def create_patch_list(
    patch_list, img, dim, shave, scale, channel, img_height, img_width
):
    """
    Loads the patch_list with patches of size (dim x dim) from the img

    Parameters
    ----------
    patch_list : list
        container for patches.
    img : np array
        3D matrix.
    dim : int
        patch dimension.
    shave : int
        shave value for patch overlapping.
    scale : int
        scale value for LR to SR.
    channel : int
        input image total channel.
    img_height : int
        input image height.
    img_width : int
        input image width.

    Returns
    -------
    None.

    """
    patch_count = 0
    row_count = 0
    column_count = 0

    # =============================================================================
    #     print('LR Image size: {}x{}'.format(img_height, img_width))
    #     print('SR Image size: {}x{}'.format(img_height*scale, img_width*scale))
    # =============================================================================
    new_i_s = 0
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
            new_i_e = new_i_s + h_e - h_s
            new_j_e = new_j_e + w_e - w_s
            patch_crop_positions = [h_s, h_e, w_s, w_e]
            SR_positions = [new_i_s, new_i_e, new_j_s, new_j_e]
            # =============================================================================
            #             print('Cropped patch position: {}-{}x{}-{}'.format(h_s, h_e, w_s, w_e))
            #             print('SR output position: {}-{}x{}-{}\n\n'.format(new_i_s, new_i_e, new_j_s, new_j_e))
            # =============================================================================
            patch_details = (
                row_count,
                column_count,
                patch_crop_positions,
                SR_positions,
                img[
                    :,
                    patch_height_start:patch_height_end,
                    patch_width_start:patch_width_end,
                ],
            )
            patch_list[patch_count] = patch_details
            new_j_s = new_j_e
            if patch_width_end == img_width:
                break
        new_i_s = new_i_e
        column_count = 0
        if patch_height_end == img_height:
            break

    if patch_count == 0:
        raise Exception("Shave size too big for given patch dimension")
    # print(len(patch_list))
    return patch_count


def batch_forward_chop(
    patch_list,
    batch_size,
    channel,
    img_height,
    img_width,
    dim,
    shave,
    scale,
    model,
    device="cuda",
    print_timer=True,
):
    """
    Create SR image from batches of patches

    Parameters
    ----------
    patch_list : list
        list of patches.
    batch_size : int
        batch size.
    channel : int
        input image channel.
    img_height : int
        input image height.
    img_width : int
        input image width.
    dim : int
        patch dimension.
    shave : int
        shave value for patch.
    scale : int
        scale for LR to SR.
    model : nn.Module
        SR model.
    device : str, optional
        GPU or CPU. The default is 'cuda'.
    print_timer : bool, optional
        Print result or not. The default is True.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    3D matrix, tuple
        output_image, tuple of timings.

    """
    logger = ut.get_logger()
    total_patches = len(patch_list)
    if batch_size > total_patches:
        sys.exit(2)
        raise Exception("Batch size greater than total number of patches")
    output_image = torch.tensor(
        np.zeros((channel, img_height * scale, img_width * scale))
    )

    cpu_to_gpu_time = 0
    gpu_to_cpu_time = 0
    batch_creating_time = 0
    total_EDSR_time = 0
    cuda_clear_time = 0
    merging_time = 0
    for start in range(1, total_patches + 1, batch_size):
        info = ""
        try:
            batch_creating_timer = ut.timer()
            batch = []
            end = start + batch_size
            if start + batch_size > total_patches:
                end = total_patches + 1
            for p in range(start, end):
                batch.append(patch_list[p][4])
            batch_creating_time += batch_creating_timer.toc()

            torch.cuda.synchronize()
            cpu_to_gpu_timer = ut.timer()
            batch = torch.stack(batch).to(device)
            torch.cuda.synchronize()
            cpu_to_gpu_time += cpu_to_gpu_timer.toc()
            info = (
                info
                + "C2G Starts: "
                + str(cpu_to_gpu_timer.t0)
                + "C2G total: "
                + str(cpu_to_gpu_time)
            )
            # =============================================================================
            #             print(batch.shape)
            #             subprocess.run("gpustat", shell=True)
            # =============================================================================
            with torch.no_grad():
                # =============================================================================
                #                 print(start, end)
                #                 print(sys.getsizeof(batch))
                # =============================================================================
                torch.cuda.synchronize()
                start_time = time.time()
                sr_batch = model(batch)
                torch.cuda.synchronize()
                end_time = time.time()
                processing_time = end_time - start_time
                total_EDSR_time += processing_time
                info = (
                    info
                    + "\tModel Starts: "
                    + str(start_time)
                    + "Model total: "
                    + str(total_EDSR_time)
                )

            torch.cuda.synchronize()
            gpu_to_cpu_timer = ut.timer()
            sr_batch = sr_batch.to("cpu")
            torch.cuda.synchronize()
            gpu_to_cpu_time += gpu_to_cpu_timer.toc()
            info = (
                info
                + "\tGPU 2 CPU Starts: "
                + str(gpu_to_cpu_timer.t0)
                + "G2C total: "
                + str(gpu_to_cpu_time)
            )
            _, _, patch_height, patch_width = sr_batch.size()
            logger.info(info)
            batch_id = 0
            merging_timer = ut.timer()
            for p in range(start, end):
                output_image[
                    :,
                    patch_list[p][3][0] : patch_list[p][3][1],
                    patch_list[p][3][2] : patch_list[p][3][3],
                ] = sr_batch[batch_id][
                    :,
                    patch_list[p][2][0] : patch_list[p][2][1],
                    patch_list[p][2][2] : patch_list[p][2][3],
                ]
                batch_id += 1

            merging_time += merging_timer.toc()
            cuda_clear_timer = ut.timer()
            ut.clear_cuda(batch, None)
            cuda_clear_time += cuda_clear_timer.toc()
        except RuntimeError as err:
            ut.clear_cuda(batch, None)
            raise Exception(err)
    model = model.to("cpu")

    if print_timer:
        print("Total upsampling time: {}\n".format(total_EDSR_time))
        print("Total CPU to GPU shifting time: {}\n".format(cpu_to_gpu_time))
        print("Total GPU to CPU shifting time: {}\n".format(gpu_to_cpu_time))
        print("Total batch creation time: {}\n".format(batch_creating_time))
        print("Total merging time: {}\n".format(merging_time))
        print("Total CUDA clear time: {}\n".format(cuda_clear_time))
        print(
            "Total time: {}\n".format(
                total_EDSR_time
                + cpu_to_gpu_time
                + gpu_to_cpu_time
                + batch_creating_time
                + cuda_clear_time
                + merging_time
            )
        )
    return output_image, (
        total_EDSR_time,
        cpu_to_gpu_time,
        gpu_to_cpu_time,
        batch_creating_time,
        cuda_clear_time,
        merging_time,
    )


def patch_batch_forward_chop(
    input_image,
    patch_dimension,
    patch_shave,
    scale,
    batch_size,
    model_type="EDSR",
    device="cuda",
    print_timer=True,
):
    """


    Parameters
    ----------
    input_image : 3D Matrix
        input image.
    patch_dimension : int
        patch dimension.
    patch_shave : int
        patch shave value.
    scale : int
        scale for LR to SR.
    batch_size : int
        batch size.
    model_type : str, optional
        model name. The default is 'EDSR'.
    device : str, optional
        GPU or CPU. The default is 'cuda'.
    print_timer : bool, optional
        print result or not. The default is True.

    Returns
    -------
    output_image : 3D Matrix
        output SR image.

    """

    model = None

    if model_type == "EDSR":
        model = md.load_edsr(device=device)
        model.eval()
    elif model_type == "RRDB":
        model = md.load_rrdb(device=device)
        model.eval()
    else:
        raise Exception("{} : Unknown model...".format(model_type))
    total_timer = ut.timer()
    channel, height, width = input_image.shape

    patch_list_timer = ut.timer()
    patch_list = {}
    create_patch_list(
        patch_list,
        input_image,
        patch_dimension,
        patch_shave,
        scale,
        channel,
        height,
        width,
    )
    patch_list_processing_time = patch_list_timer.toc()

    total_batch_processing_timer = ut.timer()
    output_image, timer_results = batch_forward_chop(
        patch_list,
        batch_size,
        channel,
        height,
        width,
        patch_dimension,
        patch_shave,
        scale,
        model=model,
        device="cuda",
        print_timer=False,
    )

    total_batch_processing_time = total_batch_processing_timer.toc()
    if model_type == "EDSR":
        output_image = output_image.int()
    total_time = total_timer.toc()
    print(len(patch_list))
    if print_timer:
        print(patch_list_processing_time)
        for t in timer_results:
            print(t)
        print(total_batch_processing_time)
        print(total_time)
    model = model.cpu()
    del model
    return output_image


def upsample(model_name, img_path, dimension, shave, batch_size, scale, device):
    file_name = img_path.split("/")[-1].split(".")[0]
    if model_name == "RRDB":
        input_image = ut.npz_loader(img_path)
        c, h, w = input_image.shape
        patch_list = {}
        create_patch_list(patch_list, input_image, dimension, shave, scale, c, h, w)
        model = md.load_rrdb(device=device)
        model.eval()
        min_dim = min(dimension, h, w)

        if min_dim != dimension:
            print(
                "\nPatch dimension is greater than the input image's minimum dimension. Changing patch dimension to input image's minimum dimension... \n "
            )
            dimension = min_dim
        output, _ = batch_forward_chop(
            patch_list,
            batch_size,
            c,
            h,
            w,
            dimension,
            shave,
            scale,
            model=model,
            device=device,
        )
        output = output.int()
        output_folder = "output_images"
        ut.save_image(
            output, output_folder, h, w, scale, output_file_name=file_name + "_x4"
        )
    elif model_name == "EDSR":
        input_image = ut.load_image(img_path)
        c, h, w = input_image.shape
        patch_list = {}
        create_patch_list(patch_list, input_image, dimension, shave, scale, c, h, w)
        model = md.load_edsr(device=device)
        model.eval()
        min_dim = min(dimension, h, w)

        if min_dim != dimension:
            print(
                "\nPatch dimension is greater than the input image's minimum dimension. Changing patch dimension to input image's minimum dimension... \n "
            )
            dimension = min_dim
        output, _ = batch_forward_chop(
            patch_list,
            batch_size,
            c,
            h,
            w,
            dimension,
            shave,
            scale,
            model=model,
            device=device,
        )
        output = output.int()
        output_folder = "output_images"
        ut.save_image(
            output, output_folder, h, w, scale, output_file_name=file_name + "_x4"
        )
    else:
        print("Unknown model...")


if __name__ == "__main__":
    # Arguments
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/slices/0.npz"
    dimension = int(sys.argv[2]) if len(sys.argv) > 2 else 293
    shave = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print_result = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
    device = str(sys.argv[6]) if len(sys.argv) > 6 else "cuda"

    # Reading image
    # img = torchvision.io.read_image(img_path)
    img = ut.npz_loader(img_path)
    c, h, w = img.shape
    # img = img.reshape((1, c, h, w))
    # plt.imshow(img[0].permute((1,2,0)))
    input_image = img.float()

    # Creating patch list from the image
    patch_list_timer = ut.timer()
    patch_list = {}
    create_patch_list(patch_list, input_image, dimension, shave, 4, c, h, w)
    patch_list_processing_time = patch_list_timer.toc()
    print("Total patch list creating time: {}".format(patch_list_processing_time))
    print(len(patch_list))
    # Loading model
    model = md.load_rrdb(device=device)
    model.eval()

    min_dim = min(dimension, h, w)

    if min_dim != dimension:
        print(
            "\nPatch dimension is greater than the input image's minimum dimension. Changing patch dimension to input image's minimum dimension... \n "
        )
        dimension = min_dim
    # Batch Processing
    batch_processing_start = time.time()
    output, _ = batch_forward_chop(
        patch_list, batch_size, c, h, w, dimension, shave, 4, model=model, device="cuda"
    )
    batch_processing_end = time.time()

    print(
        "Total batch_processing_time: {}".format(
            batch_processing_end - batch_processing_start
        )
    )

    # Saving output image
    output = output.int()
    save_start = time.time()
    fig = plt.figure(figsize=((4 * h) / 1000, (4 * w) / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    # fig = plt.figure(figsize=(4*h, 4*w))
    ax.imshow(output.permute((1, 2, 0)))
    fig.savefig(
        "output_images/" + "4x_" + img_path.split("/")[1],
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
        dpi=1000,
    )
    save_end = time.time()
    save_time = save_end - save_start
