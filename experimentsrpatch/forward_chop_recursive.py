"""
Recursive forward chop
"""
import torch
import click
import numpy as np
import sys
import modelloader as md
import utilities as ut


def forward_chop(
    input_image, model, timer, shave=10, scale=4, n_GPUs=1, min_size=160000
):
    """
    Recursive forward chop

    Parameters
    ----------
    input_image : str
        image path.
    model : str
        model.
    shave : int, optional
        overlapping value. The default is 10.
    scale : int, optional
        LR to HR scale. The default is 4.
    n_GPUs : int, optional
        number of GPUs. The default is 1.
    min_size : int, optional
        patch size. The default is 160000.

    Returns
    -------
    output : tensor
        4x output.

    """
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = input_image.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        input_image[:, :, 0:h_size, 0:w_size],
        input_image[:, :, 0:h_size, (w - w_size) : w],
        input_image[:, :, (h - h_size) : h, 0:w_size],
        input_image[:, :, (h - h_size) : h, (w - w_size) : w],
    ]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            model.eval()
            with torch.no_grad():
                lr_batch = torch.cat(lr_list[i : (i + n_GPUs)], dim=0)
                upsampling_time = ut.timer()
                sr_batch = model(lr_batch)
                upsampling_time = upsampling_time.toc()
                timer[1] += upsampling_time
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, model, timer, shave=shave, min_size=min_size)
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = input_image.new(b, c, h, w)
    merging_time = ut.timer()
    output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = sr_list[1][
        :, :, 0:h_half, (w_size - w + w_half) : w_size
    ]
    output[:, :, h_half:h, 0:w_half] = sr_list[2][
        :, :, (h_size - h + h_half) : h_size, 0:w_half
    ]
    output[:, :, h_half:h, w_half:w] = sr_list[3][
        :, :, (h_size - h + h_half) : h_size, (w_size - w + w_half) : w_size
    ]
    merging_time = merging_time.toc()
    timer[2] += merging_time
    return output


@click.command()
@click.option("--img_path", default="data/slices/12.npz", help="Path of the image")
@click.option("--model_name", default="RRDB", help="Name of the model")
@click.option("--patch_dimension", default=293, help="Patch dimension")
def main(img_path, model_name, patch_dimension):
    """
    Driver for recursive forward chop. Takes an image path and model name to upsample the image.
    """

    # Loading model and image
    img = None
    model = None
    print("\nLoading model and image... \n")
    if model_name in ["EDSR"]:
        img = ut.load_image(img_path)
        img = img.unsqueeze(0)
        model = md.load_edsr(device="cuda")
    else:
        img = ut.npz_loader(img_path)
        img = img.unsqueeze(0)
        model = md.load_rrdb(device="cuda")

    # Timers for saving stats
    timer = [0, 0, 0, 0, 0]

    print("Processing...")

    # Shfiting input image to CUDA
    total_time = ut.timer()
    cpu2gpu_time = ut.timer()
    img = img.to("cuda")
    cpu2gpu_time = cpu2gpu_time.toc()

    # Forward chopping and upsampling
    output = forward_chop(
        img, model, timer=timer, min_size=patch_dimension * patch_dimension
    )

    # Shifting output image to CPU
    gpu2cpu_time = ut.timer()
    output = output.to("cpu")
    gpu2cpu_time = gpu2cpu_time.toc()
    if model_name in ["EDSR"]:
        output = output.int()
    total_time = total_time.toc()

    timer[0] = cpu2gpu_time
    timer[-2] = gpu2cpu_time
    timer[-1] = total_time

    # Saving output
    np.savez("results/recursive_outputx4.npz", output)
    np.save("results/recursive_outputx4", output)

    # Printing processing times
    print("\nCPU 2 GPU time: ", timer[0])
    print("\nUpsampling time: ", timer[1])
    print("\nMerging time: ", timer[2])
    print("\nGPU 2 CPU time", timer[3])
    print("\nTotal execution time: ", timer[4])


def helper_rrdb_experiment(img_dimension, patch_dimension):
    # Loading model and image
    img = None
    model = None
    # =============================================================================
    #     print("\nLoading model and image... \n")
    # =============================================================================
    img = np.load("data/slices/0.npz")
    img = img.f.arr_0
    img = np.resize(img, (img_dimension, img_dimension))
    img = img[np.newaxis, :, :]
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    model = md.load_rrdb(device="cuda")

    # Timers for saving stats
    timer = [0, 0, 0, 0, 0]

    # =============================================================================
    #     print("Processing...")
    # =============================================================================

    # Shfiting input image to CUDA
    total_time = ut.timer()
    cpu2gpu_time = ut.timer()
    img = img.to("cuda")
    cpu2gpu_time = cpu2gpu_time.toc()

    # Forward chopping and upsampling
    output = forward_chop(
        img, model, timer=timer, min_size=patch_dimension * patch_dimension
    )

    # Shifting output image to CPU
    gpu2cpu_time = ut.timer()
    output = output.to("cpu")
    gpu2cpu_time = gpu2cpu_time.toc()

    total_time = total_time.toc()

    timer[0] = cpu2gpu_time
    timer[-2] = gpu2cpu_time
    timer[-1] = total_time

    # Printing processing times
    # =============================================================================
    #     print("\nCPU 2 GPU time: ", timer[0])
    #     print("\nUpsampling time: ", timer[1])
    #     print("\nMerging time: ", timer[2])
    #     print("\nGPU 2 CPU time", timer[3])
    #     print("\nTotal execution time: ", timer[4])
    # =============================================================================
    print(timer[1])
    print(timer[4])


if __name__ == "__main__":
    main()
    # helper_rrdb_experiment(int(sys.argv[1]), int(sys.argv[2]))
