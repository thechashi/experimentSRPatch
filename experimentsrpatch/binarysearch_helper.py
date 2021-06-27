"""
Binary Search Helper
"""
import sys
import torch
import time
import utilities as ut
import modelloader as md


def binary_search_helper(dimension, logger, device="cuda"):
    """
    Process random image and calculates processing time

    Parameters
    ----------
    dimension : int
        random image dimension.
    logger : logger
        keep logs.
    device : str, optional
        GPU or CPU. The default is 'cuda'.

    Returns
    -------
    total_time : float
        EDSR processing time.

    """
    model = md.load_edsr(device=device, model_details=False)
    input_image = ut.random_image(dimension)
    input_image = input_image.to(device)

    with torch.no_grad():
        start = time.time()
        output_image = model(input_image)
        state = f"\nGPU usage after loading the image {dimension}x{dimension}...\n"
        end = time.time()
        ut.get_gpu_details(device, state, logger, print_details=True)
        total_time = end - start
        ut.clear_cuda(input_image, output_image)

        state = f"\nGPU usage after clearing the image {dimension}x{dimension}...\n"
        ut.get_gpu_details(device, state, logger, print_details=True)
    return total_time


if __name__ == "__main__":
    logger = ut.get_logger()
    print(binary_search_helper(int(sys.argv[1]), logger))
