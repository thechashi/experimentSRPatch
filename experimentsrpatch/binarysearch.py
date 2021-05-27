"""Binary Search"""
import time
import math
import copy
import torch
import subprocess
import utilities as ut
import modelloader as md


def maximum_unacceptable_dimension_2n(device, logger, model):
    """
    Ge the maximum unacceptable dimension which is apower of 2

    Parameters
    ----------
    device : str
        device type.
    model : torch.nn.model
        SR model.

    Returns
    -------
    last_dimension : int
        unacceptabel dimension.

    """
    print()
    print("Getting maximum unacceptable dimension which is a power of two...")
    result1 = {}
    last_dimension = 0
    dimension = 2
    while True:
        logger.info(f"\nTesting dimension: {dimension}x{dimension} ...")

        state = f"\nGPU usage before loading the image {dimension}x{dimension}...\n"
        ut.get_gpu_details(device, state, logger, print_details=True)
# =============================================================================
# 
#         input_image = ut.random_image(dimension)
#         input_image = input_image.to(device)
# =============================================================================

        with torch.no_grad():
            try:
# =============================================================================
#                 start = time.time()
#                 output_image = model(input_image)
#                 end = time.time()
# =============================================================================
                process_output = subprocess.run(
                    ['python3', 'binarysearch_helper.py',
                     str(dimension)],
                    stdout=subprocess.PIPE,
                    text=True)
                print(process_output.stdout)
                if process_output.returncode == 0:
                    out = process_output.stdout.split('\n')
                    total_time  = out[0]
                else:
                    raise RuntimeError(process_output.stderr)
                if dimension in result1.keys():
                    result1[dimension].append(total_time)
                else:
                    result1[dimension] = [total_time]

                logger.info("\nDimension ok!\n")
                dimension *= 2
                
            except RuntimeError as err:
                logger.error("\nDimension NOT OK!\n")

                state = "\nGPU usage after dimension exception...\n"
                ut.get_gpu_details(device, state, logger, print_details=True)

                if dimension in result1.keys():
                    result1[dimension].append(math.inf)
                else:
                    result1[dimension] = [math.inf]

                last_dimension = dimension

                ut.clear_cuda(None, None)

                state = (
                    f"\nGPU usage after clearing the image {dimension}x{dimension}...\n"
                )
                ut.get_gpu_details(device, state, logger, print_details=True)
                break
    return last_dimension


def maximum_acceptable_dimension(device, logger, model, max_unacceptable_dimension):
    """
    Get amximum acceptable dimension

    Parameters
    ----------
    device : str
        device type.
    model : torch.nn.model
        SR model.
    max_unacceptable_dimension : int
        Maximum unacceptable dimension which is apower of 2.

    Returns
    -------
    last : int
        acceptable dimension.

    """
    print()
    print("Getting maximum acceptable dimension...")
    result2 = {}
    dimension = max_unacceptable_dimension
    maxm = math.inf
    minm = -math.inf
    last = 0
    while True:
        logger.info(f"\nTesting dimension: {dimension}x{dimension} ...")
        ut.clear_cuda(None, None)
        state = f"\nGPU usage before loading the image {dimension}x{dimension}...\n"
        ut.get_gpu_details(device, state, logger, print_details=True)

        with torch.no_grad():
            try:
                if last == dimension:
                    break
                process_output = subprocess.run(
                    ['python3', 'binarysearch_helper.py',
                     str(dimension)],
                    stdout=subprocess.PIPE,
                    text=True)
                print(process_output.stdout)
                if process_output.returncode == 0:
                    out = process_output.stdout.split('\n')
                    total_time  = out[0]
                else:
                    raise RuntimeError(process_output.stderr)

                last = dimension

                if dimension in result2.keys():
                    result2[dimension].append(total_time)
                else:
                    result2[dimension] = [total_time]

                minm = copy.copy(dimension)

                logger.info("\nDimension ok!\n")
                
                if maxm == math.inf:
                    dimension *= 2
                else:
                    dimension = dimension + (maxm - minm) // 2
                ut.clear_cuda(None, None)
            except RuntimeError as err:
                logger.error("\nDimension NOT OK!\n")

                state = "\nGPU usage after dimension exception...\n"
                ut.get_gpu_details(device, state, logger, print_details=True)

                maxm = copy.copy(dimension)

                if dimension in result2.keys():
                    result2[dimension].append(math.inf)
                else:
                    result2[dimension] = [math.inf]
                state = (
                    f"\nGPU usage after clearing the image {dimension}x{dimension}...\n"
                )
                ut.get_gpu_details(device, state, logger, print_details=True)
                if minm == -math.inf:
                    dimension = dimension // 2
                else:
                    dimension = minm + (maxm - minm) // 2
                ut.clear_cuda(None, None)

                continue
    return last


def do_binary_search():
    """
    Binary search function...

    Returns
    -------
    None.

    """
    logger = ut.get_logger()

    device = "cuda"

    # device information
    _, device_name = ut.get_device_details()
    total, _, _ = ut.get_gpu_details(
        device, "\nDevice info:", logger, print_details=False
    )
    log_message = (
        "\nDevice: "
        + device
        + "\tDevice name: "
        + device_name
        + "\tTotal memory: "
        + str(total)
    )
    logger.info(log_message)

    ut.clear_cuda(None, None)

    state = "Before loading model: "
    total, used, _ = ut.get_gpu_details(device, state, logger, print_details=True)

    model = md.load_edsr(device=device)

    state = "After loading model: "
    total, used, _ = ut.get_gpu_details(device, state, logger, print_details=True)

    # get the highest unacceptable dimension which is a power of 2
    max_unacceptable_dimension = maximum_unacceptable_dimension_2n(
        device, logger, model
    )
    print('\nMaximum unacceptable dimension: {}\n'.format(max_unacceptable_dimension))
    ut.clear_cuda(None, None)

    # get the maximum acceptable dimension
    max_dim = maximum_acceptable_dimension(
        device, logger, model, max_unacceptable_dimension
    )
    print('\nMaximum acceptable dimension: {}\n'.format(max_dim))
    file = open("temp_max_dim.txt", "w")
    file.write("max_dim:" + str(max_dim))
    file.close()


if __name__ == "__main__":
    do_binary_search()
