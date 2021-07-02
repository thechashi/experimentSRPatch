"""Binary Search"""
import time
import sys
import math
import copy
import torch
import subprocess
import pyfiglet
import utilities as ut
import modelloader as md
import toml
from tqdm import tqdm
from tqdm import trange
def maximum_unacceptable_dimension_2n(device, logger, model, model_name="EDSR"):
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
    print("\nGetting maximum unacceptable dimension which is a power of two...\n")
    result1 = {}
    last_dimension = 0
    dimension = 2
    last_used_memory = 0
    iteration = 0
    while True:
        iteration += 1
        _, used_memory, _ = ut.get_gpu_details(device, None, logger, print_details=False)
        leaked_memory = used_memory - last_used_memory if used_memory > last_used_memory else 0
        print('Patch Dimension: {:04}x{:04} | Used Memory: {:09.3f} | Leaked Memory: {:09.3f} | Iteration: {}'.format(dimension, dimension, used_memory, leaked_memory, iteration ))
        last_used_memory = used_memory
        #logger.info(f"\nTesting dimension: {dimension}x{dimension} ...")

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
                    ["python3", "binarysearch_helper.py", str(dimension), model_name],
                    stdout=subprocess.PIPE,
                    text=True,
                )
                #print(process_output.stdout)
                if process_output.returncode == 0:
                    out = process_output.stdout.split("\n")
                    total_time = out[0]
                else:
                    raise RuntimeError(process_output.stderr)
                if dimension in result1.keys():
                    result1[dimension].append(total_time)
                else:
                    result1[dimension] = [total_time]

                #logger.info("\nDimension ok!\n")
                dimension *= 2

            except RuntimeError as err:
                if dimension in result1.keys():
                    result1[dimension].append(math.inf)
                else:
                    result1[dimension] = [math.inf]

                last_dimension = dimension

                ut.clear_cuda(None, None)
                break
    return last_dimension


def maximum_acceptable_dimension(device, logger, model, max_unacceptable_dimension, model_name="EDSR"):
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
    print("\nGetting maximum acceptable dimension...\n")
    result2 = {}
    dimension = max_unacceptable_dimension
    maxm = math.inf
    minm = -math.inf
    last = 0
    last_used_memory = 0
    iteration = 0
    while True:
        #logger.info(f"\nTesting dimension: {dimension}x{dimension} ...")
        iteration += 1
        ut.clear_cuda(None, None)
        state = f"\nGPU usage before loading the image {dimension}x{dimension}...\n"
        ut.get_gpu_details(device, state, logger, print_details=True)
        _, used_memory, _ = ut.get_gpu_details(device, None, logger, print_details=False)
        leaked_memory = used_memory - last_used_memory if used_memory > last_used_memory else 0
        print('Patch Dimension: {:04}x{:04} | Used Memory: {:09.3f} | Leaked Memory: {:09.3f} | Iteration: {}'.format(dimension, dimension, used_memory, leaked_memory, iteration ))
        last_used_memory = used_memory
        #logger.info(f"\nTesting dimension: {dimension}x{dimension} ...")
        with torch.no_grad():
            try:
                if last == dimension:
                    break
                process_output = subprocess.run(
                    ["python3", "binarysearch_helper.py", str(dimension), model_name],
                    stdout=subprocess.PIPE,
                    text=True,
                )
                #print(process_output.stdout)
                if process_output.returncode == 0:
                    out = process_output.stdout.split("\n")
                    total_time = out[0]
                else:
                    raise RuntimeError(process_output.stderr)

                last = dimension

                if dimension in result2.keys():
                    result2[dimension].append(total_time)
                else:
                    result2[dimension] = [total_time]

                minm = copy.copy(dimension)

                #logger.info("\nDimension ok!\n")

                if maxm == math.inf:
                    dimension *= 2
                else:
                    dimension = dimension + (maxm - minm) // 2
                ut.clear_cuda(None, None)
            except RuntimeError as err:
                maxm = copy.copy(dimension)

                if dimension in result2.keys():
                    result2[dimension].append(math.inf)
                else:
                    result2[dimension] = [math.inf]
                if minm == -math.inf:
                    dimension = dimension // 2
                else:
                    dimension = minm + (maxm - minm) // 2
                ut.clear_cuda(None, None)

                continue
    return last


def do_binary_search(model_name="EDSR"):
    """
    Binary search function...

    Returns
    -------
    None.

    """
    banner = pyfiglet.figlet_format("Binary Search: " + model_name)
    print(banner)
    
    logger = ut.get_logger()
    device = ut.get_device_type()
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
    #logger.info(log_message)

    ut.clear_cuda(None, None)

    state = "Before loading model: "
    total, used, _ = ut.get_gpu_details(device, state, logger, print_details=True)
    
    if model_name not in ["EDSR", "RRDB"]:
        raise Exception('Unknown model...')

    state = "After loading model: "
    total, used, _ = ut.get_gpu_details(device, state, logger, print_details=True)

    # get the highest unacceptable dimension which is a power of 2
    max_unacceptable_dimension = maximum_unacceptable_dimension_2n(
        device, logger, None, model_name=model_name
    )
    print("\nMaximum unacceptable dimension: {}\n".format(max_unacceptable_dimension))
    ut.clear_cuda(None, None)

    # get the maximum acceptable dimension
    max_dim = maximum_acceptable_dimension(
        device, logger, None, max_unacceptable_dimension, model_name=model_name
    )
    print("\nMaximum acceptable dimension: {}\n".format(max_dim))
    ut.clear_cuda(None, None)
    # For batch processing
    config = toml.load("../batch_processing.toml")
    config["end_patch_dimension"] = max_dim
    f = open("../batch_processing.toml", "w")
    toml.dump(config, f)
    # for linear search
    config = toml.load("../config.toml")
    config["max_dim"] = max_dim
    f = open("../config.toml", "w")
    toml.dump(config, f)
# =============================================================================
#     file = open("temp_max_dim.txt", "w")
#     file.write("max_dim:" + str(max_dim))
#     file.close()
# =============================================================================


if __name__ == "__main__":
    config = toml.load('../batch_processing.toml')
    do_binary_search(config["model"])
