"""
Binary Search
-------------------
Search teh maximum acceptable patch dimension for a given model
"""
import sys
import math
import copy
import subprocess
import pyfiglet
import utilities as ut

# =============================================================================
# import experimentsrpatch.utilities as ut
# from experimentsrpatch import utilities as ut
# =============================================================================
import toml


def maximum_unacceptable_dimension_2n(device, logger, start_dim=2, model_name="EDSR"):
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
    print("\nGetting maximum unacceptable dimension which is a power of two...\n")
    result1 = {}
    last_dimension = 0
    dimension = start_dim
    last_used_memory = 0
    iteration = 0
    while True:
        # Prinitng loop status
        iteration += 1
        _, used_memory, _ = ut.get_gpu_details(
            device, None, logger, print_details=False
        )
        leaked_memory = (
            used_memory - last_used_memory if used_memory > last_used_memory else 0
        )
        print(
            "Patch Dimension: {:04}x{:04} | Used Memory: {:09.3f} | Leaked Memory: {:09.3f} | Iteration: {}".format(
                dimension, dimension, used_memory, leaked_memory, iteration
            )
        )
        last_used_memory = used_memory

        # Calling SR model for different dimension
        process_output = subprocess.run(
            ["python3", "binarysearch_helper.py", str(dimension), model_name],
            stdout=subprocess.PIPE,
            text=True,
        )
        if process_output.returncode == 0:
            out = process_output.stdout.split("\n")
            total_time = out[0]
            if dimension in result1.keys():
                result1[dimension].append(total_time)
            else:
                result1[dimension] = [total_time]
            dimension *= 2
        else:
            ut.get_gpu_details(
                device,
                "Runtime error for dimension: {}x{}".format(dimension, dimension),
                logger,
            )
            if dimension in result1.keys():
                result1[dimension].append(math.inf)
            else:
                result1[dimension] = [math.inf]

            last_dimension = dimension

            ut.clear_cuda(None, None)
            break
    return last_dimension


def maximum_acceptable_dimension(
    device, logger, model, max_unacceptable_dimension, model_name="EDSR"
):
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
    print("\nGetting maximum acceptable dimension...\n")
    result2 = {}
    dimension = max_unacceptable_dimension
    maxm = math.inf
    minm = -math.inf
    last = 0
    last_used_memory = 0
    iteration = 0
    while True:
        # Printing iterations status
        iteration += 1
        _, used_memory, _ = ut.get_gpu_details(
            device, None, logger, print_details=False
        )
        leaked_memory = (
            used_memory - last_used_memory if used_memory > last_used_memory else 0
        )
        print(
            "Patch Dimension: {:04}x{:04} | Used Memory: {:09.3f} | Leaked Memory: {:09.3f} | Iteration: {}".format(
                dimension, dimension, used_memory, leaked_memory, iteration
            )
        )
        last_used_memory = used_memory

        # Clearing cuda cache:
        ut.clear_cuda(None, None)

        # Binary Search
        if last == dimension:
            break
        process_output = subprocess.run(
            ["python3", "binarysearch_helper.py", str(dimension), model_name],
            stdout=subprocess.PIPE,
            text=True,
        )
        if process_output.returncode == 0:
            out = process_output.stdout.split("\n")
            total_time = out[0]
            last = dimension
            if dimension in result2.keys():
                result2[dimension].append(total_time)
            else:
                result2[dimension] = [total_time]
            minm = copy.copy(dimension)
            if maxm == math.inf:
                dimension *= 2
            else:
                dimension = dimension + (maxm - minm) // 2
            ut.clear_cuda(None, None)
        else:
            ut.get_gpu_details(
                device,
                "Runtime error for dimension: {}x{}".format(dimension, dimension),
                logger,
            )
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
    return last


def do_binary_search(model_name="EDSR", start_dim=2):
    """
    Binary search function...

    Returns
    -------
    None.

    """
    # Prints the header banner
    banner = pyfiglet.figlet_format("Binary Search: " + model_name)
    print(banner)

    # Getting logger
    logger = ut.get_logger()

    # Check valid model or not
    if model_name not in ["EDSR", "RRDB"]:
        logger.exception("{} model is unkknown".format(model_name))
        raise Exception("Unknown model...")

    # Device type cpu or cuda
    device = ut.get_device_type()

    if device == "cpu" and model_name not in ["EDSR"]:
        logger.exception("{} model cannot be run in CPU".format(model_name))
        raise Exception("{} model cannot be run in CPU".format(model_name))

    # Device information
    _, device_name = ut.get_device_details()

    if device == "cuda":
        logger.info("Device: {}, Device Name: {}".format(device, device_name))
        ut.get_gpu_details(
            device,
            "Before binary search: {}".format(model_name),
            logger,
            print_details=True,
        )
    else:
        logger.info("Device: {}, Device Name: {}".format(device, device_name))

    # Clearing cuda cache
    ut.clear_cuda(None, None)

    # Getting the highest unacceptable dimension which is a power of 2
    max_unacceptable_dimension = maximum_unacceptable_dimension_2n(
        device, logger, start_dim=start_dim, model_name=model_name
    )
    print("\nMaximum unacceptable dimension: {}\n".format(max_unacceptable_dimension))

    # Clearing cuda cache
    ut.clear_cuda(None, None)

    # Getting the maximum acceptable dimension
    max_dim = maximum_acceptable_dimension(
        device, logger, None, max_unacceptable_dimension, model_name=model_name
    )
    print("\nMaximum acceptable dimension: {}\n".format(max_dim))

    # Clearing cuda cache
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


if __name__ == "__main__":
    # Exception handling
    sys.excepthook = ut.exception_handler

    config = toml.load("../batch_processing.toml")
    do_binary_search(config["model"])
