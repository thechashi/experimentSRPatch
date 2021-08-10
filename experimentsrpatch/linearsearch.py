"""Linear Search"""
import os
import time
import subprocess
import torch
import toml
from tqdm import tqdm
import utilities as ut
import modelloader as md


def result_from_dimension_range(device, logger, config, model, first, last):
    """
    Get detailed result for every dimension from 1 to the last acceptable dimension

    Parameters
    ----------
    device : str
        device type.
    model : torch.nn.model
        SR model.
    first : int
        starting dimension.
    last : int
        last acceptable dimension.
    run : int, optional
        total run to average the result. The default is 10.

    Returns
    -------
    result3 : dictionary
        time for every dimension.
    memory_used : dictionary
        memory used per dimension.
    memory_free : dictionary
        memory free per dimension.

    """
    run = config["run"]
    print("\nPreparing detailed data... ")
    result3 = {}
    memory_used = {}
    memory_free = {}
    for i in range(run):
        print("\nRun: ", i + 1)
        print()
        for dim in tqdm(range(first, last + 1)):
            dimension = dim
            input_image = ut.random_image(dimension)
            input_image = input_image.to(device)
            with torch.no_grad():
                try:
                    print("\n")
                    print(input_image.shape)
                    print(input_image[0, 0, 0, 0:5])
                    start = time.time()
                    output_image = model(input_image)
                    end = time.time()
                    total_time = end - start
                    print("Processing time: ", total_time)
                    print("\n")
                    if dimension in result3.keys():
                        result3[dimension].append(total_time)
                        _, used, free = ut.get_gpu_details(
                            device, "", None, print_details=False
                        )
                        memory_used[dimension].append(used)
                        memory_free[dimension].append(free)
                    else:
                        result3[dimension] = [total_time]
                        _, used, free = ut.get_gpu_details(
                            device, "", None, print_details=False
                        )
                        memory_used[dimension] = [used]
                        memory_free[dimension] = [free]
                    ut.clear_cuda(input_image, output_image)
                except RuntimeError as err:
                    logger.exception("\nDimension NOT OK!")

                    state = "\nGPU usage after dimension exception...\n"
                    ut.get_gpu_details(device, state, logger, print_details=True)

                    output_image = None
                    ut.clear_cuda(input_image, output_image)

                    state = f"\nGPU usage after clearing the image {dimension}x{dimension}...\n"
                    ut.get_gpu_details(device, state, logger, print_details=True)
                    break
        ut.clear_cuda(None, None)
        subprocess.run("gpustat", shell=True)
    return result3, memory_used, memory_free


def do_linear_search(test=False, test_dim=32):
    """
    Linear search function...

    Returns
    -------
    None.

    """
    logger = ut.get_logger()

    device = "cuda"
    model_name = "EDSR"
    config = toml.load("../config.toml")
    run = config["run"]
    scale = int(config["scale"]) if config["scale"] else 4
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

    # =============================================================================
    #     file = open("temp_max_dim.txt", "r")
    #     line = file.read()
    #     max_dim = int(line.split(":")[1])
    # =============================================================================
    config = toml.load("../config.toml")
    max_dim = int(config["max_dim"])
    if test == False:
        detailed_result, memory_used, memory_free = result_from_dimension_range(
            device, logger, config, model, 1, max_dim
        )
    else:
        detailed_result, memory_used, memory_free = result_from_dimension_range(
            device, logger, config, model, test_dim, test_dim
        )
    if test == False:
        # get mean
        # get std
        mean_time, std_time = ut.get_mean_std(detailed_result)
        mean_memory_used, std_memory_used = ut.get_mean_std(memory_used)
        mean_memory_free, std_memory_free = ut.get_mean_std(memory_free)

        # make folder for saving results
        plt_title = "Model: {} | GPU: {} | Memory: {} MB".format(
            model_name, device_name, total
        )
        date = "_".join(str(time.ctime()).split())
        date = "_".join(date.split(":"))
        foldername = date
        os.mkdir("results/" + foldername)
        # plot data
        ut.plot_data(
            foldername,
            "dimension_vs_meantime",
            mean_time,
            "Dimensionn of Patch(nxn)",
            "Mean Processing Time: LR -> SR, Scale: {} ( {} runs )".format(scale, run),
            mode="mean time",
            title=plt_title,
        )
        ut.plot_data(
            foldername,
            "dimension_vs_stdtime",
            std_time,
            "Dimension n of Patch(nxn)",
            "Std of Processing Time: LR -> SR, Scale: {} ( {} runs )".format(
                scale, run
            ),
            mode="std time",
            title=plt_title,
        )
        ut.plot_data(
            foldername,
            "dimension_vs_meanmemoryused",
            mean_memory_used,
            "Dimension n of Patch(nxn)",
            "Mean Memory used: LR -> SR, Scale: {} ( {} runs )".format(scale, run),
            mode="mean memory used",
            title=plt_title,
        )
        ut.plot_data(
            foldername,
            "dimension_vs_stdmemoryused",
            std_memory_used,
            "Dimension n of Patch(nxn)",
            "Std Memory Used: LR -> SR, Scale: {} ( {} runs )".format(scale, run),
            mode="std memory used",
            title=plt_title,
        )
        ut.plot_data(
            foldername,
            "dimension_vs_meanmemoryfree",
            mean_memory_free,
            "Dimension n of Patch(nxn)",
            "Mean Memory Free: LR -> SR, Scale: {} ( {} runs )".format(scale, run),
            mode="mean memory free",
            title=plt_title,
        )
        ut.plot_data(
            foldername,
            "dimension_vs_stdmemoryfree",
            std_memory_free,
            "Dimension n of Patch(nxn)",
            "Std Memory Free: LR -> SR, Scale: {} ( {} runs )".format(scale, run),
            mode="std memory free",
            title=plt_title,
        )
        # save data
        ut.save_csv(
            foldername,
            "total_stat",
            device,
            device_name,
            total,
            mean_time,
            std_time,
            mean_memory_used,
            std_memory_used,
            mean_memory_free,
            std_memory_free,
        )


if __name__ == "__main__":
    do_linear_search(test=True, test_dim=32)
