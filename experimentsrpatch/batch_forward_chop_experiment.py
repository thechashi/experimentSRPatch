"""
This is the driver source code for running the batch patch experiment
"""
import sys
import os
import subprocess
import numpy as np
import pandas as pd
import time
import toml
from tqdm import tqdm
import matplotlib.pyplot as plt
import utilities as ut


np.set_printoptions(suppress=True)


def batch_range_checker(
    max_dim,
    min_dim,
    patch_shave,
    scale,
    img_path,
    logger,
    dim_gap=1,
    batch_start=1,
    device="cuda",
):
    """
    Checks maximum valid batch size for every patch dimension from min_dim to amx_dim

    Parameters
    ----------
    max_dim : int
        biggest patch dimension test.
    min_dim : int
        smallest patch dimension to test.
    patch_shave : int
        patch overlapping value.
    scale : int
        scaling value for LR to SR.
    img_path : str
        image path.
    logger : logger object
        keeps log.
    batch_start : int, optional
        smallest batch value to start from. The default is 1.
    device : str, optional
        GPU or CPU. The default is "cuda".

    Raises
    ------
    Exception
        batch size miss-match exception.

    Returns
    -------
    full_result : list of list
        stats of the experiment.

    """
    full_result = []
    for d in tqdm(range(max_dim, min_dim - 1, -dim_gap)):
        # print('\n dimension: {}, batch_start: {}\n'.format(d, batch_start))
        ut.get_gpu_details(
            device, state="GPU stat before dimension: {}".format(d), logger=logger
        )
        result = [d]
        while True:

            command = (
                "python3 "
                + "helper_batch_patch_forward_chop.py "
                + img_path
                + " "
                + str(d)
                + " "
                + str(patch_shave)
                + " "
                + str(batch_start)
                + " "
                + str(scale)
                + " "
                + str(0)
                + " "
                + device
            )
            p = subprocess.run(command, shell=True, capture_output=True)
            if p.returncode == 0:
                # logger.info('OK! Dimension: {}, Batch size : {}'.format(d, batch_start))
                batch_start += 1
            else:
                # raise Exception(p.stderr.decode())
                logger.info(
                    "Error: Dimension: {}, Batch size : {}".format(d, batch_start)
                )
                ut.get_gpu_details(
                    device, state="GPU stat after batch size error:", logger=logger
                )
                print("Error: Dimension: {}, Batch size : {}".format(d, batch_start))
                print(p.stderr.decode())
                break
        result += [batch_start - 1]

        full_result.append(result)
    return full_result


def single_patch_highest_batch_checker(
    patch_dim,
    patch_shave,
    scale,
    img_path,
    logger,
    run=3,
    batch_size_start=1,
    device="cuda",
):
    """
    Calculates timing for every possible batch size for a specific patch dimension and shave value

    Parameters
    ----------
    patch_dim : int
        dimension of patch.
    patch_shave : int
        shave value of patch.
    scale : int
        scale for LR to SR.
    img_path : str
        image path.
    run : int, optional
        total number of run for averaging values. The default is 3.
    batch_size : int, optional
        starting batch size. The default is 10.
    device : str, optional
        GPU or CPU. The default is 'cuda'.

    Returns
    -------
    full_result : list of list
        stats of the experiment.

    """
    exception = False
    full_result = []
    print("Processing...\n")
    while exception == False:
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print("\nBatch size: {}\n".format(batch_size_start))
        for r in tqdm(range(run)):
            temp = [batch_size_start]
            command = (
                "python3 "
                + "helper_batch_patch_forward_chop.py "
                + img_path
                + " "
                + str(patch_dim)
                + " "
                + str(patch_shave)
                + " "
                + str(batch_size_start)
                + " "
                + str(scale)
                + " "
                + str(0)
                + " "
                + device
            )

            p = subprocess.run(command, shell=True, capture_output=True)
            if p.returncode == 0:
                print(p.stdout.decode())
                temp += list(map(float, p.stdout.decode().split("\n")[1:10]))
                result = [result[i] + temp[i] for i in range(len(temp))]
            else:
                # raise Exception(p.stderr.decode())
                logger.info(
                    "Error: Dimension - {}, Batch size - {}".format(
                        patch_dim, batch_size_start
                    )
                )
                ut.get_gpu_details(
                    device, state="GPU stat after batch size error:", logger=logger
                )
                print(p.stderr.decode())
                exception = True
                break
        if exception == False:
            result = np.array(result) / run
            full_result.append(result)
            batch_size_start += 1
    print("Process finished!\n")
    return full_result


def plot_stat(x_data, y_data, x_label, y_label, plt_title, png_prefix, date):
    """
    Plots statistics of the single patch all possible batch experiment

    Parameters
    ----------
    x_data : numpy array
        data for x axis.
    y_data : numpy array
        data for y axis.
    x_label : str
        label for x axis.
    y_label : str
        label for y axis.
    plt_title : str
        plot title.
    png_prefix : str
        image name.
    date : str
        image name suffix.

    Returns
    -------
    None.

    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data)
    plt.savefig(
        "results/batch_forward_chop_experiment/{0}/{1}.png".format(date, png_prefix + "_" + date)
    )
    plt.show()


def process_stats_of_single_patch_batch(
    full_result, img_height, img_width, patch_dim, patch_shave, scale, model, device, date
):
    """
    Plots and saves data as csv from the single patch every possible batch experiment

    Parameters
    ----------
    full_result : panda dataframe
        result.
    img_height : int
        image height.
    img_width : int
        image width.
    patch_dim : int
        patch dimension.
    patch_shave : int
        patch shave.
    scale : int
        scale value for LR to SR.
    model : str
        EDSR.
    device : str
        GPU or CPU.

    Returns
    -------
    None.

    """
    model_name = model
    device_name = "CPU"
    total_memory = "~"
    device = device
    if device == "cuda":
        _, device_name = ut.get_device_details()
        total_memory, _, _ = ut.get_gpu_details(
            device, "\nDevice info:", logger=None, print_details=False
        )
    if device == "cuda":
        plt_title = "Model: {} | GPU: {} | Memory: {} MB".format(
            model_name, device_name, total_memory
        )
    else:
        plt_title = "Model: {} | Device: {}".format(model_name, "CPU")

# =============================================================================
#     date = "_".join(str(time.ctime()).split())
#     date = "_".join(date.split(":"))
# =============================================================================

    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 9].values).flatten(),
    )

    x_label = "Batch size (Image: {}x{}, Patch: {}x{}, Shave: {}, Scale: {})".format(
        img_height, img_width, patch_dim, patch_dim, patch_shave, scale
    )
    y_label = "Total processing time (sec): LR -> SR"

    plot_stat(
        x_data, y_data, x_label, y_label, plt_title, "total_processing_time", date
    )

    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 1].values).flatten(),
    )
    y_label = "Total patch list creation time (sec): LR -> SR"

    plot_stat(
        x_data,
        y_data,
        x_label,
        y_label,
        plt_title,
        "total_patch_list_creation_time",
        date,
    )

    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 3].values).flatten(),
    )
    y_label = "Total CPU to GPU shifting time (sec): LR -> SR"

    plot_stat(x_data, y_data, x_label, y_label, plt_title, "total_CPU_2_GPU_time", date)

    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 2].values).flatten(),
    )
    y_label = "Total EDSR processing time (sec): LR -> SR"

    plot_stat(
        x_data, y_data, x_label, y_label, plt_title, "total_edsr_processing_time", date
    )

    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 8].values).flatten(),
    )
    y_label = "Total batch processing time (sec): LR -> SR"

    plot_stat(
        x_data, y_data, x_label, y_label, plt_title, "total_batch_processing_time", date
    )

    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 4].values).flatten(),
    )
    y_label = "Total GPU to CPU shifting time (sec): LR -> SR"

    plot_stat(x_data, y_data, x_label, y_label, plt_title, "total_GPU_2_CPU_time", date)


def save_stat_csv(
    full_result, img_height, img_width, patch_dim, patch_shave, scale, model, device, file_name, folder_name, date
):
    """
    Saves data as csv

    Parameters
    ----------
    full_result : panda dataframe
        result.
    img_height : int
        image height.
    img_width : int
        image width.
    patch_dim : int
        patch dimension.
    patch_shave : int
        patch shave value.
    scale : int
        scale for LR to SR.
    model : str
        EDSR.
    device : str
        GPU or CPU.

    Returns
    -------
    None.

    """
    model_name = model
    device_name = "CPU"
    total_memory = "~"
    device = device
    _, device_name = ut.get_device_details()
    total_memory, _, _ = ut.get_gpu_details(
        device, "\nDevice info:", logger, print_details=False
    )
# =============================================================================
#     date = "_".join(str(time.ctime()).split())
#     date = "_".join(date.split(":"))
# =============================================================================
    filename = "stat_" + "single_patch_all_batch_" + date
# =============================================================================
#     folder_name = "results/" + "batch_forward_chop_experiment" + "/" + date
#     os.mkdir(folder_name)
# =============================================================================
    file = open(folder_name + "/" + file_name, "a")
    full_result.to_csv(file)
    file.close()
    meta_data_path = folder_name + "/" + "gpustat.json"
    gpu_command = "gpustat --json > " + meta_data_path
    subprocess.run(gpu_command, shell=True)
# =============================================================================
#     file.write('#' + model_name + "\n")
#     file.write('#' + device + "\n")
#     file.write('#' + device_name + "\n")
#     file.write("Memory: " + str(total_memory) + "MB\n")
#     process_details = "Image: {}x{}, Patch: {}x{}, Shave: {}, Scale: {}".format(
#         img_height, img_width, patch_dim, patch_dim, patch_shave, scale
#     )
#     file.write(str(process_details) + "\n")
# =============================================================================
# =============================================================================
#     meta_data = {}
#     meta_data["model_name"] = model_name
#     meta_data["device"] = device
#     meta_data["device_name"] = device_name
#     meta_data["total_memory"] = total_memory
#     meta_data["image_height"] = str(img_height)
#     meta_data["image_width"] = str(img_width)
#     meta_data["patch_dimension"] = str(patch_dim)
#     meta_data["shave"] = str(patch_shave)
#     meta_data["scale"] = str(scale)
#     output_file_name ="results/" + "batch_forward_chop_experiment" + "/" + date + "/" +  "metadata.toml"
#     with open(output_file_name, "w") as toml_file:
#         toml.dump(meta_data, toml_file)
# =============================================================================



if __name__ == "__main__":
    logger = ut.get_logger()
    process = sys.argv[1] if len(sys.argv) > 1 else "batch_range"
    
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    folder_name = "results/" + "batch_forward_chop_experiment" + "/" + date
    os.mkdir(folder_name)
    
    if process == "batch_range":
        config = toml.load("../batch_processing.toml")
        max_dim = int(config["end_patch_dimension"])
        min_dim = int(config["start_patch_dimension"])
        shave = int(config["shave"])
        scale = int(config["scale"])
        img_path = config["img_path"]
        batch_size_start = int(config["single_patch_batch_size_start"])
        dim_gap = int(config["dim_gap"])
        model_name = config["model"]
        device = config["device"]
        
        img = ut.load_image(img_path)
        c, h, w = img.shape
        
        if max_dim > h or max_dim > w:
            raise Exception('end_patch_dimension in batch_processing.toml is greater than input image dimension. Use a bigger input image or change end_patch_dimension. ')
        
        full_result = batch_range_checker(
            max_dim,
            min_dim,
            shave,
            scale,
            dim_gap=dim_gap,
            batch_start=batch_size_start,
            logger=logger,
            img_path=img_path,
        )
        full_result = pd.DataFrame(full_result)
        full_result.columns = ["Patch dimnesion", "Maximum Batch Size"]


        device_name = "CPU"
        total_memory = "~"
        device = device
        if device == "cuda":
            _, device_name = ut.get_device_details()
            total_memory, _, _ = ut.get_gpu_details(
                device, "\nDevice info:", logger=None, print_details=False
            )

        print("\nSaving stats...\n")
        save_stat_csv(
            full_result=full_result,
            img_height=h,
            img_width=w,
            patch_dim=str(max_dim) + "-" + str(min_dim),
            patch_shave=shave,
            scale=scale,
            model=model_name,
            device=device,
            file_name="maximum_batch_size_per_dimension.csv",
            folder_name = folder_name,
            date = date
        )

        print("\nPlotting result...\n")
        if device == "cuda":
            plt_title = "Model: {} | GPU: {} | Memory: {} MB".format(
                model_name, device_name, total_memory
            )
        else:
            plt_title = "Model: {} | Device: {}".format(model_name, "CPU")

# =============================================================================
#         date = "_".join(str(time.ctime()).split())
#         date = "_".join(date.split(":"))
# =============================================================================

        x_data, y_data = (
            np.array(full_result.iloc[:, 0].values).flatten(),
            np.array(full_result.iloc[:, 1].values).flatten(),
        )
        x_label = (
            "Dimension n of patch(nxn) (Image: {}x{}, Shave: {}, Scale: {})".format(
                h, w, shave, scale
            )
        )
        y_label = "Maximum Batch Size"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plt_title)
        plt.plot(x_data, y_data, linestyle="--", marker="o", color="b")
        plt.savefig(
            folder_name +  "/{0}.png".format(
                "maximum_batch_size" + "_" + date
            )
        )
        plt.show()

    elif process == "batch_details":
        config = toml.load("../batch_processing.toml")
        run = int(config["run"])
        dim = int(config["single_patch_dimension"])
        shave = int(config["shave"])
        scale = int(config["scale"])
        img_path = config["img_path"]
        batch_size_start = int(config["single_patch_batch_size_start"])
        model = config["model"]
        device = config["device"]

        img = ut.load_image(img_path)
        c, h, w = img.shape

        full_result = single_patch_highest_batch_checker(
            dim,
            shave,
            scale,
            logger=logger,
            img_path=img_path,
            run=run,
            batch_size_start=batch_size_start,
        )

        full_result = pd.DataFrame(full_result)
        full_result.columns = [
            "Batch size",
            "Patch list creation time",
            "EDSR processing time",
            "CPU to GPU",
            "GPU to CPU",
            "Batch creation",
            "CUDA clear time",
            "Merging time",
            "Total batch processing time",
            "Total time",
        ]
        """
        0 - batch size
        1 - patch list creation time
        2 - EDSR processing time
        3 - CPU to GPU shifting time
        4 - GPU to CPU shifting time
        5 - Batch creation
        6 - CUDA Clear time
        7 - Patch merging time
        8 - Total batch processing time
        9 - Total time
        """
        save_stat_csv(full_result, h, w, dim, shave, scale, model=model, device=device, file_name="single_patch_all_batch.csv", folder_name = folder_name, date = date)
        process_stats_of_single_patch_batch(
            full_result, h, w, dim, shave, scale, model=model, device=device, date=date
        )
