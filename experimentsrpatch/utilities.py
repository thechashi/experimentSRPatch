"""Utilities"""
import gc
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import pandas as pd
import logging
from datetime import date
import toml
from PIL import Image

def test_image():
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    print(data.shape)
    img = Image.fromarray(data, 'RGB')
    img.save('random_test.png')
    img.show()

def get_logger():
    """
    Get logger

    Returns
    -------
    logger : TYPE
        DESCRIPTION.

    """
    today = date.today().strftime("%b-%d-%Y")
    today = "_".join(str(today).split("-"))
    logfile_name = "results/logs_" + today + ".log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    stream_formatter = logging.Formatter("%(levelname)s:%(message)s")

    file_handler = logging.FileHandler(logfile_name)
    # file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def get_device_details():
    """
    Get the GPU details

    Returns
    -------
    device : torch.device
        cuda or cpu.
    device_name : str
        gpu model name.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    device_name = "cpu"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print("Device: ", device_name)
    print()
    return device, device_name


def get_gpu_details(
    device: str, state: str, logger, memory_size_format="MB", print_details=False
):
    """
    Get GPU usage

    Parameters
    ----------
    device : str
        device type.
    memory_size_format : str, optional
        MB or GB. The default is "MB".
    print_details : Boolean, optional
        flag for printing the gpu usage. The default is False.

    Returns
    -------
    total_mem : float
        total memory in gpu.
    used_mem : float
        used memory in gpu.
    free_mem : float
        free memory in gpu.

    """
    power = 2
    if memory_size_format == "MB":
        power = 2
    elif memory_size_format == "GB":
        power = 3
    if device == "cuda":
        nvmlInit()
        dev = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(dev)
        total_mem = info.total / (1024 ** power)
        used_mem = info.used / (1024 ** power)
        free_mem = info.free / (1024 ** power)
        if print_details:
            log_message = (
                state
                + "\nTotal:\t{0} {1}".format(total_mem, memory_size_format)
                + "\nUsed:\t{0} {1}".format(used_mem, memory_size_format)
                + "\nFree:\t{0} {1}".format(free_mem, memory_size_format)
                + "\n"
            )
            logger.info(log_message)
        return total_mem, used_mem, free_mem
    return None


def random_image(dimension):
    """
    Provides a random image

    Parameters
    ----------
    dimension : int
        The image dimension.

    Returns
    -------
    2D Tensor
        random image.

    """
    data = np.random.randint(0, 255, size=(dimension, dimension), dtype=np.uint8)
# =============================================================================
#     image = np.random.random((dimension, dimension)) * 255.0
#     image = np.round(image, 0)
#     print(image.shape)
# =============================================================================
    image = torch.tensor(data)
    image.unsqueeze_(0)
    image = image.repeat(3, 1, 1)
    image = image.unsqueeze(0)
    return image.float()


def clear_cuda(input_image, output_image):
    """
    Clears up the images from cuda

    Parameters
    ----------
    input_image : 2D Tensor
        input image in cuda.
    output_image : 2D Tenosr
        output_image in cuda.

    Returns
    -------
    None.

    """
    if output_image is not None:
        output_image = output_image.cpu()
        del output_image
    if input_image is not None:
        input_image = input_image.cpu()
        del input_image
    gc.collect()
    torch.cuda.empty_cache()


def get_mean_std(result3):
    """
    Ge the mean and std for every value

    Parameters
    ----------
    result3 : dictionary
        data to get mean and std from.

    Returns
    -------
    mean_dict : dictionary
        mean key value.
    std_dict : dictionary
        std key value.

    """
    mean_dict = {}
    std_dict = {}

    for key, val in result3.items():
        mean_dict[key] = np.mean(np.array(val))

    for key, val in result3.items():
        std_dict[key] = np.std(np.array(val))

    return mean_dict, std_dict


def plot_data(foldername, filename, data_dict, x_label, y_label, mode, title):
    """
    Plot data

    Parameters
    ----------
    foldername : str
        folder name.
    filename : str
        filen.
    data_dict : dictionary
        data to plot.
    x_label : str
        x label.
    y_label : str
        y label.
    mode : str
        mea or std.

    Returns
    -------
    None.

    """
    print("Plotting dimension vs ", mode)
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = filename + "_" + date
    data_dict = sorted(data_dict.items())
    x_data, y_data = zip(*data_dict)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x_data, y_data)
    plt.savefig("results/{0}/{1}.png".format(foldername, filename))
    plt.show()


def save_csv(
    foldername,
    filename,
    device,
    device_name,
    total_mem,
    mean_time,
    std_time,
    mean_mem_used,
    std_mem_used,
    mean_mem_free,
    std_mem_free,
):
    """
    Save data in CSV

    Parameters
    ----------
    foldername : str
        foldername.
    filename : str
        dilename to save data.
    device : str
        device type.
    device_name : str
        GPU model.
    total_mem : float
        total memory in GPU.
    mean_time : dictionary
        mean time vs dimension.
    std_time : dictionary
        std time vs dimension.
    mean_mem_used : dictionary
        mean memory used vs dimension.
    std_mem_used : dictionary
        std memory used vs dimension.
    mean_mem_free : dictionary
        mean free memory vs dimension.
    std_mem_free : dictionary
        std free memory vs dimension.

    Returns
    -------
    None.

    """
    config = toml.load("../config.toml")
    m_time = sorted(mean_time.items())
    s_time = sorted(std_time.items())
    m_used = sorted(mean_mem_used.items())
    s_used = sorted(std_mem_used.items())
    m_free = sorted(mean_mem_free.items())
    s_free = sorted(std_mem_free.items())
    dimension, mean_time = zip(*m_time)
    _, std_time = zip(*s_time)
    _, mean_mem_used = zip(*m_used)
    _, std_mem_used = zip(*s_used)
    _, mean_mem_free = zip(*m_free)
    _, std_mem_free = zip(*s_free)
    data_frame = pd.DataFrame(
        {
            "Dimension": list(dimension),
            "Mean Time": list(mean_time),
            "Std Time": list(std_time),
            "Mean Memory Used": list(mean_mem_used),
            "Std Memory Used": list(std_mem_used),
            "Mean Memory Free": list(mean_mem_free),
            "Std Memory Free": list(std_mem_free),
        }
    )
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = filename + "_" + date
    config['last_folder'] = foldername
    config['last_stat_csv'] = filename
    f = open("../config.toml", "w")
    toml.dump(config, f)
    f.close()
    file = open("results/" + foldername + "/" + filename, "a")
    file.write("# Device: {0} \n".format(device))
    file.write("# Device Name: {0} \n".format(device_name))
    file.write("# Total GPU memory: {0} \n".format(total_mem))
    data_frame.to_csv(file)
    file.close()
    
if __name__ == "__main__":
    print(random_image(32))
    #test_image()
