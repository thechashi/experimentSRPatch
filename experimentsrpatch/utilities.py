"""Utilities"""
import time
import torch
import math
import gc
import numpy as np
import matplotlib.pyplot as plt
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import pandas as pd
import logging
from datetime import date
import toml
import torchvision
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def create_custom_npz(
    height, width, input_file="data/slices/0.npz", output_file="data/slices/custom.npz"
):
    """
    Creates custom npz files from slice direcotry's npz files

    Parameters
    ----------
    height : int
        height of the custom npz.
    width : int
        width of the custom npz.
    input_file : str
        input file name from which this custom npz will be made.
    output_file : str
        output file name for saving the custom npz.

    Returns
    -------
    None.

    """
    image = np.load(input_file)
    image = image.f.arr_0
    image = np.resize(image, (height, width))
    np.savez(output_file, image)
    return image


def per_batch_processing_time(stat_path):
    total_patches = "Total Patches"
    total_batches_ = "Total Batches"
    maximum_batch_size = "Maximum Batch Size"
    total_batch_processing_time = "Total batch processing time"
    per_batch_processing_time_ = "Per Batch Processing Time"
    patch_dimension = "Patch Dimension"
    total_time = "Total time"
    stat_df = pd.read_csv(stat_path)

    total_batches = stat_df[total_patches] / stat_df[maximum_batch_size]
    stat_df[total_batches_] = total_batches
    per_batch_processing_time = (
        stat_df[total_batch_processing_time] / stat_df[total_batches_]
    )
    stat_df[per_batch_processing_time_] = per_batch_processing_time

    temp_file = open("results/file2.csv", "a")
    stat_df.to_csv(temp_file, header=False, index=False)
    temp_file.close()


def patch_count(img_height, img_width, patch_dimension, patch_shave):
    """
    Calculates total numebr of patches from an image with given patch dimension and shave value

    Parameters
    ----------
    img_height : int
        image height.
    img_width : int
        image width.
    patch_dimension : int
        patch dimesnion.
    patch_shave : int
        overlapping value.

    Returns
    -------
    total_patches : int
        total patch count.

    """
    actual_width = patch_dimension - 2 * patch_shave - 0
    last_start_h = img_height - patch_dimension
    total_rows = math.ceil(last_start_h / actual_width) + 1
    last_start_w = img_width - patch_dimension
    total_columns = math.ceil(last_start_w / actual_width) + 1
    # =============================================================================
    #     print(total_rows)
    #     print(total_columns)
    # =============================================================================
    total_patches = total_rows * total_columns
    return total_patches


def exception_handler(exception_type, exception, traceback):
    """
    Customized exception message

    Parameters
    ----------
    exception_type : str
        type of exception.
    exception : str
        exception message.

    Returns
    -------
    None.

    """
    print("%s: %s" % (exception_type.__name__, exception))


class timer:
    """
    Tracks timing
    """

    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        """
        Starts timer

        Returns
        -------
        None.

        """
        self.t0 = time.time()

    def toc(self, restart=False):
        """
        Ends timer and calculates time

        Parameters
        ----------
        restart : bool, optional
            restart or not. The default is False.

        Returns
        -------
        diff : float
            total time taken.

        """
        diff = time.time() - self.t0
        if restart:
            self.t0 = time.time()
        return diff

    def hold(self):
        """
        Pauses timer

        Returns
        -------
        None.

        """
        self.acc += self.toc()

    def release(self):
        """
        Unpauses timer

        Returns
        -------
        ret : float
            paused time.

        """
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        """
        Resets timer
        """
        self.acc = 0


def get_device_type():
    """
    Get device type

    Returns
    -------
    str
        type of device.

    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def test_image():
    """
    Creates, shows and saves random image

    Returns
    -------
    None.

    """
    data = np.random.randint(0, 255, size=(104, 104, 3), dtype=np.uint8)
    print(data.shape)
    img = Image.fromarray(data, "RGB")
    img.save("random_test.png")
    img.show()


def get_logger(logger_suffix=None, logfile_name=None, stream=False):
    """
    Get logger

    Returns
    -------
    logger : logger
        keeps log.

    """
    # Creating logger
    logger_suffix = logger_suffix if logger_suffix != None else str(time.time())
    logger = logging.getLogger(__name__ + logger_suffix)
    logger.setLevel(logging.INFO)

    # Logfile name by date
    if logfile_name == None:
        today = date.today().strftime("%b-%d-%Y")
        today = "_".join(str(today).split("-"))
        logfile_name = "results/logs_" + today + ".log"

    # Logfile creation
    file_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    file_handler = logging.FileHandler(logfile_name)
    # file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if stream:
        # Logging on console
        stream_formatter = logging.Formatter("%(levelname)s:%(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
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
    device_name = "~"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    return device, device_name


def load_image(img_path, show_img=False):
    """
    Loads image and returns it

    Parameters
    ----------
    img_path : str
        image path.
    show_img : bool, optional
        show image or not. The default is False.

    Returns
    -------
    img : 3D Matrix
        image.

    """
    # Reading image
    img = torchvision.io.read_image(img_path)
    if show_img:
        plt.imshow(img.permute((1, 2, 0)))
    img = img.float()
    return img

def image_shape(img_path):
    img = torchvision.io.read_image(img_path)
    print(img.shape)
    
def load_grayscale_image(img_path, show_img=False):
    """
    Loads image and returns it

    Parameters
    ----------
    img_path : str
        image path.
    show_img : bool, optional
        show image or not. The default is False.

    Returns
    -------
    img : 3D Matrix
        image.

    """
    # Reading image
    img = torchvision.io.read_image(img_path)
    if show_img:
        plt.imshow(img.permute((1, 2, 0)))
    img = img[2:, :, :].float()
    return img


def npz_loader(input_file):
    """
    Loads npz or image file. Used for RRDB.

    Parameters
    ----------
    input_file : str
        input file path.

    Returns
    -------
    image : torch tensor
        3D torch tensor.

    """
    input_file = Path(input_file)
    image_paths = [".jpg", ".png", ".jpeg", ".gif"]
    image_array_paths = [".npy", ".npz"]
    file_name = str(input_file.name).lower()
    is_file = "." + ".".join(file_name.split(".")[1:])
    if is_file in image_paths:
        image = Image.open(input_file)
        image = np.array(image.convert(mode="F"))
    if is_file == ".npz":
        image = np.load(input_file)
        image = image.f.arr_0  # Load data from inside file.
    elif is_file in image_array_paths:
        image = np.load(input_file)

    image = image[np.newaxis, :, :]
    image = torch.from_numpy(image)
    return image


def get_grayscale_image_tensor(img_path):
    """
    Creates image tensors from .jpg, .png, .jpeg, .gif, .npy, .npz files

    Parameters
    ----------
    img_path : str
        input file path.

    Returns
    -------
    image : torch tensor
        3D torch tensor.

    """
    img_path = Path(img_path)
    image_paths = [".jpg", ".png", ".jpeg", ".gif"]
    image_array_paths = [".npy", ".npz"]
    file_name = str(img_path.name).lower()
    is_file = "." + ".".join(file_name.split(".")[1:])
    if is_file in image_paths:
        image = Image.open(img_path)
        image = np.array(image.convert(mode="F"))
    if is_file == ".npz":
        image = np.load(img_path)
        image = image.f.arr_0  # Load data from inside file.
    elif is_file in image_array_paths:
        image = np.load(img_path)

    image = image[np.newaxis, :, :]
    image = torch.from_numpy(image)
    return image


def show_image(img_path, grayscale=False):
    """
    Shows .jpg, .jpeg, .png images

    Parameters
    ----------
    img_path : str
        img path.
    grayscale :  boolean, optional
        show as grayscale or not. The default is False.

    Returns
    -------
    None.

    """
    img_path = Path(img_path)
    image_paths = [".jpg", ".png", ".jpeg", ".gif"]
    file_name = str(img_path.name).lower()
    is_file = "." + ".".join(file_name.split(".")[1:])
    if is_file not in image_paths:
        print("Invalid image format")
        return
    if grayscale:
        image = Image.open(img_path).convert("L")
    else:
        image = Image.open(img_path)
    arr = np.asarray(image)
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.show()


def save_image(
    image, output_folder, input_height, input_width, scale, output_file_name="outputx4", add_date=True
):
    """
    Saves image

    Parameters
    ----------
    image : 3D Matrix
        image to save.
    output_folder : str
        output folder.
    input_height : int
        actual image height.
    input_width : int
        actual image width.
    scale : int
        scale for LR to SR.

    Returns
    -------
    None.

    """
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    if add_date:
        filename = output_file_name + "_" + date
    else:
        filename = output_file_name
    img_path = output_folder + "/" + filename + ".jpg"
    fig = plt.figure(
        figsize=((scale * input_height) / 1000, (scale * input_width) / 1000),
        dpi=100,
        frameon=False,
    )
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image.permute((1, 2, 0)))
    fig.savefig(img_path, bbox_inches="tight", transparent=True, pad_inches=0, dpi=1000)


def get_gpu_details(
    device: str, state: str, logger, memory_size_format="MB", print_details=True
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
    if memory_size_format == "KB":
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
                "\tTotal:\t{:09.3f} {}".format(total_mem, memory_size_format)
                + "\tUsed:\t{:09.3f} {}".format(used_mem, memory_size_format)
                + "\tFree:\t{:09.3f} {}".format(free_mem, memory_size_format)
                + "("
                + state
                + ")"
                + "\n"
            )
            logger.info(log_message)
        return total_mem, used_mem, free_mem
    return 0, 0, 0


def random_image(dimension, batch=True, channel=3):
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
    if channel > 1:
        image = image.repeat(channel, 1, 1)
    if batch == True:
        image = image.unsqueeze(0)
    return image.float()


def clear_cuda(input_image, output_image, gc_collect=False):
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
    if gc_collect:
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
    config["last_folder"] = foldername
    config["last_stat_csv"] = filename
    f = open("../config.toml", "w")
    toml.dump(config, f)
    f.close()
    file = open("results/" + foldername + "/" + filename, "a")
    file.write("# Device: {0} \n".format(device))
    file.write("# Device Name: {0} \n".format(device_name))
    file.write("# Total GPU memory: {0} \n".format(total_mem))
    data_frame.to_csv(file)
    file.close()

def get_ssim(img1, img2):
    if  img1.shape != img2.shape:
        print('Both image shapes have to match')
        return
    if len(img1.shape) > 3:
        print('Image shape has to be like this: (channel, height, width)')
        return
    c, h, w = img1.shape
    val = 0
    for i in range(0, c):
        i1 = img1[i, :, :]
        i2 = img2[i, :, :]
        val += ssim(i1, i2, data_range=i1.max() - i2.min())
    
    return val/c


def get_mse(img1, img2):
    if  img1.shape != img2.shape:
        print('Both image shapes have to match'  )
        return
    if len(img1.shape) > 3:
        print('Image shape has to be like this: (channel, height, width)')
        return
    c, h, w = img1.shape
    val = 0
    for i in range(0, c):
        i1 = img1[i, :, :]
        i2 = img2[i, :, :]
        val += mean_squared_error(i1, i2)
    
    return val/c
   
if __name__ == "__main__":
    #clear_cuda(None, None, False)
    img1 = np.load("output_images/test2_2187_Actual_output_x4.npy", allow_pickle=True)
    img2 = np.load("output_images/test2_2187_32_output_x4.npy", allow_pickle=True)
    img1 = img1[0,:,:,:]

    ssim_val = get_ssim(img1, img2)
    mse_val = get_mse(img1, img2)
    
    print("SSIM: ", ssim_val)
    print("MSE: ", mse_val)
# =============================================================================
#     data = pd.read_csv('results/fp16vfp32v9nt_results.csv')
#     data.plot(x='size', y=['fp16_340', 'fp32_340', 'iterative_340', 'recursive_310'], ylabel='Execution time (sec)', xlabel='Image dimension n of (nxn)', title='Recur. vs Iter. vs TRT_FP16 vs TRT_FP32 (NVIDIA GeForce GTX 860M - 2004MB)', kind='bar')
#     #data.plot(x='size', y=['mse_fp16_iterative', 'mse_fp32_iterative'], ylabel='MSE - FP16 & FP32 vs Iterative', xlabel='Image dimension n of (nxn)', title='Iterative vs TRT_FP16 AND TRT_FP32 MSE (NVIDIA GeForce GTX 860M - 2004MB)', kind='bar')
# 
#     plt.show()
# =============================================================================
    pass
