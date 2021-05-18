"""
Experiment on EDSR model
"""
import os
import gc
import time
import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import pandas as pd
from EDSR import make_model, load

def get_device_details():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    device_name = "cpu"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print("Device: ", device_name)
    print()
    return device, device_name


def get_gpu_details(device, memory_size_format="MB", print_details=False):
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
            print("******************************************************")
            print("\nGPU details:")
            print("Total memory: {0} {1}".format(total_mem, memory_size_format))
            print("Used memory: {0} {1}".format(used_mem, memory_size_format))
            print("Free memory: {0} {1}".format(free_mem, memory_size_format))
            print("****************************************************\n")
        return total_mem, used_mem, free_mem
    return None


def load_edsr(device, n_resblocks=16, n_feats=64):
    args = {
        "n_resblocks": n_resblocks,
        "n_feats": n_feats,
        "scale": [4],
        "rgb_range": 255,
        "n_colors": 3,
        "res_scale": 1,
    }
    model = make_model(args).to(device)
    load(model)
    print("\nModel details: ")
    print(model)
    print()
    return model


def random_image(dimension):
    image = np.random.random((dimension, dimension)) * 255.0
    image = torch.tensor(image)
    image.unsqueeze_(0)
    image = image.repeat(3, 1, 1)
    image = image.unsqueeze(0)
    return image.float()


def clear_cuda(input_image, output_image):
    if output_image is not None:
        output_image = output_image.cpu()
        del output_image
    if input_image is not None:
        input_image = input_image.cpu()
        del input_image
    gc.collect()
    torch.cuda.empty_cache()


def maximum_unacceptable_dimension_2n(device, model):
    print()
    print("Getting maximum unacceptable dimension which is a power of two...")
    result1 = {}
    last_dimension = 0
    dimension = 2
    while True:
        input_image = random_image(dimension)
        input_image = input_image.to(device)
        print("Testing dimension: {dimension}x{dimension} ...")
        with torch.no_grad():
            try:
                start = time.time()
                output_image = model(input_image)
                end = time.time()
                total_time = end - start
                if dimension in result1.keys():
                    result1[dimension].append(total_time)
                else:
                    result1[dimension] = [total_time]
                print("Dimension ok.")
                dimension *= 2
                clear_cuda(input_image, output_image)
            except RuntimeError as err:
                print("Dimension NOT OK!")
                print("------------------------------------------------------")
                print(err)
                if dimension in result1.keys():
                    result1[dimension].append(math.inf)
                else:
                    result1[dimension] = [math.inf]
                last_dimension = dimension
                output_image = None
                clear_cuda(input_image, output_image)
                break
    return last_dimension


def maximum_acceptable_dimension(device, model, max_unacceptable_dimension):
    print()
    print("Getting maximum acceptable dimension...")
    result2 = {}
    dimension = max_unacceptable_dimension
    maxm = math.inf
    minm = -math.inf
    last = 0
    while True:
        input_image = random_image(dimension)
        input_image = input_image.to(device)
        print("Testing dimension: {dimension}x{dimension} ...")
        with torch.no_grad():
            try:
                if last == dimension:
                    clear_cuda(input_image, output_image=None)
                    break
                start = time.time()
                output_image = model(input_image)
                end = time.time()
                total_time = end - start
                last = dimension
                if dimension in result2.keys():
                    result2[dimension].append(total_time)
                else:
                    result2[dimension] = [total_time]
                minm = copy.copy(dimension)
                print("Dimension ok.")
                if maxm == math.inf:
                    dimension *= 2
                else:
                    dimension = dimension + (maxm - minm) // 2
                clear_cuda(input_image, output_image)
            except RuntimeError as err:
                print("Dimension NOT OK!")
                print(err)
                print("------------------------------------------------------")
                maxm = copy.copy(dimension)
                if dimension in result2.keys():
                    result2[dimension].append(math.inf)
                else:
                    result2[dimension] = [math.inf]
                if minm == -math.inf:
                    dimension = dimension // 2
                else:
                    dimension = minm + (maxm - minm) // 2
                output_image = None
                clear_cuda(input_image, output_image)
                continue
    return last


def result_from_dimension_range(device, model, first, last, run=10):
    print("\nPreparing detailed data... ")
    result3 = {}
    memory_used = {}
    memory_free = {}
    for i in range(run):
        print("Run: ", i + 1)
        for dim in tqdm(range(first, last + 1)):
            dimension = dim
            input_image = random_image(dimension)
            input_image = input_image.to(device)
            with torch.no_grad():
                try:
                    start = time.time()
                    output_image = model(input_image)
                    end = time.time()
                    total_time = end - start
                    if dimension in result3.keys():
                        result3[dimension].append(total_time)
                        _, used, free = get_gpu_details(device, print_details=False)
                        memory_used[dimension].append(used)
                        memory_free[dimension].append(free)
                    else:
                        result3[dimension] = [total_time]
                        _, used, free = get_gpu_details(device, print_details=False)
                        memory_used[dimension] = [used]
                        memory_free[dimension] = [free]
                    clear_cuda(input_image, output_image)
                except RuntimeError as err:
                    print("Dimension NOT OK!")
                    print(err)
                    print("--------------------------------------------------")
                    output_image = None
                    clear_cuda(input_image, output_image)
                    break
    return result3, memory_used, memory_free


def get_mean_std(result3):
    mean_dict = {}
    std_dict = {}

    for key, val in result3.items():
        mean_dict[key] = np.mean(np.array(val))

    for key, val in result3.items():
        std_dict[key] = np.std(np.array(val))

    return mean_dict, std_dict


def plot_data(foldername, filename, data_dict, x_label, y_label, mode):
    print("Plotting dimension vs ", mode)
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = filename + "_" + date
    data_dict = sorted(data_dict.items())
    x_data, y_data = zip(*data_dict)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
    print("mt", str(len(mean_time)))
    print("st", str(len(std_time)))
    print("mU", str(len(mean_mem_used)))
    print("sU", str(len(std_mem_used)))
    print("mF", str(len(mean_mem_free)))
    print("sF", str(len(std_mem_free)))
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
    file = open("results/" + foldername + "/" + filename, "a")
    file.write("# Device: {0} \n".format(device))
    file.write("# Device Name: {0} \n".format(device_name))
    file.write("# Total GPU memory: {0} \n".format(total_mem))
    data_frame.to_csv(file)
    file.close()

def main():
    # device information
    _, device_name = get_device_details()
    device = "cuda"
    # open log file
    logfile = open("results/log.txt", "a")
    date = "_".join(str(time.ctime()).split())
    logfile.write("\n\n" + date + "\n")
    logfile.write("--------------------------------------------------------\n")
    # load model
    run = 10
    clear_cuda(None, None)
    print("Before loading model: ")
    total, used, _ = get_gpu_details(device, print_details=True)
    print("Total memory: ", total)
    model = load_edsr(device=device)
    print("After loading model: ")
    total, used, _ = get_gpu_details(device, print_details=True)
    logfile.write("\nDevice: " + device)
    logfile.write("\nDevice name: " + device_name)
    logfile.write("\nTotal memory: " + str(total))
    logfile.write("\nTotal memory used after loading model: " + str(used))
    # get the highest unacceptable dimension which is a power of 2
    max_unacceptable_dimension = maximum_unacceptable_dimension_2n(device, model)
    # get the maximum acceptable dimension
    max_dim = maximum_acceptable_dimension(device, model, max_unacceptable_dimension)
    logfile.write("\nmaximum acceptable dimension: " + str(max_dim))
    # get detailed result
    detailed_result, memory_used, memory_free = result_from_dimension_range(
        device, model, 1, max_dim, run=run
    )
    # get mean
    # get std
    mean_time, std_time = get_mean_std(detailed_result)
    mean_memory_used, std_memory_used = get_mean_std(memory_used)
    mean_memory_free, std_memory_free = get_mean_std(memory_free)

    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    foldername = date
    os.mkdir("results/" + foldername)
    # plot data
    plot_data(
        foldername,
        "dimension_vs_meantime",
        mean_time,
        "Dimension",
        "Mean Time (" + str(run) + " runs)",
        mode="mean time",
    )
    plot_data(
        foldername,
        "dimension_vs_stdtime",
        std_time,
        "Dimension",
        "Std Time (" + str(run) + " runs)",
        mode="std time",
    )
    plot_data(
        foldername,
        "dimension_vs_meanmemoryused",
        mean_memory_used,
        "Dimension",
        "Mean Memory used (" + str(run) + " runs)",
        mode="mean memory used",
    )
    plot_data(
        foldername,
        "dimension_vs_stdmemoryused",
        std_memory_used,
        "Dimension",
        "Std Memory Used (" + str(run) + " runs)",
        mode="std memory used",
    )
    plot_data(
        foldername,
        "dimension_vs_meanmemoryfree",
        mean_memory_free,
        "Dimension",
        "Mean Memory Free (" + str(run) + " runs)",
        mode="mean memory free",
    )
    plot_data(
        foldername,
        "dimension_vs_stdmemoryfree",
        std_memory_free,
        "Dimension",
        "Std Memory Free (" + str(run) + " runs)",
        mode="std memory free",
    )
    # save data
    save_csv(
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
    main()
