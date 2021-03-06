"""
Checks different shave value for a given patch dimension
"""
import sys
import toml
import time
import math
import pandas as pd
import numpy as np
import patch_calculator as pc
import matplotlib.pyplot as plt
import utilities as ut

if __name__ == "__main__":
    config = toml.load("../config.toml")
    height = int(config["img_height"])
    width = int(config["img_width"])
    scale = int(config["scale"]) if config["scale"] else 4
    filename = config["last_stat_csv"]
    foldername = config["last_folder"]
    dimension = int(config["dimension"]) if config["dimension"] else 64
    start_shave = int(config["start_shave"]) if config["start_shave"] else 3
    end_shave = (dimension // 8) * 3
    model_name = "EDSR"
    device = "cuda"
    _, device_name = ut.get_device_details()
    total_memory, _, _ = ut.get_gpu_details(
        device, "\nDevice info:", logger=None, print_details=False
    )
    # =============================================================================
    #     stat_file = open("results/" + foldername + "/" + filename, "r")
    #     lines = stat_file.readlines()
    #     device = lines[0]
    #     device_name = lines[1]
    #     total_memory = lines[2]
    # =============================================================================

    # loading patch stats from latest binary search
    df = pd.read_csv("results/" + foldername + "/" + filename, comment="#")
    dim_mean_time_list = df[["Dimension", "Mean Time"]]

    print(
        "\nCalculating processing time for different shave sizes for an image {}x{} and dimension {}...\n".format(
            height, width, dimension
        )
    )
    # calculating different processing time with different patch size for an image
    result_df = dim_mean_time_list.iloc[:, :].values
    patch_list = []
    for i in range(start_shave, end_shave):
        total_patches, patch_size = pc.total_patch(dimension, height, width, shave=i)
        patch_side = int(math.sqrt(patch_size))
        # print(i, total_patches, patch_side)
        patch_list.append([i, result_df[patch_side - 1, 1] * total_patches])
    patch_list = pd.DataFrame(patch_list)
    # plots
    print(
        "Plotting processing time for shave size from: {0} to {1} for image with shape {2}x{3} and patch size {4}...\n".format(
            start_shave, end_shave, height, width, dimension
        )
    )
    plt_title = "Model: {} | GPU: {} | Memory: {} MB".format(
        model_name, device_name, total_memory
    )
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = "shave_fullimage_" + str(height) + "_" + str(width) + "_" + date
    x_data, y_data = (
        np.array(patch_list.iloc[:, 0].values).flatten(),
        np.array(patch_list.iloc[:, 1].values).flatten(),
    )
    x_label = "Shave size for dimension {}x{} and an image ({}x{})".format(
        dimension, dimension, height, width
    )
    y_label = "Processing time (sec): LR -> SR (Scale: {})".format(scale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data)
    plt.savefig("results/{0}/{1}.png".format(foldername, filename))
    plt.show()

    # csv
    data_frame = pd.DataFrame(
        {
            "Shave": list(x_data),
            "Full Image ({}x{}) and Patch Dimension {} Processing Time:".format(
                height, width, dimension
            ): list(y_data),
        }
    )
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = "stat_" + filename
    file = open("results/" + foldername + "/" + filename, "a")
    file.write(device)
    file.write(device_name)
    file.write(str(total_memory))
    data_frame.to_csv(file)
    file.close()
