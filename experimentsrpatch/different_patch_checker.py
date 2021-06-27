"""
Checks differnt patches in a range
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
    shave = int(config["shave"]) if config["shave"] else 10
    scale = int(config["scale"]) if config["scale"] else 4
    filename = config["last_stat_csv"]
    foldername = config["last_folder"]
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

    start_dim = 4 * shave
    end_dim = min(height, width, dim_mean_time_list["Dimension"].max())

    print(
        "\nCalculating processing time for different dimensions for an image {}x{}...\n".format(
            height, width
        )
    )
    # calculating different processing time with different patch size for an image
    result_df = dim_mean_time_list.iloc[:, :].values
    post_df = result_df.copy()
    for i in range(start_dim - 1, end_dim):
        # print(i, height, width)
        total_patches, patch_size = pc.total_patch(i, height, width)
        # print(total_patches, patch_size)
        patch_side = int(math.sqrt(patch_size))
        # print(i, i*i, total_patches, patch_side, result_df[patch_side-1, 1])
        # print(result_df[patch_side-1, 1])
        post_df[i, 1] = result_df[patch_side - 1, 1] * total_patches
        # print(post_df[i, 1])
    # plots
    print(
        "Plotting processing time for patch size from: {0} to {1} for image with shape {2}x{3}...\n".format(
            start_dim, end_dim, height, width
        )
    )
    plt_title = "Model: {} | GPU: {} | Memory: {} MB".format(
        model_name, device_name, total_memory
    )
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = "patch_fullimage_" + str(height) + "_" + str(width) + "_" + date
    x_data, y_data = (
        np.array(post_df[start_dim - 1 : end_dim, 0]).flatten(),
        np.array(post_df[start_dim - 1 : end_dim, 1]).flatten(),
    )
    x_label = "Dimension n of a patch (nxn) for an input image ({}x{})".format(
        height, width
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
            "Patch Dimension": list(x_data),
            "Full Image ({}x{}) Processing Time:".format(height, width): list(y_data),
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
