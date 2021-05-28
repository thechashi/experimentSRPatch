import sys
import toml
import time
import pandas as pd
import numpy as np
import patch_calculator as pc
import matplotlib.pyplot as plt

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

    stat_file = open("results/" + foldername + "/" + filename, "r")
    lines = stat_file.readlines()
    device = lines[0]
    device_name = lines[1]
    total_memory = lines[2]

    # loading patch stats from latest binary search
    df = pd.read_csv("results/" + foldername + "/" + filename, comment="#")
    dim_mean_time_list = df[["Dimension", "Mean Time"]]

    # calculating different processing time with different patch size for an image
    result_df = dim_mean_time_list.iloc[dimension - 1, :].values
    patch_list = []
    for i in range(start_shave, end_shave):
        patch_list.append(
            [i, result_df[1] * pc.total_patch(dimension, height, width, shave=i)]
        )
    patch_list = pd.DataFrame(patch_list)
    # plots
    print(
        "Plotting processing time for shave size from: {0} to {1} for image with shape {2}x{3} and patch size {4}".format(
            start_shave, end_shave, height, width, dimension
        )
    )
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = "patch_fullimage_" + str(height) + "_" + str(width) + "_" + date
    x_data, y_data = (
        np.array(patch_list.iloc[:, 0].values).flatten(),
        np.array(patch_list.iloc[:, 1].values).flatten(),
    )
    x_label = "Shave for dimension {}x{} and an image ({}x{})".format(
        dimension, dimension, height, width
    )
    y_label = "Processing time (sec)"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_data, y_data)
    plt.savefig("results/{0}/{1}.png".format(foldername, filename))
    plt.show()

    # csv
    data_frame = pd.DataFrame(
        {
            "Shave": list(x_data),
            "Full Image ({}x{}) and Dimension {} Processing Time:".format(
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
    file.write(total_memory)
    data_frame.to_csv(file)
    file.close()
