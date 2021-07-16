"""
This is the driver source code for running the batch patch experiment
"""
import sys
import os
import subprocess
import time
import shutil
import numpy as np
import pandas as pd
import toml
from tqdm import tqdm
from tqdm import trange
import pyfiglet
import matplotlib.pyplot as plt
import utilities as ut
import batch_predictor as bp

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
    model_name="EDSR",
    device="cuda",
    temp_file_path=None,
):
    """
    Checks maximum valid batch size for every patch dimension from min_dim to max_dim

    Parameters
    ----------
    max_dim : int
        biggest patch dimension to test.
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

    # Banner for the process
    banner = pyfiglet.figlet_format("Batch Experiment: " + model_name)
    print(banner)

    # Temporary file for saving stats of every batch size
    temp_file = open(temp_file_path, "a")

    # For saving all results
    full_result = []

    # to show memory usages
    used_memory = 0
    last_used_memory = 0

    # tqdm range
    tqdm_range = trange(max_dim, min_dim - 1, -dim_gap)

    for patch_dim in tqdm_range:

        # Show memory status in the tqdm bar
        _, used_memory, _ = ut.get_gpu_details(
            device, None, logger, print_details=False
        )
        leaked_memory = (
            used_memory - last_used_memory if used_memory > last_used_memory else 0
        )
        tqdm_range.set_description(
            "Patch Dim: {:04}x{:04} | Start Batch Size: {:04} \
            | Used Memory: {:09.3f} | Leaked Memory: {:09.3f}".format(
                patch_dim, patch_dim, batch_start, used_memory, leaked_memory
            )
        )
        last_used_memory = used_memory

        # If the model couldn't process the maximum dimension in one size batch then stop
        if patch_dim < max_dim and batch_start == 0:
            raise Exception(
                "Batch execution error. Highest patch dimension couldn't be \
                processed in a batch of size 1."
            )

        # Result of current patch
        result = [patch_dim]
        error_type = [0]
        time_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Flag for jumping batch sizes when we can predict
        jump = 0

        # Start predicting when we have more than three samples
        if len(full_result) >= 3:

            # Take last three stat rows to predict
            last_three_results = full_result[-3:]

            # Prediction condition
            if (
                last_three_results[2][1] - last_three_results[1][1] >= 2
                and last_three_results[1][1] - last_three_results[0][1] >= 2
            ):
                predicted_batch = int(
                    bp.predict3(
                        tuple(last_three_results[0][0:2]),
                        tuple(last_three_results[1][0:2]),
                        tuple(last_three_results[2][0:2]),
                    )
                )
                batch_start = predicted_batch
                # Jumping batches: Flag ON
                jump = 1

        # Call subprocess for each batch size for the current patch dimension
        while True:

            # Subprocess
            command = (
                "python3 "
                + "helper_batch_patch_forward_chop.py "
                + " --img_path="
                + img_path
                + " --dimension="
                + str(patch_dim)
                + " --shave="
                + str(patch_shave)
                + " --batch_size="
                + str(batch_start)
                + " --scale="
                + str(scale)
                + " --print_result="
                + str(0)
                + " --device="
                + device
                + " --model_name="
                + model_name
            )
            p = subprocess.run(command, shell=True, capture_output=True)

            # Valid batch size
            if p.returncode == 0:

                # Get the results of the current batch size
                time_stats = list(map(float, p.stdout.decode().split("\n")[0:10]))

                # Increase batch size
                batch_start += 1

                # Logging current valid batch size for the given patch dimension
                logger.info(
                    "Patch Dimension {} - Batch Size {}...".format(
                        patch_dim, batch_start
                    )
                )

                # The batch size was a predicted batch size but wasn't valid
                # so the flag turned to -1 and now the reduced batch size is valid
                # so break
                if jump == -1:
                    # Reducing batch size to start from the last valid batch
                    # size for the next patch dimension
                    batch_start -= 1
                    break

                # The batch size was a predicted valid batch size
                elif jump == 1:
                    # Resetting the flag
                    jump = 0

            # Invalid batch size
            else:

                # Batch size greater than total batches
                if p.returncode == 2:
                    error_type[0] = 1
                if p.returncode == 2:
                    logger.error(
                        "\tDimension: {}, Batch size : {}. Batch size larger \
                        than total number of patches".format(
                            patch_dim, batch_start
                        )
                    )

                # CUDA memory error
                elif p.returncode == 1:
                    pass

                # Reducing batch size for this or next iteration depends on jump value
                batch_start -= 1

                # It wasn't a predicted batch size so go to next iteration
                if jump == 0:
                    break

                # It was a predicted batch size so don't break and change the jump  flag
                else:
                    # Stay like this till we get a valid batch size while reducing
                    jump = -1

        # Saving result for the last executed patch dimension
        result += [batch_start]
        result += error_type
        result += time_stats

        # Appending to full result
        full_result.append(result)

        # Saving each stat outcome for every patch dimension
        pd.DataFrame([result]).to_csv(temp_file, header=False, index=False)

        # Saving checkpoint to resume
        loader_config = toml.load("../loader_config.toml")
        loader_config["batch_range_experiment"]["last_patch_dim"] = patch_dim
        loader_config["batch_range_experiment"]["last_valid_batch"] = batch_start
        loader_config_file = open("../loader_config.toml", "w")
        toml.dump(loader_config, loader_config_file)
        loader_config_file.close()

    # Closing temporary csv file of stats
    temp_file.close()

    # Changing current process status
    loader_config = toml.load("../loader_config.toml")
    loader_config["batch_range_experiment"]["status"] = "finished"
    loader_config_file = open("../loader_config.toml", "w")
    toml.dump(loader_config, loader_config_file)
    loader_config_file.close()

    # Returning result
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
                + " --img_path="
                + img_path
                + " --dimension="
                + str(patch_dim)
                + " --shave="
                + str(patch_shave)
                + " --batch_size="
                + str(batch_size_start)
                + " --scale="
                + str(scale)
                + " --print_result="
                + str(0)
                + " --device="
                + device
                + " --model_name="
                + model_name
            )

            p = subprocess.run(command, shell=True, capture_output=True)
            if p.returncode == 0:
                # print(p.stdout.decode())
                temp += list(map(float, p.stdout.decode().split("\n")[1:10]))
                result = [result[i] + temp[i] for i in range(len(temp))]
            else:
                if p.returncode == 2:
                    logger.error(
                        "\tDimension: {}, Batch size : {}. Batch size larger \
                        than total number of patches".format(
                            patch_dim, batch_size_start
                        )
                    )
                elif p.returncode == 1:
                    logger.error(
                        "\tDimension: {}, Batch size : {}. CUDA out of memory".format(
                            patch_dim, batch_size_start
                        )
                    )
                # raise Exception(p.stderr.decode())
                logger.info(
                    "Error: Dimension - {}, Batch size - {}".format(
                        patch_dim, batch_size_start
                    )
                )
                ut.get_gpu_details(
                    device, state="GPU stat after batch size error:", logger=logger
                )
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
        "results/batch_forward_chop_experiment/{0}/{1}.png".format(
            date, png_prefix + "_" + date
        )
    )
    plt.show()


def process_stats_of_single_patch_batch(
    full_result,
    img_height,
    img_width,
    patch_dim,
    patch_shave,
    scale,
    model,
    device,
    date,
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
    full_result,
    img_height,
    img_width,
    patch_dim,
    patch_shave,
    scale,
    model,
    device,
    file_name,
    folder_name,
    date,
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
    file = open(folder_name + "/" + file_name, "a")
    full_result.to_csv(file, index=False)
    file.close()
    meta_data_path = folder_name + "/" + "gpustat.json"
    gpu_command = "gpustat --json > " + meta_data_path
    subprocess.run(gpu_command, shell=True)


if __name__ == "__main__":

    # Getting logger
    logger = ut.get_logger()

    # Argument for which experiment to run
    process = sys.argv[1] if len(sys.argv) > 1 else "batch_range"

    # Run either batch_range or batch_details experiment
    if process == "batch_range":
        # Loading config for the batch_range experiment
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

        # CSV columns for the stats of the batch_range experiment
        full_result_columns = [
            "Patch dimnesion",
            "Maximum Batch Size",
            "Batch Error",
            "Total Patches",
            "Patch list creation time",
            "Upsampling time",
            "CPU to GPU",
            "GPU to CPU",
            "Batch creation",
            "CUDA clear time",
            "Merging time",
            "Total batch processing time",
            "Total time",
        ]

        # Loading config to check if there was a previous unfinished process.
        loader_config = toml.load("../loader_config.toml")

        # If the previous process was finished then start a new process
        if loader_config["batch_range_experiment"]["status"] == "finished":

            # Starting timestamp for the process to start
            date = "_".join(str(time.ctime()).split())
            date = "_".join(date.split(":"))

            # Result folder path with the timestamp
            folder_name = "results/" + "batch_forward_chop_experiment" + "/" + date
            os.mkdir(folder_name)

            # Temporary csv file's path for saving stat of every patch dimension
            temp_file_name = folder_name + "/temp_maximum_batch_size_per_dimension.csv"

            # Creating a dataframe only with the columns
            full_result_df = pd.DataFrame(columns=full_result_columns)

            # Transfering the dataframe to the temporary csv file
            temp_file = open(temp_file_name, "a")
            full_result_df.to_csv(temp_file, index=False)
            temp_file.close()

            # Saving the the current process's information to resume if the machine halts
            loader_config["batch_range_experiment"]["status"] = "ongoing"
            loader_config["batch_range_experiment"]["temp_file_name"] = temp_file_name
            loader_config["batch_range_experiment"]["folder_name"] = folder_name
            loader_config["batch_range_experiment"]["last_patch_dim"] = max_dim
            loader_config["batch_range_experiment"]["min_dim"] = min_dim
            loader_config["batch_range_experiment"][
                "last_valid_batch"
            ] = batch_size_start
            loader_config["batch_range_experiment"]["shave"] = shave
            loader_config["batch_range_experiment"]["scale"] = scale
            loader_config["batch_range_experiment"]["dim_gap"] = dim_gap
            loader_config["batch_range_experiment"]["model_name"] = model_name
            loader_config["batch_range_experiment"]["device"] = device
            loader_config["batch_range_experiment"]["img_path"] = img_path

        # When previous process was unfinished
        else:
            # Show message to user if there is an unfinished process
            print(
                "\nUnfinished batch experiment detected. Would you like to continue it?\n"
            )

            # Take user input to know if the user wants to continue the previous process
            reply = input("Press  Y/y for Yes and N/n for No:\n")

            # If YES continue from the previous checkpoint
            if reply == "Y" or reply == "y":

                # Prcoess continuation message
                print("Continuing unfinished process...\n")

                # File where the process needs to continue saving the stats
                temp_file_name = loader_config["batch_range_experiment"][
                    "temp_file_name"
                ]

                # Starting parameters for continuing the process
                folder_name = loader_config["batch_range_experiment"]["folder_name"]
                min_dim = int(loader_config["batch_range_experiment"]["min_dim"])
                shave = int(loader_config["batch_range_experiment"]["shave"])
                scale = int(loader_config["batch_range_experiment"]["scale"])
                img_path = loader_config["batch_range_experiment"]["img_path"]
                batch_size_start = int(
                    loader_config["batch_range_experiment"]["last_valid_batch"]
                )
                dim_gap = int(loader_config["batch_range_experiment"]["dim_gap"])
                max_dim = (
                    int(loader_config["batch_range_experiment"]["last_patch_dim"])
                    - dim_gap
                )
                model_name = loader_config["batch_range_experiment"]["model_name"]
                device = loader_config["batch_range_experiment"]["device"]

            # If NO remove the unfinished process and start a new one
            else:
                # Remove the data of the previous unfinished process
                folder_name = loader_config["batch_range_experiment"]["folder_name"]
                shutil.rmtree(folder_name)

                # Starting a new process
                print("Starting new process...\n")

                # Starting timestamp for the process to start
                date = "_".join(str(time.ctime()).split())
                date = "_".join(date.split(":"))

                # Result folder path with the timestamp
                folder_name = "results/" + "batch_forward_chop_experiment" + "/" + date
                os.mkdir(folder_name)

                # Temporary csv file's path for saving stat of every patch dimension
                temp_file_name = (
                    folder_name + "/temp_maximum_batch_size_per_dimension.csv"
                )

                # Creating a dataframe only with the columns
                full_result_df = pd.DataFrame(columns=full_result_columns, index=False)

                # Transfering the dataframe to the temporary csv file
                temp_file = open(temp_file_name, "a")
                full_result_df.to_csv(temp_file)
                temp_file.close()

                # Saving the the current process's information to resume if the machine halts
                loader_config["batch_range_experiment"]["status"] = "ongoing"
                loader_config["batch_range_experiment"][
                    "temp_file_name"
                ] = temp_file_name
                loader_config["batch_range_experiment"]["folder_name"] = folder_name
                loader_config["batch_range_experiment"]["last_patch_dim"] = max_dim
                loader_config["batch_range_experiment"]["min_dim"] = min_dim
                loader_config["batch_range_experiment"][
                    "last_valid_batch"
                ] = batch_size_start
                loader_config["batch_range_experiment"]["shave"] = shave
                loader_config["batch_range_experiment"]["scale"] = scale
                loader_config["batch_range_experiment"]["dim_gap"] = dim_gap
                loader_config["batch_range_experiment"]["model_name"] = model_name
                loader_config["batch_range_experiment"]["device"] = device
                loader_config["batch_range_experiment"]["img_path"] = img_path

        # Saving loader config for resuming the process if needed
        loader_config_file = open("../loader_config.toml", "w")
        toml.dump(loader_config, loader_config_file)
        loader_config_file.close()

        # Channels, height, and width of the image
        c, h, w = 0, 0, 0

        if model_name not in ["RRDB"]:
            # Loads normal JPEG image
            img = ut.load_image(img_path)
            c, h, w = img.shape
        else:
            # Loads special image for models like RRDB...
            img = ut.npz_loader(img_path)
            c, h, w = img.shape

        # If the maximum dimension of the patch is greater than images height
        # or width then stop the process
        if max_dim > h or max_dim > w:
            raise Exception(
                "end_patch_dimension in batch_processing.toml is greater than \
                input image dimension. Use a bigger input image or change end_patch_dimension. "
            )

        # Batch range checker process
        full_result = batch_range_checker(
            max_dim,
            min_dim,
            shave,
            scale,
            dim_gap=dim_gap,
            batch_start=batch_size_start,
            logger=logger,
            img_path=img_path,
            model_name=model_name,
            temp_file_path=temp_file_name,
        )

        # Final csv file name for the stats
        file_name = "maximum_batch_size_per_dimension.csv"

        # Time stamp for the final process
        date = "_".join(str(time.ctime()).split())
        date = "_".join(date.split(":"))

        # Renaming previous folder with  new folder name
        new_folder_name = "results/" + "batch_forward_chop_experiment" + "/" + date
        os.rename(folder_name, new_folder_name)
        folder_name = new_folder_name

        # Changing previous temporary file's name to the final name
        print("\nSaving stats...\n")
        temp_file_path = folder_name + "/temp_maximum_batch_size_per_dimension.csv"
        actual_file_path = folder_name + "/" + file_name
        os.rename(temp_file_path, actual_file_path)

        # Saving gpu stat for the finished process
        meta_data_path = folder_name + "/" + "gpustat.json"
        gpu_command = "gpustat --json > " + meta_data_path
        subprocess.run(gpu_command, shell=True)

        # Meta information to show in the plot
        device_name = "CPU"
        total_memory = "~"
        device = device
        if device == "cuda":
            _, device_name = ut.get_device_details()
            total_memory, _, _ = ut.get_gpu_details(
                device, "\nDevice info:", logger=None, print_details=False
            )

        # Loading result from csv to plot
        full_result = pd.read_csv(actual_file_path)

        # Plotting result message
        print("\nPlotting result...\n")

        # Creating plot title
        if device == "cuda":
            plt_title = "Model: {} | GPU: {} | Memory: {} MB".format(
                model_name, device_name, total_memory
            )
        else:
            plt_title = "Model: {} | Device: {}".format(model_name, "CPU")

        # X axis - Patch dimension, Y axis - Maximum batch size
        x_data, y_data = (
            np.array(full_result.iloc[:, 0].values).flatten(),
            np.array(full_result.iloc[:, 1].values).flatten(),
        )

        # X and y labels
        x_label = (
            "Dimension n of patch(nxn) (Image: {}x{}, Shave: {}, Scale: {})".format(
                h, w, shave, scale
            )
        )
        y_label = "Maximum Batch Size"

        # Plotting
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plt_title)
        plt.plot(x_data, y_data, linestyle="--", marker="o", color="b")

        # Saving plot
        plt.savefig(folder_name + "/{0}.png".format("maximum_batch_size" + "_" + date))

        # Showing plot
        plt.show()

    # Batch_details experiment: produce stats for all possible batch sizes
    elif process == "batch_details":

        # Loading paramters for the batch_details process
        config = toml.load("../batch_processing.toml")
        run = int(config["run"])
        dim = int(config["single_patch_dimension"])
        shave = int(config["shave"])
        scale = int(config["scale"])
        img_path = config["img_path"]
        batch_size_start = int(config["single_patch_batch_size_start"])
        model = config["model"]
        device = config["device"]

        # Channels, height, and width of the image
        c, h, w = 0, 0, 0

        if model not in ["RRDB"]:
            # Loads normal JPEG image
            img = ut.load_image(img_path)
            c, h, w = img.shape
        else:
            # Loads special image for models like RRDB...
            img = ut.npz_loader(img_path)
            c, h, w = img.shape

        # Batch details experiment
        full_result = single_patch_highest_batch_checker(
            dim,
            shave,
            scale,
            logger=logger,
            img_path=img_path,
            run=run,
            batch_size_start=batch_size_start,
        )

        # Converting full result into a dataframe
        full_result = pd.DataFrame(full_result)

        # Dataframe's column
        full_result.columns = [
            "Batch size",
            "Total Patches",
            "Patch list creation time",
            "Upsampling time",
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
        2 - Upsampling time
        3 - CPU to GPU shifting time
        4 - GPU to CPU shifting time
        5 - Batch creation
        6 - CUDA Clear time
        7 - Patch merging time
        8 - Total batch processing time
        9 - Total time
        """

        # Saving full result
        save_stat_csv(
            full_result,
            h,
            w,
            dim,
            shave,
            scale,
            model=model,
            device=device,
            file_name="single_patch_all_batch.csv",
            folder_name=folder_name,
            date=date,
        )

        # Plotting the full result
        process_stats_of_single_patch_batch(
            full_result, h, w, dim, shave, scale, model=model, device=device, date=date
        )
