import subprocess
import time
import toml
import pandas as pd
import custom_upsampler as cu
import forward_chop_recursive as fcr
import click
import utilities as ut


@click.command()
@click.option("--chop_type", default="recursive", help="Forward chop type")
@click.option("--start_dim", default="256", help="Image starting dimension")
@click.option("--end_dim", default="5000", help="Image ending dimension")
@click.option("--dim_gap", default="64", help="Image dimension gap ")
@click.option("--shave", default="10", help="Shave value")
@click.option("--scale", default=4, help="Scale value")
@click.option(
    "--stat_csv", default="results/rrdb_860GTX.csv", help="Iterative chop stat"
)
def run_experiment(chop_type, start_dim, end_dim, dim_gap, shave, scale, stat_csv):
    """
    Driver for runnign three experiments

    Parameters
    ----------
    chop_type : char
        experiment type.
    start_dim : int
        image starting dimension.
    end_dim : int
        image end dimension.
    dim_gap : int
        image dimension gap.
    shave : int
        overlapping value.
    scale : int
        lr to hr scale.
    stat_csv : char
        stat related to iterative experiment.

    Returns
    -------
    None.

    """
    logger = ut.get_logger()
    if chop_type == "iterative":
        full_result_columns = [
            "Patch Dimension",
            "Maximum Batch Size",
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
            "Image Dimension",
        ]
        date = "_".join(str(time.ctime()).split())
        date = "_".join(date.split(":"))
        full_result_df = pd.DataFrame(columns=full_result_columns)
        file_name = "patch_iterative_result" + date + ".csv"
        file = open("results/" + file_name, "a")
        full_result_df.to_csv(file, index=False)

        # all_stat = []
        for d in range(int(start_dim), int(end_dim), int(dim_gap)):
            print("Image dimension: ", d)
            command = (
                "python3 "
                + "custom_upsampler.py "
                + " "
                + stat_csv
                + " "
                + str(d)
                + " "
                + str(shave)
                + " "
                + str(scale)
            )
            p = subprocess.run(command, shell=True, capture_output=True)
            if p.returncode == 0:

                # Get the results of the current batch size
                time_stats = list(map(float, p.stdout.decode().split("\n")[0:12]))
                time_stats.append(d)
                stat_df = pd.DataFrame([time_stats])
                stat_df.to_csv(file, header=False, index=False)
            else:
                break
        file.close()
    elif chop_type == "recursive":
        print("\nRecursive chopping...")
        # =============================================================================
        #         config = toml.load("../batch_processing.toml")
        # =============================================================================
        max_patch_dimenison = 400
        full_result_columns = [
            "Patch Dimension",
            "Upsampling time",
            "Total time",
            "Image Dimension",
        ]
        date = "_".join(str(time.ctime()).split())
        date = "_".join(date.split(":"))
        full_result_df = pd.DataFrame(columns=full_result_columns)
        file_name = "recursive_result" + date + ".csv"
        file = open("results/" + file_name, "a")
        full_result_df.to_csv(file, index=False)

        # all_stat = []
        for d in range(int(start_dim), int(end_dim), int(dim_gap)):
            print("Image dimension: ", d)
            command = (
                "python3 "
                + "forward_chop_recursive.py "
                + " "
                + str(d)
                + " "
                + str(max_patch_dimenison)
            )
            p = subprocess.run(command, shell=True, capture_output=True)
            if p.returncode == 0:

                # Get the results of the current batch size
                # print(p.stdout.decode().split("\n"))
                time_stats = [max_patch_dimenison]
                time_stats += list(map(float, p.stdout.decode().split("\n")[0:2]))
                time_stats.append(d)
                stat_df = pd.DataFrame([time_stats])
                stat_df.to_csv(file, header=False, index=False)
            else:
                logger.error(
                    "CUDA memory error at image dimension: {} and patch dimension: {}".format(
                        d, max_patch_dimenison
                    )
                )
                break
        file.close()
    elif chop_type == "patch_iterative":
        print("\nIterative chopping...")
        config = toml.load("../batch_processing.toml")
        max_patch_dimenison = config["end_patch_dimension"]
        full_result_columns = [
            "Patch Dimension",
            "Upsampling time",
            "Cropping time",
            "Shifting time",
            "Clearing time",
            "Total time",
            "Image Dimension",
        ]
        date = "_".join(str(time.ctime()).split())
        date = "_".join(date.split(":"))
        full_result_df = pd.DataFrame(columns=full_result_columns)
        file_name = "patch_iterative_result" + date + ".csv"
        file = open("results/" + file_name, "a")
        full_result_df.to_csv(file, index=False)

        # all_stat = []
        for d in range(int(start_dim), int(end_dim), int(dim_gap)):
            print("Image dimension: ", d)
            command = (
                "python3 "
                + "forward_chop.py "
                + " "
                + str(d)
                + " "
                + str(max_patch_dimenison)
            )
            p = subprocess.run(command, shell=True, capture_output=True)
            if p.returncode == 0:

                # Get the results of the current batch size
                # print(p.stdout.decode().split("\n"))
                time_stats = [max_patch_dimenison]
                time_stats += list(map(float, p.stdout.decode().split("\n")[0:5]))
                time_stats.append(d)
                stat_df = pd.DataFrame([time_stats])
                stat_df.to_csv(file, header=False, index=False)
            else:
                logger.error(
                    "CUDA memory error at image dimension: {} and patch dimension: {}".format(
                        d, max_patch_dimenison
                    )
                )
                break
        file.close()


if __name__ == "__main__":
    run_experiment()
