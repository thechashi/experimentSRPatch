"""
Experiment on SR models
"""
import os
import time
import linearsearch as ls
import binarysearch as bs
import modelloader as md
import utilities as ut

#from EDSR import make_model, load
# =============================================================================
# def load_edsr(device, n_resblocks=16, n_feats=64):
#     """
#     Loads the EDSR model
# 
#     Parameters
#     ----------
#     device : str
#         device type.
#     n_resblocks : int, optional
#         number of res_blocks. The default is 16.
#     n_feats : int, optional
#         number of features. The default is 64.
# 
#     Returns
#     -------
#     model : torch.nn.model
#         EDSR model.
# 
#     """
#     args = {
#         "n_resblocks": n_resblocks,
#         "n_feats": n_feats,
#         "scale": [4],
#         "rgb_range": 255,
#         "n_colors": 3,
#         "res_scale": 1,
#     }
#     model = make_model(args).to(device)
#     load(model)
#     print("\nModel details: ")
#     print(model)
#     print()
#     return model
# =============================================================================
def main():
    """
    Main function

    Returns
    -------
    None.

    """
    # device information
    _, device_name = ut.get_device_details()
    device = "cuda"
    # open log file
    logfile = open("results/log.txt", "a")
    date = "_".join(str(time.ctime()).split())
    logfile.write("\n\n" + date + "\n")
    logfile.write("--------------------------------------------------------\n")
    # load model
    run = 10
    ut.clear_cuda(None, None)
    print("Before loading model: ")
    total, used, _ = ut.get_gpu_details(device, logfile, print_details=True)
    print("Total memory: ", total)
    model = md.load_edsr(device=device)
    print("After loading model: ")
    total, used, _ = ut.get_gpu_details(device, logfile, print_details=True)
    logfile.write("\nDevice: " + device)
    logfile.write("\nDevice name: " + device_name)
    logfile.write("\nTotal memory: " + str(total))
    logfile.write("\nTotal memory used after loading model: " + str(used))
    # get the highest unacceptable dimension which is a power of 2
    max_unacceptable_dimension = bs.maximum_unacceptable_dimension_2n(device, model, logfile)
    # get the maximum acceptable dimension
    max_dim = bs.maximum_acceptable_dimension(device, model, max_unacceptable_dimension, logfile)
    logfile.write(f"\nMaximum acceptable dimension is {max_dim}x{max_dim}\n")
    print(f"\nMaximum acceptable dimension is {max_dim}x{max_dim}\n")
    # get detailed result
    detailed_result, memory_used, memory_free = ls.result_from_dimension_range(
        device, model, 1, max_dim, logfile, run=run
    )
    # get mean
    # get std
    mean_time, std_time = ut.get_mean_std(detailed_result)
    mean_memory_used, std_memory_used = ut.get_mean_std(memory_used)
    mean_memory_free, std_memory_free = ut.get_mean_std(memory_free)
    # make folder for saving results
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    foldername = date
    os.mkdir("results/" + foldername)
    # plot data
    ut.ut.plot_data(
        foldername,
        "dimension_vs_meantime",
        mean_time,
        "Dimension",
        "Mean Time (" + str(run) + " runs)",
        mode="mean time",
    )
    ut.plot_data(
        foldername,
        "dimension_vs_stdtime",
        std_time,
        "Dimension",
        "Std Time (" + str(run) + " runs)",
        mode="std time",
    )
    ut.plot_data(
        foldername,
        "dimension_vs_meanmemoryused",
        mean_memory_used,
        "Dimension",
        "Mean Memory used (" + str(run) + " runs)",
        mode="mean memory used",
    )
    ut.plot_data(
        foldername,
        "dimension_vs_stdmemoryused",
        std_memory_used,
        "Dimension",
        "Std Memory Used (" + str(run) + " runs)",
        mode="std memory used",
    )
    ut.plot_data(
        foldername,
        "dimension_vs_meanmemoryfree",
        mean_memory_free,
        "Dimension",
        "Mean Memory Free (" + str(run) + " runs)",
        mode="mean memory free",
    )
    ut.plot_data(
        foldername,
        "dimension_vs_stdmemoryfree",
        std_memory_free,
        "Dimension",
        "Std Memory Free (" + str(run) + " runs)",
        mode="std memory free",
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
    logfile.close()


if __name__ == "__main__":
    main()
