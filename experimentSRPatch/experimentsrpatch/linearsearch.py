import time
import torch
from tqdm import tqdm
import utilities as ut
def result_from_dimension_range(device, model, first, last, logfile, run=10):
    """
    Get detailed result for every dimension from 1 to the last acceptable dimension

    Parameters
    ----------
    device : str
        device type.
    model : torch.nn.model
        SR model.
    first : int
        starting dimension.
    last : int
        last acceptable dimension.
    run : int, optional
        total run to average the result. The default is 10.

    Returns
    -------
    result3 : dictionary
        time for every dimension.
    memory_used : dictionary
        memory used per dimension.
    memory_free : dictionary
        memory free per dimension.

    """
    print("\nPreparing detailed data... ")
    result3 = {}
    memory_used = {}
    memory_free = {}
    for i in range(run):
        print("Run: ", i + 1)
        for dim in tqdm(range(first, last + 1)):
            dimension = dim
            input_image = ut.random_image(dimension)
            input_image = input_image.to(device)
            with torch.no_grad():
                try:
                    start = time.time()
                    output_image = model(input_image)
                    end = time.time()
                    total_time = end - start
                    if dimension in result3.keys():
                        result3[dimension].append(total_time)
                        _, used, free = ut.get_gpu_details(device, logfile, print_details=False)
                        memory_used[dimension].append(used)
                        memory_free[dimension].append(free)
                    else:
                        result3[dimension] = [total_time]
                        _, used, free = ut.get_gpu_details(device, logfile, print_details=False)
                        memory_used[dimension] = [used]
                        memory_free[dimension] = [free]
                    ut.clear_cuda(input_image, output_image)
                except RuntimeError as err:
                    print("\nDimension NOT OK!")
                    print("------------------------------------------------------")
                    print(err)
                    logfile.write("\nDimension NOT OK!\n")
                    logfile.write(f"Run: {run}")
                    logfile.write(str(err))
                    logfile.write("\nAfter loading the images...\n")
                    ut.get_gpu_details(device, logfile, print_details=True)
                    output_image = None
                    ut.clear_cuda(input_image, output_image)
                    logfile.write("\nAfter clearing the images...\n")
                    ut.get_gpu_details(device, logfile, print_details=True)
                    logfile.write("\n------------------------------------------\n")
                    break
    return result3, memory_used, memory_free