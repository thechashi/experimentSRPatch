import time
import math
import copy
import torch
import utilities as ut
def maximum_unacceptable_dimension_2n(device, model, logfile):
    """
    Ge the maximum unacceptable dimension which is apower of 2

    Parameters
    ----------
    device : str
        device type.
    model : torch.nn.model
        SR model.

    Returns
    -------
    last_dimension : int
        unacceptabel dimension.

    """
    print()
    print("Getting maximum unacceptable dimension which is a power of two...")
    result1 = {}
    last_dimension = 0
    dimension = 2
    while True:
        print("\n#########################################################\n")
        logfile.write("\n##################################################\n")
        print(f"Testing dimension: {dimension}x{dimension} ...")
        logfile.write(f"\nTesting dimension: {dimension}x{dimension} ...")
        logfile.write("\nBefore loading the images...\n")
        print("\nBefore loading the images...\n")
        ut.get_gpu_details(device, logfile, print_details=True)
        input_image = ut.random_image(dimension)
        input_image = input_image.to(device)
        
        with torch.no_grad():
            try:
                start = time.time()
                output_image = model(input_image)
                logfile.write("\nAfter loading the images...\n")
                print("\nAfter loading the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
                end = time.time()
                total_time = end - start
                if dimension in result1.keys():
                    result1[dimension].append(total_time)
                else:
                    result1[dimension] = [total_time]
                print("Dimension ok.")
                logfile.write("\nDimension ok.")
                dimension *= 2
                ut.clear_cuda(input_image, output_image)
                logfile.write("\nAfter clearing the images...\n")
                print("\nAfter clearing the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
                print("\n#########################################################\n")
                logfile.write("\n##################################################\n")
            except RuntimeError as err:
                print("\nDimension NOT OK!")
                print("------------------------------------------------------")
                print(err)
                logfile.write("\nDimension NOT OK!\n")
                logfile.write(str(err))
                logfile.write("\nAfter loading the images...\n")
                print("\nAfter loading the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
                if dimension in result1.keys():
                    result1[dimension].append(math.inf)
                else:
                    result1[dimension] = [math.inf]
                last_dimension = dimension
                output_image = None
                ut.clear_cuda(input_image, output_image)
                logfile.write("\nAfter clearing the images...\n")
                print("\nAfter clearing the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
                logfile.write("\n------------------------------------------\n")
                print("\n#########################################################\n")
                logfile.write("\n##################################################\n")
                break
    return last_dimension


def maximum_acceptable_dimension(device, model, max_unacceptable_dimension, logfile):
    """
    Get amximum acceptable dimension

    Parameters
    ----------
    device : str
        device type.
    model : torch.nn.model
        SR model.
    max_unacceptable_dimension : int
        Maximum unacceptable dimension which is apower of 2.

    Returns
    -------
    last : int
        acceptable dimension.

    """
    print()
    print("Getting maximum acceptable dimension...")
    result2 = {}
    dimension = max_unacceptable_dimension
    maxm = math.inf
    minm = -math.inf
    last = 0
    while True:
        print("\n#########################################################\n")
        logfile.write("\n##################################################\n")
        print(f"Testing dimension: {dimension}x{dimension} ...")
        logfile.write(f"\nTesting dimension: {dimension}x{dimension} ...")
        logfile.write("\nBefore loading the images...\n")
        print("\nBefore loading the images...\n")
        ut.get_gpu_details(device, logfile, print_details=True)
        input_image = ut.random_image(dimension)
        input_image = input_image.to(device)
        with torch.no_grad():
            try:
                if last == dimension:
                    ut.clear_cuda(input_image, output_image=None)
                    logfile.write("\nAfter clearing the images...\n")
                    print("\nAfter clearing the images...\n")
                    ut.get_gpu_details(device, logfile, print_details=True)
                    break
                start = time.time()
                output_image = model(input_image)
                logfile.write("\nAfter loading the images...\n")
                print("\nAfter loading the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
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
                ut.clear_cuda(input_image, output_image)
                logfile.write("\nAfter clearing the images...\n")
                print("\nAfter clearing the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
                print("\n#########################################################\n")
                logfile.write("\n##################################################\n")
            except RuntimeError as err:
                print("\nDimension NOT OK!")
                print("------------------------------------------------------")
                print(err)
                logfile.write("\nDimension NOT OK!\n")
                logfile.write(str(err))
                logfile.write("\nAfter loading the images...\n")
                print("\nAfter loading the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
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
                ut.clear_cuda(input_image, output_image)
                logfile.write("\nAfter clearing the images...\n")
                print("\nAfter clearing the images...\n")
                ut.get_gpu_details(device, logfile, print_details=True)
                logfile.write("\n------------------------------------------\n")
                print("\n#########################################################\n")
                logfile.write("\n##################################################\n")
                continue
    return last