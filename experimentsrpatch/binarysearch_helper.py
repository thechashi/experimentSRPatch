"""
Binary Search Helper
"""
import sys
import torch
import time
import utilities as ut
import modelloader as md


def binary_search_helper(dimension, logger, model_name="EDSR", device="cuda"):
    """
    Process random image and calculates processing time

    Parameters
    ----------
    dimension : int
        random image dimension.
    logger : logger
        keep logs.
    device : str, optional
        GPU or CPU. The default is 'cuda'.

    Returns
    -------
    total_time : float
        EDSR processing time.

    """
    total_time = 0
    try:
        model = None
        if model_name == "EDSR":
            model = md.load_edsr(device=device)
        elif model_name == "RRDB":
            model = md.load_rrdb(device=device)
        else:
            raise Exception('Unknown model...')
            
        input_image = ut.random_image(dimension)
        if model_name == "RRDB":
            input_image = input_image[:, 2:, :, :]
        input_image = input_image.to(device)
    
        with torch.no_grad():
            start = time.time()
            output_image = model(input_image)
            end = time.time()
            total_time = end - start
            ut.clear_cuda(input_image, output_image)
        model.cpu()
        del model
    except RuntimeError as err:
        logger.error( "Runtime error for dimension: {}x{}: " + err)
        sys.exit(1)
    return total_time


if __name__ == "__main__":
    sys.excepthook = ut.exception_handler
    logger = ut.get_logger()
    print(binary_search_helper(int(sys.argv[1]), logger, model_name=sys.argv[2]))
