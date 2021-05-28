"""
Total Patch Calculation given height and width
"""
import copy


def total_patch(dimension, height, width, shave=10, scale=4):
    """
    Total Patch calculation

    Parameters
    ----------
    dimension : int
        min_size dimension.
    height : int
        actual image height.
    width : int
        actual image width.
    shave : int
        base overlapping value.
    scale : int
        hr scale size.

    Returns
    -------
    int
        total number of patches.

    """
    min_size = dimension * dimension
    size = height * width
    h_tracker = []
    w_trakcer = []
    h_size = 0
    w_size = 0
    count = 0
    h, w = copy.copy(height), copy.copy(width)
    while size >= min_size:
        h_half, w_half = height // 2, width // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += h_size % scale
        w_size += w_size % scale
        size = h_size * w_size
        height, width = h_size, w_size
        count += 1

    if h * w == min_size:
        count = 0
    return 4 ** count


if __name__ == "__main__":
    total_patch(100, 100, 100, 10, 4)
