import numpy as np
import torch
import experimentsrpatch.utilities as ut

def create_custom_npz(input_file, height, width):
    image = np.load(input_file)
    #print(type(image.f.arr_0.dtype))
    image = image.f.arr_0
    image = np.resize(image, (height, width))
    np.savez("11.npz", image)

if __name__ == "__main__":
    create_custom_npz("0.npz", 700, 400)
    for i in range(11):
        name = str(i) + ".npz"
        image = ut.npz_loader(name)
        c, h, w = image.shape
        print("Name: ", name)
        print("C, H, W: ", c, h , w)
# =============================================================================
#     image = np.load("0.npz")
#     #print(type(image.f.arr_0.dtype))
#     image = image.f.arr_0
#     for  i in range(128, 4000, 64):
#         image = np.resize(image, (i, i))
#         image = image[np.newaxis, :, :]
#         image = torch.from_numpy(image)
#         c, h, w = image.shape
#         print("C, H, W: ", c, h , w)
# =============================================================================
    pass