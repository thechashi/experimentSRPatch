import numpy as np
import experimentsrpatch.utilities as ut

def create_custom_npz(input_file, height, width):
    image = np.load(input_file)
    image = image.f.arr_0
    image = np.resize(image, (height, width))
    np.savez("10.npz", image)

if __name__ == "__main__":
    create_custom_npz("0.npz", 4000, 2300)
    for i in range(11):
        name = str(i) + ".npz"
        image = ut.npz_loader(name)
        c, h, w = image.shape
        print("Name: ", name)
        print("C, H, W: ", c, h , w)
        