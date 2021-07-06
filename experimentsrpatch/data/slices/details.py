import experimentsrpatch.utilities as ut


for i in range(10):
    name = str(i) + ".npz"
    image = ut.npz_loader(name)
    c, h, w = image.shape
    print("Name: ", name)
    print("C, H, W: ", c, h , w)
    