import numpy as np
from PIL import Image
from pathlib import Path

with np.load("1.npz") as data:
    print(data.values)
    
def npz_loader(input_file):
    """
    :param input_file: directory of images
    :return: image: returns the image matrix
    """
    input_file = Path(input_file)
    image_paths = [".jpg", ".png", ".jpeg", ".gif"]
    image_array_paths = [".npy", ".npz"]
    file_name = str(input_file.name).lower()
    is_file = "." + ".".join(file_name.split(".")[1:])
    if is_file in image_paths:
        image = Image.open(input_file)
        image = np.array(image.convert(mode="F"))
    if is_file == ".npz":
        image = np.load(input_file)
        image = image.f.arr_0  # Load data from inside file.
    elif is_file in image_array_paths:
        image = np.load(input_file)
    return image

i = npz_loader("1.npz")
print(type(i))
print(i.shape)
print(i)

img = Image.fromarray(i)
img.show()
    
    