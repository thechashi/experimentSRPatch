import numpy as np
from matplotlib import pyplot as plt
import click

def plot_image(img_path):
    """
    Plots npy file

    Parameters
    ----------
    img_path : Text
        npy gfile path.

    Returns
    -------
    None.

    """
    data = np.load(img_path)
    print("The image is: ")
    print(data)
    print('Output data type: ', type(data.dtype))

def show_image(img_path):
    """
    Plot image from npy file

    Parameters
    ----------
    img_path : TEXT
        npy file path.

    Returns
    -------
    None.

    """
    data = np.load(img_path)
    plt.imshow(data[0,:,:], cmap='gray')
    plt.show()

@click.command()
@click.option('--img_path', default='results/outputx4.npy', help='Path of the .npy file')
@click.option('--plot_img', default=False, help='Switch for plotting image or not')
@click.option('--show_img', default=True, help='Switch for showing image or not')
def  main(plot_img, show_img, img_path):
    if plot_img:
        plot_image(img_path)
    if show_img:
        show_image(img_path)
if __name__ == "__main__":
    main()
    

