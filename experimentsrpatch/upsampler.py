"""
Upsampler for a given image and model
"""
import toml
import pyfiglet
import click
import batch_patch_forward_chop as bpfc


@click.command()
@click.option("--img_path", default="data/slices/0.npz", help="Path of the image")
@click.option("--model_name", default="RRDB", help="Name of the model")
def main(img_path, model_name):
    """
    Driver for upsampler

    Parameters
    ----------
    img_path : str
        image path.
    model_name : str
        model name.

    Returns
    -------
    None.

    """
    banner = pyfiglet.figlet_format("Upsampler: " + model_name)
    print(banner)

    config = toml.load("../upsampler_config.toml")
    batch_size = config["batch_size"]
    patch_dimension = config["patch_dimension"]
    patch_shave = config["patch_shave"]
    scale = config["scale"]

    bpfc.upsample(
        model_name,
        img_path,
        patch_dimension,
        patch_shave,
        batch_size,
        scale,
        device="cuda",
    )


if __name__ == "__main__":
    main()

# =============================================================================
# How to process input images with different size( here height > width):
#     a) When the height of the input image is less than the maximum acceptable patch dimesnion(MAPD):
#         - process the input image in one go
#     b) When the height of the input is bigger than MAPD but the width is smaller than MAPD:
#         - divide the image into patches with fastest patch dimension
#         - batch process
#     c) When the height and width of the input is bigger than the MAPD but smaller than 2.3k:
#         - find out total batches from total patches and maximum batch size
#         - calculate per batch processing time from the csv
#         - calculate per batch processing time for each dimension  with the maximum batch size and total number of patches
#         - find out the patch dimension with the fastest batch processing time
#         - divide the image into the patches with the fastest batch processing time
#         - batch process
#     d) When the width is smaller than 2.3k and height is bigger than 2.3K;
#         - divide the image into patches with fastest patch dimension
#         - batch process
#     e) When the height and width are both bigger than 2.3k:
#         - divide the image into patches with fastest patch dimension
#         - batch process
# =============================================================================
