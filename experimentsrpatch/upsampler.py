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
