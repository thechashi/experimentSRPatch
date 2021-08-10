"""
Helper for batch forward chopping experiment
"""
import sys
import numpy as np
import click
import batch_patch_forward_chop as bpfc
import utilities as ut


@click.command()
@click.option("--img_path", default="data/slices/0.npz", help="Path of the image")
@click.option("--dimension", default=110, help="Patch dimension")
@click.option("--shave", default=10, help="Shave value for patch overlapping")
@click.option("--batch_size", default=2, help="Number of pathes per batch")
@click.option("--scale", default=4, help="Scale size for output")
@click.option("--print_result", default=False, help="Save output as np matrix or not")
@click.option("--device", default="cuda", help="Device type")
@click.option("--model_name", default="RRDB", help="Name of the model")
def main(
    img_path, dimension, shave, batch_size, scale, print_result, device, model_name
):
    """
    Takes an image path and model name and produce the high resolution version
    of that image with the help of that model
    """
    if model_name == "EDSR":
        input_image = ut.load_image(img_path)
    elif model_name == "RRDB":
        # input_image = ut.load_grayscale_image(img_path)
        input_image = ut.npz_loader(img_path)
        # print(input_image.shape)
    else:
        raise Exception("{} : Unknown model...".format(model_name))

    output_image = bpfc.patch_batch_forward_chop(
        input_image,
        dimension,
        shave,
        scale,
        batch_size,
        model_type=model_name,
        print_timer=True,
    )
    if print_result:
        np.savez("results/outputx4.npz", output_image)
        np.save("results/outputx4", output_image)
        # c, h, w = input_image.shape
        # ut.save_image(output_image, "results/", h, w, scale)


if __name__ == "__main__":
    main()
# =============================================================================
#     img_path = sys.argv[1] if len(sys.argv) > 1 else "data/slices/10.npz"
#     dimension = int(sys.argv[2]) if len(sys.argv) > 2 else 145
#     shave = int(sys.argv[3]) if len(sys.argv) > 3 else 12
#     batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
#     scale = int(sys.argv[5]) if len(sys.argv) > 5 else 4
#     print_result = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
#     device = str(sys.argv[7]) if len(sys.argv) > 7 else "cuda"
#     model_name = str(sys.argv[8]) if len(sys.argv) > 8 else "RRDB"
#
#     if model_name == "EDSR":
#         input_image = ut.load_image(img_path)
#     elif model_name == "RRDB":
#         # input_image = ut.load_grayscale_image(img_path)
#         input_image = ut.npz_loader(img_path)
#         # print(input_image.shape)
#     else:
#         raise Exception("{} : Unknown model...".format(model_name))
#
#     output_image = bpfc.patch_batch_forward_chop(
#         input_image,
#         dimension,
#         shave,
#         scale,
#         batch_size,
#         model_type=model_name,
#         print_timer=True,
#     )
#     if print_result:
#         np.savez("results/outputx4.npz", output_image)
#         np.save("results/outputx4", output_image)
#         #c, h, w = input_image.shape
#         #ut.save_image(output_image, "results/", h, w, scale)
#
# =============================================================================
