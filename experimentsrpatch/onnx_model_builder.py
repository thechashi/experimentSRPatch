import torch
import torch.onnx
import click
import numpy as np
import modelloader as md
import utilities as ut


@click.command()
@click.option("--model_name", default="RRDB", help="Name of the model")
@click.option("--patch_size", default=100, help="input patch dimension n of nxn")
@click.option("--onnx_model_name", default="rrdb.onnx", help="Path of the image")
def main(model_name, patch_size, onnx_model_name):
    device = ut.get_device_type()
    model = None
    if model_name == "RRDB":
        model = md.load_rrdb(device)
        image = ut.create_custom_npz(patch_size, patch_size)
        image = image[np.newaxis, :, :]
        image = torch.from_numpy(image)
        dummy_input = image.unsqueeze(0).to(device)
    elif model_name == "EDSR":
        model = md.load_edsr(device)
        dummy_input = ut.random_image(patch_size).to(device)
    else:
        print("Unknown model!")
        return
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        "inference_models/" + onnx_model_name,
        verbose=False,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )


main()
