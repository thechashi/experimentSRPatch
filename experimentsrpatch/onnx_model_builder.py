import sys
import numpy as np
import torch
import torch.onnx
import subprocess
import utilities as ut
import modelloader as md


def build_onnx_model(model_name, patch_size, onnx_model_name, device="cuda"):
    """
    Builds ONNX model with a fixed input shape

    Parameters
    ----------
    model_name : str
        Upsampler model name. (i.e. RRDB, EDSR).
    patch_size : int
        Input image dimension n of nxn.
    onnx_model_name : str
        output ONNX model name.

    Returns
    -------
    None.

    """
# =============================================================================
#     print('Before loading model: ')
#     subprocess.run("gpustat", shell = True)
#     print()
#     total_time = 0
#     model = None
#     if model_name == "EDSR":
#         model = md.load_edsr(device=device)
#         print('After loading model: ')
#         subprocess.run("gpustat", shell = True)
#         print()
#     elif model_name == "RRDB":
#         model = md.load_rrdb(device=device)
#     else:
#         raise Exception("Unknown model...")
#     model.eval()
#     input_image = ut.random_image(patch_size)
#     if model_name == "RRDB":
#         input_image = input_image[:, 2:, :, :]
#     dummy_input = input_image.to(device)
# =============================================================================
# =============================================================================
#     print('Before loading model: ')
#     subprocess.run("gpustat", shell = True)
#     print()
# =============================================================================
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
# =============================================================================
#         print('After loading model: ')
#         subprocess.run("gpustat", shell = True)
#         print()
# =============================================================================
        dummy_input = ut.random_image(patch_size).to(device)
# =============================================================================
#         print('After loading image: ')
#         subprocess.run("gpustat", shell = True)
#         print()
# =============================================================================
# =============================================================================
#         b, c, h, w = 1, 3, patch_size, patch_size
#         dummy_input = torch.rand(b, c, h, w, requires_grad=False).to(device)
# =============================================================================
    else:
        print("Unknown model!")
        return
    print(dummy_input.shape)
    model.eval()
    with torch.no_grad():

# =============================================================================
#         print('Before processing: ')
#         subprocess.run("gpustat", shell = True)
#         print()
# =============================================================================
# =============================================================================
#         output = model(input_image )
#         print('After processing: ')
#         subprocess.run("gpustat", shell = True)
#         print()
# =============================================================================
        try:
            torch.onnx.export(
                model,
                dummy_input,
                "inference_models/" + onnx_model_name,
                verbose=False,
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
            )
        except RuntimeError as err:
            sys.exit(1)
# =============================================================================
#         print('After processing: ')
#         subprocess.run("gpustat", shell = True)
#         print()
# =============================================================================
        
if __name__ == "__main__":
    # =============================================================================
    #     # build sample onnx model
    build_onnx_model(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    # =============================================================================