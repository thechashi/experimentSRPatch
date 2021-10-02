import click
import toml
import subprocess
import utilities as ut

@click.command()
@click.option("--model_name", default="EDSR", help="Upsampler model name")
@click.option("--patch_dim", default=None, help="Input dimension of dummy input")
@click.option("--use_precision", default="fp32", help="Enables FP16 or FP32 precision for layers that support it, in addition to FP32")
@click.option("--verbose", default=False, help="Print the steps while conversion")
def build_onnx_trt(model_name, patch_dim, use_precision, verbose):
    if patch_dim == None:
        config = toml.load("../config.toml")
        patch_dim = int(config["max_dim"])
    else:
        patch_dim = int(patch_dim)
        
    
    # pytorch to onnx model
    if verbose:
        print("Building ONNX model from the PyTorch model...")
    onnx_model_name = model_name.lower() + "_" + str(use_precision)+ "_" + \
    str(patch_dim) + ".onnx"
# =============================================================================
#     omb.build_onnx_model(model_name, patch_dim, onnx_model_name)
# =============================================================================
# =============================================================================
#     command1 = "python3 onnx_model_builder.py " + str(model_name) + " " + \
#         str(patch_dim) + " " + str(onnx_model_name)
# =============================================================================
    command1 = ["python3", "onnx_model_builder.py", str(model_name), str(patch_dim), str(onnx_model_name)]
    #subprocess.run(command1, shell=True)
    
    while True:
        process_output = subprocess.run(
            command1,
            stdout=subprocess.PIPE,
            text=True,
        )
        if process_output.returncode != 0:
            ut.clear_cuda(None, None)
            patch_dim -=1
            print('Memory out. Decreasing patch size. New patch_size = {}'.format(patch_dim))
# =============================================================================
#             command1 = "python3 onnx_model_builder.py " + str(model_name) + " " + \
#                 str(patch_dim) + " " + str(onnx_model_name)
# =============================================================================
            command1 = ["python3", "onnx_model_builder.py", str(model_name), str(patch_dim), str(onnx_model_name)]
        else:
            ut.clear_cuda(None, None)
            # for linear search
            config = toml.load("../config.toml")
            config["max_dim"] = patch_dim
            f = open("../config.toml", "w")
            toml.dump(config, f)
            break
    
    # onnx to trt
    if verbose:
        print("Building TRT engine from the ONNX model...")
    trt_model = "inference_models/" + model_name.lower() + "_" + str(use_precision) + \
        "_" + str(patch_dim) + ".trt"
# =============================================================================
#     otu.build_trt_engine("inference_models/"+onnx_model_name, trt_model)
# =============================================================================
    if use_precision == "fp32":
        command2 = "python3 onnx_trt_util.py " + "inference_models/"+onnx_model_name + " " + \
            str(trt_model) + " 0"
    elif use_precision == "fp16":
        command2 = "python3 onnx_trt_util.py " + "inference_models/"+onnx_model_name + " " + \
            str(trt_model) + " 1"
    subprocess.run(command2, shell=True)
    
if __name__ == "__main__":
    build_onnx_trt()