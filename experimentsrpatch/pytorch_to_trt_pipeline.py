import click
import toml
import onnx_trt_util as otu

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
    onnx_model_name = model_name.lower() + "_" + str(use_precision) + "_" + ".onnx"
    otu.build_onnx_model(model_name, patch_dim, onnx_model_name)
    
    # onnx to trt
    if verbose:
        print("Building TRT engine from the ONNX model...")
    trt_model = "inference_models/" + model_name.lower() + "_" + str(use_precision) + "_" + ".trt"
    otu.build_trt_engine("inference_models/"+onnx_model_name, trt_model)
    
if __name__ == "__main__":
    build_onnx_trt()