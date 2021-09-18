"""
Source code for turning a PyTorch model to ONNX model and then turning that
ONNX model and then turning that onnx model to TensorRT engine.

Also contians code for inferencing with the TensorRT engine
"""
import torch
import torch.onnx
import numpy as np
import subprocess
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import utilities as ut
import modelloader as md


def build_onnx_model(model_name, patch_size, onnx_model_name):
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
        b, c, h, w = 1, 3, patch_size, patch_size
        dummy_input = torch.rand(b, c, h, w, requires_grad=True).to(device)
    else:
        print("Unknown model!")
        return
    with torch.no_grad():
        model.eval()
        output = model(dummy_input)
        torch.onnx.export(
            model,
            dummy_input,
            "inference_models/" + onnx_model_name,
            verbose=False,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
        )


def build_trt_engine(onnx_model, trt_model):
    """
    Runs terminal command for turning a ONNX model to TensorRT engine

    Parameters
    ----------
    onnx_model : str
        input ONNX model name.
    trt_model : str
        output TensorRT engine name.

    Returns
    -------
    bool
        returns True after succesfully creating an engine, False otherwise.

    """
    command = (
        "trtexec --onnx="
        + onnx_model
        + " --saveEngine="
        + trt_model
        + " --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
    )
    process_output = subprocess.run(
        command, stdout=subprocess.PIPE, text=True, shell=True
    )
    if process_output.returncode != 0:
        print("Unable to create TensorRT engine!")
        return False
    else:
        print("TensorRT engine created successfully")
        return True


def trt_inference(trt_engine, img, patch_size, scale=4, use_fp16=False):
    """
    Inference with TensorRT enigne

    Parameters
    ----------
    trt_engine : str
        TensorRT engine path.
    img : Tensor
        input image.
    patch_size : int
        input image dimension n of nxn.
    scale : int, optional
        Enlarging scale. The default is 4.
    use_fp16 : bool, optional
        precision type. The default is True.

    Returns
    -------
    output : Numpy Array
        output image.

    """
    USE_FP16 = use_fp16
    target_dtype = np.float16 if USE_FP16 else np.float32
    b, c, h, w = img.shape
    input_batch = np.ascontiguousarray(img, dtype=np.float32)
    f = open(trt_engine, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # need to set input and output precisions to FP16 to fully enable it
    output = np.empty(
        [b, c, patch_size * scale, patch_size * scale], dtype=target_dtype
    )

    # allocate device memory
    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict(batch):  # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()

        return output

    output = predict(input_batch)

    return output


if __name__ == "__main__":
    # =============================================================================
    #     # build sample onnx model
# =============================================================================
#     build_onnx_model(model_name="RRDB", patch_size=120, onnx_model_name="rrdb.onnx")
# =============================================================================
    # =============================================================================

    # =============================================================================
    #     # build smaple trt engine
    build_trt_engine("inference_models/rrdb.onnx", "inference_models/rrdb.trt")
    # =============================================================================

    # sample inference with trt engine
# =============================================================================
#     img_path = "data/test7.jpg"
#     input_batch = ut.load_image(img_path).unsqueeze(0).numpy()
#     print(input_batch)
#     print(input_batch.shape)
#     print(type(input_batch))
#     b, c, h, w = input_batch.shape
# 
#     trt_engine = "inference_models/edsr.trt"
#     output = trt_inference(trt_engine, input_batch, h)
#     print(output)
#     print(output.shape)
#     print(type(output))
# 
#     output = torch.tensor(output).int()
#     output_folder = "output_images"
#     file_name = "data/test7.jpg".split("/")[-1].split(".")[0]
#     ut.save_image(output[0], output_folder, 120, 120, 4, output_file_name=file_name + "_x4")
# 
# =============================================================================
