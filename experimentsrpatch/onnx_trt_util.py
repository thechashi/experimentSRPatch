"""
Source code for turning a PyTorch model to ONNX model and then turning that
ONNX model and then turning that onnx model to TensorRT engine.

Also contians code for inferencing with the TensorRT engine
"""
import sys
import torch
import torch.onnx
import numpy as np
import subprocess
import tensorrt as trt
import pycuda.autoinit
import subprocess
import pycuda.driver as cuda
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


def build_trt_engine(onnx_model, trt_model, use_fp16=False):
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
    print(use_fp16)
    if use_fp16:
        command = (
            "trtexec --onnx="
            + onnx_model
            + " --saveEngine="
            + trt_model
            + " --explicitBatch"
            + " --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
        )
    else:
        command = (
            "trtexec --onnx="
            + onnx_model
            + " --saveEngine="
            + trt_model
            + " --explicitBatch"
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
    input_batch = np.ascontiguousarray(img, dtype=target_dtype)
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
    #     build_onnx_model(model_name="EDSR", patch_size=345, onnx_model_name="edsr.onnx")
    # =============================================================================
    # =============================================================================

    # =============================================================================
    #     # build smaple trt engine
    build_trt_engine(sys.argv[1], sys.argv[2], int(sys.argv[3]))
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
#
# =============================================================================
