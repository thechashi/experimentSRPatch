import tensorrt as trt
import torch
import numpy as np
import click
import pycuda.driver as cuda
import utilities as ut

USE_FP16 = True


def predict(input_img, trt_engine_path, scale):
    """
    Inference through trt engine

    Parameters
    ----------
    input_img : TYPE
        DESCRIPTION.
    trt_engine_path : TYPE
        DESCRIPTION.
    scale : TYPE
        DESCRIPTION.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    target_dtype = np.float16 if USE_FP16 else np.float32

    input_batch = input_img

    # input_batch = ut.npz_loader(img_path).numpy()
    # input_batch = ut.load_image(img_path).unsqueeze(0).numpy()
    input_batch = np.ascontiguousarray(input_batch, dtype=np.float16)
    b, c, h, w = input_batch.shape
    f = open(trt_engine_path, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # need to set input and output precisions to FP16 to fully enable it
    output = np.empty([b, c, scale * h, scale * w], dtype=target_dtype)

    # allocate device memory
    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # transfer input data to device
    cuda.memcpy_htod_async(d_input, input_batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output


@click.command()
@click.option(
    "--trt_engine_path", default="inference_models/edsr.trt", help="Path of trt engine"
)
@click.option("--input_img", help="Input for the trt engine")
@click.option("--scale", default=4, help="Scalar for enlarging the image")
def main(input_img, trt_engine_path, scale):
    input_batch = ut.load_image(input_img).unsqueeze(0).numpy()
    output = predict(input_batch, trt_engine_path, scale)
    print(output)
    print(type(output))
    print(output.shape)


if __name__ == "__main__":
    main()
