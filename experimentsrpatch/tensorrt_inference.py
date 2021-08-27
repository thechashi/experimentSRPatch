import tensorrt as trt
import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import utilities as ut

USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32

# input_batch = ut.random_image(100).numpy()
img_path = "data/test2.jpg"
input_batch = ut.load_grayscale_image(img_path).numpy()

f = open("edsr.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()


# need to set input and output precisions to FP16 to fully enable it
output = np.empty([1, 3, 400, 400], dtype=target_dtype)

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
print(output)
print(type(output))

output = torch.tensor(output).int()
output_folder = "output_images"
file_name = img_path.split("/")[-1].split(".")[0]
ut.save_image(output[0], output_folder, 100, 100, 4, output_file_name=file_name + "_x4")
