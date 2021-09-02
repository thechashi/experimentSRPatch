import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import utilities as ut
import numpy as np
def build_engine(model_file, max_ws=512*1024*1024, fp16=False):
    print("building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.\
                                                  EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            #last_layer = network.get_layer(network.num_layers - 1)
            #network.mark_output(last_layer.get_output(0))
            engine = builder.build_engine(network, config=config)
            return engine
engine = build_engine("edsr.onnx")

with open('edsr.trt', 'wb') as f:
    f.write(bytearray(engine.serialize()))
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)    
runtime = trt.Runtime(TRT_LOGGER)
with open('edsr.trt', 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    edsr_context = engine.create_execution_context()
    # input_batch = ut.random_image(100).numpy()
    img_path = "data/test2.jpg"
    input_batch = ut.npz_loader(img_path).numpy()
    # need to set input and output precisions to FP16 to fully enable it
    output_d = np.empty([1, 3, 400, 400], dtype=np.float32)
    '''
    memory allocation for inputs
    '''
    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    '''memory allocation for outputs'''
    d_output = cuda.mem_alloc(1 * output_d.nbytes)
    
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()
    
    
    def predict(batch):  # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream) 
        # execute model
        edsr_context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()
    
        return output
    
    
    output = predict(input_batch)
    print(output)
    print(type(output))
    print(output.shape)
    print(output.max())
    print(output.min())
