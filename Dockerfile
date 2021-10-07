FROM nvcr.io/nvidia/tensorrt:21.09-py3
RUN mkdir /app
COPY requirements.txt /app
WORKDIR /app
RUN python3 -m pip install -r requirements.txt
WORKDIR /workspace/srexp
#CMD python3 tensorrt_model_builder.py
#CMD trtexec --onnx=edsr.onnx --saveEngine=edsr.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
#CMD python3 tensorrt_inference.py


