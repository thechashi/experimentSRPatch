FROM nvcr.io/nvidia/tensorrt:21.07-py3
RUN mkdir /app
COPY requirements.txt /app
WORKDIR /app
RUN python3 -m pip install poetry
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install nvidia-pyindex
RUN python3 -m pip install nvidia-tensorrt
RUN python3 -m pip install 'pycuda<2021.1'
WORKDIR /workspace