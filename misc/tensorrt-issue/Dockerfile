FROM nvcr.io/nvidia/tensorrt:21.07-py3
RUN mkdir /app
COPY requirements.txt /app
WORKDIR /app
RUN python3 -m pip install -r requirements.txt
WORKDIR /workspace/srexp
