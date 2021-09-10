# -*- coding: utf-8 -*-

import subprocess
import click 
@click.command()
@click.option("--onnx_model_name", default="rrdb.onnx", help="Name of the onnx model")
@click.option("--trt_model_name", default="rrdb.trt", help="Name of the trt model")
def main(onnx_model_name, trt_model_name):
    command = "trtexec --onnx=" +  onnx_model_name + " --saveEngine=inference_models/" + trt_model_name + " --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
    subprocess.run(command, shell=True)
    
main()
    