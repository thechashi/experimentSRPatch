import torch
import torch.onnx
import click
import modelloader as md
import utilities as ut
import torch.nn as nn

model = md.load_edsr(device="cuda")
with torch.no_grad():
    model.eval()
    dummy_input = ut.random_image(100).cuda()
    b, c, h, w = 1, 3, 100, 100
    inp = torch.rand(b, c, h, w, requires_grad=True).cuda()
    output = model(inp)
    torch.onnx.export(model,
                      inp,
                      "edsr.onnx",
                      verbose=False,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes = {'input':{0: 'batch', 2:'height', 3:'width'},
                                      'output':{0: 'batch', 2:'height', 3:'width'}
                                      }
                      )

@click.command()
@click.option("--model_name", default="RRDB", help="Name of the model")
@click.option("--onnx_model_name", default="rrdb.onnx", help="Path of the image")
def main(model_name, onnx_model_name):
    device = ut.get_device_type()
    model = None
    if model_name == "RRDB":
        model = md.load_rrdb(device)
        dummy_input = ut.random_image(100, batch=False, channel=1).to(device)
    elif model_name == "EDSR":
        model = md.load_edsr(device)
        dummy_input = ut.random_image(100).to(device)
    model.eval()
    
    torch.onnx.export(model, dummy_input, "inference_models/" + onnx_model_name, verbose=False)
    
        

