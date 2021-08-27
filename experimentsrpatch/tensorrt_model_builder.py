import torch
import torch.onnx
import modelloader as md
import utilities as ut

model = md.load_edsr(device="cuda")
dummy_input = ut.random_image(128).cuda()
torch.onnx.export(model, dummy_input, "edsr.onnx", verbose=False)
