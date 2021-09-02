import torch
import torch.onnx
import modelloader as md
import utilities as ut

model = md.load_edsr(device="cuda")
model.eval()
dummy_input = ut.random_image(100).cuda()
torch.onnx.export(model, dummy_input, "edsr.onnx", verbose=False)
