import modelloader as md

model = md.load_edsr(device="cuda")

import torch  

dummy_input = torch.randn(3, 3, 1, 1, device='cuda')

import torch.onnx
torch.onnx.export(model, dummy_input, "edsr.onnx", verbose=False)
