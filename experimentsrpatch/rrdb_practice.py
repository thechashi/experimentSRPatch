import torch
import modelloader as md
import utilities as ut

model = md.load_rrdb(device="cuda")

img = ut.npz_loader("data/slices/0.npz")

img = img[:, 0:100, 0:100]
img = img.unsqueeze(0)
img = img.to("cuda")

model.eval()
with torch.no_grad():
    output = model(img)
