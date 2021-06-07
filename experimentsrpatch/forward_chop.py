import math
import numpy as np
import utilities as ut
import modelloader as md
import subprocess
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
def forward_chop_iterative(x, model = None, shave=10, min_size=6400):
        dim = round(math.sqrt(min_size))
        b, c, h, w = x.size()
# =============================================================================
#         print('type', type(x))
#         print('cuda:', x.is_cuda)
#         print()
# =============================================================================
        device = 'cuda'
        count = 0
        output = torch.tensor(np.zeros((b, c, h*4, w*4)))
# =============================================================================
#         print('input_size: ', x.size())
#         print('output_size: ', output.shape)
# =============================================================================
        
        new_i_s = 0
        for i in range(0, h, dim-2*shave):
            new_j_s = 0
            new_j_e = 0
            print('Loop: ', count)
            print('Before:')
            subprocess.run("gpustat", shell=True)
            for j in range(0, w, dim-2*shave):
                count += 1
                h_s = i
                h_e = min(h, i+dim)
                w_s = j
                w_e = min(w, j+dim)
                lr = x[:, :, h_s:h_e, w_s:w_e]
                lr = lr.to(device)
                sr = model(lr)
                sr = sr.to('cpu')
                #sr = sr.detach().numpy()
# =============================================================================
#                 print('h: {}x{} w: {}x{}'.format(h_s, h_e, w_s, w_e))
#                 print('current dim: {}x{}'.format(h_e-h_s,w_e-w_s))
# =============================================================================
                n_h = (h_e-h_s)*4
                n_w = (w_e-w_s)*4

# =============================================================================
#                 print('next_dimension: {}x{}'.format(n_h, n_w))
# =============================================================================
                n_h_s, n_h_e, n_w_s, n_w_e = 0, 0, 0, 0

                if h_s == 0:
                    n_h_s = 0
                else:
                    n_h_s = shave*4
                if h_e == h:
                    n_h_e = (h_e-h_s)*4

                else:
                    n_h_e = ((h_e-h_s) - shave)*4
                    
                new_i_e = new_i_s + n_h_e - n_h_s
                
                if w_s == 0:
                    n_w_s = 0
                else:
                    n_w_s = shave*4
                if w_e == w:
                    n_w_e = (w_e-w_s)*4
                else:
                    n_w_e = ((w_e-w_s) - shave)*4
                new_j_e = new_j_e + n_w_e - n_w_s 
                output[:, :, new_i_s:new_i_e, new_j_s:new_j_e] \
                        = sr[:, :, n_h_s:n_h_e, n_w_s:n_w_e]
# =============================================================================
#                 print('new -> h: {}x{} w: {}x{}'.format(n_h_s, n_h_e, n_w_s, n_w_e))
#                 print('h-> {}:{}, w-> {}:{}'.format(new_i_s, new_i_e, new_j_s, new_j_e))
#                 print()
# =============================================================================
                if w_e == w:
                    break
                new_j_s = new_j_e
                ut.clear_cuda(lr, None)
            print('After:')
            subprocess.run("gpustat", shell=True)
            print()
            new_i_s = new_i_e
            if h_e == h:
                break
# =============================================================================
#         print(count)
#         print(output.shape)
# =============================================================================
        return output

if __name__ == "__main__":
    dimension = 32
    device = 'cuda'
    img = torchvision.io.read_image('test2.jpg')
    
    c, h, w = img.shape
    img = img.reshape((1,c, h, w))
    plt.imshow(img[0].permute((1,2,0)))
    input_image = img.float()

# =============================================================================
#     model = md.load_edsr(device=device)
#     out = forward_chop_iterative(input_image, model = model)
#     out = out.int()
#     
#     plt.imshow(out[0].permute((1,2,0)))
# =============================================================================

