import math
import numpy as np
import utilities as ut
import modelloader as md
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from PIL import Image
def forward_chop_iterative(x, model = None, shave=10, min_size=1024):
        dim = int(math.sqrt(min_size)) # getting patch dimension
        b, c, h, w = x.size() # current image batch, channel, height, width
        device = 'cuda'
        patch_count = 0
        output = torch.tensor(np.zeros((b, c, h*4, w*4)))
        total_time = 0
        x = x.to(device)
        
        new_i_s = 0 
        for i in tqdm(range(0, h, dim-2*shave)):
            new_j_s = 0
            new_j_e = 0
            for j in range(0, w, dim-2*shave):
                patch_count += 1
                h_s, h_e = i, min(h, i+dim) # patch height start and end
                w_s, w_e = j, min(w, j+dim) # patch width start and end
                lr = x[:, :, h_s:h_e, w_s:w_e]
                
                with torch.no_grad():
                    # EDSR processing
                    start = time.time()
                    sr = model(lr)
                    end = time.time()
                    processing_time = end - start
                    total_time += processing_time
                
                # new cropped patch's dimension (h and w)
                n_h_s, n_h_e, n_w_s, n_w_e = 0, 0, 0, 0
                
                n_h_s = 0 if h_s == 0 else (shave*4)
                n_h_e = ((h_e-h_s)*4) if h_e == h else (((h_e-h_s) - shave)*4)
                new_i_e = new_i_s + n_h_e - n_h_s
                
                n_w_s = 0 if w_s ==0 else (shave*4)
                n_w_e = ((w_e-w_s)*4) if w_e == w else (((w_e-w_s) - shave)*4)
                new_j_e = new_j_e + n_w_e - n_w_s 
                
                # corpping image in gpu
                sr_small = sr[:, :, n_h_s:n_h_e, n_w_s:n_w_e]
                sr_small = sr_small.to('cpu')
                output[:, :, new_i_s:new_i_e, new_j_s:new_j_e] = sr_small
                del sr_small

                if w_e == w:
                    break
                new_j_s = new_j_e
                ut.clear_cuda(lr, sr)
            new_i_s = new_i_e
            if h_e == h:
                break
        print('Patch dimension: {}x{}'.format(dim, dim))
        print('Total pacthes: ', patch_count)
        print('Total EDSR Processing time: ', total_time)
        return output

if __name__ == "__main__":
    
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'test2.jpg'
    dimension = int(sys.argv[2])if len(sys.argv) > 2 else 32
    shave = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    
    device = 'cuda'
    img = torchvision.io.read_image(img_path)
    
    c, h, w = img.shape
    img = img.reshape((1,c, h, w))
    plt.imshow(img[0].permute((1,2,0)))
    input_image = img.float()

    model = md.load_edsr(device=device)
    model.eval()
    st = time.time()
    out = forward_chop_iterative(input_image, shave=shave, min_size=dimension*dimension, model = model)
    et = time.time()
    tt = et - st
    print('Total forward chopping time: ', tt)
    out = out.int()
    
    plt.imshow(out[0].permute((1,2,0)))
    plt.savefig('result_imagex4.png', bbox_inches='tight')

