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

def create_patch_queue(queue, img, dim, shave, img_height, img_width):
    patch_count = 0
    for patch_height_start in range(0, img_height, dim-2*shave):
        if (img_height < patch_height_start+dim):
            break
        for patch_width_start in range(0, img_width, dim-2*shave):
            if (img_width < patch_width_start+dim):
                break
            patch_count += 1
            patch_height_end = min(img_height, patch_height_start+dim)
            patch_width_end = min(img_width, patch_width_start+dim)
            print('{}x{}:{}x{}'.format(patch_height_start,patch_height_end, patch_width_start, patch_width_end ))
            #patch = img[:, :, patch_height_start:patch_height_end, patch_width_start:patch_width_end]
            #queue.append(patch)
            queue.append(img[:, :, patch_height_start:patch_height_end, patch_width_start:patch_width_end][0])
            if patch_width_end == img_width:
                    break
        if patch_height_end == img_height:
            break
    if patch_count == 0:
        raise Exception('Shave size too big for given patch dimension')
    return patch_count
    
def forward_chop_iterative(x, model = None, shave=10, min_size=1024, device = 'cuda', print_result=True):
    dim = int(math.sqrt(min_size)) # getting patch dimension
    b, c, h, w = x.size() # current image batch, channel, height, width
    device = device
    patch_count = 0
    output = torch.tensor(np.zeros((b, c, h*4, w*4)))
    total_time = 0
    total_crop_time = 0
    total_shift_time = 0
    total_clear_time = 0
    if device == "cuda":
        x = x.to(device)
    
    new_i_s = 0 
    for i in range(0, h, dim-2*shave):
        new_j_s = 0
        new_j_e = 0
        # CHECKING 
        if (h < i+dim):
            break
        for j in range(0, w, dim-2*shave):
            # CHECKING
            if (w < j+dim):
                break
            patch_count += 1
            h_s, h_e = i, min(h, i+dim) # patch height start and end
            w_s, w_e = j, min(w, j+dim) # patch width start and end
            print('Patch no: {} : {}-{}x{}-{}\n'.format(patch_count, h_s, h_e, w_s, w_e))
            
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
            
            # corpping image in 
            crop_start = time.time()
            sr_small = sr[:, :, n_h_s:n_h_e, n_w_s:n_w_e]
            crop_end = time.time()
            crop_time = crop_end - crop_start
            total_crop_time += crop_time
            
            shift_start = time.time()
            if device == "cuda":
                sr_small = sr_small.to('cpu')
            shift_end = time.time()
            shift_time = shift_end - shift_start
            total_shift_time += shift_time
            output[:, :, new_i_s:new_i_e, new_j_s:new_j_e] = sr_small
            del sr_small

            if w_e == w:
                break
            new_j_s = new_j_e
            clear_start = time.time()
            if device == "cuda":
                ut.clear_cuda(lr, sr)
            clear_end = time.time()
            clear_time = clear_end - clear_start
            total_clear_time += clear_time
        new_i_s = new_i_e

        if h_e == h:
            break
    if print_result == True:
        print('Patch dimension: {}x{}'.format(dim, dim))
        print('Total pacthes: ', patch_count)
        print('Total EDSR Processing time: ', total_time)
        print('Total crop time: ', total_crop_time)
        print('Total shift time: ', total_shift_time)
        print('Total clear time: ', total_clear_time)
    return output, total_time, total_crop_time, total_shift_time, total_clear_time

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'data/test2.jpg'
    dimension = int(sys.argv[2])if len(sys.argv) > 2 else 50
    shave = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print_result = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    device =  str(sys.argv[5]) if len(sys.argv) > 5 else 'cuda'
    img = torchvision.io.read_image(img_path)
    
    c, h, w = img.shape
    img = img.reshape((1,c, h, w))
    #plt.imshow(img[0].permute((1,2,0)))
    input_image = img.float()
    
    Q = []
    create_patch_queue(Q, input_image, dimension, shave, h, w)
    batch_one = torch.stack(Q[:2]).to(device)
    batch_two = torch.stack(Q[2:]).to(device)
    
    #Q = torch.stack(Q)
    model = md.load_edsr(device=device)
    model.eval()
    
    with torch.no_grad():
        # EDSR processing
        start = time.time()
        sr_batch = model(batch_one)
        end = time.time()
        processing_time = end - start

    
    sr_batch = sr_batch.cpu().int()

    fig = plt.figure(figsize=((4*h)/1000, (4*w)/1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #fig = plt.figure(figsize=(4*h, 4*w))
    ax.imshow(sr_batch[0].permute((1,2,0)))  
    fig.savefig('result_imagex4_1.png',bbox_inches='tight',transparent=True, pad_inches=0, dpi=1000)
    fig = plt.figure(figsize=((4*h)/1000, (4*w)/1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #fig = plt.figure(figsize=(4*h, 4*w))
    ax.imshow(sr_batch[1].permute((1,2,0)))  
    fig.savefig('result_imagex4_1.png',bbox_inches='tight',transparent=True, pad_inches=0, dpi=1000)
# =============================================================================
#     st = time.time()
#     out_tuple = forward_chop_iterative(input_image, shave=shave, min_size=dimension*dimension, model = model, device = device, print_result=print_result)
#     out = out_tuple[0]
#     et = time.time()
#     tt = et - st
# =============================================================================
    
# =============================================================================
#     if print_result == True:
#         print('Total forward chopping time: ', tt)
#         print('\nSaving...\n')
#         out = out.int()
#         save_start = time.time()
#         fig = plt.figure(figsize=((4*h)/1000, (4*w)/1000), dpi=100, frameon=False)
#         ax = plt.Axes(fig, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         fig.add_axes(ax)
#         #fig = plt.figure(figsize=(4*h, 4*w))
#         ax.imshow(out[0].permute((1,2,0)))  
#         fig.savefig('result_imagex4.png',bbox_inches='tight',transparent=True, pad_inches=0, dpi=1000)
#         save_end = time.time()
#         save_time = save_end -save_start
#         print('Saving time: {}'.format(save_time))
#     else:
#         print(dimension, *out_tuple[1:], )
# =============================================================================
