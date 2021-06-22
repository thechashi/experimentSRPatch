import sys
import time
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import modelloader as md
def create_patch_list(patch_list, img, dim, shave, scale, channel, img_height, img_width):
    patch_count = 0
    row_count = 0
    column_count = 0
    
    print('LR Image size: {}x{}'.format(img_height, img_width))
    print('SR Image size: {}x{}'.format(img_height*scale, img_width*scale))
    new_i_s = 0 
    for patch_height_start in range(0, img_height, dim-2*shave):
        row_count += 1
        right_most = False
        bottom_most = False
        left_increased = 0
        top_increased = 0
        new_j_s = 0
        new_j_e = 0
        for patch_width_start in range(0, img_width, dim-2*shave):
            column_count += 1
            patch_count += 1
            patch_height_end = min(img_height, patch_height_start+dim)
            patch_width_end = min(img_width, patch_width_start+dim)
            
            if (img_height < patch_height_start+dim):
                bottom_most = True
                old_patch_height_start = patch_height_start
                patch_height_start = img_height - dim
                top_increased = old_patch_height_start - patch_height_start
                
            if (img_width < patch_width_start+dim):
                right_most =True
                old_patch_width_start = patch_width_start
                patch_width_start = img_width - dim
                left_increased = old_patch_width_start - patch_width_start   
                
            left_crop, top_crop, right_crop, bottom_crop = 0, 0, shave*scale, shave*scale
            
            if  patch_width_start != 0:
                if right_most == True:
                    left_crop =  (shave+left_increased) * scale
                else:
                    left_crop =  shave*scale
                    
            if patch_height_start != 0:
                if bottom_most == True:
                    top_crop = (shave+top_increased) * scale
                else:
                    top_crop = shave*scale

            if patch_width_end == img_width:
                right_crop = 0
                
            if patch_height_end == img_height:
                bottom_crop = 0
                    
# =============================================================================
#             print('Patch no: {}, Row: {}, Column: {}\n'.format(patch_count, row_count, column_count))
#             print('{}x{}:{}x{}'.format(patch_height_start, patch_height_end, patch_width_start, patch_width_end ))
#             print('SR Patch size: {}x{}'.format(dim*scale, dim*scale))
# =============================================================================
            
            h_s, h_e, w_s, w_e = 0+ top_crop, dim*scale-bottom_crop, 0+left_crop, dim*scale-right_crop
            new_i_e = new_i_s + h_e - h_s
            new_j_e = new_j_e + w_e - w_s
            patch_crop_positions = [h_s, h_e, w_s, w_e]
            SR_positions = [new_i_s, new_i_e, new_j_s, new_j_e]
# =============================================================================
#             print('Cropped patch position: {}-{}x{}-{}'.format(h_s, h_e, w_s, w_e))
#             print('SR output position: {}-{}x{}-{}\n\n'.format(new_i_s, new_i_e, new_j_s, new_j_e))
# =============================================================================
            patch_details = (row_count, column_count, patch_crop_positions, SR_positions, img[ :, patch_height_start:patch_height_end, patch_width_start:patch_width_end])
            patch_list[patch_count] = patch_details
            new_j_s = new_j_e
            if patch_width_end == img_width:
                    break
        new_i_s = new_i_e        
        column_count = 0
        if patch_height_end == img_height:
            break
        
    if patch_count == 0:
        raise Exception('Shave size too big for given patch dimension')
    return patch_count


def batch_forward_chop(patch_list, batch_size, channel, img_height, img_width, dim, shave, scale,  model, device='cuda'):
    total_patches = len(patch_list)
    output_image = torch.tensor(np.zeros(( channel, img_height*scale, img_width*scale)))
    height_start, width_start = 0, 0
    total_EDSR_time = 0
    for start in range(1, total_patches + 1, batch_size):
        batch = []
        end = start+batch_size
        if start+batch_size + batch_size > total_patches:
            end = total_patches + 1
        for p in range(start, end):
            batch.append(patch_list[p][4]) 
        batch = torch.stack(batch).to(device)
        
        with torch.no_grad():
            # EDSR processing
            start_time = time.time()
            sr_batch = model(batch)
            end_time = time.time()
            processing_time = end_time - start_time
            total_EDSR_time += processing_time
            
        sr_batch = sr_batch.to('cpu')
        _, _, patch_height, patch_width = sr_batch.size()
        
        batch_len = len(sr_batch)
        batch_id = 0
        for p in range(start, end):
# =============================================================================
#             print('Patch: {}'.format(p))
#             print(patch_list[p][3])
#             print(patch_list[p][2])
#             print(sr_batch[batch_id].size())
# =============================================================================
            output_image[:, patch_list[p][3][0]:patch_list[p][3][1], patch_list[p][3][2]:patch_list[p][3][3] ] =  sr_batch[batch_id][:, patch_list[p][2][0]: patch_list[p][2][1], patch_list[p][2][2]: patch_list[p][2][3]]
            batch_id += 1
    print('Total EDSR Processing time: {}\n'.format(total_EDSR_time))
    return output_image
        



if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'data/test2.jpg'
    dimension = int(sys.argv[2])if len(sys.argv) > 2 else 45
    shave = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print_result = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    device =  str(sys.argv[5]) if len(sys.argv) > 5 else 'cuda'
    img = torchvision.io.read_image(img_path)
    
    c, h, w = img.shape
    #img = img.reshape((1, c, h, w))
    #plt.imshow(img[0].permute((1,2,0)))
    input_image = img.float()
    
    patch_list = {}
    create_patch_list(patch_list, input_image, dimension, shave, 4, c, h, w)
    
    model = md.load_edsr(device=device)
    model.eval()
    
    batch_processing_start = time.time()
    output = batch_forward_chop(patch_list, 6, c, h, w, dimension, shave, 4, model = model, device = 'cuda')
    batch_processing_end = time.time()
    
    print('Total batch_processing_time: {}'.format(batch_processing_end-batch_processing_start))
    output = output.int()
    save_start = time.time()
    fig = plt.figure(figsize=((4*h)/1000, (4*w)/1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #fig = plt.figure(figsize=(4*h, 4*w))
    ax.imshow(output.permute((1,2,0)))  
    fig.savefig('result_imagex4.png',bbox_inches='tight',transparent=True, pad_inches=0, dpi=1000)
    save_end = time.time()
    save_time = save_end -save_start