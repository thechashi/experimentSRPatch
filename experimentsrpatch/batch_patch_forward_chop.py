import sys
import torchvision
import torch
def create_patch_list(patch_list, img, dim, shave, scale, channel, img_height, img_width):
    patch_count = 0
    row_count = 0
    column_count = 0
    
    for patch_height_start in range(0, img_height, dim-2*shave):
        row_count += 1
        right_most = False
        bottom_most = False
        left_increased = 0
        top_increased = 0
        for patch_width_start in range(0, img_width, dim-2*shave):
            column_count += 1
            patch_count += 1
            patch_height_end = min(img_height, patch_height_start+dim)
            patch_width_end = min(img_width, patch_width_start+dim)
            
            if (img_height < patch_height_start+dim):
                bottom_most = True
                old_patch_height_start = patch_height_start
                patch_height_start = img_height - dim
                left_increased = old_patch_height_start - patch_height_start
                
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
                    
            print('Patch no: {}, Row: {}, Column: {}\n'.format(patch_count, row_count, column_count))
            print('{}x{}:{}x{}\n\n'.format(patch_height_start,patch_height_end, patch_width_start, patch_width_end ))
            patch_details = (row_count, column_count, left_crop, top_crop, right_crop, bottom_crop, img[ :, patch_height_start:patch_height_end, patch_width_start:patch_width_end])
            patch_list[patch_count] = patch_details
            
            if patch_width_end == img_width:
                    break
                
        column_count = 0
        if patch_height_end == img_height:
            break
        
    if patch_count == 0:
        raise Exception('Shave size too big for given patch dimension')
    return patch_count


def batch_forward_chop(patch_list, batch_size, model, device='cuda'):
    total_patches = len(patch_list)
    batches = []
    for start in range(1, total_patches + 1, batch_size):
        batch = []
        end = start+batch_size
        if start+batch_size + batch_size > total_patches:
            end = total_patches + 1
        for p in range(start, end):
            batch.append(patch_list[p][6]) 
        batch = torch.stack(batch)
        
        batches.append(batch)
        
        
    return batches


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
    batches = batch_forward_chop(patch_list, 6)
    