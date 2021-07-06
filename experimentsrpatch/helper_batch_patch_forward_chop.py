"""
Helper for batch forward chopping experiment
"""
import sys
import batch_patch_forward_chop as bpfc
import utilities as ut

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/slices/10.npz"
    dimension = int(sys.argv[2]) if len(sys.argv) > 2 else 145
    shave = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    scale = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    print_result = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
    device = str(sys.argv[7]) if len(sys.argv) > 7 else "cuda"
    model_name = str(sys.argv[8]) if len(sys.argv) > 8 else "RRDB"
    
    if model_name == "EDSR":
        input_image = ut.load_image(img_path)
    elif model_name == "RRDB":
        #input_image = ut.load_grayscale_image(img_path)
        input_image = ut.npz_loader(img_path)
        print(input_image.shape)
    else:
        raise Exception('{} : Unknown model...'.format(model_name))
        
    output_image = bpfc.patch_batch_forward_chop(
        input_image, dimension, shave, scale, batch_size, model_type=model_name, print_timer=True
    )
    if print_result:
        c, h, w = input_image.shape
        ut.save_image(output_image, "results/", h, w, scale)
