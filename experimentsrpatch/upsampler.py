import sys
import toml
import pyfiglet
import batch_patch_forward_chop as bpfc
if __name__ == "__main__":
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "EDSR"
    img_path = sys.argv[2] if len(sys.argv) > 2 else "data/test5.jpg"
    
    banner = pyfiglet.figlet_format("Upsampler: " + model_name)
    
    print(banner)
    
    config = toml.load('../upsampler_config.toml')
    batch_size = config['batch_size']
    patch_dimension = config['patch_dimension']
    patch_shave = config['patch_shave']
    scale = config['scale']
    
    bpfc.upsample(model_name, img_path, patch_dimension, patch_shave, batch_size, scale, device="cuda")
    
    