import os
import click
import subprocess
import pandas
import toml
def get_filelist_from_folder(folder, file_format):
    filelist = os.listdir(folder)
    for file in filelist[:]:
        if not(file.endswith(file_format)):
            filelist.remove(file)
    return filelist



@click.command()
@click.option("--mode", default="TRT", help="TRT or TORCH model")
@click.option("--model_name", default="EDSR", help="Model name. e.g EDSR, RRDB etc")
@click.option("--trt_path", default=None, help="Path of the trt engine")
@click.option("--use_fp16", default=False, help="Use precision FP16 or FP32")
@click.option("--save_mode", default="npy", help="Save mode: npy, npz, png")
def get_forward_chop_stats(mode, model_name, trt_path, use_fp16, save_mode):
    png_files = get_filelist_from_folder('data/diff_sizes/', '.jpg')
    png_files = ['data/diff_sizes/'+file for file in png_files]
    print('Found {} files.'.format(len(png_files)))
    config = toml.load('../config.toml')
    patch_size = config['max_dim']
    for file in png_files:
        img_path = file
        print('Processing {}...\n'.format(file))
        if mode == "REC":
            patch_size=310
            command = "python3 forward_chop_recursive.py " + \
            " --img_path=" + str(img_path) + \
            " --model_name=" + str(model_name) + \
            " --patch_dimension=" + str(patch_size)
            print(command)
        else:
            command = "python forward_chop.py " + "--mode=" + str(mode) + \
                " --model_name=" + str(model_name) + \
                " --trt_path=" + str(trt_path) + \
                " --img_path=" + str(img_path) + \
                " --patch_size=" + str(patch_size) + \
                " --use_fp16=" + str(use_fp16) + \
                " --save_mode=" + str(save_mode)
                
        print(command)
        p = subprocess.run(command, shell=True, capture_output=True)

        # Valid batch size
        if p.returncode == 0:
            # Get the results of the current batch size
            print(p.stdout.decode())
            #time_stats = list(map(float, p.stdout.decode().split("\n")[0:10]))
        else:
            print('Error executing the process')
        
if __name__ == "__main__":
    get_forward_chop_stats()
    
    
        
