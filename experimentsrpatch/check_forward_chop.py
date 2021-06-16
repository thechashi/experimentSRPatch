import sys
import subprocess
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import utilities as ut
import matplotlib.pyplot as plt


def check_differnet_patches_in_forward_chop(min_dim, max_dim, shave, image_path, run = 1, device="cuda"):
    
    model_name = "EDSR"
    device_name = "CPU"
    total_memory = "~"
    device = device
    if device == "cuda":
        _, device_name = ut.get_device_details()
        total_memory, _, _ = ut.get_gpu_details(
            device, "\nDevice info:", logger=None, print_details=False
        )
    print_result = "0" 
    full_result = []
    break_flag = 0
    for d in tqdm(range(min_dim, max_dim+1)):
        s = [0,0,0,0,0]
        s_t = []
        for r in range(run):
            temp = [d]
            command = "python3 " + "forward_chop.py " + image_path + " " +   str(d) + " "  + str(shave) + " " + print_result + " " + device
            p = subprocess.run(command, shell=True, capture_output = True)
            if p.returncode == 0:
                temp += list(map(float, p.stdout.decode().split()[1:]))
                s = [s[i] + temp[i] for i in range(len(temp))]
            else:
                raise Exception(p.stderr)
                break
        s = np.array(s) / run
        full_result.append(s)
        
        
    full_result = pd.DataFrame(full_result)
    full_result.columns = ['Dimension', 'EDSR Processing Time', 'Cropping time', 'Shifting Time', 'CUDA Cleanign Time']
    
    if device =="cuda":
        plt_title = 'Model: {} | GPU: {} | Memory: {} MB'.format(model_name, device_name, total_memory)
    else:
        plt_title = 'Model: {} | Device: {}'.format(model_name, 'CPU')
    
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    
    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 1].values).flatten(),
    )
    x_label = "Dimension n of patch(nxn)"
    y_label = "Processing time (sec): LR -> SR"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data)
    plt.savefig("results/forward_chop_experiment/{0}.png".format('processing_time_'+date))
    plt.show()
    
    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 2].values).flatten(),
    )
    x_label = "Dimension n of patch(nxn)"
    y_label = "Cropping time (sec): LR -> SR"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data)
    plt.savefig("results/forward_chop_experiment/{0}.png".format('cropping_time_'+date))
    plt.show()
    
    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 3].values).flatten(),
    )
    x_label = "Dimension n of patch(nxn)"
    y_label = "Shifting time (sec): LR -> SR"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data)
    plt.savefig("results/forward_chop_experiment/{0}.png".format('shfiting_time_'+date))
    plt.show()
    
    x_data, y_data = (
        np.array(full_result.iloc[:, 0].values).flatten(),
        np.array(full_result.iloc[:, 4].values).flatten(),
    )
    x_label = "Dimension n of patch(nxn)"
    y_label = "Cleaning time (sec): LR -> SR"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data)
    plt.savefig("results/forward_chop_experiment/{0}.png".format('cuda_cleaning_time_'+date))
    plt.show()
    
    filename = "stat_" + 'EDSR_forward_processing_iterative_' + date
    file = open("results/" + 'forward_chop_experiment' + "/" + filename, "a")
    file.write(device+'\n')
    file.write(device_name+'\n')
    file.write('Memory: ' + str(total_memory) + 'MB\n')
    full_result.to_csv(file)
    file.close()


        
if __name__ == "__main__":
    start_dim = int(sys.argv[1])
    end_dim = int(sys.argv[2])
    shave = int(sys.argv[3])
    image_path = sys.argv[4]
    run = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    device = sys.argv[6] if len(sys.argv) > 6 else "cuda"
    check_differnet_patches_in_forward_chop(start_dim, end_dim, shave, image_path, run, device)
