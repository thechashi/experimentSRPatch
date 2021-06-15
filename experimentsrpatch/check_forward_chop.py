import sys
import subprocess
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_differnet_patches_in_forward_chop(min_dim, max_dim, shave, image_path):
    full_result = []
    for d in range(min_dim, max_dim+1):
        print(d)
        s = [d]
        p = subprocess.run("python3 forward_chop.py test2.jpg 40 10 0", shell=True, capture_output = True)
        s += list(map(float, p.stdout.decode().split()[1:]))
        full_result.append(s)
    full_result = pd.DataFrame(full_result)
    full_result.columns = ['Dimension', 'EDSR Processing Time', 'Cropping time', 'Shifting Time', 'CUDA Cleanign Time']
    plt_title = 'EDSR Processing time'
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
    #plt.savefig("results/{0}/{1}.png".format(foldername, filename))
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
    #plt.savefig("results/{0}/{1}.png".format(foldername, filename))
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
    #plt.savefig("results/{0}/{1}.png".format(foldername, filename))
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
    #plt.savefig("results/{0}/{1}.png".format(foldername, filename))
    plt.show()


        
if __name__ == "__main__":
    check_differnet_patches_in_forward_chop(40,100, 10, 'test2.jpg')
