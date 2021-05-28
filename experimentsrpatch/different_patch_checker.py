import sys
import toml
import time
import pandas as pd
import numpy as np
import patch_calculator as pc
import matplotlib.pyplot as plt

config = toml.load("../config.toml")
height = int(config['img_height'])
width = int(config['img_width'])
shave = int(config['shave']) if config['shave'] else 10
scale = int(config['scale']) if config['scale'] else 4
filename = config['last_stat_csv']
foldername = config['last_folder']



df = pd.read_csv('results/' +foldername + '/' + filename, comment = '#')
dim_mean_time_list = df[['Dimension', 'Mean Time']]

start_dim = 4*shave 
end_dim = min(height, width, dim_mean_time_list['Dimension'].max())
print(dim_mean_time_list['Dimension'].max())

result_df = dim_mean_time_list.iloc[start_dim-1:end_dim, :].values
for i in range(len(result_df)):
    result_df[i, 1] = result_df[i, 1]*pc.total_patch(result_df[i,0], height, width)

# plots

print('Plotting processing time for patch size from: {0} to {1} for image with shape {2}x{3}'.format(start_dim, end_dim, height, width))
date = "_".join(str(time.ctime()).split())
date = "_".join(date.split(":"))
filename = 'fullimage_' + str(height) + '_' + str(width) + "_" + date
x_data, y_data = np.array(result_df[:, 0]).flatten(), np.array(result_df[:, 1]).flatten()
x_label = 'Patch dimension (1 side) for an image ({}x{})'.format(height, width)
y_label = 'Processing time (sec)'
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.plot(x_data, y_data)
plt.savefig("results/{0}/{1}.png".format(foldername, filename))
plt.show()



    


