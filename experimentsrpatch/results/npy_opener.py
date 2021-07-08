import numpy as np
data = np.load('outputx4.npy')
print("The image is: ")
print(data)
print('Output data type: ', type(data.dtype))

from matplotlib import pyplot as plt

plt.imshow(data[0,:,:], cmap='gray')
plt.show()

