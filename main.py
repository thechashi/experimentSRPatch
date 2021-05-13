import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import time
import math
import copy
from EDSR import make_model, load

device = 'cuda'
args = {'n_resblocks' : 16,
        'n_feats' : 64,
        'scale' : [4],
        'rgb_range' : 255,
        'n_colors' : 3,
        'res_scale' : 1,
        }

model = make_model(args).to(device)
load(model)
dir(model)

# =============================================================================
# dimension = 256
# B = np.random.random((dimension,dimension))*255.
# # print(B)
# plt.imshow(B)
# plt.show()
# B = torch.tensor(B)
# B.unsqueeze_(0)
# B = B.repeat(3, 1, 1)
# B = B.unsqueeze(0)
# =============================================================================

result2 = {}
for i in range(10):
  dimension = 2
  print('run: ', i)
  while(True):
    # print('Dimension: ', dimension)
    # print(max)
    # print(min)
    # print(dimension)
    B = np.random.random((dimension,dimension))*255.
    # print(B)
    # plt.imshow(B)
    # plt.show()
    B = torch.tensor(B)
    B.unsqueeze_(0)
    B = B.repeat(3, 1, 1)
    B = B.unsqueeze(0)


    B = B.float().to(device)
    with torch.no_grad():
      try:
        start = time.time()
        O = model(B)
        end = time.time()
        total_time = end - start
        # print(total_time)
        # print('-------------------')
        if dimension in result2.keys():
          result2[dimension].append(total_time)
        else:
          result2[dimension] = [total_time]
        min = copy.copy(dimension)
        dimension *= 2
      except RuntimeError as e:
        print(e)
        if dimension in result2.keys():
          result2[dimension].append(math.inf)
        else:
          result2[dimension] = [math.inf]
        break


    O=O.cpu()
    O=O.detach().numpy()
    # plt.imshow(O[0][0])
    # plt.show()
    B = B.cpu()
    # print(B.shape)
    # print(O.shape)
    del B
    del O
    gc.collect()
    torch.cuda.empty_cache()

for k, v in result2.items():
  result2[k] = sum(v)/float(len(v))
print(result2)


# part 2
result = {}

for i in range(10):
  print('run: ', i)
  dimension = 1024
  max = math.inf
  min = -math.inf
  last = 0
  while(True):
    # print('Dimension: ', dimension)
    B = np.random.random((dimension,dimension))*255.
    # print(B)
    # plt.imshow(B)
    # plt.show()
    B = torch.tensor(B)
    B.unsqueeze_(0)
    B = B.repeat(3, 1, 1)
    B = B.unsqueeze(0)
    B = B.float().to(device)
    with torch.no_grad():
      try:
        if last == dimension:
          break
        start = time.time()
        O = model(B)
        end = time.time()
        total_time = end - start
        # print(total_time)
        # print('-------------------')
        last = dimension
        if dimension in result.keys():
          result[dimension].append(total_time)
        else:
          result[dimension] = [total_time]
        min = copy.copy(dimension)
        if max == math.inf:
          dimension *= 2
        else:
          dimension = dimension + (max-min)//2
      except RuntimeError as e:
        #print(e)
        max = copy.copy(dimension)
        if dimension in result.keys():
          result[dimension].append(math.inf)
        else:
          result[dimension] = [math.inf]
        if min == -math.inf:
          dimension = dimension//2
        else:
          dimension = min + (max-min)//2
        continue
        # break


    O=O.cpu()
    O=O.detach().numpy()
    # plt.imshow(O[0][0])
    # plt.show()
    B = B.cpu()
    del B
    del O
    gc.collect()
    torch.cuda.empty_cache()

for k, v in result.items():
  result[k] = sum(v)/float(len(v))
dic_items = result.items()
result = sorted(dic_items)
print(result)

result = dict(result)
z = {**result2, **result}
print(z)

import matplotlib.pylab as plt

r = sorted(z.items()) # sorted by key, return a list of tuples

x, y = zip(*r) # unpack a list of pairs into two tuples
plt.xlabel('Dimension')
plt.ylabel('Average time (10 runs)')
plt.plot(x, y)
plt.show()