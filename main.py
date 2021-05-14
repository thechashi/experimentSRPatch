import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import time
import math
import copy
from tqdm import tqdm
from EDSR import make_model, load

device = 'cuda'
def loadEDSR(n_resblocks = 16, n_feats=64, scale = [4]): 
  args = {'n_resblocks' : n_resblocks,
          'n_feats' : n_feats,
          'scale' : scale,
          'rgb_range' : 255,
          'n_colors' : 3,
          'res_scale' : 1,
          }

  model = make_model(args).to(device)
  load(model)
  dir(model)
  return model

def randomImage(dimension):
    B = np.random.random((dimension,dimension))*255.
    B = torch.tensor(B)
    B.unsqueeze_(0)
    B = B.repeat(3, 1, 1)
    B = B.unsqueeze(0)
    return B.float()

def clearCuda(inputImage, output_image):
  output_image=output_image.cpu()
  inputImage = inputImage.cpu()
  del inputImage
  del output_image
  gc.collect()
  torch.cuda.empty_cache()

model = loadEDSR()

#part 1
print()
print('Getting maximum unacceptable dimension which is a power of two...')
result1 = {}
lastDimension = 0
dimension = 2
while(True):
  input_image = randomImage(dimension)
  input_image = input_image.to(device)
  print('Testing dimension: {0}x{1} ...'.format(dimension,dimension))
  with torch.no_grad():
    try:
      start = time.time()
      output_image = model(input_image)
      end = time.time()
      total_time = end - start
      if dimension in result1.keys():
        result1[dimension].append(total_time)
      else:
        result1[dimension] = [total_time]
      print('Dimension ok.')
      dimension *= 2
      clearCuda(input_image, output_image)
    except RuntimeError as e:
      print('Dimension NOT OK!')
      print('-----------------------------------------------------------')
      print(e)
      if dimension in result1.keys():
        result1[dimension].append(math.inf)
      else:
        result1[dimension] = [math.inf]
      lastDimension = dimension
      clearCuda(input_image, output_image)
      break

for k, v in result1.items():
  result1[k] = sum(v)/float(len(v)) 


# part 2
print()
print('Getting maximum acceptable dimension...')
result2 = {}
dimension = 1024
maxm = math.inf
minm = -math.inf
last = 0
while(True):
  input_image = randomImage(dimension)
  input_image = input_image.to(device)
  print('Testing dimension: {0}x{1} ...'.format(dimension,dimension))
  with torch.no_grad():
    try:
      if last == dimension:
        break
      start = time.time()
      output_image = model(input_image)
      end = time.time()
      total_time = end - start
      last = dimension
      if dimension in result2.keys():
        result2[dimension].append(total_time)
      else:
        result2[dimension] = [total_time]
      minm = copy.copy(dimension)
      print('Dimension ok.')
      if maxm == math.inf:
        dimension *= 2
      else:
        dimension = dimension + (maxm-minm)//2
      clearCuda(input_image, output_image)
    except RuntimeError as e:
      print('Dimension NOT OK!')
      print(e)
      print('-----------------------------------------------------------')
      maxm = copy.copy(dimension)
      if dimension in result2.keys():
        result2[dimension].append(math.inf)
      else:
        result2[dimension] = [math.inf]
      if minm == -math.inf:
        dimension = dimension//2
      else:
        dimension = minm + (maxm-minm)//2
      clearCuda(input_image, output_image)
      continue

for k, v in result2.items():
  result2[k] = sum(v)/float(len(v))
# print(result2) 

dic_items = result2.items()
result2 = sorted(dic_items)
result2 = dict(result2)
# print(result2)

z = {**result1, **result2}
final = {}
for key, value in z.items():
  if value == math.inf:
    continue
  final[key] = value

print('\nValid Results: ')
print(final)

print('\nPlotting detailed results...')
date = "_".join(str(time.ctime()).split())
date = "_".join(date.split(':'))
filename = 'dim2n_vs_time' + "_" + date

r = sorted(final.items())

x, y = zip(*r)
plt.xlabel('Dimension (2^n)')
plt.ylabel('Average time')
plt.plot(x, y)
plt.savefig("figures/{}.png".format(filename))
plt.show()

# part 3
print_every = 50
print('\nPreparing detailed data... ')
result3 = {}
for i in range(10):
  print('Run: ', i+1)
  for d in tqdm(range(1, last+1)):
    dimension = d
    input_image = randomImage(dimension)
    input_image = input_image.to(device)
    # if(d%print_every == 0):
      # print('Testing dimension: {0}x{1} ...'.format(dimension,dimension))
    with torch.no_grad():
      try:
        start = time.time()
        output_image = model(input_image)
        end = time.time()
        total_time = end - start
        if dimension in result3.keys():
          result3[dimension].append(total_time)
        else:
          result3[dimension] = [total_time]
        # if(d%print_every == 0):
          # print('Dimension ok.')
        clearCuda(input_image, output_image)
      except RuntimeError as e:
        # print('Dimension NOT OK!')
        # print('-----------------------------------------------------------')
        print(e)
        if dimension in result3.keys():
          result3[dimension].append(math.inf)
        else:
          result3[dimension] = [math.inf]
        clearCuda(input_image, output_image)
        break

mean_dict = {}
std_dict = {}

for k, v in result3.items():
  mean_dict[k] = np.mean(np.array(v))
print(mean_dict) 

for k, v in result3.items():
  std_dict[k] = np.std(np.array(v))
print(std_dict) 

print('Plotting detailed results...')
date = "_".join(str(time.ctime()).split())
date = "_".join(date.split(':'))
filename = 'dim_vs_meantime' + "_" + date

mean_dict = sorted(mean_dict.items())
x, y = zip(*mean_dict)
plt.xlabel('Dimension')
plt.ylabel('Mean time (10 runs)')
plt.plot(x, y)
plt.savefig("figures/{}.png".format(filename))
plt.show()

date = "_".join(str(time.ctime()).split())
date = "_".join(date.split(':'))
filename = 'dim_vs_stdtime' + "_" + date

std_dict = sorted(std_dict.items())
x, y = zip(*std_dict)
plt.xlabel('Dimension')
plt.ylabel('Std of time (10 runs)')
plt.plot(x, y)
plt.savefig("figures/{}.png".format(filename))
plt.show()