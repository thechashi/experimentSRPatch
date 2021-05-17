import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import time
import math
import copy
from tqdm import tqdm
from EDSR import make_model, load
from pynvml import *
import pandas as pd
import os
def getDeviceDetails():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    device_name = 'cpu'
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(0)
        print('Device: ', device_name)
    print()
    return device, device_name   
        
def getGPUDetails(device, memory_size_format = 'MB', print_details = False):
    power = 2
    if memory_size_format == 'MB':
        power = 2
    elif memory_size_format == 'GB':
        power = 3
    if device == 'cuda':
# =============================================================================
#         t = torch.cuda.get_device_properties(0).total_memory/(1024**power)
#         r = torch.cuda.memory_reserved(0)/(1024**power)
#         c = torch.cuda.memory_reserved(0)/(1024**power)
#         a = torch.cuda.memory_allocated(0)/(1024**power)
#         f = r-a  # free inside reserved
# =============================================================================
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        t = info.total/(1024**power)
        u = info.used/(1024**power)
        f = info.free/(1024**power)
        if print_details:
            print('******************************************************')
            print('\nGPU details:')
            print('Total memory: {0} {1}'.format(t, memory_size_format))
            print('Used memory: {0} {1}'.format(u, memory_size_format))
            print('Free memory: {0} {1}'.format(f, memory_size_format))
            print('****************************************************\n')
        return t, u, f
    else:
        return None

def loadEDSR(device, n_resblocks=16, n_feats=64, scale=[4]):
    args = {
        "n_resblocks": n_resblocks,
        "n_feats": n_feats,
        "scale": scale,
        "rgb_range": 255,
        "n_colors": 3,
        "res_scale": 1,
    }
    """
    """
    model = make_model(args).to(device)
    load(model)
    print('\nModel details: ')
    print(model)
    print()
    return model


def randomImage(dimension):
    B = np.random.random((dimension, dimension)) * 255.0
    B = torch.tensor(B)
    B.unsqueeze_(0)
    B = B.repeat(3, 1, 1)
    B = B.unsqueeze(0)
    return B.float()


def clearCuda(inputImage, output_image):
    if output_image != None:
        output_image = output_image.cpu()
        del output_image
    if inputImage != None:
        inputImage = inputImage.cpu()
        del inputImage
    gc.collect()
    torch.cuda.empty_cache()



def maximumUnacceptableDimension2n(device, model):
    print()
    print("Getting maximum unacceptable dimension which is a power of two...")
    result1 = {}
    lastDimension = 0
    dimension = 2
    while True:
        input_image = randomImage(dimension)
        input_image = input_image.to(device)
        print("Testing dimension: {0}x{1} ...".format(dimension, dimension))
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
                print("Dimension ok.")
                dimension *= 2
                clearCuda(input_image, output_image)
            except RuntimeError as e:
                print("Dimension NOT OK!")
                print("------------------------------------------------------")
                print(e)
                if dimension in result1.keys():
                    result1[dimension].append(math.inf)
                else:
                    result1[dimension] = [math.inf]
                lastDimension = dimension
                output_image = None
                clearCuda(input_image, output_image)
                break
    return lastDimension

def maximumAcceptableDimension(device, model, maxUnacceptableDimension):
    print()
    print("Getting maximum acceptable dimension...")
    result2 = {}
    dimension = maxUnacceptableDimension
    maxm = math.inf
    minm = -math.inf
    last = 0
    while True:
        input_image = randomImage(dimension)
        input_image = input_image.to(device)
        print("Testing dimension: {0}x{1} ...".format(dimension, dimension))
        with torch.no_grad():
            try:
                if last == dimension:
                    clearCuda(input_image, output_image=None)
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
                print("Dimension ok.")
                if maxm == math.inf:
                    dimension *= 2
                else:
                    dimension = dimension + (maxm - minm) // 2
                clearCuda(input_image, output_image)
            except RuntimeError as e:
                print("Dimension NOT OK!")
                print(e)
                print("------------------------------------------------------")
                maxm = copy.copy(dimension)
                if dimension in result2.keys():
                    result2[dimension].append(math.inf)
                else:
                    result2[dimension] = [math.inf]
                if minm == -math.inf:
                    dimension = dimension // 2
                else:
                    dimension = minm + (maxm - minm) // 2
                output_image = None
                clearCuda(input_image, output_image)
                continue
    return last

def resultFromDimensionRange(device, model, first, last, run=10):
    print("\nPreparing detailed data... ")
    result3 = {}
    memoryUsed = {}
    memoryFree = {}
    for i in range(run):
        print("Run: ", i + 1)
        for d in tqdm(range(1, last + 1)):
            dimension = d
            input_image = randomImage(dimension)
            input_image = input_image.to(device)
            with torch.no_grad():
                try:
                    start = time.time()
                    output_image = model(input_image)
                    end = time.time()
                    total_time = end - start
                    if dimension in result3.keys():
                        result3[dimension].append(total_time)
                        t, u, f = getGPUDetails(device, print_details=False)
                        memoryUsed[dimension].append(u)
                        memoryFree[dimension].append(f)
                    else:
                        result3[dimension] = [total_time]
                        t, u, f = getGPUDetails(device, print_details=False)
                        memoryUsed[dimension] = [u]
                        memoryFree[dimension] = [f]
                    clearCuda(input_image, output_image)
                except RuntimeError as e:
                    print("Dimension NOT OK!")
                    print(e)
                    print("--------------------------------------------------")
                    output_image = None
                    clearCuda(input_image, output_image)
                    break
    return result3, memoryUsed, memoryFree

def getMeanStd(result3):
    mean_dict = {}
    std_dict = {}
    
    for k, v in result3.items():
        mean_dict[k] = np.mean(np.array(v))
    
    for k, v in result3.items():
        std_dict[k] = np.std(np.array(v))
        
    return mean_dict, std_dict

def plotData(foldername, filename, data_dict, xLabel, yLabel, mode):
    print("Plotting dimension vs ", mode)
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = filename + "_" + date
    foldername = foldername
    data_dict = sorted(data_dict.items())
    x, y = zip(*data_dict)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(x, y)
    plt.savefig("results/{0}/{1}.png".format(foldername, filename))
    plt.show()

def saveCSV(foldername, filename, device, device_name, total_mem, meanTime,
            stdTime, meanMemUsed, stdMemUsed, meanMemFree, stdMemFree):
    mT = sorted(meanTime.items())
    sT = sorted(stdTime.items())
    mU = sorted(meanMemUsed.items())
    sU = sorted(stdMemUsed.items())
    mF = sorted(meanMemFree.items())
    sF = sorted(stdMemFree.items())
    dimension, meanTime = zip(*mT)
    _, stdTime = zip(*sT)
    _, meanMemUsed = zip(*mU)
    _, stdMemUsed = zip(*sU)
    _, meanMemFree = zip(*mF)
    _, stdMemFree = zip(*sF)
    print('mt', str(len(meanTime)))
    print('st', str(len(stdTime)))
    print('mU', str(len(meanMemUsed)))
    print('sU', str(len(stdMemUsed)))
    print('mF', str(len(meanMemFree)))
    print('sF', str(len(stdMemFree)))
    df = pd.DataFrame({'Dimension': list(dimension),
                       'Mean Time': list(meanTime),
                       'Std Time': list(stdTime),
                       'Mean Memory Used': list(meanMemUsed),
                       'Std Memory Used': list(stdMemUsed),
                       'Mean Memory Free': list(meanMemFree),
                       'Std Memory Free': list(stdMemFree),
                       })
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = filename + "_" + date
    file = open('results/'+foldername+'/'+filename, 'a')
    file.write('# Device: {0} \n'.format(device))
    file.write('# Device Name: {0} \n'.format(device_name))
    file.write('# Total GPU memory: {0} \n'.format(total_mem))
    df.to_csv(file)
    file.close()
    pass

    

def main():
    # device information
    _, device_name = getDeviceDetails()
    device = 'cuda'
    # load model
    run = 10
    clearCuda(None, None)
    print('Before loading model: ')
    total, used, free = getGPUDetails(device, print_details=True)
    print('Total memory: ', total)
    model = loadEDSR(device=device)
    print('After loading model: ')
    total, used, free = getGPUDetails(device, print_details=True)
    # get the highest unacceptable dimension which is a power of 2
    maxUnacceptabelDimension = maximumUnacceptableDimension2n(device, model)
    # get the maximum acceptable dimension
    maxDim = maximumAcceptableDimension(device, model, maxUnacceptabelDimension)
    # get detailed result
    detailedResult, memoryUsed, memoryFree = resultFromDimensionRange(device,
                                                                      model,
                                                                      1,
                                                                      maxDim,
                                                                      run=run)
    # get mean
    # get std
    meanTime, stdTime = getMeanStd(detailedResult)
    meanMemoryUsed, stdMemoryUsed =  getMeanStd(memoryUsed)
    meanMemoryFree, stdMemoryFree =  getMeanStd(memoryFree)
    
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    foldername = date
    os.mkdir('results/'+foldername)
    # plot data
    plotData(foldername, 'dimension_vs_meantime', meanTime, 'Dimension',
             'Mean Time ('+str(run)+' runs)', mode='mean time')
    plotData(foldername, 'dimension_vs_stdtime', stdTime, 'Dimension',
             'Std Time ('+str(run)+' runs)', mode='std time')
    plotData(foldername, 'dimension_vs_meanmemoryused', meanMemoryUsed, 'Dimension',
             'Mean Memory used ('+str(run)+' runs)', mode='mean memory used')
    plotData(foldername, 'dimension_vs_stdmemoryused', stdMemoryUsed, 'Dimension',
             'Std Memory Used ('+str(run)+' runs)', mode='std memory used')
    plotData(foldername, 'dimension_vs_meanmemoryfree', meanMemoryFree, 'Dimension',
             'Mean Memory Free ('+str(run)+' runs)', mode='mean memory free')
    plotData(foldername, 'dimension_vs_stdmemoryfree', stdMemoryFree, 'Dimension',
             'Std Memory Free ('+str(run)+' runs)', mode='std memory free')
    # save data
    saveCSV(foldername, 'total_stat', device, device_name, total, meanTime, stdTime,
            meanMemoryUsed, stdMemoryUsed, meanMemoryFree, stdMemoryFree)
if __name__ == "__main__":
    main()
