import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import time
import math
import copy
from tqdm import tqdm
from EDSR import make_model, load




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
    dir(model)
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
                print("-----------------------------------------------------------")
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
                print("-----------------------------------------------------------")
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
    for i in range(run):
        print("Run: ", i + 1)
        for d in tqdm(range(1, last + 1)):
            dimension = d
            input_image = randomImage(dimension)
            input_image = input_image.to(device)
            print(dimension)
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
                    clearCuda(input_image, output_image)
                except RuntimeError as e:
                    print("Dimension NOT OK!")
                    print(e)
                    print("-----------------------------------------------------------")
                    if dimension in result3.keys():
                        result3[dimension].append(math.inf)
                    else:
                        result3[dimension] = [math.inf]
                    output_image = None
                    clearCuda(input_image, output_image)
                    break
    return result3

def getMeanStd(result3):
    mean_dict = {}
    std_dict = {}
    
    for k, v in result3.items():
        mean_dict[k] = np.mean(np.array(v))
    
    for k, v in result3.items():
        std_dict[k] = np.std(np.array(v))
        
    return mean_dict, std_dict

def plotData(filename, data_dict, xLabel, yLabel, mode):
    print("Plotting dimension vs ", mode)
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    filename = filename + "_" + date
    
    data_dict = sorted(data_dict.items())
    x, y = zip(*data_dict)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(x, y)
    plt.savefig("results/{}.png".format(filename))
    plt.show()

def saveCSV():
    pass

def main():
    # load model
    device = "cuda"
    run = 10
    model = loadEDSR(device=device)
    # get the highest unacceptable dimension which is a power of 2
    maxUnacceptabelDimension = maximumUnacceptableDimension2n(device, model)
    # get the maximum acceptable dimension
    maxDim = maximumAcceptableDimension(device, model, maxUnacceptabelDimension)
    # get detailed result
    detailedResult = resultFromDimensionRange(device, model, 1, maxDim, run=run)
    # get mean
    # get std
    meanResult, stdResult = getMeanStd(detailedResult)
    # plot data
    plotData('dimension_vs_meantime', meanResult, 'Dimension', 'Mean Time ('+str(run)+' runs)', mode='mean time')
    plotData('dimension_vs_std', meanResult, 'Dimension', 'Std Time ('+str(run)+' runs)', mode='std time')
    # save data
    
if __name__ == "__main__":
    main()
