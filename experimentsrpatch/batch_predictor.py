#!/usr/bin/env python3
from scipy import interpolate
import numpy as np
import scipy.interpolate

sizes = list(range(114, 61, -4))
bsizes = [88, 94, 101, 109, 118, 128, 139, 152, 166, 183, 202, 224, 251, 282]

data = list(zip(sizes, bsizes))


def predict(a, b):
    # returns next output
    p = b[1] - a[1] + b[1]
    return (b[0] - (a[0] - b[0]), p)


for i in range(len(data) - 2):
    output = predict(data[i], data[i + 1])

    print(
        data[i],
        data[i + 1],
        "\t-->\t",
        data[i + 2],
        "\t",
        output,
        "\tError:",
        output[1] - data[i + 2][1],
    )


def predict3(a, b, c):
    x = [a[0], b[0], c[0]]
    y = [a[1], b[1], c[1]]
    xnew = c[0] - (a[0] - b[0])
    #print("Input:", x,y, xnew)
    f = scipy.interpolate.interp1d(x, y, kind="quadratic", fill_value="extrapolate")
    return f(xnew)
    
for i in range(len(data) - 3):
    a,b,c = data[i], data[i+1], data[i+2]
    output = predict3(a,b,c)
    print( "Error: ",  output - data[i+3][1], "\t", output, "\t", data[i+3][1] )