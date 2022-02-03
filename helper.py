import numpy as np
import random
from matplotlib import pyplot as plt

## sigmoid activion
def sig(num):
    if -np.min(num) > np.log(np.finfo(type(1.1)).max):
        return np.zeros(num.shape)  
    return np.divide(1,np.add(1,np.exp(-num)))

## derviative of sigmoid
def sig_deriv(num):
    res=sig(num)
    return np.multiply(res,np.subtract(1,res))

## tahn
def tanh(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
## RelU
def Relu(num):
    return np.where(num>0,num,0)


## Relu derivative
def Relu_deriv(num):
    num=np.where(num>0,num,0)
    num=np.where(num==0,num,1)

    return num

## leaky RelU
def L_Relu(num):
    return np.where(num>0,num,0.01*num)

## leaky RelU deriv
def L_Relu_D(num):
    num=np.where(num<=0,num,1)
    num=np.where(num>=0,num,0.01)

    return num

