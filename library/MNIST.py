from torch import utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def getMNIST(number_of_samples):
    # Transform PIL image into a tensor. The values are in the range [0, 1]
    t = transforms.ToTensor()

    # Load datasets for training and apply the given transformation.
    mnist = datasets.MNIST(root='data', train=True, download=True, transform=t)
    input = []
    output = []

    for i in range(number_of_samples):
        output_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        output_array[mnist[i][1]] = 1
        input.append(np.array(mnist[0][0][0]).flatten().T)
        output.append(output_array)
    return np.array(input), np.array(output)


