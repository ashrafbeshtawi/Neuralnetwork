import random
import numpy as np

## returns x,y as input
## and result as 1 for both x & y positive
## else 0
def get_xy_problems(number_of_samples,from_range,to_range):
    input = []
    output = []
    for i in range(number_of_samples):
        x = random.uniform(from_range,to_range)
        y = random.uniform(from_range,to_range)
        input.append([x,y])
        if (x < 0 and y > 0):
            output.append([1])
        else:
            output.append([0]) 
    return np.array(input),np.array(output)



