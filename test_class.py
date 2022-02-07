from main import NeuralNetwork 
import numpy as np

nn_mutated = NeuralNetwork([2,3,4,2],'xavier')

layers = nn_mutated.layers.copy()
layer_size = np.random.randint(1,max(layers))


exit()
#print(nn_mutated.get_bias())
for i in range(len(nn_mutated.get_bias())):
    print(nn_mutated.get_bias()[i].shape)

print('---------')
for i in range(len(nn_mutated.get_weights())):
    print(nn_mutated.get_weights()[i].shape)


print(nn_mutated.get_weights()[0])