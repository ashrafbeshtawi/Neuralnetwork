from main import NeuralNetwork 
import numpy as np
import helper

nn = NeuralNetwork([1,2,3,4],'xavier')
for i in range(100):
    ## copy the layers sizes & the max layer size
    layers = nn.layers.copy()
    ## layer size is random and between 1 neuron to the biggest available layer +- 10%
    layer_size = np.random.randint(1,max(layers)+1)
    layer_size = layer_size +  np.random.uniform(0,1) * layer_size
    layer_size = max(1,int(layer_size))
    layer_position = max(1,np.random.randint(0,len(layers)))

    layers.insert(layer_position,layer_size)
    nn_mutated = NeuralNetwork(layers,'xavier')
    ## save new bias
    nn.get_bias().insert(layer_position-1,nn_mutated.get_bias()[layer_position-1])
    ## save new weights
    del nn.get_weights()[layer_position-1]
    nn.get_weights().insert(layer_position-1,nn_mutated.get_weights()[layer_position-1])
    nn.get_weights().insert(layer_position,nn_mutated.get_weights()[layer_position])
    ## save new layer shape
    nn.layers = layers
    nn.predict([1],helper.Relu,helper.sig)
    print(i)

##copy weights
print(layers,nn.layers)
for i in range(len(nn.weights)):
    print(nn.weights[i].shape)
print('---------')
for i in range(len(nn_mutated.weights)):
    print(nn_mutated.weights[i].shape)
print('*********')
##copy weights
for i in range(len(nn.bias)):
    print(nn.bias[i].shape)
print('---------')
for i in range(len(nn_mutated.bias)):
    print(nn_mutated.bias[i].shape)  

exit()
## get the layer and weights on this position
bias = nn_mutated.get_bias()
Weight = nn_mutated.get_weights()

print(Weight)

#print(layers, layer_size, layer_position)
exit()
#print(nn_mutated.get_bias())
for i in range(len(nn_mutated.get_bias())):
    print(nn_mutated.get_bias()[i].shape)

print('---------')
for i in range(len(nn_mutated.get_weights())):
    print(nn_mutated.get_weights()[i].shape)


print(nn_mutated.get_weights()[0])