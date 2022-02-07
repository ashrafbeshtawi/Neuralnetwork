from ast import operator
from operator import ge
from main import NeuralNetwork, test_print
import helper
import data
import random
import numpy as np

## CONSTANTS
NUMBER_OF_GENERATIONS = 100
GENERATION_SIZE = 1000
SELECTION_RATIO = 0.5
STARTING_STRUCTURE = [2,2,1]
HIDDEN_LAYERS_ACTIVIATION = helper.Relu
LAST_LAYER_ACTIVATION = helper.tanh


## chose & mutate Nueral network
def mutate(generation,old_population_limit):
    ## pick a NN
    NN = generation[random.randint(0,old_population_limit-1)]['Neuralnetwork']
    w_old = NN.get_weights()
    b_old = NN.get_bias()
    w_copy = []
    b_copy = []
    ### copy weights
    for i in range(len(w_old)):
        w_copy.append(np.copy(w_old[i]))
    ## copy bias
    for i in range(len(b_old)):
        b_copy.append(np.copy(b_old[i]))

    ## chose to mutate weights or bias
    chance = random.randint(0,1)
    ## mutate weights
    if(chance == 0):  
        ## pick random layer
        layer = random.randint(0,len(w_copy)-1)
        ## pick connection
        connection = random.randint(0,w_copy[layer].shape[1]-1)
        ## mutating connection
        muation = random.uniform(-0.1, 0.1)
        connection_value = w_copy[layer][0,connection]
        w_copy[layer][0,connection] = connection_value + muation * connection_value
    ## mutate bias
    else:
        ## pick random layer
        layer = random.randint(0,len(b_copy)-1)
        ## pick neuron
        neuron = random.randint(0,b_copy[layer].shape[0]-1)
        ## mutating bias
        muation = random.uniform(-0.1, 0.1)
        random_push = random.uniform(-0.1, 0.1)
        bias_value = b_copy[layer][neuron]
        b_copy[layer][neuron] = bias_value + muation * bias_value + random_push
    
    ## save new NN
    nn_mutated = NeuralNetwork([],'xavier')
    nn_mutated.set_weights(w_copy)
    nn_mutated.set_bias(b_copy)

    generation += [{'Neuralnetwork':nn_mutated,'performace':0}]




## saved in in (Neuralnetwork,performace) pairs
generation = []

input,output = data.get_xy_problems(200,-1,1)
## generate first generation
for i in range(GENERATION_SIZE):
    nn = NeuralNetwork(STARTING_STRUCTURE,'xavier')
    performance = nn.test(input,output,HIDDEN_LAYERS_ACTIVIATION,LAST_LAYER_ACTIVATION)
    ## save generation with performace
    generation.append({'Neuralnetwork':nn,'performace':performance[0]})

for i in range(NUMBER_OF_GENERATIONS):
    ### sorting the generation
    generation.sort(key=lambda x:x['performace'],reverse = True)

    ### calculte the survived ratio & the to be created networks
    survived = int(SELECTION_RATIO * len(generation))
    to_be_created = len(generation) - survived
    ### terminating bed networks
    generation = generation[0:survived]

    ### print results
    print('Generation:',i)
    print('Best Fix: ',generation[0]['performace']*100,'Worst Fix: ',generation[-1]['performace']*100)
    ### Start mutations & creating generations
    for i in range(to_be_created):
            mutate(generation,survived)
    ### test generation
    for i in range(len(generation)):
            performance = generation[i]['Neuralnetwork'].test(input,output,HIDDEN_LAYERS_ACTIVIATION,LAST_LAYER_ACTIVATION)
            generation[i]['performace'] = performance[0]
    

### best fit
best_nn = generation[0]['Neuralnetwork']
#test_print(best_nn.get_weights(),best_nn.get_bias(),input,output,HIDDEN_LAYERS_ACTIVIATION,LAST_LAYER_ACTIVATION)


input,output = data.get_xy_problems(1000,-1,1)
print(best_nn.test(input,output,HIDDEN_LAYERS_ACTIVIATION,LAST_LAYER_ACTIVATION)*100)









