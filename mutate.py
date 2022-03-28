from ast import operator
from operator import ge
from lib import NeuralNetwork, test_print
import helper
import data
import random
import numpy as np

def muatete_run():
    ### GENERATION CONFIG
    NUMBER_OF_GENERATIONS = 100
    GENERATION_SIZE = 500
    ### NN CONFIG
    STARTING_STRUCTURE = [16,2,2,3,4,5,1]
    HIDDEN_LAYERS_ACTIVIATION = helper.sig
    LAST_LAYER_ACTIVATION = helper.tanh
    ### MUTAION CONFIG
    ## rare mutation means new layer
    ## normal mutation is adjestment to available layer
    CHANCE_OF_RARE_MUTAION = 50
    SELECTION_RATIO = 0.5

    ## chose & mutate Nueral network
    def mutate(generation,old_population_limit):
        ## pick a NN
        NN = generation[random.randint(0,old_population_limit-1)]['Neuralnetwork']
        w_old = NN.get_weights()
        b_old = NN.get_bias()
        w_copy = []
        b_copy = []
        layers_copy = NN.get_layers().copy()
        ### copy weights
        for i in range(len(w_old)):
            w_copy.append(np.copy(w_old[i]))
        ## copy bias
        for i in range(len(b_old)):
            b_copy.append(np.copy(b_old[i]))

        ## chose to mutate weights or bias
        rare_mutation = random.randint(0,100)
        if(rare_mutation < CHANCE_OF_RARE_MUTAION):
            ## layer size is random and between 1 neuron to the biggest available layer + random number
            layer_size = np.random.randint(1,max(layers_copy)+1)
            layer_size = layer_size +  np.random.uniform(0,1) * layer_size
            layer_size = max(1,int(layer_size))
            layer_position = max(1,np.random.randint(0,len(layers_copy)-1))
            ## generate new dummy nn with the desired new layer
            layers_copy.insert(layer_position,layer_size)
            nn_mutated = NeuralNetwork(layers_copy,'xavier')
            ## save new bias
            b_copy.insert(layer_position-1,nn_mutated.get_bias()[layer_position-1])
            ## save new weights
            del w_copy[layer_position-1]
            w_copy.insert(layer_position-1,nn_mutated.get_weights()[layer_position-1])
            w_copy.insert(layer_position,nn_mutated.get_weights()[layer_position])
        else:
            chance = random.randint(0,100)
            ## mutate weights
            if(chance >= 50):  
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
        nn_mutated = NeuralNetwork(layers_copy,'xavier')
        nn_mutated.set_weights(w_copy)
        nn_mutated.set_bias(b_copy)

        generation += [{'Neuralnetwork':nn_mutated,'performace':0}]




    ## saved in in (Neuralnetwork,performace) pairs
    generation = []

    #input,output = data.get_xy_problems(200,-1,1)
    #input,output = data.logical(200)
    input,output = data.fourones(200)


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
        print('Best Neural Network: ',generation[0]['performace']*100,'%','Worst Neural Network: ',generation[-1]['performace']*100,'%')
        print('Best NN structure: ',generation[0]['Neuralnetwork'].get_layers(),'Worst NN structure:: ',generation[-1]['Neuralnetwork'].get_layers())
        NeuralNetwork_layers = generation[0]['Neuralnetwork'].get_layers()

        ### Start mutations & creating generations
        for i in range(to_be_created):
                mutate(generation,survived)
        ### test generation
        for i in range(len(generation)):
                performance = generation[i]['Neuralnetwork'].test(input,output,HIDDEN_LAYERS_ACTIVIATION,LAST_LAYER_ACTIVATION)
                generation[i]['performace'] = performance[0]
        

    ### best fit
    best_nn = generation[0]['Neuralnetwork']


    #input,output = data.get_xy_problems(1000,-1,1)
    input,output = data.fourones(1000)

    print(best_nn.test(input,output,HIDDEN_LAYERS_ACTIVIATION,LAST_LAYER_ACTIVATION)*100)









