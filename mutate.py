from ast import operator
from operator import ge
import main
import data
import random
import numpy as np
NUMBER_OF_GENERATION = 50
GENERATION_SIZE = 200
SELECTION_RATIO = 0.5
MUTATION_RATIO = 1
NUMBERS_PER_NETWORK = 100


## chose & mutate Nueral network
def mutate(generation,old_population_limit):
    ## pick a NN
    NN = generation[random.randint(0,old_population_limit-1)]
    w_copy = []
    b_copy = []
    ### copy weights
    for i in range(len(NN['weights'])):
        w_copy.append(np.copy(NN['weights'][i]))
    ## copy bias
    for i in range(len(NN['bias'])):
        b_copy.append(np.copy(NN['bias'][i]))



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
    generation += [{'weights':w_copy,'bias':b_copy,'performace':0}]


## create copies (kids) of NN
def copy(generation,old_population_limit):
    ## pick a NN
    NN = generation[random.randint(0,old_population_limit-1)]
    w_copy = NN['weights'].copy()
    b_copy = NN['bias'].copy()
    generation += [{'weights':w_copy,'bias':b_copy,'performace':0}]


## saved in in (w,b,performace) pairs
generation = []

input,output = data.get_xy_problems(200,-1,1)
## generate first generation
for i in range(GENERATION_SIZE):
    w, b =main.get_component([2,2,1],'xavier')
    performance = main.test(w,b,input,output,main.Relu,main.sig)
    ## save generation with performace
    generation.append({'weights':w,'bias':b,'performace':performance[0]})

for i in range(NUMBER_OF_GENERATION):
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
        chance = random.uniform(0, 1)
        if (chance <= MUTATION_RATIO):
            mutate(generation,survived)
        else:
            copy(survived)
    ### test generation
    for i in range(len(generation)):
            performance = main.test(generation[i]['weights'],generation[i]['bias'],input,output,main.Relu,main.sig)
            generation[i]['performace'] = performance[0]
    

main.test_print(generation[0]['weights'],generation[i]['bias'],input,output,main.Relu,main.sig)








