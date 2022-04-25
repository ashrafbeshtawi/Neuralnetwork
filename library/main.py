import _thread

from numpy import outer
import visual as visual
import mutate as mutate
import helper as helper
import data as data
import MNIST

### Data visualisation Config
SHOW_VISUAL = True
PRINT_RESULTS = True
PLOT_INTERVAL = 1500
### GENERATION CONFIG
NUMBER_OF_GENERATIONS = 100
GENERATION_SIZE = 500
### NN CONFIG
HIDDEN_LAYERS_SHAPE = 'random' ### list like [16,1,1] OR 'random' for random shapes
MAX_HIDDEN_LAYERS = 10 ## only if hidden layers shape = random
MAX_NEURONS_IN_HIDDEN_LAYERS = 5 ## only if hidden layers shape = random
HIDDEN_LAYERS_ACTIVIATION = helper.sig
LAST_LAYER_ACTIVATION = helper.tanh
### MUTAION CONFIG
## rare mutation means new layer
## normal mutation is adjestment to available layer
CHANCE_OF_RARE_MUTAION = 50
SELECTION_RATIO = 0.5

### Problem to be solved
input, output = MNIST.getMNIST(10)
#input,output = data.get_xy_problems(200,-1,1)
#input,output = data.logical(200)
#input,output = data.fourones(200)
#### Main Section
shared_Neural_Network = {}

if (SHOW_VISUAL): 
    _thread.start_new_thread(
    mutate.muatete_run,
        (
        shared_Neural_Network, input, output, HIDDEN_LAYERS_SHAPE,
        GENERATION_SIZE, NUMBER_OF_GENERATIONS, HIDDEN_LAYERS_ACTIVIATION,
        LAST_LAYER_ACTIVATION, SELECTION_RATIO, CHANCE_OF_RARE_MUTAION,
        PRINT_RESULTS,
        MAX_HIDDEN_LAYERS,
        MAX_NEURONS_IN_HIDDEN_LAYERS
        )
    )   
    visual.runGraph(shared_Neural_Network, PLOT_INTERVAL)
else:
    mutate.muatete_run(
        shared_Neural_Network, input, output, HIDDEN_LAYERS_SHAPE,
        GENERATION_SIZE, NUMBER_OF_GENERATIONS, HIDDEN_LAYERS_ACTIVIATION,
        LAST_LAYER_ACTIVATION, SELECTION_RATIO, CHANCE_OF_RARE_MUTAION,
        PRINT_RESULTS,
        MAX_HIDDEN_LAYERS,
        MAX_NEURONS_IN_HIDDEN_LAYERS
          )
      

