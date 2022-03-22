import matplotlib.pyplot as plt 
import matplotlib.animation as ani
from main import NeuralNetwork 

##dummy nn
nn = NeuralNetwork([2,3,7,6,5,5,6],'xavier')
##
PLOT_HEIGHT = 1000
PLOT_WIDTH = 1000
fig = plt.figure() 
plt.xlim(0, PLOT_WIDTH)
plt.ylim(0, PLOT_HEIGHT)
graph, = plt.plot([], [], 'o')



def anim(i):
    layers = nn.get_layers()
    x = []
    y = []
    ## draw neurons
    current_x = 0
    distance = PLOT_WIDTH/(len(layers)+1)
    for i in range(len(layers)):
        current_x = current_x + distance
        x = x + [current_x]*layers[i]
    
    for i in range(len(layers)):
        distance = PLOT_HEIGHT/(layers[i]+1)
        current_y = 0
        for j in range(layers[i]):
            current_y = current_y + distance
            y = y + [current_y]

    ## draw connections
    for i in range(len(layers)-1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                x_values = [x[sum(layers[0:i])+j], x[sum(layers[0:i+1])+k]]
                y_values = [y[sum(layers[0:i])+j], y[sum(layers[0:i+1])+k]]
                plt.plot(x_values, y_values, color = 'red', linestyle="-")


    graph.set_data(x,y)
    return graph
    

animator = ani.FuncAnimation(fig, anim, interval = 1000)


plt.show()