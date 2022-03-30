import matplotlib.pyplot as plt 
import matplotlib.animation as ani

PLOT_HEIGHT = 1000
PLOT_WIDTH = 1000
fig = plt.figure(figsize=(8, 6)) 
plt.xlim(0, PLOT_WIDTH)
plt.ylim(0, PLOT_HEIGHT)
graph, = plt.plot([], [], 'o', markersize=12)

def runGraph(shared_Neural_Network, plot_interval):
    animator = ani.FuncAnimation(fig, anim, fargs= (shared_Neural_Network,), interval = plot_interval)
    plt.show()

def anim(i, shared_Neural_Network):
    if (shared_Neural_Network == {}) :
        return
    fig.clear()
    layers = shared_Neural_Network['Neuralnetwork'].get_layers()
    weights =shared_Neural_Network['Neuralnetwork'].get_weights()
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
    plt.plot(x, y, 'o', markersize=12)
    plt.title('Generation: '+str(shared_Neural_Network['generation'])+' \n Accuracy: '+str(shared_Neural_Network['performace']))
    ## draw connections
    for i in range(len(layers)-1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                x_values = [x[sum(layers[0:i])+j], x[sum(layers[0:i+1])+k]]
                y_values = [y[sum(layers[0:i])+j], y[sum(layers[0:i+1])+k]]
                
                if (weights[i][0][j]>0) :
                    plt.plot(x_values, y_values, color = 'blue', linestyle="-")
                else:
                    plt.plot(x_values, y_values, color = 'red', linestyle="-")

    return None
    


