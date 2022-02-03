from main import NeuralNetwork 

nn = NeuralNetwork([2,4,1],'xavier')

nn.set_bias(4)
print(nn.get_bias())