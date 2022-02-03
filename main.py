import numpy as np
import random
from matplotlib import pyplot as plt


class NeuralNetwork:
    ## init the neural network
    def get_component(self,layers,init_type):
        created_layers=[]
        bias=[]
        ##create weight matricies of layers 
        for i in range(len(layers)-1):
            m=layers[i+1]
            n=layers[i]
            ## init type
            layer=None
            if init_type=="xavier":
                layer=np.random.normal(0, m, (m,n))
            elif init_type=="random_x":
                layer=np.multiply(np.random.rand(m,n),np.sqrt(2/m)) 
            elif init_type=="random":
                layer=np.random.rand(m,n)

            created_layers.append(layer)
        
        ## create bias
        for i in range(1,len(layers)):
            n=layers[i]
            ## init type
            if init_type=="xavier":
                layer=np.zeros(n)
            elif init_type=="random":
                layer=np.zeros(n)

            bias.append(layer)

        return created_layers,bias

    def __init__(self,layers, init_type) -> None:
        self.weights, self.bias = self.get_component(layers,init_type)
    
    def get_weights(self):
        return self.weights

    def set_weights(self,weights):
        self.weights = weights 

    def get_bias(self):
        return self.bias 

    def set_bias(self,bias):
        self.bias = bias        

    #predict : makre prediction and return activation of each layer
    def predict(input,weights,bias,activation_func,last_layer_activ_func):
        activation=input
        z=[]
        layer_activation=[input]
        for i in range(len(weights)):
            z_temp=np.add(np.dot(weights[i],activation),bias[i])

            if(i==len(weights)-1):
                activation=last_layer_activ_func(z_temp)
            else:
                activation=activation_func(z_temp)

            z.append(z_temp)
            layer_activation.append(activation)
        return layer_activation,z
#print component of NT
def print_component(weights,bias):
    for i in range(len(weights)):
        print("weight: \n",weights[i],end="   ")
        #if(i==0):
        #    print(weights[0].shape[1]*["input layer"])
        #elif(i==len(weights)-1):
        #    print(bias[i-1])
        #    print("-----------")
        #    print(weights[len(weights)-1].shape[1]*["ouput layer"])
        #else:
        print("bias: ",bias[i])

        print("-----------")










# cost function
def cost_calculate(prediction,correct):
    return np.divide(np.power(np.subtract(correct,prediction),2),2)



def train(input,output,weights_i,bias_i,activation_func,deriv,last_layer_activ_func,last_layer_activ_func_deriv,epoch_size,iterations,learning_rate):

    weights=weights_i
    bias=bias_i
    ##reads input
    x,y=input.shape
    a,b=output.shape
    #number of backpros steps
    back_prpg_steps=len(weights_i)

    if(x!=a):
        print("dimension error")
        print(x,a)
        return False
    
    ## training for epoch
    number_of_epoches=iterations//epoch_size
    print("All samples:",iterations,"Number of epoches",number_of_epoches,"Epoch size",epoch_size)

    ## number of neuronas in the last layer
    number_nuerons_last_layer=weights[-1].shape[0]

    ## going throw the samples
    for i in range(number_of_epoches):
        ##calculating  error
        error=np.zeros([number_nuerons_last_layer])
        #print("error reset",error,error.shape)
        ## cost function result
        cost=np.zeros([number_nuerons_last_layer])
        ## going throw epoches
        for j in range(epoch_size):
            #picking an example randomly
            index=random.randint(0, x-1)
            sample=input[index].T
            correct_answer=output[index].T
            #make the prediction

            results,z=predict(sample,weights,bias,activation_func,last_layer_activ_func)

            #calculate local cost
            local_cost=cost_calculate(results[-1],correct_answer)
            #calculate local error
            local_error=np.subtract(results[-1],correct_answer)
            # adding the cost
            cost=np.add(cost,local_cost)
            # adding the error
            error=np.add(error,local_error)




        ## end of epoch
        ##  back propagation

        error=error[None]
        for j in range(back_prpg_steps-1,-1,-1):

            ## choose the derivative function
            my_deriv=None
            if(j==back_prpg_steps-1):
                my_deriv=last_layer_activ_func_deriv
            else:
                my_deriv=deriv
            
            #calculating the fixed value 
            new_deriv=my_deriv(z[j]) 
            fixed=np.multiply(new_deriv,error)
            fixed=np.reshape(fixed,(1,fixed.shape[1]))

            

            #bias correction
            bias_correction=np.multiply(fixed,learning_rate).T
            ## calculating weights correction
            ## we clone the lines of the last layer
            # !!!! danger section may contain errors
            #last_layer_activation_expanded= np.array([results[j]]*fixed.shape[0]) 
            ## we clone the colomns of  fixed
            #fixed_expanded= np.transpose([fixed]*results[j].shape[0])
            ##
            #print("target",results[j].shape,results[j][None].shape,results[j][None].T.shape)
            #print("actual",np.asmatrix(results[j]).T.shape,fixed[None].shape)
            #print("result",np.asmatrix(results[j]).T,np.asmatrix(results[j]).T.shape)
            weights_correction=np.dot(np.asmatrix(results[j]).T,fixed).T

            

            ## elementwise multiplication
            #weights_correction=np.multiply(last_layer_activation_expanded,fixed_expanded)
            #print("old",weights_correction)

            ## with respect to the learn rate
            weights_correction=np.multiply(weights_correction,learning_rate)
            # !!! end of danger

            ## error for next layer
            ##flip the weights
            w_f=weights[j].T
            #print("T",weights[neuron_activation_index].T)

            ## we clone the lines of  fixed
            #f_f= np.transpose([fixed]*w_f.shape[0])
            #print(w_f,f_f)


            error=np.multiply(w_f.dot(fixed.T).T,learning_rate)



            ## refresh the weights
            #print(weights[j])
            #print(weights_correction)
            weights[j]=np.subtract(weights[j],weights_correction)
            weights[j]= np.asarray(weights[j])
            ##refresh bias
            bias[j]= np.subtract(bias[j],bias_correction)[-1]
            #print("weight corrections",weights_correction,weights_correction.shape)
            #print("bias corrections",bias_correction,bias_correction.shape)



    #print("component",weights,"\n bias",bias)
    return weights,bias





def test(w,b,input,output,activ,last_activ):
    ##total performance
    total=np.array([0])
    sum=0
    ##
    weights=w
    bias=b
    ##
    x,y=input.shape
    a,b=output.shape

    for i in range(x):
            #reading input and correct answer
            sub_input=input[i].T
            correct=output[i].T
            #make prediction
            results,z=predict(sub_input,weights,bias,activ,last_activ)
            #calculate cost
            local_cost=np.abs(np.subtract(np.round(results[-1]),correct))         

            ## calculate performence
            total=total+np.average(local_cost)
            sum=sum+1
            i=i+1
    return 1-(total/sum)



def test_print(w,b,input,output,activ,last_activ):
    ##total performance
    total=np.array([0])
    sum=0
    ##
    weights=w
    bias=b
    ##
    x,y=input.shape
    a,b=output.shape

    for i in range(x):
            #reading input and correct answer
            sub_input=input[i].T
            correct=output[i].T
            #make prediction
            results,z=predict(sub_input,weights,bias,activ,last_activ)
            #calculate cost
            local_cost=np.abs(np.subtract(np.round(results[-1]),correct))

            print(sub_input,np.round(results[-1]),correct)
            
            ## calculate performence
            total=total+np.average(local_cost)
            sum=sum+1
            i=i+1

            






