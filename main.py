import numpy as np
import random
from matplotlib import pyplot as plt

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



## init
def get_component(layers,init_type):
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
        elif init_type=="random":
            layer=np.multiply(np.random.rand(m,n),np.sqrt(2/m)) 

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

## sigmoid activion
def sig(num):
    if -np.min(num) > np.log(np.finfo(type(1.1)).max):
        return np.zeros(num.shape)  
    return np.divide(1,np.add(1,np.exp(-num)))

## derviative of sigmoid
def sig_deriv(num):
    res=sig(num)
    return np.multiply(res,np.subtract(1,res))

## RelU
def Relu(num):
    return np.where(num>0,num,0)


## Relu derivative
def Relu_deriv(num):
    num=np.where(num>0,num,0)
    num=np.where(num==0,num,1)

    return num

## leaky RelU
def L_Relu(num):
    return np.where(num>0,num,0.01*num)

## leaky RelU deriv
def L_Relu_D(num):
    num=np.where(num<=0,num,1)
    num=np.where(num>=0,num,0.01)

    return num




#predict : makre prediction and return activation of each layer
def predict(input,weights,bias,activation_func,last_layer_activ_func):
    activation=input
    z=[]
    layer_activation=[input]
    for i in range(len(weights)):
        #print(np.dot(weights[i],activation),bias[i])
        #print(np.add(np.dot(weights[i],activation),bias[i]))
        z_temp=np.add(np.dot(weights[i],activation),bias[i])


        if(i==len(weights)-1):
            activation=last_layer_activ_func(z_temp)
        else:
            activation=activation_func(z_temp)


        z.append(z_temp)
        layer_activation.append(activation)
    return layer_activation,z


# cost function
def cost_calculate(prediction,correct):
    return np.divide(np.power(np.subtract(correct,prediction),2),2)



def train(input,output,weights_i,bias_i,activation_func,deriv,last_layer_activ_func,last_layer_activ_func_deriv,epoch_size,iterations,learning_rate):
    ##total performance
    total=np.array([0])
    sum=0
    ##
    weights=weights_i
    bias=bias_i
    ##
    x,y=input.shape
    a,b=output.shape
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

        ## error of the last layer
        #print("error",error,error.shape)
        #print("new back probgation")
        error=error[None]
        for j in range(back_prpg_steps-1,-1,-1):

            #print("error",error,error.shape)
            #print("weights",weights[j],weights[j].shape)
            #print("bias",bias[j],bias[j].shape)



            ## choose the derivative function
            my_deriv=None
            if(j==back_prpg_steps-1):
                my_deriv=last_layer_activ_func_deriv
            else:
                my_deriv=deriv
            
            #calculating the fixed value 
            ## fixed= activation'(z).error
            new_deriv=my_deriv(z[j]) #np.reshape(my_deriv(z[j]),-1)
            #error=np.reshape(error,-1)
            fixed=np.multiply(new_deriv,error)
            #print("fixed old",fixed,fixed.shape)
            fixed=np.reshape(fixed,(1,fixed.shape[1]))
            #print("fixed new",fixed,fixed.shape)
            

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
"""
    ## training for epoch
    i=0
    cost=None
    for j in range(x//epoch):

        ##training for single input
        for k in range(epoch):

            #reading input and correct answer
            sub_input=input[i].T
            correct=output[i].T
            #make prediction
            results,z=predict(sub_input,weights,bias,activation_func,last_layer_activ_func)
            #calculate cost
            local_cost=cost_calculate(np.round(results[-1]),correct)

            if cost==None:
                cost=local_cost
            else:
                cost=np.add(cost,local_cost)
            ## calculate performence
            total=total+np.average(local_cost)
            sum=sum+1
            i=i+1
            #print("####################################################")
            #print("Cost: ",local_cost,"input ",sub_input, "correct ",correct, " result", results[-1],"performance",1-total/sum)
        cost=None

        #back propagation
        correct_for_this_layer=correct
        neuron_activation_index=-1
        learning_rate=0.1


        for j in range(back_prpg_steps-1,-1,-1):
            
            ## correcting bias
            ## fixed values   
            ## fixed= sig'(z).2(a(L)-y)

           # print("sig",sig_deriv(z[neuron_activation_index]),z[neuron_activation_index])
            if(j==back_prpg_steps-1):
                my_deriv=last_layer_activ_func_deriv
            else:
                my_deriv=deriv


            fixed=np.multiply(my_deriv(z[neuron_activation_index]),np.subtract(results[neuron_activation_index],correct_for_this_layer))
            bias_correction=np.multiply(fixed,learning_rate)
            #print("fixed",fixed)
            #print("deriv",my_deriv(z[neuron_activation_index]))


            ## correcting weights
            ## correction= a(L-1).fixed

            ## we clone the lines of the last layer
            last_layer_activation_expanded= np.array([results[neuron_activation_index-1]]*fixed.shape[0]) 
            ## we clone the colomns of  fixed
            fixed_expanded= np.transpose([fixed]*results[neuron_activation_index-1].shape[0])
            ## elementwise multiplication
            weights_correction=np.multiply(last_layer_activation_expanded,fixed_expanded)

            
            #print("last_layer_activation_expanded",last_layer_activation_expanded)
            #print("fixed_expanded",fixed_expanded)
            #print("weights_correction",weights_correction)
            #print("bias_correction",bias_correction)
            
            ## with respect to the learn rate
            weights_correction=np.multiply(weights_correction,learning_rate)




            ## cost for next layer
            ## correct= w(L).fixed0

            ##flip the weights
            w_f=weights[neuron_activation_index].T
            #print("T",weights[neuron_activation_index].T)

            ## we clone the lines of  fixed
            f_f= np.transpose([fixed]*w_f.shape[0])



            ## refresh the weights
            weights[neuron_activation_index]=np.subtract(weights[neuron_activation_index],weights_correction)
            ##refresh bias
            bias[neuron_activation_index]= np.subtract(bias[neuron_activation_index],bias_correction)
            
            ##calculate cost for the next layer 
            cost_next=w_f.dot(f_f)
            correct_for_this_layer=np.multiply(cost_next.diagonal(),learning_rate)
            neuron_activation_index=neuron_activation_index-1

    return weights,bias
"""




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
            local_cost=cost_calculate(np.round(results[-1]),correct)
            ## plot ... to be removed later
            #print(sub_input,np.round(results[-1]),correct)
            
            c=None
            if np.round(results[-1])==1:
                c="red"
            else:
                c="blue"
            plt.scatter(sub_input[0],sub_input[1],color=c)

            ## calculate performence
            total=total+np.average(local_cost)
            sum=sum+1
            i=i+1
    plt.show()


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
            local_cost=cost_calculate(np.round(results[-1]),correct)

            print(sub_input,np.round(results[-1]),correct)
            


            ## calculate performence
            total=total+np.average(local_cost)
            sum=sum+1
            i=i+1

            

            
    









"""
x,y=get_component([3,10,10,1],"xavier")
k=x.copy()
l=y.copy()

#x=np.multiply(sig_deriv(5.1),np.multiply(2,np.subtract(0.99704446,0)))

#print(sig_deriv(5.1))
#exit()


input=[]
output=[]
for i in range(500):
    a=random.randint(0, 1)# int(np.random.uniform(-1000, 1000))
    b=random.randint(0, 1)#int(np.random.uniform(-1000, 1000))
    c=random.randint(0, 1)#int(np.random.uniform(-1000, 1000))
    input_sub=[a,b,c]
    r=None
    if((a<0 and b<0 ) or (a>0 and b>0)):
        #print("1 point",a,b)
        r=1
    else:
        r=0
                # xor
    output_sub=[(a and not b and not c) or (b and not a and not c) or (c and not a and not b) or (a and b and c)]
    input.append(input_sub)
    output.append(output_sub)


input=np.array(input)
output=np.array(output)

test_print(x,y,input[0:10],output[0:10],Relu,sig)

#print(x,y)
#w,b=train(input,output,x,y,L_Relu,L_Relu_D,sig,sig_deriv,1,9000,0.07)
#test(w,b,input[0:100],output[0:100],Relu,sig)

w,b=train(input,output,k,l,sig,sig_deriv,sig,sig_deriv,1,90000,0.07)
test_print(w,b,input[0:30],output[0:30],sig,sig)





"""






