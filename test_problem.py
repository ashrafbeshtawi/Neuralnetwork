import main as NN
import numpy as np

def Problem2D_Sig():
    x,y=NN.get_component([2,10,10,1],"xavier")
    k=x.copy()
    l=y.copy()
    input=[]
    output=[]
    for i in range(500):
        a= int(np.random.uniform(-1000, 1000))/1000
        b=int(np.random.uniform(-1000, 1000))/1000
        input_sub=[a,b]
        r=None
        if((a<0 and b<0 ) or (a>0 and b>0)):
            r=1
        else:
            r=0

        output_sub=[r]
        input.append(input_sub)
        output.append(output_sub)


    input=np.array(input)
    output=np.array(output)

    #NN.test(x,y,input[0:100],output[0:100],NN.sig,NN.sig)

    w,b=NN.train(input,output,k,l,NN.sig,NN.sig_deriv,NN.sig,NN.sig_deriv,1,40000,0.07)
    NN.test(w,b,input[0:100],output[0:100],NN.sig,NN.sig)


def Problem2D_Relu():
    x,y=NN.get_component([2,10,10,1],"xavier")
    k=x.copy()
    l=y.copy()
    input=[]
    output=[]
    for i in range(500):
        a= int(np.random.uniform(-1000, 1000))/1000
        b=int(np.random.uniform(-1000, 1000))/1000
        input_sub=[a,b]
        r=None
        if((a<0 and b<0 ) or (a>0 and b>0)):
            r=1
        else:
            r=0

        output_sub=[r]
        input.append(input_sub)
        output.append(output_sub)


    input=np.array(input)
    output=np.array(output)

    #NN.test(x,y,input[0:100],output[0:100],NN.Relu,NN.sig)

    w,b=NN.train(input,output,k,l,NN.Relu,NN.Relu_deriv,NN.sig,NN.sig_deriv,100,900000,0.03)
    NN.test(w,b,input[0:100],output[0:100],NN.Relu,NN.sig)


def Problem2D_LeakyRelu():
    x,y=NN.get_component([2,10,1],"xavier")
    k=x.copy()
    l=y.copy()
    input=[]
    output=[]
    for i in range(500):
        a= int(np.random.uniform(-1000, 1000))/1000
        b=int(np.random.uniform(-1000, 1000))/1000
        input_sub=[a,b]
        r=None
        if((a<0 and b<0 ) or (a>0 and b>0)):
            r=1
        else:
            r=0

        output_sub=[r]
        input.append(input_sub)
        output.append(output_sub)


    input=np.array(input)
    output=np.array(output)

    #NN.test(x,y,input[0:100],output[0:100],NN.L_Relu,NN.sig)

    w,b=NN.train(input,output,k,l,NN.L_Relu,NN.L_Relu_D,NN.L_Relu,NN.L_Relu_D,1,40000,0.008)
    NN.test(w,b,input[0:100],output[0:100],NN.L_Relu,NN.sig)

Problem2D_Sig()
#Problem2D_Relu()
Problem2D_LeakyRelu()