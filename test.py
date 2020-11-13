from main import predict,sig,Relu,Relu_deriv,get_component,cost_calculate,L_Relu,L_Relu_D
import numpy as np



def test_predict_1():
    input=np.array([8,9])
    w=[np.array([[1,2],[3,4],[5,6]])]
    b=[np.array([[1,2,3]])]
    result=predict(input,w,b,Relu,sig)
    print("correct",np.add(np.dot(w,input),b))
    print("result",result[1])

def test_cost():
    x=cost_calculate(4,3)
    print("correct")


def test_relu():
    x=np.array([1,2,3,-4,6,0])
    #print(Relu(x))
    #print(Relu_deriv(x))
    print(x)
    print(L_Relu(x))
    print(L_Relu_D(x))
test_relu()

