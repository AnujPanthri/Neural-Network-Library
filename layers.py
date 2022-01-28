import numpy as np
from activations import *

class layer:
    def out_shape(self):
        raise NotImplementedError()
    def config(self):
        raise NotImplementedError()
    def forward(self):
        raise NotImplementedError()
    def backward(self):
        raise NotImplementedError()
    
    
class input_layer(layer):
    def __init__(self,nodes):
        self.nodes=nodes
        self.type="input_layer"
        self.name=self.type
        self.theta=np.array([])
        self.bias=np.array([])
    def forward(self,X):
        self.a=X # m*n where n are the number of features and m is number of example
        return self.a
    def out_shape(self):
        return ('m',self.nodes)


class dense(layer):
    def __init__(self,nodes,activation='relu',last_layer=False,use_bias=True):
        self.nodes=nodes
        self.activation=eval(activation+'()')
        self.type="dense"
        self.name=self.type
        self.last_layer=last_layer
        self.use_bias=use_bias
    def config(self,shape_of_last):
        prev=shape_of_last[-1]
        curr=self.nodes
        self.theta=np.random.rand(prev*curr).reshape((prev,curr))
        self.bias=np.array([])
        if self.use_bias:
            self.bias=np.random.rand(curr).reshape((1,curr))
    def out_shape(self):
        return ('m',self.nodes)

    def forward(self,a):
        self.a_prev=a
        self.z=np.matmul(a,self.theta)
        if self.use_bias:
            self.z=self.z+self.bias
        self.a=self.activation.forward(self.z)
        return self.a
    def backward(self,grad): # grad: dL/da
        m=grad.shape[0]
        da_dz=self.activation.backward(self.z)
        dJ_dz=grad*da_dz # element-wise multiplication
        
        self.d_theta=(1/m)*np.matmul(self.a_prev.T,dJ_dz) # dL/dtheta
        if self.use_bias:
            self.d_bias=(1/m)*np.sum(dJ_dz,axis=0,keepdims=True) # dL/dbias
        da_prev=np.matmul(dJ_dz,self.theta.T)  # dL/da_prev
        return da_prev