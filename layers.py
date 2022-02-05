import copy
import numpy as np
from activations import *
from weight_initializers import *

class layer:
    def __init__(self):
        self.theta=np.array([])
        self.d_theta=np.array([])
        self.originaltheta=np.array([])
        self.bias=np.array([])
        self.d_bias=np.array([])
        self.originalbias=np.array([])
    def reset(self):
        self.theta=copy.deepcopy(self.originaltheta)
        self.bias=copy.deepcopy(self.originalbias)
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
        super().__init__()
        self.nodes=nodes
        self.type="input_layer"
        self.name=self.type
    def forward(self,X):
        self.a=X # m*n where n are the number of features and m is number of example
        return self.a
    def out_shape(self):
        return ('m',self.nodes)


class dense(layer):
    def __init__(self,nodes,activation='relu',last_layer=False,use_bias=True,intializer="random_initializer"):
        super().__init__()
        self.nodes=nodes
        self.activation_name=activation
        self.activation=eval(activation+'()')
        self.type="dense"
        self.name=self.type
        self.last_layer=last_layer
        self.use_bias=use_bias
        self.intializer=intializer
    def config(self,shape_of_last):
        prev=shape_of_last[-1]
        curr=self.nodes
        
        self.originaltheta=eval(self.intializer+"((prev,curr))")
        self.theta=copy.deepcopy(self.originaltheta)
        self.bias=np.array([])
        if self.use_bias:
            # self.originalbias=np.random.randn(1,curr)
            self.originalbias=np.zeros((1,curr))
            self.bias=copy.deepcopy(self.originalbias)
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
        self.da_prev=da_prev
        return da_prev