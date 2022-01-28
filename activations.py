import numpy as np
import copy

class activation:
    def forward(self):
        raise NotImplementedError()
    def backward(self):
        raise NotImplementedError()

class linear(activation):
    def forward(self,z):
        return z
    def backward(self,z):
        return np.ones_like(z)

class relu(activation):      
    def forward(self,z):
        out=copy.deepcopy(z)
        out[(z<=0)]=0
        return out
    def backward(self,z):
        out=copy.deepcopy(z)
        out=(z>0)*1
        return out

class sigmoid(activation):
    def forward(self,z):
        z=1/(1+np.exp(-z))
        return z
    def backward(self,z):
        z=self.forward(z)*(1-self.forward(z))
        return z