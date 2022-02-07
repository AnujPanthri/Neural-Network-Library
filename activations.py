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


class softmax(activation):
    def forward(self,z):
        # z=np.exp(z)/np.sum(np.exp(z) ,axis=1, keepdims=True)
        z=np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)) ,axis=1, keepdims=True)
        return z
        # e = np.exp(z-np.max(z))
        # s = np.sum(e, axis=1, keepdims=True)
        # return e/s
    def backward(self,z):
        softmax=self.forward(z)
        same=softmax*(1.-softmax) # for i==j
        not_same=-(softmax[:,:,None]*softmax[:,None,:]) # for i!=j
        # print(not_same.shape)
        same_idx=np.arange(0,not_same.shape[-1],1)
        not_same[:,same_idx,same_idx]=same

        # return same
        return not_same