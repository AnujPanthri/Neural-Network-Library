import numpy as np

class loss:
    def forward(self):
        raise NotImplementedError()
    def backward(self):
        raise NotImplementedError()

class binary_crossentropy(loss):
    def forward(self,y_true,y_pred):
        self.y_true=y_true
        self.y_pred=y_pred
        self.m=y_true.shape[0]
        epsilon=1e-6
        J=-(1/self.m)*np.sum(y_true*np.log(y_pred+epsilon)+(1-y_true)*np.log(1-y_pred+epsilon))
        return J
    def backward(self):
        dJ=(self.y_pred-self.y_true)/(self.y_pred*(1-self.y_pred))
        return dJ  # dJ/da


class mse(loss):
    def forward(self,y_true,y_pred):
        self.y_true=y_true
        self.y_pred=y_pred
        self.m=y_true.shape[0]
        epsilon=1e-6
        J=(1/(2*self.m))*np.sum((y_pred-y_true)**2)
        return J
    def backward(self):
        dJ=(self.y_pred-self.y_true)
        return dJ  # dJ/da