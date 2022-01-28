import numpy as np
import matplotlib.pyplot as plt
from layers import *
from activations import *
from losses import *
    
        

class nn:
    def __init__(self,use_bias="user"):
        self.layers=[]
        self.lr=1
        self.cost_hist=[]
        self.loss=binary_crossentropy()
        self.use_bias=use_bias
    def add(self,layer):
        self.layers.append(layer)
    def make_model(self):
        for i in range(len(self.layers)-1):
            self.layers[i].name=self.layers[i].name+"_"+str((i+1))
            out=self.layers[i].out_shape()
            if self.use_bias=="True":
                self.layers[i+1].use_bias=True
            if self.use_bias=="False":
                self.layers[i+1].use_bias=False
            self.layers[i+1].config(out)
            
        self.layers[i+1].name=self.layers[i+1].name+"_"+str((i+1+1))
    def summary(self):
        print("\n\nsummary:\n")
        for layer in self.layers:
            print(layer.name,"\ttheta shape:",layer.theta.shape,"\tbias shape:",layer.bias.shape,"\tOutput shape:",layer.out_shape())
        print("\n\n")
    def forward(self,X):
        for layer in self.layers:
            X=layer.forward(X)
        return X

    
    def backward(self,y):
        #alway call backward after calling forward
        m=y.shape[0]
        dJ_da=self.loss.backward()
        grads=dJ_da
        # print("grads:",grads.shape)
        for i in range(len(self.layers))[::-1][:len(self.layers)-1]:
            grads=self.layers[i].backward(grads)
        
        for i in range(len(self.layers))[::-1][:len(self.layers)-1]:# update
            # print(self.layers[i].name)
            self.layers[i].theta-=self.lr*self.layers[i].d_theta
            if self.layers[i].use_bias:
                self.layers[i].bias-=self.lr*self.layers[i].d_bias

    def fit(self,X,y_true,iter=10,verbose=1):
        
        for i in range(1,iter+1):
            print("epoch",i,":  ",end="")
            y_pred=self.forward(X)
            self.cost_hist.append(self.loss.forward(y_true,y_pred))
            self.backward(y_true)
            if verbose==1:
                print("Loss:{:.4f}".format(self.cost_hist[-1]))
    def predict(self,X):
        out=self.forward(X)
        return out
    def plot_training(self):
        x=np.arange(1,len(self.cost_hist)+1,1)
        y=self.cost_hist
        plt.figure()
        plt.plot(x,y)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()


if __name__=='__main__':
    model=nn()
    model.add(input_layer((2)))
    model.add(dense(3,activation='sigmoid'))
    model.add(dense(5,activation='sigmoid'))
    model.add(dense(1,activation='sigmoid',last_layer=True))
    model.make_model()
    model.summary()

    X=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1],
            ])
    y=np.array([[0],
                [1],
                [1],
                [1],
                ])

    print("X:",X.shape)
    out=model.predict(X)
    print("out:",out.shape)


    model.fit(X,y,iter=1)