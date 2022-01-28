import numpy as np
import copy
import matplotlib.pyplot as plt


def linear(z):
    return z
def linear_derivative(z):
    return np.ones_like(z)
def relu(z):
    out=copy.deepcopy(z)
    out[(z<=0)]=0
    return out
def relu_derivative(z):
    out=copy.deepcopy(z)
    out=(z>0)*1
    return out
def sigmoid(z):
    z=1/(1+np.exp(-z))
    return z
def sigmoid_derivative(z):
    z=sigmoid(z)*(1-sigmoid(z))
    return z
# class layer:
#     def forward(self):
#         raise NotImplementedError()
#     def out_shape(self):
#         raise NotImplementedError()
#     def __name__(self):
#         raise NotImplementedError()
    
class input_layer:
    def __init__(self,nodes):
        self.nodes=nodes
        self.name="input_layer"
        self.theta=np.array([])
    def forward(self,X):
        self.a=X.T # n*m where n are the number of features and m is number of example
        self.a=np.r_[np.ones([1,self.a.shape[-1]]),self.a]
        return self.a
    def out_shape(self):
        return (self.nodes,'m')

class dense:
    def __init__(self,nodes,activation='relu',last_layer=False):
        self.nodes=nodes
        self.activation=activation
        self.name="dense"
        self.last_layer=last_layer
    def config(self,shape_of_last):
        prev=shape_of_last[0]
        curr=self.nodes
        # np.random.seed(133)
        self.theta=np.random.rand((curr*(prev+1))).reshape((curr,prev+1))
        # print("theta:",self.theta.shape)
    def out_shape(self):
        return (self.nodes,'m')

    def forward(self,a):
        self.a_prev=a
        self.z=np.matmul(self.theta,a)
        self.a=eval(self.activation+"(self.z)")
        if not self.last_layer:
            self.a=np.r_[np.ones([1,self.a.shape[-1]]),self.a]

        return self.a
    def backward(self,grad): # grad: dL/da
        m=grad.shape[-1]
        da_dz=eval(self.activation+"_derivative(self.z)")
        if not self.last_layer:
            da_dz=np.r_[np.ones([1,self.z.shape[-1]]),da_dz]
        # print("grad",grad.shape)
        dJ_dz=grad*da_dz
        if not self.last_layer:
            dJ_dz=dJ_dz[1:,:]
        # print("dJ_dz",dJ_dz)  # correct till this line
        self.d_theta=(1/m)*np.matmul(dJ_dz,self.a_prev.T)
        # print("dtheta:",self.d_theta)
        da_prev=np.matmul(self.theta.T,dJ_dz)
        # print("da_prev:",da_prev.shape)
        return da_prev
        
class binary_crossentropy:
    def forward(self,y_true,y_pred):
        self.y_true=y_true
        self.y_pred=y_pred
        self.m=y_true.shape[-1]
        epsilon=1e-6
        J=-(1/self.m)*np.sum(y_true*np.log(y_pred+epsilon)+(1-y_true)*np.log(1-y_pred+epsilon))
        return J
    def backward(self):
        dJ=(self.y_pred-self.y_true)/(self.y_pred*(1-self.y_pred))
        return dJ
class nn:
    def __init__(self):
        self.layers=[]
        self.lr=1
        self.cost_hist=[]
        self.loss=binary_crossentropy()
    def add(self,layer):
        self.layers.append(layer)
    def make_model(self):
        for i in range(len(self.layers)-1):
            self.layers[i].name=self.layers[i].name+"_"+str((i+1))
            out=self.layers[i].out_shape()
            self.layers[i+1].config(out)
        self.layers[i+1].name=self.layers[i+1].name+"_"+str((i+1+1))
    def summary(self):
        print("\n\nsummary:\n")
        for layer in self.layers:
            print(layer.name,"\ttheta shape:",layer.theta.shape,"\tOutput shape:",layer.out_shape())
        print("\n\n")
    def forward(self,X):
        for layer in self.layers:
            X=layer.forward(X)
        return X

    
    def backward(self,y):
        #alway call backward after calling forward
        m=y.shape[-1]
        dJ_da=self.loss.backward()
        grads=dJ_da
        
        for i in range(len(self.layers))[::-1][:len(self.layers)-1]:
            grads=self.layers[i].backward(grads)
        
        for i in range(len(self.layers))[::-1][:len(self.layers)-1]:# update
            # print(self.layers[i].name)
            self.layers[i].theta-=self.lr*self.layers[i].d_theta

    def fit(self,X,y_true,iter=10):
        y_true=y_true.T
        
        for i in range(1,iter+1):
            print("epoch",i,":  ",end="")
            y_pred=self.forward(X)
            self.cost_hist.append(self.loss.forward(y_true,y_pred))
            self.backward(y_true)
            print("Loss:{:.4f}".format(self.cost_hist[-1]))
    def predict(self,X):
        out=self.forward(X)
        return out.T
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
    # model.add(dense(5,activation='sigmoid'))
    # model.add(dense(3,activation='sigmoid'))
    model.add(dense(2,activation='sigmoid'))
    model.add(dense(1,activation='sigmoid',last_layer=True))
    model.make_model()
    model.summary()
    # model.layers[0].nsdae()

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
    # model.layers[-1].theta[0,0]=-10
    # model.layers[-1].theta[0,1]=-1
    # model.layers[-1].theta[0,2]=-100
    out=model.predict(X)
    print("out:",out.shape)


    model.fit(X,y,iter=1)