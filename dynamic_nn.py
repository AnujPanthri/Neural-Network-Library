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
        
        self.z=np.matmul(self.theta,a)
        self.a=eval(self.activation+"(self.z)")
        if not self.last_layer:
            self.a=np.r_[np.ones([1,self.a.shape[-1]]),self.a]
        # self.a=relu(self.z)
        # print(self.z)
        # print(self.a)
        return self.a
class nn:
    def __init__(self):
        self.layers=[]
        self.lr=1
        self.cost_hist=[]
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
        # X=X[1:,:]
        return X

    def loss(self,y_true,y_pred):
        m=y_true.shape[-1]
        epsilon=1e-6
        J=-(1/m)*np.sum(y_true*np.log(y_pred+epsilon)+(1-y_true)*np.log(1-y_pred+epsilon))
        return J
    def backward(self,y):
        #alway call backward after calling forward
        m=y.shape[-1]
        self.layers[-1].dz=self.layers[-1].a-y  # for sigmoid and binary crossentropy loss(logistic loss)
        self.layers[-1].d_theta=(1/m)*np.matmul(self.layers[-1].dz,self.layers[-1-1].a.T) 

        for i in range(len(self.layers))[::-1][1:len(self.layers)-1]:
            # print(self.layers[i].name,self.layers[i].activation)
            g_derv=eval(self.layers[i].activation+"_derivative(self.layers[i].z)")
            g_derv=np.r_[np.ones([1,self.layers[i].z.shape[-1]]),g_derv]
            # print("theta,dz:",self.layers[i+1].theta.T.shape,self.layers[i+1].dz.shape)
            # print(np.matmul(self.layers[i+1].theta.T,self.layers[i+1].dz).shape)
            self.layers[i].dz=np.matmul(self.layers[i+1].theta.T,self.layers[i+1].dz) *g_derv
            self.layers[i].dz=self.layers[i].dz[1:,:]
            # print(f"dz{i+1}:",self.layers[i].dz.shape,f"a{(i-1)+1}:",self.layers[i-1].a.T.shape)
            self.layers[i].d_theta=(1/m)*np.matmul(self.layers[i].dz,self.layers[i-1].a.T)
            # print(f"d_theta{i+1}:",self.layers[i].d_theta.shape)
            # self.layers[i].theta-=self.lr*self.layers[i].d_theta #old

        for i in range(len(self.layers))[::-1][:len(self.layers)-1]:# update
            # print(self.layers[i].name)
            self.layers[i].theta-=self.lr*self.layers[i].d_theta

    def fit(self,X,y_true,iter=10):
        y_true=y_true.T
        
        for i in range(1,iter+1):
            print("epoch",i,":  ",end="")
            y_pred=self.forward(X)
            self.backward(y_true)
            # print("y_true:",y_true.shape)
            # print("y_pred:",y_pred.shape)
            # print("Loss:",self.loss(y_true,y_pred))
            self.cost_hist.append(self.loss(y_true,y_pred))
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


    model.fit(X,y,iter=1000)
