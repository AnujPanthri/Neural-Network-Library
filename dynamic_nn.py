import copy
import numpy as np
import matplotlib.pyplot as plt
from layers import *
from activations import *
from losses import *
import data_preprocessing
import math
    
        

class nn:
    def __init__(self,use_bias="user"):
        self.layers=[]
        self.lr=0.01
        self.cost_hist=[]
        self.val_cost_hist=[]
        self.metrics_names=[]
        self.metrics=[]
        self.val_metrics_hist=[]
        self.metrics_hist=[]
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
    def compile(self,loss=False,metrics=False,metrics_names=False):
        if loss:
            self.loss=loss
        if metrics:
            self.metrics.extend(metrics)
            self.metrics_names.extend(metrics_names)
            for _ in range(len(self.metrics)):
                self.metrics_hist.append(list())
                self.val_metrics_hist.append(list())
    def summary(self):
        print("\n\nsummary:\n")
        for layer in self.layers:
            print(layer.name,"\ttheta shape:",layer.theta.shape,"\tbias shape:",layer.bias.shape,"\tOutput shape:",layer.out_shape())
        print("\n\n")
    def reset(self):
        for layer in self.layers:
            layer.reset()
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
        return grads
    def update_parameters(self):
        for i in range(len(self.layers))[::-1][:len(self.layers)-1]:# update
            # print(self.layers[i].name)
            self.layers[i].theta-=self.lr*self.layers[i].d_theta
            if self.layers[i].use_bias:
                self.layers[i].bias-=self.lr*self.layers[i].d_bias
    def get_gradients(self):
        d_theta_vec=np.array([])
        d_bias_vec=np.array([])
        for i in range(1,len(self.layers)):
            d_theta_vec=np.r_[d_theta_vec,self.layers[i].d_theta.ravel()]
            d_bias_vec=np.r_[d_bias_vec,self.layers[i].d_bias.ravel()]
        return d_theta_vec,d_bias_vec
    def get_parameters(self):
        theta_vec=np.array([])
        bias_vec=np.array([])
        for i in range(1,len(self.layers)):
            theta_vec=np.r_[theta_vec,self.layers[i].theta.ravel()]
            bias_vec=np.r_[bias_vec,self.layers[i].bias.ravel()]
        return theta_vec,bias_vec
    def set_parameters(self,theta_vec,bias_vec):
        t_start=0
        b_start=0
        for i in range(1,len(self.layers)):
            t_end=t_start+int(np.prod(self.layers[i].theta.shape))
            b_end=b_start+int(np.prod(self.layers[i].bias.shape))
            # print(t_start,t_end)
            # print(b_start,b_end)
            # print(self.layers[i].theta.shape,self.layers[i].bias.shape)
            self.layers[i].theta=theta_vec[t_start:t_end].reshape(self.layers[i].theta.shape)
            self.layers[i].bias=bias_vec[b_start:b_end].reshape(self.layers[i].bias.shape)
            # print(self.layers[i].theta.shape,self.layers[i].bias.shape)
            t_start=t_end
            b_start=b_end

    def fit(self,X,y_true,val_data=False,iter=10,verbose=1):
        ''' 1.Forward 
            2.Loss
            3.Backward
            4.Update
            5.Show metrics
        '''
        for i in range(1,iter+1):
            if verbose==1:
                print("epoch",i,":",end="")
            y_pred=self.forward(X)
            self.cost_hist.append(self.loss.forward(y_true,y_pred))
            self.backward(y_true)
            self.update_parameters()

            self.print_loss_and_metrics(y_true,y_pred,train=True,verbose=verbose)
            if isinstance(val_data,tuple):
                val_pred=self.forward(val_data[0])
                val_true=val_data[1]
                self.val_cost_hist.append(self.loss.forward(val_true,val_pred))
                self.print_loss_and_metrics(val_true,val_pred,train=False,verbose=verbose)
            if verbose==1:
                print()
            #callback
    def print_loss_and_metrics(self,y_true,y_pred,train=False,verbose=1):
        if verbose==1:
            if train:
                print("  Loss:{:.4f}".format(self.loss.forward(y_true,y_pred)),end='')
            else:
                print("  Val_Loss:{:.4f}".format(self.loss.forward(y_true,y_pred)),end='')
        for i,metric in enumerate(self.metrics):
            if train:
                self.metrics_hist[i].append(metric(y_true,y_pred))
                if verbose==1:
                    print("  ",self.metrics_names[i],':{:.4f}'.format(self.metrics_hist[i][-1]),end='')
            else:
                self.val_metrics_hist[i].append(metric(y_true,y_pred))
                if verbose==1:
                    print("  Val_",self.metrics_names[i],':{:.4f}'.format(self.val_metrics_hist[i][-1]),end='')
            

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
        plt.ylim((0,plt.ylim()[1]+1))
        plt.show()
    def learning_curve(self,X,y_true,val_data=False,no_of_steps=100):
        start=1
        end=X.shape[0]
        cost_hist=[]
        val_cost_hist=[]
        m_list=np.linspace(start,end,no_of_steps)
        m_list=m_list.astype('int')
        print(m_list)
        # lr_list=np.logspace(start,end,no_of_steps)
        epochs=5000
        # epochs=10000
        for m in m_list:
            sub_x,sub_y=X[:m,:],y_true[:m,:]
            print("sub set:",sub_x.shape,sub_y.shape)
            for _ in range(epochs):
                y_pred=self.forward(sub_x)
                self.loss.forward(sub_y,y_pred)
                self.backward(sub_y)
                self.update_parameters()
            # self.fit(sub_x,sub_y,iter=epochs,verbose=0)
            y_pred=self.forward(sub_x)
            cost_hist.append(self.loss.forward(sub_y,y_pred))
            if isinstance(val_data,tuple):
                val_y_pred=self.forward(val_data[0])
                val_cost_hist.append(self.loss.forward(val_data[1],val_y_pred))
            self.reset()
        self.reset()
    
        # low_grad_idx=np.argmin(grad_list)
        plt.figure()
        print(len(cost_hist))
        print(len(m_list))
        plt.plot(m_list,cost_hist,'b')
        if isinstance(val_data,tuple):
            plt.plot(m_list,val_cost_hist,'g')
        # plt.plot(lr_list[low_grad_idx],self.cost_hist[low_grad_idx],'ro')
        plt.xlabel("number of training examples m:")
        plt.ylabel("loss")
        plt.legend(["training data","val data"])
        plt.show()
        self.cost_hist=[]

    def lr_finder(self,X,y_true,start=0.001,end=10,no_of_steps=100,show=True):
        self.cost_hist=[]
        best_loss=np.inf
        original_lr=self.lr
        grad_list=[]
        
        # lr_list=np.linspace(start,end,no_of_steps)
        # lr_list=np.logspace(start,end,no_of_steps)
        lr_list=[i*0.001 for i in range(no_of_steps)]
        for lr in lr_list:
            self.lr=lr
            y_pred=self.forward(X)
            self.loss.forward(y_true,y_pred)
            grad_list.append(np.sum(self.backward(y_true)))
            self.update_parameters()
            y_pred=self.forward(X)
            self.cost_hist.append(self.loss.forward(y_true,y_pred))
            if self.cost_hist[-1]<best_loss:
                best_loss=self.cost_hist[-1]
            if self.cost_hist[-1]>best_loss:
                break
        lr_list=lr_list[:len(grad_list)]
        self.reset()
    
        low_grad_idx=np.argmin(grad_list)
        low_cost_idx=np.argmin(self.cost_hist)-10
        
        if show:
            plt.figure()
            print(len(self.cost_hist))
            print(len(grad_list))
            plt.plot(lr_list,self.cost_hist)
            plt.plot(lr_list[low_grad_idx],self.cost_hist[low_grad_idx],'ro')
            plt.plot(lr_list[low_cost_idx],self.cost_hist[low_cost_idx],'go')
            plt.xlabel("lr")
            plt.ylabel("loss")
            plt.show()
        self.lr=original_lr
        self.cost_hist=[]
        # return lr_list[low_grad_idx]
        return lr_list[low_cost_idx],lr_list[low_grad_idx]


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


    model.fit(X,y,iter=10)