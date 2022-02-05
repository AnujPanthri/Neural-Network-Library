import numpy as np
import matplotlib.pyplot as plt
import copy

def show(model,X,y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    data_as_input=np.c_[xx.ravel(), yy.ravel()]
    zz=model.predict(data_as_input)
    zz=(zz>=0.5)*1
    zz=zz.reshape(xx.shape)

    plt.figure(figsize=(10,7))
    plt.title(f"Prediction(decision boundary)")
    plt.plot(X[(y==0)[:,0],0],X[(y==0)[:,0],1],"go")
    plt.plot(X[(y==1)[:,0],0],X[(y==1)[:,0],1],"bo")
    plt.contourf(xx, yy, zz, cmap='Paired')
    plt.xlabel('X')
    plt.ylabel('y')

    if model.use_bias=="False":
        plt.savefig("without_bias.png",dpi=200)
    if model.use_bias=="True" or model.use_bias=="user":
        plt.savefig("with_bias.png",dpi=200)
    plt.show()
def reg_show(model,X,y,y_scaler=False):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    h = 100
    xx= np.linspace(x_min, x_max, h)
    data_as_input=xx.reshape(-1,1)
    out=model.predict(data_as_input)
    out=out.reshape(-1,1)
    if y_scaler:
        out=y_scaler.transform_to_normal(out)
    plt.figure(figsize=(10,7))
    plt.title(f"Prediction(decision boundary)")
    plt.plot(X,y,"go")
    plt.plot(xx,out,"ro")
    plt.xlabel('X')
    plt.ylabel('y')

    # if model.use_bias=="False":
    #     plt.savefig("without_bias.png",dpi=200)
    # if model.use_bias=="True" or model.use_bias=="user":
    #     plt.savefig("with_bias.png",dpi=200)
    plt.show()

def split_data(X,y,split=80):
    X=copy.deepcopy(X)
    y=copy.deepcopy(y)
    rand_num=np.random.random_integers(1,1000)
    np.random.seed(rand_num)
    np.random.shuffle(X)
    np.random.seed(rand_num)
    np.random.shuffle(y)
    idx=int((split/100)*X.shape[0])
    x_train,y_train=X[:idx],y[:idx]
    x_test,y_test=X[idx:],y[idx:]
    return x_train,y_train,x_test,y_test
    

def acc(y_true,y_pred,show=False):
        y_pred=(y_pred>=0.5)*1
        total=np.sum(np.ones_like(y_true))
        match=np.sum((y_true==y_pred)*1)
        acc=(match/total)*100
        if show:
            print("acc:","{:.2f}".format(acc))
        return acc

class plotter:
    def __init__(self,graph_name="",y_axis="N/A"):
        plt.figure()
        # plt.style.use('seaborn')
        plt.title(graph_name)
        plt.xlabel("epochs")
        plt.ylabel(y_axis)
    def add(self,y,y_name="notgiven"):
        x=np.arange(1,len(y)+1,1)
        plt.plot(x,y,label=y_name)
    
        plt.ylim((0,plt.ylim()[1]))
        plt.legend(loc='best')
    def show(self):
        plt.show()


