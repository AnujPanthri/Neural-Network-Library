import numpy as np
import matplotlib.pyplot as plt

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
