import dynamic_nn as dn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from data import load_planar_dataset

model=dn.nn()
model.add(dn.input_layer((2)))

# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu'))


model.add(dn.dense(5,activation='sigmoid'))
model.add(dn.dense(5,activation='sigmoid'))
model.add(dn.dense(1,activation='sigmoid',last_layer=True))
model.make_model()
model.summary()
model.lr=0.8
model.lr=2
# model.lr=0.05
# model.lr=0.5
# model.loss=dn.mse()



X,y = datasets.make_moons(n_samples=1000,shuffle=True, noise=0.09, random_state=4)
# X,y = datasets.make_blobs(n_samples=1000,n_features=20,centers=2,random_state=12)
# X,y = datasets.make_circles(n_samples=1000, noise=0.1)
y = np.reshape(y,(len(y),1))
# X,y = load_planar_dataset()
print('X:',X.shape)
print('y:',y.shape)


def show():
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    data_as_input=np.c_[xx.ravel(), yy.ravel()]
    # data_as_input=self.ready_data(data_as_input)
    # print('data_as_input',data_as_input.shape)
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
    plt.show()

show()
# model.fit(X,y,iter=5000)
model.fit(X,y,iter=10000)
# model.fit(X,y,iter=50000)
model.plot_training()
show()