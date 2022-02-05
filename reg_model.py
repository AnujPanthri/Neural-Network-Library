import dynamic_nn as dn
import numpy as np
import matplotlib.pyplot as plt
from data import *
from helper import *
from gradient_checking import *
# from feature_scaling import *
model=dn.nn()
# model=dn.nn(use_bias="False")
model.add(dn.input_layer((1)))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
# model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
# model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
# model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
# model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
# model.add(dn.dense(5,activation='linear'))
# model.add(dn.dense(5,activation='sigmoid'))
# model.add(dn.dense(2,activation='sigmoid'))
model.add(dn.dense(1,activation='linear',last_layer=True))
# model.add(dn.dense(1,activation='relu',last_layer=True))
# model.add(dn.dense(1,activation='sigmoid',last_layer=True))
model.make_model()
model.summary()

model.loss=dn.mse()
# if training is unstable try reducing lr ,or if training slow try increasing the lr

# model.lr=0.4

model.lr=0.005 # relu
model.lr=0.00008
model.lr=0.00001

model.lr=0.00008 # relu
model.lr=0.0008 # relu


# model.lr=0.2 # sigmoid



# X=np.arange(1,20+1,1).reshape(-1,1)
# X=np.arange(1,40+1,1).reshape(-1,1)
# y=X**2
from sklearn import datasets
X,y=datasets.make_regression(n_samples=100, n_features=1,noise=4,random_state=33)

y=y.reshape(-1,1) 


# p_x=dn.data_preprocessing.polynomial(order=4)
f_s_x=dn.data_preprocessing.feature_scaling(X)
f_s_y=dn.data_preprocessing.feature_scaling(y)
X=f_s_x.transform(X)
y=f_s_y.transform(y)

# reg_show(model,X,y)
# X=f_s_x.transform_to_normal(X)
# y=f_s_y.transform_to_normal(y)
# reg_show(model,X,y)

print('X:',X.shape)
print('y:',y.shape)

x_train,y_train,x_test,y_test=split_data(X,y)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


gradient_checking(model,X,y) # gradient checking before training
model.fit(x_train,y_train,iter=500,verbose=0)
gradient_checking(model,X,y) # gradient checking after training


# lr=model.lr_finder(x_train,y_train,0.0001,2,1000)
# print("recommended lr:",lr)
# model.learning_curve(x_train,y_train,(x_test,y_test),6)


# reg_show(model,x_train,y_train)

# model.fit(x_train,y_train,iter=10000,verbose=1)
model.fit(x_train,y_train,val_data=(x_test,y_test),iter=10000,verbose=1)
model.plot_training() # plot training graph (loss vs number of iterations). if this graph is not smooth training is unstable


reg_show(model,x_train,y_train)
# reg_show(model,f_s_x.transform_to_normal(x_train),f_s_y.transform_to_normal(y_train),f_s_y)

plot=plotter(graph_name="loss",y_axis="loss")
plot.add(model.cost_hist,y_name="train_loss")
plot.add(model.val_cost_hist,y_name="val_loss")
plot.show()

# print(model.loss.forward(y_test,model.forward(x_test)))
# plt.figure()
# plt.plot(f_s_x.transform_to_normal(X)[::4],f_s_y.transform_to_normal(y)[::4])
# plt.plot(f_s_x.transform_to_normal(X)[::4],f_s_y.transform_to_normal(y)[::4],'go')
# plt.show(block=False)

# while(True):
#     test_data=np.array([[float(input("enter number:"))]])
#     # test_data=f_s_x.transform(test_data)
#     out=model.predict(f_s_x.transform(test_data))
#     out=f_s_y.transform_to_normal(out)
#     plt.plot(test_data,out,'rx')
#     plt.show(block=False)
#     print(test_data[0,0],":",out[0,0])