import dynamic_nn as dn
import numpy as np
import matplotlib.pyplot as plt
from data import *
from helper import *
from confusion_matrix import *
from gradient_checking import *
from data_preprocessing import *

model=dn.nn()
# model=dn.nn(use_bias="False")
# model.add(dn.input_layer((2)))
model.add(dn.input_layer((15)))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu'))
model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
# model.add(dn.dense(5,activation='sigmoid'))
# model.add(dn.dense(5,activation='sigmoid'))
# model.add(dn.dense(5,activation='sigmoid'))
# model.add(dn.dense(5,activation='sigmoid'))
model.add(dn.dense(1,activation='sigmoid',last_layer=True))
model.make_model()
model.summary()
model.compile(loss=dn.binary_crossentropy(),metrics=[acc],metrics_names=['acc'])

# if training is unstable try reducing lr ,or if training slow try increasing the lr

model.lr=2.1  # sigmoid
model.lr=0.9  # sigmoid
# model.lr=1.2  # sigmoid
# model.lr=0.4  # relu

model.lr=2  # sigmoid
model.lr=0.008  # sigmoid





''' Experiment with these data also , note you might have to adjust lr(learning rate) according to each data '''
# X,y = load_planar_dataset() # data loading
# X,y = load_moon() # data loading         
# X,y = load_circle() # data loading
# X,y = load_blob() # data loading
X,y = load_heart_disease() # data loading  

scaler=feature_scaling(X)
X=scaler.transform(X)
# X=scaler.transform_to_normal(X)

print('X:',X.shape)
print('y:',y.shape)

x_train,y_train,x_test,y_test=split_data(X,y)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

u_s=undersampling(x_train,y_train)
x_train,y_train=u_s.get_data()
undersampling(x_train,y_train)  #  just to see the new data
input('press enter to continue.....')



gradient_checking(model,X,y)

model.fit(x_train,y_train,iter=500,verbose=0)

gradient_checking(model,X,y)




# lr=model.lr_finder(x_train,y_train,0.0001,2,1000)
# model.lr=lr
# print("recommended lr:",lr)
# model.learning_curve(x_train,y_train,(x_test,y_test),6)

# show(model,X,y)
for _ in range(10):
    x_train,y_train=u_s.get_data()
    # lr=model.lr_finder(x_train,y_train,0.0001,2,1000)
    # model.lr=lr
    model.fit(x_train,y_train,val_data=(x_test,y_test),iter=5000,verbose=1)

# model.plot_training() # plot training graph (loss vs number of iterations). if this graph is not smooth training is unstable

# show(model,X,y)

plot=plotter(graph_name="loss",y_axis="loss")
plot.add(model.cost_hist,y_name="train_loss")
plot.add(model.val_cost_hist,y_name="val_loss")
plot.show()

plot=plotter(graph_name="accuracy",y_axis="acc")
plot.add(model.metrics_hist[0],y_name="train_acc")
plot.add(model.val_metrics_hist[0],y_name="val_acc")
plot.show()



y_pred_train=model.predict(x_train)
y_pred_test=model.predict(x_test)

print("Train acc:",acc(y_train,y_pred_train))
print("Test acc:",acc(y_test,y_pred_test))


y_test=y_test.astype("int")
y_pred_test=(model.predict(x_test)>=0.5)*1

print(y_test.shape,y_pred_test.shape)
con=confusion(y_test,y_pred_test,classes=2)
print("confusion_matrix:\n",con)
f1score=f1
all=f1score(con)
print("Names:",all[0])
print("Precision:",all[1])
print("Recall:",all[2])
print("F1:",all[3])
print("Support:",all[4])
