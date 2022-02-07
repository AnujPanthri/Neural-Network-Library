import dynamic_nn as dn
import numpy as np
import matplotlib.pyplot as plt
from data import *
from helper import *
from confusion_matrix import *
from gradient_checking import *

model=dn.nn()
# model=dn.nn(use_bias="False")
model.add(dn.input_layer((64)))
# model.add(dn.dense(5,activation='relu'))
# model.add(dn.dense(5,activation='relu',intializer="he_initializer"))
# model.add(dn.dense(5,activation='sigmoid'))
model.add(dn.dense(20,activation='sigmoid'))
model.add(dn.dense(20,activation='sigmoid'))
# model.add(dn.dense(5,activation='sigmoid'))
# model.add(dn.dense(5,activation='sigmoid'))
model.add(dn.dense(10,activation='softmax',last_layer=True))
# model.add(dn.dense(2,activation='sigmoid',last_layer=True))
model.make_model()
model.summary()
# model.compile(loss=dn.binary_crossentropy(),metrics=[acc],metrics_names=['acc'])
model.compile(loss=dn.categorical_crossentropy(),metrics=[categorical_acc],metrics_names=['acc'])
# model.compile(loss=dn.categorical_crossentropy())

# if training is unstable try reducing lr ,or if training slow try increasing the lr

# model.lr=2.1  # sigmoid
model.lr=0.3  # relu
# model.lr=0.2  # relu
# model.lr=0.001  # relu


# model.lr=0.0000000001  # relu





''' Experiment with these data also , note you might have to adjust lr(learning rate) according to each data '''
# X,y = load_planar_dataset() # data loading
# X,y = load_moon() # data loading         
# X,y = load_circle() # data loading
# X,y = load_blob() # data loading
from sklearn import datasets
X,y=datasets.load_digits(return_X_y=True)
y=y.reshape(-1,1)

y=dn.data_preprocessing.one_hot(y,classes=10)
print('X:',X.shape)
print('y:',y.shape)

# print(y[-5:,:])


x_train,y_train,x_test,y_test=split_data(X,y)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# y_pred=model.predict(x_test)
# print("loss:",model.loss.forward(y_test,y_pred))

# # print("grads:",get_grads(model,X,y))
# gradient_checking(model,X,y) # gradient checking before training
# model.fit(x_train,y_train,iter=500,verbose=0)
# gradient_checking(model,X,y) # gradient checking after training

# print("X:",X.min(),X.max())
# input("press enter to continue....")
model.learning_curve(x_train,y_train,(x_test,y_test),6)
model.fit(x_train,y_train,val_data=(x_test,y_test),iter=5000,verbose=1)
# model.plot_training() # plot training graph (loss vs number of iterations). if this graph is not smooth training is unstable

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
y_test=np.argmax(y_test,axis=-1)[...,None]
y_pred_test=(model.predict(x_test)>=0.5)*1
y_pred_test=np.argmax(y_pred_test,axis=-1)[...,None]

print(y_test.shape,y_pred_test.shape)
con=confusion(y_test,y_pred_test,classes=10)
print("confusion_matrix:\n",con)
f1score=f1
all=f1score(con)
print("Names:",all[0])
print("Precision:",all[1])
print("Recall:",all[2])
print("F1:",all[3])
print("Support:",all[4])






as_image=x_test.reshape([-1,8,8])
out=y_pred_test
# print(as_image)
fig=plt.figure()
ex=50
col=8
import math
plt.title("Predictions")
plt.axis('off')
for i in range(ex):
    fig.add_subplot(math.ceil(ex/col),col,(i+1))
    plt.title(f"{out[i,0]}")
    plt.imshow(as_image[i,:],cmap='gray')
    plt.axis('off')
fig.subplots_adjust(hspace=0.433)
fig.tight_layout()
plt.show()
